import torch
import torch.nn as nn
import pytorch_lightning as pl

from polarbev.config import get_cfg
from polarbev.models.polarbev import PolarBEV
from polarbev.losses import SpatialRegressionLoss, SegmentationLoss
from polarbev.metrics import IntersectionOverUnion, PanopticMetric
from polarbev.utils.instance import predict_instance_segmentation_and_trajectories
from polarbev.utils.visualisation import visualise_output


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        # see config.py for detailss
        self.hparams = hparams
        # pytorch lightning does not support saving YACS CfgNone
        cfg = get_cfg(cfg_dict=self.hparams)
        self.cfg = cfg
        self.n_classes = len(self.cfg.SEMANTIC_SEG.WEIGHTS)

        # Bird's-eye view extent in meters
        assert self.cfg.LIFT.X_BOUND[1] > 0 and self.cfg.LIFT.Y_BOUND[1] > 0
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        # Model
        self.model = PolarBEV(cfg)

        # Losses
        self.losses_fn = nn.ModuleDict()
        self.losses_fn['segmentation'] = SegmentationLoss(
            class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.WEIGHTS),
            use_top_k=self.cfg.SEMANTIC_SEG.USE_TOP_K,
            top_k_ratio=self.cfg.SEMANTIC_SEG.TOP_K_RATIO,
            future_discount=self.cfg.FUTURE_DISCOUNT,
        )
        self.losses_fn['instance_center'] = SpatialRegressionLoss(
            norm=2, future_discount=self.cfg.FUTURE_DISCOUNT
        )
        self.losses_fn['instance_offset'] = SpatialRegressionLoss(
            norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
        )

        # Uncertainty weighting
        self.model.segmentation_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.model.centerness_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.model.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.metric_iou_val = IntersectionOverUnion(self.n_classes)
        self.metric_panoptic_val = PanopticMetric(n_classes=self.n_classes)

        self.training_step_count = 0

    def shared_step(self, batch, is_train):
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        lidar2imgs = batch['lidar2imgs']
        
        output, inter_seg, inter_instance_offset, inter_instance_center = self.model(
            image, intrinsics, extrinsics, lidar2imgs
        )

        loss = {}
        loss['segmentation'] = 0.0
        loss['instance_center'] = 0.0
        loss['instance_offset'] = 0.0
        for i in range(len(inter_seg) - 1):
            loss['segmentation'] += 0.5 * self.losses_fn['segmentation'](
                inter_seg[i], batch['segmentation'])
            loss['instance_center'] += 0.5 * self.losses_fn['instance_center'](
                inter_instance_center[i], batch['centerness'])
            loss['instance_offset'] += 0.5 * self.losses_fn['instance_offset'](
                inter_instance_offset[i], batch['offset'])
            
        segmentation_factor = 1 / torch.exp(self.model.segmentation_weight)
        loss['segmentation'] = segmentation_factor * (loss['segmentation'] + self.losses_fn['segmentation'](
            output['segmentation'], batch['segmentation']))

        centerness_factor = 1 / (2*torch.exp(self.model.centerness_weight))
        loss['instance_center'] = centerness_factor * (loss['instance_center'] + self.losses_fn['instance_center'](
            output['instance_center'], batch['centerness']))

        offset_factor = 1 / (2*torch.exp(self.model.offset_weight))
        loss['instance_offset'] = offset_factor * (loss['instance_offset'] + self.losses_fn['instance_offset'](
            output['instance_offset'], batch['offset']))

        loss['segmentation_uncertainty'] = 0.5 * self.model.segmentation_weight
        loss['centerness_uncertainty'] = 0.5 * self.model.centerness_weight
        loss['offset_uncertainty'] = 0.5 * self.model.offset_weight

        # Metrics
        if not is_train:
            seg_prediction = output['segmentation'].detach()
            seg_prediction = torch.argmax(seg_prediction, dim=2, keepdims=True)
            self.metric_iou_val(seg_prediction, batch['segmentation'])

            pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                output, compute_matched_centers=False
            )

            self.metric_panoptic_val(pred_consistent_instance_seg, batch['instance'])

        return output, batch, loss

    def visualise(self, labels, output, batch_idx, prefix='train'):
        visualisation_video = visualise_output(labels, output, self.cfg)
        name = f'{prefix}_outputs'
        if prefix == 'val':
            name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(name, visualisation_video, global_step=self.training_step_count, fps=2)

    def training_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, True)
        self.training_step_count += 1
        for key, value in loss.items():
            self.logger.experiment.add_scalar(key, value, global_step=self.training_step_count)
        if self.training_step_count % self.cfg.VIS_INTERVAL == 0:
            self.visualise(labels, output, batch_idx, prefix='train')
        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, False)
        for key, value in loss.items():
            self.log('val_' + key, value)

        if batch_idx == 0:
            self.visualise(labels, output, batch_idx, prefix='val')

    def shared_epoch_end(self, step_outputs, is_train):
        # log per class iou metrics
        class_names = ['background', 'dynamic']
        if not is_train:
            scores = self.metric_iou_val.compute()
            for key, value in zip(class_names, scores):
                self.logger.experiment.add_scalar('val_iou_' + key, value, global_step=self.training_step_count)
            self.metric_iou_val.reset()

        if not is_train:
            scores = self.metric_panoptic_val.compute()
            for key, value in scores.items():
                for instance_name, score in zip(['background', 'vehicles'], value):
                    if instance_name != 'background':
                        self.logger.experiment.add_scalar(f'val_{key}_{instance_name}', score.item(),
                                                          global_step=self.training_step_count)
            self.metric_panoptic_val.reset()

        self.logger.experiment.add_scalar('segmentation_weight',
                                          1 / (torch.exp(self.model.segmentation_weight)),
                                          global_step=self.training_step_count)
        self.logger.experiment.add_scalar('centerness_weight',
                                          1 / (2 * torch.exp(self.model.centerness_weight)),
                                          global_step=self.training_step_count)
        self.logger.experiment.add_scalar('offset_weight', 1 / (2 * torch.exp(self.model.offset_weight)),
                                          global_step=self.training_step_count)
        if self.cfg.INSTANCE_FLOW.ENABLED:
            self.logger.experiment.add_scalar('flow_weight', 1 / (2 * torch.exp(self.model.flow_weight)),
                                              global_step=self.training_step_count)

    def training_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, True)

    def validation_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, False)

    def configure_optimizers(self):
        params = self.model.parameters()

        optimizer = torch.optim.AdamW(
            params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.SCHEDULER.MAX_LR, total_steps=self.cfg.SCHEDULER.TOTAL_STEPS, pct_start=self.cfg.SCHEDULER.PCT_START,\
            cycle_momentum=self.cfg.SCHEDULER.CYCLE_MOMENTUM, div_factor=self.cfg.SCHEDULER.DIV_FACTOR, final_div_factor=self.cfg.SCHEDULER.FINAL_DIV_FACTOR)


        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
