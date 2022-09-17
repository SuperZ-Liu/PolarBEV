from argparse import ArgumentParser

import torch
from tqdm import tqdm

from polarbev.data import prepare_dataloaders
from polarbev.trainer import TrainingModule
from polarbev.metrics import IntersectionOverUnion, PanopticMetric
from polarbev.utils.network import preprocess_batch
from polarbev.utils.instance import predict_instance_segmentation_and_trajectories

# 30mx30m, 100mx100m
EVALUATION_RANGES = {'30x30': (70, 130),
                     '100x100': (0, 200)
                     }


def eval(checkpoint_path, dataroot, version):
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1

    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version

    _, valloader = prepare_dataloaders(cfg)

    panoptic_metrics = {}
    iou_metrics = {}
    n_classes = len(cfg.SEMANTIC_SEG.WEIGHTS)
    for key in EVALUATION_RANGES.keys():
        panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).to(
            device)
        iou_metrics[key] = IntersectionOverUnion(n_classes).to(device)

    for i, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        lidar2imgs = batch['lidar2imgs']

        with torch.no_grad():
            # Evaluate with mean prediction
            output, inter_seg, inter_instance_offset, inter_instance_center = model(image, intrinsics, extrinsics, lidar2imgs)

        # Consistent instance seg
        pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            output, compute_matched_centers=False, make_consistent=True
        )

        segmentation_pred = output['segmentation'].detach()
        segmentation_pred = torch.argmax(segmentation_pred, dim=2, keepdims=True)

       
        for i, (key, grid) in enumerate(EVALUATION_RANGES.items()):
           
                limits = slice(grid[0], grid[1])
                panoptic_metrics[key](pred_consistent_instance_seg[..., limits, limits].contiguous().detach(),
                                      batch['instance'][..., limits, limits].contiguous()
                                      )


                iou_metrics[key](segmentation_pred[..., limits, limits].contiguous(),
                                batch['segmentation'][..., limits, limits].contiguous()
                                )


    results = {}
    for key, grid in EVALUATION_RANGES.items():
        panoptic_scores = panoptic_metrics[key].compute()
        for panoptic_key, value in panoptic_scores.items():
            results[f'{panoptic_key}'] = results.get(f'{panoptic_key}', []) + [100 * value[1].item()]

        iou_scores = iou_metrics[key].compute()
        results['iou'] = results.get('iou', []) + [100 * iou_scores[1].item()]

    for panoptic_key in ['iou']:
        print(panoptic_key)
        print(' & '.join([f'{x:.1f}' for x in results[panoptic_key]]))


if __name__ == '__main__':
    parser = ArgumentParser(description='Fiery evaluation')
    parser.add_argument('--checkpoint', default='./fiery.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='./nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')

    args = parser.parse_args()

    eval(args.checkpoint, args.dataroot, args.version)
