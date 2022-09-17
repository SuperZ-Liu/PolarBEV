import torch
import torch.nn as nn
import torch.nn.functional as F
from polarbev.models.img2bev import Camera2BEV
from polarbev.models.encoder import Encoder
from polarbev.models.decoder import RingDecoder
from polarbev.utils.network import pack_sequence_dim, unpack_sequence_dim, set_bn_momentum
from polarbev.utils.geometry import calculate_birds_eye_view_parameters
import numpy as np


def cart2polar(input_xy):
    rho = torch.sqrt(input_xy[..., 0] ** 2 + input_xy[..., 1] ** 2)
    phi = torch.atan2(input_xy[..., 1], input_xy[..., 0])
    return torch.stack((rho, phi), dim=2)

def polar2cat(input_xy_polar):
    x = input_xy_polar[..., 0] * torch.cos(input_xy_polar[..., 1])
    y = input_xy_polar[..., 0] * torch.sin(input_xy_polar[..., 1])
    return torch.cat((x, y), dim=-1)

class PolarBEV(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.encoder_downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE
        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS

        self.hq_channels = self.encoder_out_channels

        eps = 1e-6
        x = torch.arange(bev_start_position[0], self.cfg.LIFT.X_BOUND[1] + eps, self.cfg.LIFT.X_BOUND[2])
        y = torch.arange(bev_start_position[1], self.cfg.LIFT.Y_BOUND[1] + eps, self.cfg.LIFT.Y_BOUND[2])
        yy, xx = torch.meshgrid([y, x])
        xy_cord = torch.stack([xx, yy], dim=-1)
        self.polar_cord = cart2polar(xy_cord)
        
        
        polar_grid = []
        polar_r = torch.sqrt(self.bev_start_position[0] ** 2 + self.bev_start_position[1] ** 2)
        self.polar_r = polar_r
        polar_r_resolution = torch.sqrt(self.bev_resolution[0] ** 2 + self.bev_resolution[1] ** 2)

        self.polar_r_resolution = polar_r_resolution
        polar_grid.append((polar_r + polar_r_resolution) // polar_r_resolution)
        polar_grid.append(self.bev_dimension[0] + self.bev_dimension[1])
        # polar_grid: [R_dimention, C_dimention]
        # polar_grid.append(100)
        # polar_grid.append(400)
        self.polar_grid = polar_grid
        
        intervals = [(polar_r + polar_r_resolution) / polar_grid[0]]
        intervals.append(2 * np.pi / polar_grid[1])
        self.intervals = intervals

        rho = torch.linspace(0., 1., int(polar_grid[0].item() + 1))
        phi = torch.linspace(0., 1., int(polar_grid[1].item() + 1))
        
        phi, rho = torch.meshgrid(phi[..., :-1], rho[..., :-1])
        
        # inital BEV surface
        zz = torch.ones([int(polar_grid[1].item()), int(polar_grid[0].item())]) * 0.55
        pc_range = [polar_r + polar_r_resolution, 2*np.pi, self.cfg.LIFT.Z_BOUND[0], self.cfg.LIFT.Z_BOUND[1]]

        self.hq_number = int(polar_grid[0].item() * polar_grid[1].item())
        self.img2bev = Camera2BEV(int(polar_grid[1].item()), int(polar_grid[0].item()), self.hq_number, self.hq_channels, n_layers=3, n_cameras=6, n_levels=1, rho=rho, phi=phi, zz=zz, intervals=intervals, pc_range=pc_range, img_size=(self.cfg.IMAGE.FINAL_DIM[1], self.cfg.IMAGE.FINAL_DIM[0]), n_classes=len(self.cfg.SEMANTIC_SEG.WEIGHTS))
        
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD
       
        if self.cfg.MODEL.SUBSAMPLE:
            assert self.cfg.DATASET.NAME == 'lyft'
            self.receptive_field = 3
            self.n_future = 5

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        # Encoder
        self.encoder = Encoder(cfg=self.cfg.MODEL.ENCODER)

        self.decoder = RingDecoder(
            in_channels=self.hq_channels,
            n_classes=len(self.cfg.SEMANTIC_SEG.WEIGHTS),
            predict_future_flow=self.cfg.INSTANCE_FLOW.ENABLED,
        )

        set_bn_momentum(self, self.cfg.MODEL.BN_MOMENTUM)


    def forward(self, image, intrinsics, extrinsics, lidar2imgs):
        # start = time.time()
        output = {}

        # Only process features from the past and present
        image = image[:, :self.receptive_field].contiguous()
        intrinsics = intrinsics[:, :self.receptive_field].contiguous()
        extrinsics = extrinsics[:, :self.receptive_field].contiguous()
        lidar2imgs = lidar2imgs[:, :self.receptive_field].contiguous()
        
        x, inter_seg, inter_instance_offset, inter_instance_center, xy_cord = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics, lidar2imgs)

        states = x

        bev_output = self.decoder(states[:, -1:])

        # resample to cart
        b, s, c, h, w = bev_output['segmentation'].shape
        bev_output['segmentation'] = pack_sequence_dim(bev_output['segmentation'])
        bev_output['segmentation'] = F.pad(bev_output['segmentation'], (0, 0, int(self.polar_grid[1].item())//2, int(self.polar_grid[1].item()) - int(self.polar_grid[1].item())//2), mode='circular')
        bev_output['segmentation'] = F.grid_sample(bev_output['segmentation'], xy_cord)
        bev_output['segmentation'] = bev_output['segmentation'].view(b*s, c, self.bev_dimension[0].item(), self.bev_dimension[1].item()).permute(0, 1, 3, 2)
        bev_output['segmentation'] = unpack_sequence_dim(bev_output['segmentation'], b, s)
        b, s, c, h, w = bev_output['instance_center'].shape
        bev_output['instance_center'] = pack_sequence_dim(bev_output['instance_center'])
        bev_output['instance_center'] = F.pad(bev_output['instance_center'], (0, 0, int(self.polar_grid[1].item())//2, int(self.polar_grid[1].item()) - int(self.polar_grid[1].item())//2), mode='circular')
        bev_output['instance_center'] = F.grid_sample(bev_output['instance_center'], xy_cord)
        bev_output['instance_center'] = bev_output['instance_center'].view(b*s, c, self.bev_dimension[0].item(), self.bev_dimension[1].item()).permute(0, 1, 3, 2)
        bev_output['instance_center'] = unpack_sequence_dim(bev_output['instance_center'], b, s)
        b, s, c, h, w = bev_output['instance_offset'].shape
        bev_output['instance_offset'] = pack_sequence_dim(bev_output['instance_offset'])
        bev_output['instance_offset'] = F.pad(bev_output['instance_offset'], (0, 0, int(self.polar_grid[1].item())//2, int(self.polar_grid[1].item()) - int(self.polar_grid[1].item())//2), mode='circular')
        bev_output['instance_offset'] = F.grid_sample(bev_output['instance_offset'], xy_cord)
        bev_output['instance_offset'] = bev_output['instance_offset'].view(b*s, c, self.bev_dimension[0].item(), self.bev_dimension[1].item()).permute(0, 1, 3, 2)
        bev_output['instance_offset'] = unpack_sequence_dim(bev_output['instance_offset'], b, s)
        output = {**output, **bev_output}
        
        return output, inter_seg, inter_instance_offset, inter_instance_center

    def encoder_forward(self, x):
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x = self.encoder(x)
        x = x.view(b, n, *x.shape[1:])
        # [batch, n_cameras, height, weight, channels]
        x = x.permute(0, 1, 3, 4, 2)

        return x

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics, lidar2imgs):
        b, s, n, c, h, w = x.shape
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)
        lidar2imgs = pack_sequence_dim(lidar2imgs)

        x = self.encoder_forward(x)
        
        x_cord = self.polar_cord[..., 0:1] / (self.polar_r + self.polar_r_resolution)
        x_cord = (x_cord - 0.5) * 2

        x_cord = x_cord.unsqueeze(0).repeat(b*s, 1, 1, 1).view(b*s, -1, 1)

        y_cord = self.polar_cord[..., 1:2] / (np.pi) / 2
        y_cord = y_cord.unsqueeze(0).repeat(b*s, 1, 1, 1).view(b*s, -1, 1)
        xy_cord = torch.cat((x_cord, y_cord), dim=-1)
        xy_cord = xy_cord.view(b*s, -1, 1, 2)
        xy_cord = xy_cord.to(x)
        x, inter_seg, inter_instance_offset, inter_instance_center, reg = self.img2bev(x, lidar2imgs)
        
        x = x.permute(1, 2, 0).view(b * s, -1, int(self.polar_grid[1].item()), int(self.polar_grid[0].item()))


        for i in range(len(inter_seg) - 1):
            inter_seg[i] = inter_seg[i].permute(1, 2, 0).view(b*s, -1, int(self.polar_grid[1].item()), int(self.polar_grid[0].item()))
            inter_seg[i] = F.pad(inter_seg[i], (0, 0, int(self.polar_grid[1].item())//2, int(self.polar_grid[1].item()) - int(self.polar_grid[1].item())//2), mode='circular')
            inter_seg[i] = F.grid_sample(inter_seg[i], xy_cord)
            inter_seg[i] = inter_seg[i].view(b, s, -1, self.bev_dimension[0].item(), self.bev_dimension[1].item()).permute(0, 1, 2, 4, 3)
            
            inter_instance_center[i] = inter_instance_center[i].permute(1, 2, 0).view(b*s, -1, int(self.polar_grid[1].item()), int(self.polar_grid[0].item()))
            inter_instance_center[i] = F.pad(inter_instance_center[i], (0, 0, int(self.polar_grid[1].item())//2, int(self.polar_grid[1].item()) - int(self.polar_grid[1].item())//2), mode='circular')
            inter_instance_center[i] = F.grid_sample(inter_instance_center[i], xy_cord)
            inter_instance_center[i] = inter_instance_center[i].view(b, s, -1, self.bev_dimension[0].item(), self.bev_dimension[1].item()).permute(0, 1, 2, 4, 3)
            
            inter_instance_offset[i] = inter_instance_offset[i].permute(1, 2, 0).view(b*s, -1, int(self.polar_grid[1].item()), int(self.polar_grid[0].item()))
            inter_instance_offset[i] = F.pad(inter_instance_offset[i], (0, 0, int(self.polar_grid[1].item())//2, int(self.polar_grid[1].item()) - int(self.polar_grid[1].item())//2), mode='circular')
            inter_instance_offset[i] = F.grid_sample(inter_instance_offset[i], xy_cord)
            inter_instance_offset[i] = inter_instance_offset[i].view(b, s, -1, self.bev_dimension[0].item(), self.bev_dimension[1].item()).permute(0, 1, 2, 4, 3)
            
        x = unpack_sequence_dim(x, b, s)
        return x, inter_seg, inter_instance_offset, inter_instance_center, xy_cord