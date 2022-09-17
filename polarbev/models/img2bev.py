import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class RingConv(nn.Module):
    def __init__(self, rho_dims, phi_dims, in_channels, out_channels):
        super().__init__()
        self.phi_dims = phi_dims
        self.rho_dims = rho_dims
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        L, bs, c = x.shape
        x = x.permute(1, 2, 0).reshape(bs, c, self.phi_dims, self.rho_dims)
        x = F.pad(x, (0, 0, 1, 1), mode='circular')
        x = self.conv1(x)
        x = F.pad(x, (0, 0, 1, 1), mode='circular')
        x = self.conv2(x)
        x = x.view(bs, c, -1).permute(2, 0, 1)
        
        return x


class Camera2BEV(nn.Module):
    def __init__(self, phi_dims, rho_dims, query_number, embed_dims, n_layers, n_cameras, n_levels, rho, phi, zz, intervals, pc_range, img_size, n_classes):
        super().__init__()
        self.n_levels = n_levels
        self.n_layers = n_layers
        self.img_size = img_size
        self.pc_range = pc_range
        self.rho = rho
        self.phi = phi
        self.zz = zz
        self.embed_dims = embed_dims
        self.intervals = intervals
        self.query_number = query_number
        

        self.phi_query = nn.Embedding(phi_dims, embed_dims * 2)
        self.rho_query = nn.Embedding(rho_dims, embed_dims * 2)
        self.reference_points = nn.Linear(embed_dims, 1)
        
        
        self.reg_branchs = nn.ModuleList()
        self.iter_refines = nn.ModuleList()
        self.ring_convs = nn.ModuleList()
        self.seg_heads = nn.ModuleList()
        self.instance_center_heads = nn.ModuleList()
        self.instance_offset_heads = nn.ModuleList()
        for i in range(n_layers):
            self.iter_refines.append(CrossAttention(embed_dims, n_cameras, n_levels, dropout=0.0))
            self.ring_convs.append(RingConv(rho_dims, phi_dims, embed_dims, embed_dims))
            self.reg_branchs.append(nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, 1)
            ))
            self.seg_heads.append(nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, n_classes),
            ))
            self.instance_offset_heads.append(nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, 2),
            ))
            self.instance_center_heads.append(nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, 1),
                nn.Sigmoid()
            ))
            
    def forward(self, x, lidar2imgs):
        bs, n_cameras, h, w, c = x.shape
        x = x.permute(0, 1, 4, 2, 3)
        # QxNxC
        query_embeds = (self.phi_query.weight.unsqueeze(1) + self.rho_query.weight.unsqueeze(0)).flatten(0, 1)
        query_embed, output = torch.split(query_embeds.unsqueeze(1).repeat(1, bs, 1), self.embed_dims, dim=-1)
        
        reference_points = self.reference_points(query_embed)

        zz = self.zz.to(x)
        zz = zz.unsqueeze(0).repeat(bs, 1, 1).view(bs, -1, 1)
        zz = (inverse_sigmoid(zz) + reference_points.permute(1, 0, 2)).sigmoid()

        rho = self.rho.to(x)
        phi = self.phi.to(x)
        
        reg = []
        inter_seg = []
        inter_instance_offset = []
        inter_instance_center = []
        
        for i in range(self.n_layers):
            # mask: [bs, c, n_q, n_c, 1]
            output, mask = self.iter_refines[i](output, query_embed, rho, phi, zz, x, lidar2imgs, self.intervals, self.pc_range, self.img_size)

            output = self.ring_convs[i](output)
            
            tmp = self.reg_branchs[i](output.permute(1, 0, 2))
            zz = (inverse_sigmoid(zz) + tmp).sigmoid()
            reg.append(zz)


            inter_seg.append(self.seg_heads[i](output))
            inter_instance_offset.append(self.instance_offset_heads[i](output))
            inter_instance_center.append(self.instance_center_heads[i](output))
        
        return output, inter_seg, inter_instance_offset, inter_instance_center, reg
    

class CrossAttention(nn.Module):
    def __init__(self, hidden_dims, n_cameras, n_levels, dropout, n_points=4):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.n_cameras = n_cameras
        self.n_levels = n_levels
        self.num_points = n_points
        self.dropout = nn.Dropout(p=dropout)
        
        assert n_levels == 1
        
        self.attention_weights = nn.Linear(hidden_dims, n_cameras * n_points)
        self.output_poj = nn.Linear(hidden_dims, hidden_dims)
        
    def forward(self, query, query_pos, rho, phi, zz_cord, img_feats, lidar2imgs, intervals, pc_range, img_size):
        inp_residual = query
        query = query + query_pos
        
        query = query.permute(1, 0, 2)
        bs, num_query, _ = query.size()
        
        attention_weights = self.attention_weights(query).view(bs, 1, num_query, self.n_cameras, self.num_points)
        
        output, mask = feature_sampling(img_feats, rho, phi, zz_cord, intervals, pc_range, lidar2imgs, img_size)

        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)
        
        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1)
        output = output.permute(2, 0, 1)

        output = self.output_poj(output)
        
        return self.dropout(output) + inp_residual, mask
    

def feature_sampling(img_feats, rho, phi, zz_cord, intervals, pc_range, lidar2imgs, img_size):
    bs, _, _ = zz_cord.shape

    rho = rho.unsqueeze(0).repeat(bs, 1, 1).view(bs, -1, 1)
    phi = phi.unsqueeze(0).repeat(bs, 1, 1).view(bs, -1, 1)
    rho = rho * pc_range[0] + intervals[0] / 2
    phi = phi * pc_range[1] - np.pi + intervals[1] / 2
    xx_cord = rho * torch.cos(phi)
    yy_cord = rho * torch.sin(phi)

    zz_cord = zz_cord * (pc_range[3] - pc_range[2]) + pc_range[2]
    
    xyz_cord = torch.cat([xx_cord, yy_cord, zz_cord], dim=-1)
    xyz_cord = torch.cat((xyz_cord, torch.ones_like(xyz_cord[..., :1])), -1)
    bs, num_query = xyz_cord.size()[:2]
    n_camera = lidar2imgs.size(1)

    xyz_cord = xyz_cord.view(bs, 1, num_query, 4).repeat(1, n_camera, 1, 1).unsqueeze(-1)
    lidar2imgs = lidar2imgs.view(bs, n_camera, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    xyz_cord_cam = torch.matmul(lidar2imgs, xyz_cord).squeeze(-1)
    eps = 1e-5
    mask = (xyz_cord_cam[..., 2:3] > eps)
    xyz_cord_cam = xyz_cord_cam[..., 0:2] / torch.maximum(xyz_cord_cam[..., 2:3], torch.ones_like(xyz_cord_cam[..., 2:3]) * eps)
    
    xyz_cord_cam[..., 0] /= img_size[0]
    xyz_cord_cam[..., 1] /= img_size[1]
       
    xyz_cord_cam = (xyz_cord_cam - 0.5) * 2
    mask = (mask & (xyz_cord_cam[..., 0:1] > -1.)
                 & (xyz_cord_cam[..., 0:1] < 1.)
                 & (xyz_cord_cam[..., 1:2] > -1.)
                 & (xyz_cord_cam[..., 1:2] < 1.))

    mask = mask.view(bs, n_camera, 1, num_query, 1).permute(0, 2, 3, 1, 4)
    mask = torch.nan_to_num(mask)

    bs, N, C, H, W = img_feats.shape
    img_feats = img_feats.view(bs*N, C, H, W)
    xyz_cord_cam = xyz_cord_cam.view(bs*N, num_query, 1, 2)
    sampled_feat = F.grid_sample(img_feats, xyz_cord_cam)
    sampled_feat = sampled_feat.view(bs, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
    
    return sampled_feat, mask