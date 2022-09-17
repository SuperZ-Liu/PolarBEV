import torch.nn as nn
from torchvision.models.resnet import resnet18
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, down_sample=None, norm_layer=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=(0, 1), bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=(0, 1), bias=False)
        self.bn2 = norm_layer(out_channels)
        self.down_sample = down_sample
        
    def forward(self, x):
        residual = x
        
        out = F.pad(x, (0, 0, 1, 1), mode='circular')
        out = self.relu(self.bn1(self.conv1(out)))
        
        out = F.pad(out, (0, 0, 1, 1), mode='circular')
        out = self.bn2(self.conv2(out))
        
        if self.down_sample is not None:
            residual = self.down_sample(x)
        out = self.relu(out + residual)
        return out
    
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)
        self.bn = norm_layer(out_channels)
    
    def forward(self, x):
        return self.bn(self.conv(x))


class RingUpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scale_factor = scale_factor
        
    def forward(self, x, x_skip):
        x = F.pad(x, (0, 0, 1, 1), mode='circular')
        x = F.interpolate(x, size=(x_skip.shape[2]+2, x_skip.shape[3]), mode='bilinear', align_corners=False)
        x = x[:, :, 1:-1, :]
        x = self.bn(self.conv(x))
        return x + x_skip


class RingDecoder(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=(0, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, 1, norm_layer=nn.BatchNorm2d),
            BasicBlock(64, 64, 1, norm_layer=nn.BatchNorm2d)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, 2, Downsample(64, 128, nn.BatchNorm2d), nn.BatchNorm2d),
            BasicBlock(128, 128, 1, norm_layer=nn.BatchNorm2d)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, 2, down_sample=Downsample(128, 256, nn.BatchNorm2d), norm_layer=nn.BatchNorm2d),
            BasicBlock(256, 256, 1, norm_layer=nn.BatchNorm2d)
        )
        self.predict_future_flow = predict_future_flow
        
        shared_out_channels = in_channels
        self.up3_skip = RingUpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = RingUpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = RingUpsamplingAdd(64, shared_out_channels, scale_factor=2)
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=(0, 1), bias=False),
            nn.BatchNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=(0, 1), bias=False),
            nn.BatchNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=(0, 1), bias=False),
            nn.BatchNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=(0, 1), bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )
            
    def forward(self, x):
        # import pdb; pdb.set_trace()
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        
        # (H, W)
        skip_x = {'1': x}
        x = F.pad(x, (0, 0, 3, 3), mode='circular')
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x
        
        # (H/8, W/8)
        x = self.layer3(x)
        
        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])
        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])
        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])
        
        x = F.pad(x, (0, 0, 1, 1), mode='circular')
        
        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None
        return {
            'segmentation': segmentation_output.view(b, s, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, s, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, s, *instance_offset_output.shape[1:]),
            'instance_flow': instance_future_output.view(b, s, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
        }
