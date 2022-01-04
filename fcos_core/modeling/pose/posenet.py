import torch
from torch import nn

import numpy as np

from .layers import Conv, Hourglass, Pool, Residual, HeatmapLoss
from .heatmap import GenerateHeatmap

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class PoseNetModule(nn.Module):
    # def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
    def __init__(self, cfg, in_channels, c5_channels):
        super(PoseNetModule, self).__init__()
        
        nstack = cfg.MODEL.POSENET.N_STACK
        inp_dim = in_channels
        oup_dim = cfg.MODEL.POSENET.OUTPUT_DIM
        bn = False
        increase = 0
        
        sigma = cfg.MODEL.POSENET.SIGMA
        
        self.fuse_pose_type = cfg.MODEL.POSENET.FUSE_TYPE
        
        self.nstack = nstack
        
        """
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        """
        
        self.align_dims = Merge(c5_channels, in_channels)  # e.g. from 2048 in C5 --> to 256 in P5
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.heatmapLoss = HeatmapLoss()
        self.gen_heatmap = GenerateHeatmap(oup_dim, sigma)
        
        self.align_dims_combined = Merge(in_channels*nstack, in_channels)  # e.g. from T*256 --> to 256
    
    """
    def forward(self, imgs):
        ## our posenet
        x = imgs.permute(0, 3, 1, 2)  # x of size 1,3,inpdim,inpdim
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)
    """
    
    def forward(self, c5_feature, keypoints):
        # print("start:", c5_feature.shape)
        c5_feature = self.align_dims(c5_feature)  # e.g. from 2048 in C5 --> to 256 in P5
        # print("end:", c5_feature.shape)
        
        # c5_feature.shape [batch, dim, size_h, size_w], size_h % 2^4(16) == 0, and size_w % 2^4(16) == 0
        
        combined_hm_preds = []
        combined_hm_features = []
        for i in range(self.nstack):
            hg = self.hgs[i](c5_feature)
            mid_feature = self.features[i](hg)
            preds = self.outs[i](mid_feature)
            # print(i, hg.shape, mid_feature.shape, preds.shape)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                c5_feature = c5_feature + self.merge_preds[i](preds) + self.merge_features[i](mid_feature)
                combined_hm_features.append(c5_feature)
            else:
                combined_hm_features.append(mid_feature)
        # return torch.stack(combined_hm_preds, 1)
        
        '''If we have better feature fusion method?'''
        
        if self.fuse_pose_type == "LAST":
            """Method 1: only use the last aux_feature is not so good"""
            aux_feature = combined_hm_features[-1]

        if self.fuse_pose_type == "FULL":
            """Method 2: concatnate all pose features from all layers, and then align dimensions"""
            combined_aux_feature = torch.cat(combined_hm_features, dim=1)
            aux_feature = self.align_dims_combined(combined_aux_feature)
        
        
        if self.training:
            heatmap_losses = self.calc_loss(combined_hm_preds, keypoints)
            return aux_feature, heatmap_losses
        else:
            return aux_feature, []
            

    def calc_loss(self, combined_hm_preds, keypoints):
        traget_size = [int(combined_hm_preds[-1].shape[-2]), int(combined_hm_preds[-1].shape[-1])]
        # print("traget_size:", traget_size)
        
        heatmaps = []
        for kpts_i in keypoints:
            keypoints_revised = kpts_i.resize(traget_size).get_keypoints()
            keypoints_revised_np = keypoints_revised.cpu().numpy()
            heatmap_per_im_np = self.gen_heatmap(keypoints_revised_np, traget_size)
            heatmaps.append(heatmap_per_im_np)
        
        heatmaps = torch.from_numpy(np.array(heatmaps)).cuda()  # heatmaps.shape is [batch, kpts_num, img_h, img_w]
        # heatmaps = torch.as_tensor(heatmaps, dtype=torch.float32, device=keypoints_revised.devide)
        
        combined_loss = self.heatmapLoss(self.nstack, combined_hm_preds, heatmaps)
        
        losses = {"loss_heatmap": combined_loss}
        return losses


def build_posenet(cfg, in_channels, c5_channels):
    if cfg.MODEL.POSENET_ON:
        return PoseNetModule(cfg, in_channels, c5_channels)
    else:
        return []