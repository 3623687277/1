import torch
import torch.nn as nn

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils
import sys
from pathlib import Path
from ....utils import box_utils, calibration_kitti, common_utils, object3d_kitti
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ....datasets.kitti.kitti_dataset import KittiDataset
# import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication
# import os
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']='/home/ubuntu/anaconda3/envs/pcdet/lib/python3.9/site-packages/cv2/qt/plugins/platforms'

cmap = plt.cm.jet
def knn(x, k):  # 如果没有使用该参数，则默认使用最近邻数量为20
    # print("---")
    # print(x.shape) # ([32, 3, 1024])
    x1=x[:,1:4]
    # print(x.shape)
    inner = -2 * torch.matmul(x1.transpose(2, 1), x1)
    xx = torch.sum(x1 ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    # print(idx.shape) #([32, 1024, 20])
    return idx


def get_graph_feature(x, k=10, idx=None):
    batch_size = 1
    #batch_size = args.batch_size,
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    # print(idx_base.shape)
    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    # print(feature.shape,'feature')
    feature = feature.view(batch_size, num_points, k, num_dims)
    # print(feature.shape,'featurefeature')
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # print(x.shape,'x')
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    # print(feature.shape) #([32, 6, 1024, 20])
    return feature
def vis_pointcloud(points, colors=None):
    """
    渲染显示雷达点云
    :param points:    numpy.ndarray  `N x 3`
    :param colors:    numpy.ndarray  `N x 3`  (0, 255)
    :return:
    """
    app = QApplication(sys.argv)

    if colors is not None:
        colors = colors / 255
        colors = np.hstack((colors, np.ones(shape=(colors.shape[0], 1))))
    else:
        colors = (1, 1, 1, 1)
    og_widget = gl.GLViewWidget()
    point_size = np.zeros(points.shape[0], dtype=np.float16) + 0.1

    points_item1 = gl.GLScatterPlotItem(pos=points, size=point_size, color=colors, pxMode=False)
    og_widget.addItem(points_item1)

    # 作为对比
    # points_item2 = gl.GLScatterPlotItem(pos=points, size=point_size, color=(1, 1, 1, 1), pxMode=False)
    # points_item2.translate(0, 0, 0)
    # og_widget.addItem(points_item2)

    og_widget.show()
    sys.exit(app.exec_())


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class VoxelSetAbstraction(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(64)
        self.bn11 = nn.BatchNorm2d(32)

        self.bn5 = nn.BatchNorm1d(2048)

        # self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
        #                            self.bn1,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(512 * 2, 32, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, bias=False),
                                    self.bn11,
                                    nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(512, 2048, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(2048 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 40)

        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]
            cur_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg[src_name].POOL_RADIUS,
                nsamples=SA_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps])

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            mlps = SA_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k]

            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg['raw_points'].POOL_RADIUS,
                nsamples=SA_cfg['raw_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            c_in += sum([x[-1] for x in mlps])

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:7]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        # if len(keypoints.shape) == 3:
        #     batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1, 1)
        #     keypoints = torch.cat((batch_idx.float(), keypoints.view(-1, 6)), dim=1)
        return keypoints

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints_color = self.get_sampled_points(batch_dict)
        keypoints = keypoints_color[:, :,0:3].contiguous()    #keypoints[1,2048,3]
        # keypoints = keypoints.squeeze(dim=0)
        # print(keypoints.shape,keypoints)
        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride']
            )
            point_features_list.append(point_bev_features)

        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3).contiguous()
        # new_xyz = keypoints[:,:, 1:4].contiguous()
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            # xyz = raw_points[:, 1:4]
            # xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            # for bs_idx in range(batch_size):
            #     xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
            # point_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None
            #
            # pooled_points, pooled_features = self.SA_rawpoints(
            #     xyz=xyz.contiguous(),
            #     xyz_batch_cnt=xyz_batch_cnt,
            #     new_xyz=new_xyz,
            #     new_xyz_batch_cnt=new_xyz_batch_cnt,
            #     features=point_features,
            # )
            # keypoints_color = keypoints_color.unsqueeze(dim=0)
            feature = keypoints_color.permute(0, 2, 1).contiguous()
            x = feature    #[1,6,2048]
            # print(x.shape)
            x = get_graph_feature(x, k=10)  # [1, 12, 2048, 10]
            # print(x.shape)
            x = self.conv1(x)  # [1, 64, 2048, 10]
            # print(x.shape)
            x1 = x.max(dim=-1, keepdim=False)[0]  # [1, 64, 2048]
            # print(x1.shape)
            x = get_graph_feature(x1, k=10)  # [1, 128, 2048, 10]
            # print(x.shape)
            x = self.conv2(x)  # [1, 64, 2048, 10]
            # print(x.shape)  #
            x2 = x.max(dim=-1, keepdim=False)[0]  # [1,64,2048]
            # print(x2.shape)  #
            x = get_graph_feature(x2, k=10)  # [1, 128, 2048, 10]
            # print(x.shape)  #
            x = self.conv3(x)  # [1, 128, 2048, 10]
            # print(x.shape)  #
            x3 = x.max(dim=-1, keepdim=False)[0]  # [1, 128, 2048]
            # print(x3.shape)  #
            x = get_graph_feature(x3, k=10)  # [1, 256, 2048, 10]
            # print(x.shape)  #
            x = self.conv4(x)  # [1, 256, 2048, 10]
            # print(x.shape)  #
            x4 = x.max(dim=-1, keepdim=False)[0]  # [1, 256, 2048]
            # print(x4.shape)  #
            x = torch.cat((x1, x2, x3, x4), dim=1)  # [1, 512, 2048]
            x = get_graph_feature(x, k=10)
            x = self.conv6(x)
            # x = self.conv7(x)
            # x = self.conv8(x)
            # x = self.conv9(x)
            # x = self.conv10(x)
            # x = self.conv11(x)

            # print(x.shape)
            x5 = x.max(dim=-1, keepdim=False)[0]  # [1, 32, 2048]
            x5 = x5.squeeze(dim=0)
            pooled_features = x5.permute(1, 0).contiguous()
            # print(self.SA_rawpoints)


            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            pooled_points, pooled_features = self.SA_layers[k](
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=batch_dict['multi_scale_3d_features'][src_name].features.contiguous(),
            )
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

        point_features = torch.cat(point_features_list, dim=2)

        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)

        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = point_coords  # (BxN, 4)
        # batch_dict['point_coords'] =keypoints
        return batch_dict
