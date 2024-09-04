import os
import os.path as osp
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

import torch
import torch.nn as nn
import torchvision.ops as tv_ops
import torch.nn.functional as F
import torchvision.transforms as transforms


from torch_scatter import scatter, scatter_softmax, scatter_max, scatter_log_softmax
from extensions.ray_aabb.jit import ray_aabb
from extensions.pcl_aabb.jit import pcl_aabb


import constants
import models.pointnet as pnet
import models.resnet_dilated as resnet_dilated
import models.implicit_net as im_net
# from models.SwinTransformer import SwinTransformerSys
# from models.FPN import FPN_Swin
import models.FPN as FPN
from models.keypoints.resnetUnet import OfficialResNetUnet
from models.keypoints.generate_features import GFM
import models.kpts_encoder as kpts_encoder
import models.fusion as fusion_models
import utils.point_utils as point_utils
import utils.vis_utils as vis_utils
import utils.loss_utils as loss_utils
from utils.training_utils import *

class LIDF(nn.Module):
    def __init__(self, opt, device):
        super(LIDF, self).__init__()
        self.opt = opt
        self.device = device
        # build models
        self.build_model()

    def build_model(self):
        # positional embedding
        if self.opt.model.pos_encode:
            self.embed_fn, embed_ch = im_net.get_embedder(self.opt.model.multires)
            self.embeddirs_fn, embeddirs_ch = im_net.get_embedder(self.opt.model.multires_views)
        else:
            self.embed_fn, embed_ch = im_net.get_embedder(self.opt.model.multires, i=-1)
            self.embeddirs_fn, embeddirs_ch = im_net.get_embedder(self.opt.model.multires_views, i=-1)
            assert embed_ch == embeddirs_ch == 3
        
        # rgb model
        if self.opt.model.rgb_model_type == 'resnet':
            self.resnet_model = resnet_dilated.Resnet34_8s(inp_ch=self.opt.model.rgb_in, out_ch=self.opt.model.rgb_out).to(self.device)
        elif self.opt.model.rgb_model_type == 'swin':
            # self.rgb_swin_model = FPN.FPN_Swin(self.opt).to(self.device)
            self.rgb_swin_model = FPN.FPN_Swin2(self.opt).to(self.device)
            if self.opt.model.SWIN.USE_PRETRAIN_CKPT:
                self.rgb_swin_model.init_weights()
        else:
            raise NotImplementedError('Does not support RGB model: {}'.format(self.opt.model.rgb_model_type))
        
        # pointnet model
        if self.opt.model.pnet_model_type == 'twostage':
            self.pnet_model = pnet.PointNet2Stage(input_channels=self.opt.model.pnet_in,
                                    output_channels=self.opt.model.pnet_out, gf_dim=self.opt.model.pnet_gf).to(self.device)
        elif self.opt.model.pnet_model_type == 'pointattention':
            self.pnet_model = pnet.PointNet2StagePointAttention(input_channels=self.opt.model.pnet_in,
                                    output_channels=self.opt.model.pnet_out, gf_dim=self.opt.model.pnet_gf).to(self.device)
        elif self.opt.model.pnet_model_type == 'twostage_voxelfusion':
            self.pnet_model = pnet.PointNet2Stage(input_channels=self.opt.model.pnet_in,
                                    output_channels=self.opt.model.pnet_out - 32, gf_dim=self.opt.model.pnet_gf).to(self.device)
        elif self.opt.model.pnet_model_type == 'twostage_pointfusion':
            self.pnet_model = pnet.PointNet2Stage_PointFusion(input_channels=self.opt.model.pnet_in,
                                    output_channels=self.opt.model.pnet_out, gf_dim=self.opt.model.pnet_gf).to(self.device)
        else:
            raise NotImplementedError('Does not support PNET model: {}'.format(self.opt.model.pnet_model_type))
        
        # decoder input dim
        if self.opt.model.rgb_embedding_type == 'ROIAlign':
            if self.opt.model.rgb_model_type == 'resnet':
                dec_inp_dim = self.opt.model.pnet_out + self.opt.model.rgb_out * (self.opt.model.roi_out_bbox**2) \
                                + 2 * embed_ch + embeddirs_ch
            elif self.opt.model.rgb_model_type == 'swin':
                dec_inp_dim = self.opt.model.pnet_out + self.opt.model.rgb_out * (self.opt.model.roi_out_bbox**2) \
                                + 2 * embed_ch + embeddirs_ch
            if self.opt.model.use_hand_features:
                hand_feature_dim = 3 * self.opt.model.num_hand_kpts
                if self.opt.model.use_2d_hand_features:
                    hand_feature_dim -= self.opt.model.num_hand_kpts
                if self.opt.model.use_kpts_encoder:
                    self.kpts_encoder = kpts_encoder.KeypointsEncoder(input_dim=hand_feature_dim,
                                                                    hidden_dim=self.opt.model.keypoints_encoder.hidden_dim,
                                                                    output_dim=self.opt.model.keypoints_encoder.output_dim).to(self.device)
                    hand_feature_dim = self.opt.model.keypoints_encoder.output_dim
                if self.opt.model.use_relative_hand_feature:
                    hand_feature_dim *= 2
                dec_inp_dim += hand_feature_dim
            
            # print('####### dec_inp_dim', dec_inp_dim)
        else:
            raise NotImplementedError('Does not support RGB embedding: {}'.format(self.opt.model.rgb_embedding_type))
        
        # fusion model
        if self.opt.model.fusion.use_fusion:
            if self.opt.model.fusion.fusion_type == 'deep_fusion':
                self.resnet_model = resnet_dilated.Resnet34_8s_no_interpolate(inp_ch=self.opt.model.rgb_in, out_ch=self.opt.model.rgb_out).to(self.device)
                self.fusion_model = fusion_models.DeepFusion().to(self.device)
                self.opt.model.rgb_embedding_type = 'deep_fusion'
            elif self.opt.model.fusion.fusion_type == 'cross_atten':
                self.fusion_model = fusion_models.CrossAttentionFusion(in_channel=self.opt.model.pnet_out,
                                                                       depth=1, num_heads=4,
                                                                       ).to(self.device)
            elif self.opt.model.fusion.fusion_type == 'adaptive_fusion':
                self.fusion_model = fusion_models.AdaptiveFusion(rgb_embed_dim=self.opt.model.rgb_out * (self.opt.model.roi_out_bbox**2),
                                                                 xyz_embed_dim=self.opt.model.pnet_out,
                                                                 ).to(self.device)
            elif self.opt.model.fusion.fusion_type == 'gated_fusion':
                self.fusion_model = fusion_models.GatedFusion(input_dim=self.opt.model.pnet_out).to(self.device)
                dec_inp_dim += self.opt.model.pnet_out
        
        
        # offset decoder
        if self.opt.model.offdec_type == 'IMNET':
            self.offset_dec = im_net.IMNet(inp_dim=dec_inp_dim, out_dim=1, 
                                    gf_dim=self.opt.model.imnet_gf, use_sigmoid=self.opt.model.use_sigmoid).to(self.device)
        elif self.opt.model.offdec_type == 'IEF':
            self.offset_dec = im_net.IEF(self.device, inp_dim=dec_inp_dim, out_dim=1, gf_dim=self.opt.model.imnet_gf, 
                                    n_iter=self.opt.model.n_iter, use_sigmoid=self.opt.model.use_sigmoid).to(self.device)
        else:
            raise NotImplementedError('Does not support Offset Decoder Type: {}'.format(self.opt.model.offdec_type))
        
        # prob decoder
        if self.opt.loss.prob_loss_type == 'ray':
            prob_out_dim = 1
        if self.opt.model.probdec_type == 'IMNET':
            self.prob_dec = im_net.IMNet(inp_dim=dec_inp_dim, out_dim=prob_out_dim, 
                                gf_dim=self.opt.model.imnet_gf, use_sigmoid=self.opt.model.use_sigmoid).to(self.device)
        else:
            raise NotImplementedError('Does not support Prob Decoder Type: {}'.format(self.opt.model.probdec_type))
        
        # uncertainty decoder
        if self.opt.loss.use_uncertainty_loss:
            uncer_inp_dim = self.opt.model.pnet_out + self.opt.model.rgb_out * (self.opt.model.roi_out_bbox**2)
            self.uncer_dec = im_net.IMNet(inp_dim=uncer_inp_dim, out_dim=1,
                                gf_dim=self.opt.model.imnet_gf, use_sigmoid=True).to(self.device)
            self.voxel_2_ray = pnet.Voxel2Ray(self.opt.model.pnet_out, self.opt.model.pnet_out).to(self.device)
            
        

        # load keypoints predict module
        self.GFM = GFM()
        if not self.opt.exp_type == 'test_real' and self.opt.model.use_hand_features and self.opt.model.use_pred_keypoints:
            if self.opt.model.keypoints.use_depth:
                self.kpts_model = OfficialResNetUnet(self.opt.model.keypoints.net, 
                                    self.opt.model.num_hand_kpts, pretrain=True, 
                                    deconv_dim=128,
                                    out_dim_list=[self.opt.model.num_hand_kpts * 3, self.opt.model.num_hand_kpts, 
                                                  self.opt.model.num_hand_kpts]).to(self.device)
                kpts_model_checkpoint = torch.load(self.opt.model.keypoints.ckpts_dir, map_location=self.device)
                for param in self.kpts_model.parameters():
                    param.requires_grad = False
                restore(self.kpts_model, kpts_model_checkpoint['backbone_d'])
        

        # loss function
        self.pos_loss_fn = nn.L1Loss()
        if self.opt.loss.pos_loss_type == 'combine':
            self.pos_loss_fn = loss_utils.CombinedLoss()
        print('loss_fn at GPU {}'.format(self.opt.gpu_id))

    def prepare_data(self, batch, exp_type, pred_mask):
        if self.opt.exp_type == 'test_real':
            data_dict = batch
            return data_dict # for real world exp
        
        # fetch data
        batch = to_gpu(batch, self.device)
        rgb_img = batch['rgb']
        bs = rgb_img.shape[0]
        h,w = rgb_img.shape[2],rgb_img.shape[3]
        corrupt_mask = batch['corrupt_mask'].squeeze(1)
        xyz = batch['xyz']
        xyz_corrupt = batch['xyz_corrupt']
        if 'valid_mask' in batch.keys():
            valid_mask = batch['valid_mask'].squeeze(1)
        else:
            valid_mask = 1 - corrupt_mask
        
        # flat h and w dim
        xyz_flat = xyz.permute(0, 2, 3, 1).contiguous().reshape(bs,-1,3)
        xyz_corrupt_flat = xyz_corrupt.permute(0, 2, 3, 1).contiguous().reshape(bs,-1,3)
        
        self.camera_params = (batch['fx'][0].item(), batch['fy'][0].item(), batch['cx'][0].item(), batch['cy'][0].item())

        # arrange data in a dictionary
        data_dict = {
            'bs': bs,
            'h': h,
            'w': w,
            'rgb_img': rgb_img,
            'corrupt_mask': corrupt_mask,
            'valid_mask': valid_mask,
            'xyz_flat': xyz_flat,
            'xyz_corrupt_flat': xyz_corrupt_flat,
            'depth': batch['depth'],
            'depth_corrupt': batch['depth_corrupt'],
            'fx': batch['fx'].float(),
            'fy': batch['fy'].float(),
            'cx': batch['cx'].float(),
            'cy': batch['cy'].float(),
            'xres': batch['xres'].float(),
            'yres': batch['yres'].float(),
            'hand_kpts': batch['hand_kpts'],
            # 'item': batch['item'],
            # 'item_path': batch['item_path'],
        }
        
        if exp_type == 'test':
            data_dict.update({
                'item': batch['item'],
                'hand_kpts_uv': batch['hand_kpts_uv'],
            })
        
        # add pred_mask
        if exp_type != 'train':
            if self.opt.mask_type == 'pred':
                data_dict['pred_mask'] = pred_mask
                data_dict['valid_mask'] = 1 - pred_mask
            elif self.opt.mask_type == 'all':
                data_dict['pred_mask'] = torch.ones_like(data_dict['corrupt_mask'])
                inp_zero_mask = (batch['depth_corrupt'] == 0).squeeze(1).float()
                data_dict['valid_mask'] = 1 - inp_zero_mask

        return data_dict

    def get_valid_points(self, data_dict):
        '''
            If valid_sample_num == -1, use all valid points. Otherwise uniformly sample valid points in a small block.
            valid_idx: (valid_point_num,2), 1st dim is batch idx, 2nd dim is flattened img idx.
        '''
        bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
        if self.opt.grid.valid_sample_num != -1: # sample valid points
            valid_idx = point_utils.sample_valid_points(data_dict['valid_mask'], self.opt.grid.valid_sample_num, block_x=8, block_y=8)
        else: # get all valid points
            valid_mask_flat = data_dict['valid_mask'].reshape(bs,-1)
            valid_idx = torch.nonzero(valid_mask_flat, as_tuple=False)
        valid_bid = valid_idx[:,0]
        valid_flat_img_id = valid_idx[:,1]
        # get rgb and xyz for valid points.
        valid_xyz = data_dict['xyz_corrupt_flat'][valid_bid, valid_flat_img_id]
        rgb_img_flat = data_dict['rgb_img'].permute(0,2,3,1).contiguous().reshape(bs,-1,3)
        valid_rgb = rgb_img_flat[valid_bid, valid_flat_img_id]
        # update intermediate data in data_dict
        data_dict.update({
            'valid_bid': valid_bid,
            'valid_flat_img_id': valid_flat_img_id,
            'valid_xyz': valid_xyz,
            'valid_rgb': valid_rgb,
        })

    def get_occ_vox_bound(self, data_dict):
        ##################################
        #  Get occupied voxel in a batch
        ##################################
        # setup grid properties
        xmin = torch.Tensor(constants.XMIN).float().to(self.device)
        xmax = torch.Tensor(constants.XMAX).float().to(self.device)
        min_bb = torch.min(xmax- xmin).item()
        part_size = min_bb / self.opt.grid.res
        # we need half voxel margin on each side
        xmin = xmin - 0.5 * part_size
        xmax = xmax + 0.5 * part_size
        # get occupied grid
        occ_vox_bid_global_coord, revidx, valid_v_pid, \
        valid_v_rel_coord, idx_grid = point_utils.batch_get_occupied_idx(
                    data_dict['valid_xyz'], data_dict['valid_bid'].unsqueeze(-1),
                    xmin=xmin, xmax=xmax, 
                    crop_size=part_size, overlap=False)
        # images in current minibatch do not have occupied voxels
        if occ_vox_bid_global_coord.shape[0] == 0:
            print('No occupied voxel', data_dict['item_path'])
            return False
        occ_vox_bid = occ_vox_bid_global_coord[:,0]
        occ_vox_global_coord = occ_vox_bid_global_coord[:,1:]
        ''' compute occupied voxel bound '''
        bound_min = xmin.unsqueeze(0) + occ_vox_global_coord * part_size
        bound_max = bound_min + part_size
        voxel_bound = torch.cat((bound_min,bound_max),1)
        # update data_dict
        data_dict.update({
            'xmin': xmin,
            'part_size': part_size,
            'revidx': revidx,
            'valid_v_pid': valid_v_pid,
            'valid_v_rel_coord': valid_v_rel_coord,
            'occ_vox_bid': occ_vox_bid,
            'occ_vox_global_coord': occ_vox_global_coord,
            'voxel_bound': voxel_bound,    
        })
        # print('valid_v_pid', valid_v_pid.shape)
        return True

    def get_miss_ray(self, data_dict, exp_type):
        #####################################
        # compute ray dir and img grid index 
        #####################################
        bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
        fx,fy = data_dict['fx'], data_dict['fy']
        cx,cy = data_dict['cx'], data_dict['cy']
        y_ind, x_ind = torch.meshgrid(torch.arange(h), torch.arange(w))
        x_ind = x_ind.unsqueeze(0).repeat(bs,1,1).float().to(self.device)
        y_ind = y_ind.unsqueeze(0).repeat(bs,1,1).float().to(self.device)
        # img grid index, (bs,h*w,2)
        img_ind_flat = torch.stack((x_ind,y_ind),-1).reshape(bs,h*w,2).long()
        cam_x = x_ind - cx.reshape(-1,1,1)
        cam_y = (y_ind - cy.reshape(-1,1,1)) * fx.reshape(-1,1,1) / fy.reshape(-1,1,1)
        cam_z = fx.reshape(-1,1,1).repeat(1,h,w)
        ray_dir = torch.stack((cam_x,cam_y,cam_z),-1)
        ray_dir = ray_dir / torch.norm(ray_dir,dim=-1,keepdim=True)
        ray_dir_flat = ray_dir.reshape(bs,-1,3)
        
        ###################################
        # sample miss points 
        # (miss_point_num,2): 1st dim is batch idx, second dim is flatted img idx.
        ###################################
        if exp_type != 'train' and self.opt.mask_type in ['pred', 'all']:
            pred_mask_flat = data_dict['pred_mask'].view(bs,-1)
            miss_idx = torch.nonzero(pred_mask_flat, as_tuple=False)
        else:
            corrupt_mask_flat = data_dict['corrupt_mask'].view(bs,-1)
            miss_idx = torch.nonzero(corrupt_mask_flat, as_tuple=False)
        if exp_type == 'train' and self.opt.grid.miss_sample_num != -1 and bs*self.opt.grid.miss_sample_num < miss_idx.shape[0]:            
            ''' randomly sample miss point. make them as continuous as possible '''
            miss_bid = miss_idx[:,0]
            # get max miss ray cnt for all examples inside a minibatch
            miss_bid_nodup, _, miss_bid_cnt = torch.unique_consecutive(miss_bid,dim=0,return_counts=True,return_inverse=True)
            # make sure cnt is sorted and fill in zero if non exist
            miss_bid_cnt_sorted = scatter(miss_bid_cnt, miss_bid_nodup, 
                            dim=0, dim_size=bs, reduce="sum")
            miss_bid_sid_eid = torch.cumsum(miss_bid_cnt_sorted, 0)
            miss_bid_sid_eid = torch.cat((torch.Tensor([0]).long().to(self.device), miss_bid_sid_eid),0)
            sample_list = []
            # iterate over examples in a batch
            for i in range(miss_bid_sid_eid.shape[0]-1):
                cur_sid = miss_bid_sid_eid[i].item()
                cur_eid = miss_bid_sid_eid[i+1].item()
                cur_cnt = miss_bid_cnt_sorted[i].item()
                if cur_cnt > self.opt.grid.miss_sample_num: # sample random miss points
                    start_range = cur_cnt - self.opt.grid.miss_sample_num + 1
                    start_id = np.random.choice(start_range) + cur_sid
                    sample_list.append(miss_idx[start_id:start_id+self.opt.grid.miss_sample_num])
                else: # add all miss points
                    sample_list.append(miss_idx[cur_sid:cur_eid])
            miss_idx = torch.cat(sample_list,0)
        
        total_miss_sample_num = miss_idx.shape[0]
        miss_bid = miss_idx[:,0]
        miss_flat_img_id = miss_idx[:,1]
        # get ray dir and img index for sampled miss point
        miss_ray_dir = ray_dir_flat[miss_bid, miss_flat_img_id]
        miss_img_ind = img_ind_flat[miss_bid, miss_flat_img_id]
        # update data_dict
        data_dict.update({
            'miss_bid': miss_bid, # num of miss pixels (0 to batch_size - 1)
            'miss_flat_img_id': miss_flat_img_id,
            'miss_ray_dir': miss_ray_dir,
            'miss_img_ind': miss_img_ind,
            'total_miss_sample_num': total_miss_sample_num
        })

    def compute_ray_aabb(self, data_dict):
        ################################## 
        #    Run ray AABB slab test
        #    mask: (occ_vox_num_in_batch, miss_ray_num_in_batch)
        #    dist: (occ_vox_num_in_batch, miss_ray_num_in_batch,2). store in voxel dist and out voxel dist
        ##################################
        mask, dist = ray_aabb.forward(data_dict['miss_ray_dir'], data_dict['voxel_bound'], 
                            data_dict['miss_bid'].int(), data_dict['occ_vox_bid'].int())
        mask = mask.long()
        dist = dist.float()

        # get idx of ray-voxel intersect pair
        intersect_idx = torch.nonzero(mask, as_tuple=False)
        occ_vox_intersect_idx = intersect_idx[:,0]
        miss_ray_intersect_idx = intersect_idx[:,1]
        # images in current mini batch do not have ray occ vox intersection pair.
        if intersect_idx.shape[0] == 0:
            print('No miss ray and occ vox intersection pair', data_dict['item_path'])
            return False
        data_dict.update({
            'mask': mask,
            'dist': dist,
            'occ_vox_intersect_idx': occ_vox_intersect_idx,
            'miss_ray_intersect_idx': miss_ray_intersect_idx, # num of ray-voxel pair (0 to miss_ray_num - 1)
        })
        # debug
        # print('occ_vox_intersect_idx', occ_vox_intersect_idx.shape)
        # print('miss_ray_intersect_idx', miss_ray_intersect_idx.shape)
        # print(miss_ray_intersect_idx[0:100])
        # print('miss_ray_intersect_idx.max', miss_ray_intersect_idx.max())
        # raise NotImplementedError('stop')
        
        return True

    def compute_gt(self, data_dict):
        if self.opt.exp_type == 'test_real':
            return
        ###########################################
        #    Compute Groundtruth for position and ray termination label
        ###########################################
        # get gt pos for sampled missing point
        gt_pos = data_dict['xyz_flat'][data_dict['miss_bid'], data_dict['miss_flat_img_id']]
        # pcl_mask(i,j) indicates if j-th missing point gt pos inside i-th voxel
        pcl_mask = pcl_aabb.forward(gt_pos, data_dict['voxel_bound'], data_dict['miss_bid'].int(), data_dict['occ_vox_bid'].int())
        pcl_mask = pcl_mask.long()
        # compute gt label for ray termination
        pcl_label = pcl_mask[data_dict['occ_vox_intersect_idx'], data_dict['miss_ray_intersect_idx']]
        pcl_label_float = pcl_label.float()

        # get intersected voxels
        unique_intersect_vox_idx, occ_vox_intersect_idx_nodup2dup = torch.unique(data_dict['occ_vox_intersect_idx'], sorted=True, dim=0, return_inverse=True)
        intersect_voxel_bound = data_dict['voxel_bound'][unique_intersect_vox_idx]
        intersect_vox_bid = data_dict['occ_vox_bid'][unique_intersect_vox_idx]
        # get sampled valid pcl inside intersected voxels
        valid_intersect_mask = pcl_aabb.forward(data_dict['valid_xyz'], intersect_voxel_bound.contiguous(), data_dict['valid_bid'].int(), intersect_vox_bid.int().contiguous())
        valid_intersect_mask = valid_intersect_mask.long()
        try:
            valid_intersect_nonzero_idx = torch.nonzero(valid_intersect_mask, as_tuple=False)
        except:
            print(data_dict['valid_xyz'].shape)
            print(valid_intersect_mask.shape)
            print(unique_intersect_vox_idx.shape, intersect_voxel_bound.shape)
            print(data_dict['item_path'])
        valid_xyz_in_intersect = data_dict['valid_xyz'][valid_intersect_nonzero_idx[:,1]]
        valid_rgb_in_intersect = data_dict['valid_rgb'][valid_intersect_nonzero_idx[:,1]]
        valid_bid_in_intersect = data_dict['valid_bid'][valid_intersect_nonzero_idx[:,1]]
        # update data_dict
        data_dict.update({
            'gt_pos': gt_pos,
            'pcl_label': pcl_label,
            'pcl_label_float': pcl_label_float,
            'valid_xyz_in_intersect': valid_xyz_in_intersect,
            'valid_rgb_in_intersect': valid_rgb_in_intersect,
            'valid_bid_in_intersect': valid_bid_in_intersect
        })

    def get_embedding(self, data_dict):
        ########################### 
        #   Get embedding
        ##########################
        bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
        ''' Positional Encoding '''
        # compute intersect pos
        intersect_dist = data_dict['dist'][data_dict['occ_vox_intersect_idx'], data_dict['miss_ray_intersect_idx']]
        intersect_enter_dist, intersect_leave_dist = intersect_dist[:,0], intersect_dist[:,1]

        intersect_dir = data_dict['miss_ray_dir'][data_dict['miss_ray_intersect_idx']]
        intersect_enter_pos = intersect_dir * intersect_enter_dist.unsqueeze(-1)
        intersect_leave_pos = intersect_dir * intersect_leave_dist.unsqueeze(-1)


        intersect_voxel_bound = data_dict['voxel_bound'][data_dict['occ_vox_intersect_idx']]
        intersect_voxel_center = (intersect_voxel_bound[:,:3] + intersect_voxel_bound[:,3:]) / 2.
        if self.opt.model.intersect_pos_type == 'rel':
            inp_enter_pos = intersect_enter_pos - intersect_voxel_center
            inp_leave_pos = intersect_leave_pos - intersect_voxel_center
        else:
            inp_enter_pos = intersect_enter_pos
            inp_leave_pos = intersect_leave_pos

        # positional encoding
        intersect_enter_pos_embed = self.embed_fn(inp_enter_pos)
        intersect_leave_pos_embed = self.embed_fn(inp_leave_pos)
        intersect_dir_embed = self.embeddirs_fn(intersect_dir)    
        
        ''' RGB Embedding ''' 
        miss_ray_intersect_img_ind = data_dict['miss_img_ind'][data_dict['miss_ray_intersect_idx']]
        miss_ray_intersect_bid = data_dict['miss_bid'][data_dict['miss_ray_intersect_idx']]
        
        if self.opt.model.rgb_model_type == 'resnet':
            full_rgb_feat = self.resnet_model(data_dict['rgb_img'])
        elif self.opt.model.rgb_model_type == 'swin':
            full_rgb_feat = self.rgb_swin_model(data_dict['rgb_img'])
        else:
            raise NotImplementedError('Does not support RGB model: {}'.format(self.opt.model.rgb_model_type))
        # ROIAlign to pool features
        if self.opt.model.rgb_embedding_type == 'ROIAlign':
            # compute input boxes for ROI Align
            miss_ray_intersect_ul = miss_ray_intersect_img_ind - self.opt.model.roi_inp_bbox // 2
            miss_ray_intersect_br = miss_ray_intersect_img_ind + self.opt.model.roi_inp_bbox // 2
            # clamp is done in original image coords
            miss_ray_intersect_ul[:,0] = torch.clamp(miss_ray_intersect_ul[:,0], min=0., max=w-1)
            miss_ray_intersect_ul[:,1] = torch.clamp(miss_ray_intersect_ul[:,1], min=0., max=h-1)
            miss_ray_intersect_br[:,0] = torch.clamp(miss_ray_intersect_br[:,0], min=0., max=w-1)
            miss_ray_intersect_br[:,1] = torch.clamp(miss_ray_intersect_br[:,1], min=0., max=h-1)
            roi_boxes = torch.cat((miss_ray_intersect_bid.unsqueeze(-1), miss_ray_intersect_ul, miss_ray_intersect_br),-1).float()
            # sampled rgb features for ray-voxel intersect pair. (pair num,rgb_feat_len,roi_out_bbox,roi_out_bbox)
            spatial_scale = 1.0
            intersect_rgb_feat = tv_ops.roi_align(full_rgb_feat, roi_boxes, 
                                    output_size=self.opt.model.roi_out_bbox,
                                    spatial_scale=spatial_scale,
                                    aligned=True)

            try:
                intersect_rgb_feat = intersect_rgb_feat.reshape(intersect_rgb_feat.shape[0],-1)
                # print(f"intersect_rgb_feat: {intersect_rgb_feat.shape}")
            except:
                print(intersect_rgb_feat.shape)
                print(roi_boxes.shape)
                print(data_dict['miss_ray_intersect_idx'].shape, miss_ray_intersect_bid.shape, miss_ray_intersect_img_ind.shape)
                print(data_dict['total_miss_sample_num'])
                print(data_dict['item_path'])
        elif self.opt.model.rgb_embedding_type == 'deep_fusion':
            intersect_rgb_feat = None
        else:
            raise NotImplementedError('Does not support RGB embedding type: {}'.format(self.opt.model.rgb_embedding_type))

        '''  Voxel Embedding '''
        valid_v_rgb = data_dict['valid_rgb'][data_dict['valid_v_pid']]
        # print(f'valid_v_rgb: {valid_v_rgb.shape}')
        if self.opt.model.pnet_pos_type == 'rel': # relative position w.r.t voxel center
            if self.opt.model.pnet_in == 6:
                pnet_inp = torch.cat((data_dict['valid_v_rel_coord'], valid_v_rgb),-1)
            elif self.opt.model.pnet_in == 3:
                pnet_inp = data_dict['valid_v_rel_coord']
        elif self.opt.model.pnet_pos_type == 'combine':
            assert self.opt.model.pnet_in == 6
            assert self.opt.model.pnet_model_type == 'twostage_pointfusion'
            valid_v_abs_coord = data_dict['valid_v_rel_coord'] + intersect_voxel_center[data_dict['revidx']]
            pnet_inp = torch.cat((data_dict['valid_v_rel_coord'], valid_v_abs_coord),-1)
        else:
            raise NotImplementedError('Does not support Pnet pos type: {}'.format(self.opt.model.pnet_pos_type))
        # pointnet forward
        if self.opt.model.pnet_model_type == 'twostage' or self.opt.model.pnet_model_type == 'pointattention':
            occ_voxel_feat = self.pnet_model(inp_feat=pnet_inp, vox2point_idx=data_dict['revidx'])
            intersect_voxel_feat = occ_voxel_feat[data_dict['occ_vox_intersect_idx']]
        elif self.opt.model.pnet_model_type == 'twostage_voxelfusion':
            occ_voxel_feat = self.pnet_model(inp_feat=pnet_inp, vox2point_idx=data_dict['revidx'])
            intersect_voxel_feat = occ_voxel_feat[data_dict['occ_vox_intersect_idx']]
            
            intersect_voxel_center_uv = self.GFM.xyz2uvd(intersect_voxel_center, self.camera_params)[:, :2]            
            intersect_voxel_center_ul = intersect_voxel_center_uv - self.opt.model.pnet_voxel_fusion.roi_out_bbox // 2
            intersect_voxel_center_br = intersect_voxel_center_uv + self.opt.model.pnet_voxel_fusion.roi_out_bbox // 2
            
            intersect_voxel_center_ul[:,0] = torch.clamp(intersect_voxel_center_ul[:,0], min=0., max=w-1)
            intersect_voxel_center_ul[:,1] = torch.clamp(intersect_voxel_center_ul[:,1], min=0., max=h-1)
            intersect_voxel_center_br[:,0] = torch.clamp(intersect_voxel_center_br[:,0], min=0., max=w-1)
            intersect_voxel_center_br[:,1] = torch.clamp(intersect_voxel_center_br[:,1], min=0., max=h-1)
            roi_boxes = torch.cat((miss_ray_intersect_bid.unsqueeze(-1), intersect_voxel_center_ul, intersect_voxel_center_br),-1).float()
            
            intersect_voxel_rgb_feat = tv_ops.roi_align(full_rgb_feat, roi_boxes,
                                            output_size=self.opt.model.pnet_voxel_fusion.roi_out_bbox,
                                            spatial_scale=spatial_scale,
                                            aligned=True)
            intersect_voxel_rgb_feat = intersect_voxel_rgb_feat.reshape(intersect_voxel_rgb_feat.shape[0],-1)
            intersect_voxel_feat = torch.cat([intersect_voxel_feat, intersect_voxel_rgb_feat], -1)
        
        elif self.opt.model.pnet_model_type == 'twostage_pointfusion':
            full_rgb_feat_flat = full_rgb_feat.permute(0,2,3,1).contiguous().reshape(bs,-1,full_rgb_feat.shape[1])
            point_rgb_feat = full_rgb_feat_flat[data_dict['valid_bid'], data_dict['valid_flat_img_id']][data_dict['valid_v_pid']]
            occ_voxel_feat = self.pnet_model(inp_feat=pnet_inp, vox2point_idx=data_dict['revidx'], point_rgb_feat=point_rgb_feat)
            intersect_voxel_feat = occ_voxel_feat[data_dict['occ_vox_intersect_idx']]
        else:
            raise NotImplementedError('Does not support pnet model type: {}'.format(self.opt.model.pnet_model_type))
        if self.opt.model.rgb_embedding_type == 'deep_fusion':
            intersect_voxel_feat = None
            data_dict.update({
                'occ_voxel_feat': occ_voxel_feat
            })
        
        
        '''  Hand_kpts Embedding '''
        if self.opt.model.use_hand_features:
            if not self.opt.model.use_pred_keypoints:
                if self.opt.exp_type == 'test_real':
                    hand_kpts_feat = data_dict['pred_keypoints_xyz']
                    hand_kpts_feat = hand_kpts_feat.to(self.device)
                else:
                    hand_kpts_feat = data_dict['hand_kpts'].squeeze(1) # B, 21, 3
            else:
                pred_kpts_offset, pred_kepts_feat = self.kpts_model(data_dict['depth_corrupt'])
                pred_kpts_uvd = self.GFM.offset2joint_weight(pred_kpts_offset, data_dict['depth_corrupt'], 0.8)
                hand_kpts_feat = self.GFM.uvd2xyz(pred_kpts_uvd, self.camera_params)
            
            if self.opt.model.use_2d_hand_features:
                hand_kpts_feat = hand_kpts_feat[:, :, :2]
            
            if self.opt.model.use_relative_pos:
                first_hand_kpt = hand_kpts_feat[:,0,:].unsqueeze(1)
                hand_kpts_feat_rel = hand_kpts_feat - first_hand_kpt
                hand_kpts_feat_rel = hand_kpts_feat_rel.reshape(bs, -1)
                intersect_hand_feat_rel_pos = hand_kpts_feat_rel[data_dict['miss_bid'][data_dict['miss_ray_intersect_idx']]]
                data_dict.update({
                    'hand_kpts_feat_rel_pos': hand_kpts_feat_rel})
            
            hand_kpts_feat = hand_kpts_feat.reshape(bs, -1)
            hand_kpts_feat_bid = data_dict['miss_bid'][data_dict['miss_ray_intersect_idx']]
            intersect_hand_feat = hand_kpts_feat[hand_kpts_feat_bid]
            
            if self.opt.model.use_kpts_encoder:
                intersect_hand_feat = self.kpts_encoder(intersect_hand_feat)
            if self.opt.model.use_relative_hand_feature:
                if self.opt.model.use_2d_hand_features:
                    intersect_hand_feat_relative = intersect_voxel_center[:, :2].repeat(1, 21)
                else:
                    intersect_hand_feat_relative = intersect_voxel_center.repeat(1, 21)
                intersect_hand_feat_relative = intersect_hand_feat - intersect_hand_feat_relative
                if self.opt.model.use_relative_pos:
                    intersect_hand_feat = torch.cat([intersect_hand_feat_rel_pos, intersect_hand_feat_relative], -1)
                else:
                    intersect_hand_feat = torch.cat([intersect_hand_feat, intersect_hand_feat_relative], -1)
        else:
            intersect_hand_feat = None
            hand_kpts_feat = None
        # print(f"intersect_hand_feat: {intersect_hand_feat.shape}")
        # print(f'intersect_voxel_center: {intersect_voxel_center.shape}')
        # print(f"intersect_hand_feat_relative: {intersect_hand_feat_relative.shape}")
        # raise NotImplementedError('stop')
        
        # uncertainty embedding
        if self.opt.loss.use_uncertainty_loss:
            miss_ray_intersect_img_ind = data_dict['miss_img_ind']
            miss_ray_intersect_bid = data_dict['miss_bid']
            
            miss_ray_intersect_ul = miss_ray_intersect_img_ind - self.opt.model.roi_inp_bbox // 2
            miss_ray_intersect_br = miss_ray_intersect_img_ind + self.opt.model.roi_inp_bbox // 2
            # clamp is done in original image coords
            miss_ray_intersect_ul[:,0] = torch.clamp(miss_ray_intersect_ul[:,0], min=0., max=w-1)
            miss_ray_intersect_ul[:,1] = torch.clamp(miss_ray_intersect_ul[:,1], min=0., max=h-1)
            miss_ray_intersect_br[:,0] = torch.clamp(miss_ray_intersect_br[:,0], min=0., max=w-1)
            miss_ray_intersect_br[:,1] = torch.clamp(miss_ray_intersect_br[:,1], min=0., max=h-1)
            roi_boxes = torch.cat((miss_ray_intersect_bid.unsqueeze(-1), miss_ray_intersect_ul, miss_ray_intersect_br),-1).float()
            intersect_rgb_feat_uncer = tv_ops.roi_align(full_rgb_feat, roi_boxes, 
                                    output_size=self.opt.model.roi_out_bbox,
                                    spatial_scale=spatial_scale,
                                    aligned=True)
            intersect_rgb_feat_uncer = intersect_rgb_feat_uncer.reshape(intersect_rgb_feat_uncer.shape[0],-1)
            intersect_voxel_feat_uncer = self.voxel_2_ray(intersect_voxel_feat, data_dict['miss_ray_intersect_idx'])
            data_dict.update({
                'intersect_rgb_feat_uncer': intersect_rgb_feat_uncer,
                'intersect_voxel_feat_uncer': intersect_voxel_feat_uncer
            })
        
        # update data_dict
        data_dict.update({
            'intersect_dir': intersect_dir,
            'intersect_enter_dist': intersect_enter_dist,
            'intersect_leave_dist': intersect_leave_dist,
            'intersect_enter_pos': intersect_enter_pos,
            'intersect_leave_pos': intersect_leave_pos,
            'intersect_enter_pos_embed': intersect_enter_pos_embed,
            'intersect_leave_pos_embed': intersect_leave_pos_embed,
            'intersect_dir_embed': intersect_dir_embed,
            'full_rgb_feat': full_rgb_feat,
            'intersect_rgb_feat': intersect_rgb_feat,
            'intersect_voxel_feat': intersect_voxel_feat,
            'intersect_hand_feat': intersect_hand_feat,
            'full_hand_feat': hand_kpts_feat,
        })

    def feature_fusion(self, data_dict):
        if self.opt.model.fusion.fusion_type == 'concat':
            fused_feature = torch.cat((data_dict['intersect_voxel_feat'].contiguous(), data_dict['intersect_rgb_feat'].contiguous()), -1)
            # if self.opt.model.use_hand_features:
            #     fused_feature = torch.cat((fused_feature, data_dict['intersect_hand_feat'].contiguous()), -1)
        elif self.opt.model.fusion.fusion_type == 'adaptive_fusion':
            fused_feature = self.fusion_model(data_dict)
        elif self.opt.model.fusion.fusion_type == 'cross_atten':
            fused_feature = self.fusion_model(tuple([data_dict['intersect_rgb_feat'], data_dict['intersect_voxel_feat']]))
            fused_feature = fused_feature.squeeze(1)
        elif self.opt.model.fusion.fusion_type == 'deep_fusion':
            fused_voxel_feature_list = []
            for b in range(data_dict['bs']):
                rgb_feature = data_dict['full_rgb_feat'][b].unsqueeze(0)
                voxel_feature = data_dict['occ_voxel_feat'][data_dict['occ_vox_bid']==b].unsqueeze(0)
                fused_voxel_feature = self.fusion_model(rgb_feature.contiguous(), voxel_feature.contiguous())
                fused_voxel_feature_list.append(fused_voxel_feature)
            fused_voxel_feature = torch.cat(fused_voxel_feature_list, 1).squeeze(0)
            # print(f'fused_feature: {fused_feature.shape}')
            fused_feature = fused_voxel_feature[data_dict['occ_vox_intersect_idx']]
        elif self.opt.model.fusion.fusion_type == 'gated_fusion':
            fused_feature = self.fusion_model(data_dict['intersect_voxel_feat'].contiguous(), data_dict['intersect_rgb_feat'].contiguous())
        
        else:
            raise NotImplementedError('Does not support Fusion Type: {}'.format(self.opt.model.fusion.fusion_type))
        return fused_feature
    
    def get_pred(self, data_dict, exp_type, epoch):
        ######################################################## 
        # Concat embedding and send to decoder 
        ########################################################
        if self.opt.model.fusion.use_fusion:
            fused_feature = self.feature_fusion(data_dict)
            if self.opt.model.use_hand_features:
                fused_feature = torch.cat((fused_feature, data_dict['intersect_hand_feat'].contiguous()), -1)
            inp_embed = torch.cat((fused_feature.contiguous(), data_dict['intersect_enter_pos_embed'].contiguous(),
                                data_dict['intersect_leave_pos_embed'].contiguous(), data_dict['intersect_dir_embed'].contiguous()),-1)
        else:
            if not self.opt.model.use_hand_features:
                inp_embed = torch.cat(( data_dict['intersect_voxel_feat'].contiguous(), data_dict['intersect_rgb_feat'].contiguous(),
                                    data_dict['intersect_enter_pos_embed'].contiguous(),
                                    data_dict['intersect_leave_pos_embed'].contiguous(), data_dict['intersect_dir_embed'].contiguous()),-1)
            else:
                inp_embed = torch.cat(( data_dict['intersect_voxel_feat'].contiguous(), data_dict['intersect_rgb_feat'].contiguous(), 
                                    data_dict['intersect_hand_feat'].contiguous(),
                                        data_dict['intersect_enter_pos_embed'].contiguous(),
                                        data_dict['intersect_leave_pos_embed'].contiguous(), data_dict['intersect_dir_embed'].contiguous()),-1)
        # print(f"inp_embed: {inp_embed.shape}")
        
        pred_offset = self.offset_dec(inp_embed)
        pred_prob_end = self.prob_dec(inp_embed)
        
        if self.opt.loss.use_uncertainty_loss:
            inp_embed_uncer = torch.cat(( data_dict['intersect_rgb_feat_uncer'].contiguous(), 
                                         data_dict['intersect_voxel_feat_uncer'].contiguous()), -1)
            pred_uncer = self.uncer_dec(inp_embed_uncer)
            data_dict.update({
                'pred_uncer': pred_uncer
            })
        
        # scale pred_offset from (0,1) to (offset_range[0], offset_range[1]).
        pred_scaled_offset = pred_offset * (self.opt.grid.offset_range[1] - self.opt.grid.offset_range[0]) + self.opt.grid.offset_range[0]
        pred_scaled_offset = pred_scaled_offset * np.sqrt(3) * data_dict['part_size']
        pair_pred_pos = data_dict['intersect_enter_pos'] + pred_scaled_offset * data_dict['intersect_dir']
        # we detach the pred_prob_end. we don't want pos loss to affect ray terminate score.
        if self.opt.loss.prob_loss_type == 'ray':
            pred_prob_end_softmax = scatter_softmax(pred_prob_end.detach()[:,0], data_dict['miss_ray_intersect_idx'])
        # training uses GT pcl_label to get max_pair_id (voxel with largest prob)
        if exp_type == 'train' and epoch < self.opt.model.maxpool_label_epo:
            _, max_pair_id = scatter_max(data_dict['pcl_label_float'], data_dict['miss_ray_intersect_idx'],
                                dim_size=data_dict['total_miss_sample_num'])
        # test/valid uses pred_prob_end_softmax to get max_pair_id (voxel with largest prob)
        else:
            _, max_pair_id = scatter_max(pred_prob_end_softmax, data_dict['miss_ray_intersect_idx'],
                            dim_size=data_dict['total_miss_sample_num'])
        if self.opt.model.scatter_type == 'Maxpool':
            dummy_pos = torch.zeros([1,3]).float().to(self.device)
            pair_pred_pos_dummy = torch.cat((pair_pred_pos, dummy_pos),0)
            pred_pos = pair_pred_pos_dummy[max_pair_id]    
        else:
            raise NotImplementedError('Does not support Scatter Type: {}'.format(self.opt.model.scatter_type))
        
        assert pred_pos.shape[0] == data_dict['total_miss_sample_num']
        # update data_dict
        data_dict.update({
            'pair_pred_pos': pair_pred_pos,
            'max_pair_id': max_pair_id,
            'pred_prob_end': pred_prob_end,
            'pred_prob_end_softmax': pred_prob_end_softmax,
            'pred_pos': pred_pos,
        })
        # print(f"pred_offset: {pred_offset.shape}, pred_prob_end: {pred_prob_end.shape}, pred_pos: {pred_pos.shape}")

    def compute_loss(self, data_dict, exp_type, epoch):
        if self.opt.exp_type == 'test_real':
            return
        bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
        ''' position loss '''
        if self.opt.loss.pos_loss_type == 'single':
            if not self.opt.loss.hard_neg:
                if not self.opt.loss.use_uncertainty_loss:
                    if not self.opt.loss.use_depth_pos_loss:
                        pos_loss = self.pos_loss_fn(data_dict['pred_pos'], data_dict['gt_pos'])
                    else:
                        pos_loss = self.pos_loss_fn(data_dict['pred_pos'][:, 2], data_dict['gt_pos'][:, 2])
                else: # uncertainty loss
                    # s = 2 * torch.log(data_dict['pred_uncer'] + 1e-6)
                    s = data_dict['pred_uncer']
                    pos_loss = torch.mean(torch.exp(-s) * abs(data_dict['pred_pos'] - data_dict['gt_pos']) + self.opt.loss.uncertainty_lambda * s)
            else:
                pos_loss_unreduce = torch.mean((data_dict['pred_pos'] - data_dict['gt_pos']).abs(),-1)
                k = int(pos_loss_unreduce.shape[0] * self.opt.loss.hard_neg_ratio)
                pos_loss_topk,_ = torch.topk(pos_loss_unreduce, k)
                pos_loss = torch.mean(pos_loss_topk)
        elif self.opt.loss.pos_loss_type == 'combined':
            pos_loss = self.pos_loss_fn(data_dict['pred_pos'], data_dict['gt_pos'])

        ''' Ending probability loss '''
        if self.opt.loss.prob_loss_type == 'ray':
            pred_prob_end_log_softmax = scatter_log_softmax(data_dict['pred_prob_end'][:,0], data_dict['miss_ray_intersect_idx'])
            pcl_label_idx = torch.nonzero(data_dict['pcl_label'], as_tuple=False).reshape(-1)
            prob_loss_unreduce = -1*pred_prob_end_log_softmax[pcl_label_idx]
            if not self.opt.loss.hard_neg:
                prob_loss = torch.mean(prob_loss_unreduce)
            else:
                k = int(prob_loss_unreduce.shape[0] * self.opt.loss.hard_neg_ratio)
                prob_loss_topk,_ = torch.topk(prob_loss_unreduce, k)
                prob_loss = torch.mean(prob_loss_topk)
            
        ''' surface normal loss '''
        if exp_type == 'train':
            gt_pcl = data_dict['xyz_flat'].clone()
            pred_pcl = data_dict['xyz_flat'].clone()
        else:
            gt_pcl = data_dict['xyz_corrupt_flat'].clone()
            pred_pcl = data_dict['xyz_corrupt_flat'].clone()        
        
        gt_pcl[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['gt_pos']
        gt_pcl = gt_pcl.reshape(bs,h,w,3).permute(0,3,1,2).contiguous()
        gt_surf_norm_img,_,_ = point_utils.get_surface_normal(gt_pcl)
        gt_surf_norm_flat = gt_surf_norm_img.permute(0,2,3,1).contiguous().reshape(bs,h*w,3)
        gt_surf_norm = gt_surf_norm_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]

        pred_pcl[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['pred_pos']
        pred_pcl = pred_pcl.reshape(bs,h,w,3).permute(0,3,1,2).contiguous()
        pred_surf_norm_img, dx, dy = point_utils.get_surface_normal(pred_pcl)
        pred_surf_norm_flat = pred_surf_norm_img.permute(0,2,3,1).contiguous().reshape(bs,h*w,3)
        pred_surf_norm = pred_surf_norm_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]

        # surface normal loss
        cosine_val = F.cosine_similarity(pred_surf_norm, gt_surf_norm, dim=-1)
        surf_norm_dist = (1 - cosine_val) / 2.
        if not self.opt.loss.hard_neg:
            surf_norm_loss = torch.mean(surf_norm_dist)
        else:
            k = int(surf_norm_dist.shape[0] * self.opt.loss.hard_neg_ratio)
            surf_norm_dist_topk,_ = torch.topk(surf_norm_dist, k)
            surf_norm_loss = torch.mean(surf_norm_dist_topk)
        # angle err
        angle_err = torch.mean(torch.acos(torch.clamp(cosine_val,min=-1,max=1)))
        angle_err = angle_err / np.pi * 180.

        # smooth loss
        dx_dist = torch.sum(dx*dx,1)
        dx_dist_flat = dx_dist.reshape(bs,h*w)
        miss_dx_dist = dx_dist_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]
        
        dy_dist = torch.sum(dy*dy,1)
        dy_dist_flat = dy_dist.reshape(bs,h*w)
        miss_dy_dist = dy_dist_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]
        
        if not self.opt.loss.hard_neg:
            smooth_loss = torch.mean(miss_dx_dist) + torch.mean(miss_dy_dist)
        else:
            k = int(miss_dx_dist.shape[0] * self.opt.loss.hard_neg_ratio)
            miss_dx_dist_topk,_ = torch.topk(miss_dx_dist, k)
            miss_dy_dist_topk,_ = torch.topk(miss_dy_dist, k)
            smooth_loss = torch.mean(miss_dx_dist_topk) + torch.mean(miss_dy_dist_topk)
        
        ''' loss net '''
        loss_net = self.opt.loss.pos_w * pos_loss + self.opt.loss.prob_w * prob_loss
        if self.opt.loss.surf_norm_w > 0 and epoch >= self.opt.loss.surf_norm_epo:
            loss_net += self.opt.loss.surf_norm_w * surf_norm_loss
        if self.opt.loss.smooth_w > 0 and epoch >= self.opt.loss.smooth_epo:
            loss_net += self.opt.loss.smooth_w * smooth_loss

        
        #######################
        # Evaluation Metric
        #######################
        # ending accuracy for missing point
        _, pred_label = scatter_max(data_dict['pred_prob_end_softmax'], data_dict['miss_ray_intersect_idx'],
                                dim_size=data_dict['total_miss_sample_num'])
        _, gt_label = scatter_max(data_dict['pcl_label'], data_dict['miss_ray_intersect_idx'],
                                dim_size=data_dict['total_miss_sample_num'])
        acc = torch.sum(torch.eq(pred_label, gt_label).float()) / torch.numel(pred_label)

        # position L2 error: we don't want to consider 0 depth point in the position L2 error.
        zero_mask = torch.sum(data_dict['gt_pos'].abs(),dim=-1)
        zero_mask[zero_mask!=0] = 1.
        elem_num = torch.sum(zero_mask)
        if elem_num.item() == 0:
            err = torch.Tensor([0]).float().to(self.device)
        else:
            err = torch.sum(torch.sqrt(torch.sum((data_dict['pred_pos'] - data_dict['gt_pos'])**2,-1))*zero_mask) / elem_num
        # compute depth errors following cleargrasp
        zero_mask_idx = torch.nonzero(zero_mask, as_tuple=False).reshape(-1)


        if exp_type != 'train':
            gt_xyz = data_dict['xyz_flat'].clone()
            gt_xyz = gt_xyz.reshape(bs,h,w,3)
            gt_depth = gt_xyz[:,:,:,2].unsqueeze(1)
            pred_xyz = data_dict['xyz_corrupt_flat'].clone()
            pred_xyz[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['pred_pos']
            pred_xyz = pred_xyz.reshape(bs,h,w,3)
            pred_depth = pred_xyz[:,:,:,2].unsqueeze(1)
            seg_mask = data_dict['corrupt_mask'].unsqueeze(1)
            data_dict.update({'gt_depth_img': gt_depth,
                              'pred_depth_img': pred_depth,})
        
            a1, a2, a3, rmse, rmse_log, log10, abs_rel, abs_rel, mae, sq_rel, num_valid = \
                loss_utils.get_metrics_depth_restoration_inference(gt_depth, pred_depth, 224, 126, seg_mask)
            
            if self.opt.loss.use_uncertainty_loss:
                pred_uncer_img = torch.zeros_like(data_dict['xyz_corrupt_flat'])
                pred_uncer_img[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['pred_uncer']
                pred_uncer_img = pred_uncer_img.reshape(bs,h,w,3)
                pred_uncer_img = pred_uncer_img[:,:,:,0].unsqueeze(1)
                data_dict.update({'pred_uncer_img': pred_uncer_img})
            

        # update data_dict
        data_dict.update({
            'zero_mask_idx': zero_mask_idx,
            'gt_surf_norm_img': gt_surf_norm_img,
            'pred_surf_norm_img': pred_surf_norm_img
        })

        # loss dict
        loss_dict = {
            'pos_loss': pos_loss,
            'prob_loss': prob_loss,
            'surf_norm_loss': surf_norm_loss,
            'smooth_loss': smooth_loss,
            'loss_net': loss_net,
            'acc': acc,
            'err': err,
            'angle_err': angle_err,
        }
        if exp_type != 'train':
            loss_dict['num_valid'] = num_valid
            loss_dict.update({
                'a1': a1,
                'a2': a2,
                'a3': a3,
                'rmse': rmse,
                'rmse_log': rmse_log,
                'log10': log10,
                'abs_rel': abs_rel,
                'mae': mae,
                'sq_rel': sq_rel,
            })

        return loss_dict

    def forward(self, batch, exp_type, epoch, pred_mask=None):
        loss_dict = {}
        # prepare input and gt data
        data_dict = self.prepare_data(batch, exp_type, pred_mask)
        
        # get valid points data
        self.get_valid_points(data_dict)
        
        # get occupied voxel data
        occ_vox_flag = self.get_occ_vox_bound(data_dict)
        if exp_type == 'train' and self.opt.dist.ddp:
            # have to set barrier to wait for all processes finished forward pass
            dist.barrier()
            success_num = torch.Tensor([occ_vox_flag]).to(self.device)
            dist.all_reduce(success_num, op=dist.ReduceOp.SUM)
            # at least one gpu fails: clear grad buffer and return
            if success_num[0] < self.opt.dist.ngpus_per_node:
                print('gpu {}: {}'.format(self.opt.gpu_id, success_num[0]))
                return False, data_dict, loss_dict
        elif not occ_vox_flag:
            return False, data_dict, loss_dict
        
        # get miss ray data
        self.get_miss_ray(data_dict, exp_type)
        miss_sample_flag = (data_dict['total_miss_sample_num'] != 0)
        if exp_type == 'train' and self.opt.dist.ddp:
            # have to set barrier to wait for all processes finished forward pass
            dist.barrier()
            success_num = torch.Tensor([miss_sample_flag]).to(self.device)
            dist.all_reduce(success_num, op=dist.ReduceOp.SUM)
            # at least one gpu fails: clear grad buffer and return
            if success_num[0] < self.opt.dist.ngpus_per_node:
                print('gpu {}: {}'.format(self.opt.gpu_id, success_num[0]))
                return False, data_dict, loss_dict
        elif not miss_sample_flag:
            return False, data_dict, loss_dict

        # ray AABB slab test
        intersect_pair_flag = self.compute_ray_aabb(data_dict)
        if exp_type == 'train' and self.opt.dist.ddp:
            # have to set barrier to wait for all processes finished forward pass
            dist.barrier()
            success_num = torch.Tensor([intersect_pair_flag]).to(self.device)
            dist.all_reduce(success_num, op=dist.ReduceOp.SUM)
            # at least one gpu fails: clear grad buffer and return
            if success_num[0] < self.opt.dist.ngpus_per_node:
                print('gpu {}: {}'.format(self.opt.gpu_id, success_num[0]))
                return False, data_dict, loss_dict
        elif not intersect_pair_flag:
            return False, data_dict, loss_dict
        
        # compute gt
        self.compute_gt(data_dict)
        # get embedding
        self.get_embedding(data_dict)
        # get prediction
        self.get_pred(data_dict, exp_type, epoch)
        # compute loss
        loss_dict = self.compute_loss(data_dict, exp_type, epoch)
        return True, data_dict, loss_dict