import torch
import torch.nn as nn
from models.keypoints.loss import SmoothL1Loss
from utils.training_utils import *
from models.keypoints.resnetUnet import OfficialResNetUnet, OfficialResNetUnet_RGB2offset_3D
from models.keypoints.generate_features import GFM

# from matplotlib import pyplot as plt

class Pipeline(nn.Module):
    def __init__(self, opt, device):
        super(Pipeline, self).__init__()
        self.opt = opt
        self.device = device
        
        # build models
        self.build_model()

    def build_model(self):
        joint_num = self.opt.model.num_hand_kpts
        
        if self.opt.model.keypoints.use_rgb:
            self.backbone_rgb = OfficialResNetUnet_RGB2offset_3D(self.opt.model.keypoints.net, 
                                                            joint_num, 
                                                            pretrain=True, deconv_dim=128,
                                                            out_dim_list=[joint_num * 3, joint_num, joint_num]).to(self.device)
        if self.opt.model.keypoints.use_depth:
            self.backbone_d = OfficialResNetUnet(self.opt.model.keypoints.net, 
                                                joint_num, pretrain=True, 
                                                deconv_dim=128,
                                                out_dim_list=[joint_num * 3, joint_num, joint_num]).to(self.device)
        self.GFM = GFM()
        # loss functions
        self.L1Loss = SmoothL1Loss()
        self.BECLoss = torch.nn.BCEWithLogitsLoss()
        self.L2Loss = torch.nn.MSELoss()
        
    def prepare_data(self, batch, exp_type, pred_mask):
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
        # xyz_flat = xyz.permute(0, 2, 3, 1).contiguous().reshape(bs,-1,3)
        # xyz_corrupt_flat = xyz_corrupt.permute(0, 2, 3, 1).contiguous().reshape(bs,-1,3)
    
        self.params = (batch['fx'][0].item(), batch['fy'][0].item(), batch['cx'][0].item(), batch['cy'][0].item())
        
        # arrange data in a dictionary
        data_dict = {
            'bs': bs,
            'h': h,
            'w': w,
            'rgb_img': rgb_img,
            'xyz_corrupt': xyz_corrupt,
            'corrupt_mask': corrupt_mask,
            'valid_mask': valid_mask,
            'hand_mask': batch['hand_mask'],
            # 'xyz_flat': xyz_flat,
            # 'xyz_corrupt_flat': xyz_corrupt_flat,
            'depth': batch['depth'],
            'depth_corrupt': batch['depth_corrupt'],
            'fx': batch['fx'].float(),
            'fy': batch['fy'].float(),
            'cx': batch['cx'].float(),
            'cy': batch['cy'].float(),
            'xres': batch['xres'].float(),
            'yres': batch['yres'].float(),
            'hand_kpts': batch['hand_kpts'],
            'hand_kpts_uv': batch['hand_kpts_uv'],
        }
        
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
    
    def forward(self, batch, exp_type, epoch, pred_mask=None):
        # prepare input and gt data
        data_dict = self.prepare_data(batch, exp_type, pred_mask)
        
        if self.opt.model.keypoints.use_rgb:
            pred_rgb_offset, pred_rgb_feat = self.backbone_rgb(data_dict['rgb_img'])
            pred_rgb_keypoints_uvd = self.GFM.offset2joint_weight(pred_rgb_offset, data_dict['depth'], 0.8)
            pred_rgb_keypoints_xyz = self.GFM.uvd2xyz(pred_rgb_keypoints_uvd, self.params)
            data_dict['pred_rgb_keypoints_uvd'] = pred_rgb_keypoints_uvd
            data_dict['pred_rgb_keypoints_xyz'] = pred_rgb_keypoints_xyz
        
        if self.opt.model.keypoints.use_depth:
            if self.opt.model.keypoints.use_corrupt_depth:
                pred_depth_offset, pred_depth_feat = self.backbone_d(data_dict['depth_corrupt'])
                pred_depth_keypoints_uvd = self.GFM.offset2joint_weight(pred_depth_offset, data_dict['depth_corrupt'], 0.8)
            elif self.opt.model.keypoints.use_hand_depth:
                hand_depth = data_dict['hand_mask'] / 255 * data_dict['depth_corrupt']
                pred_depth_offset, pred_depth_feat = self.backbone_d(hand_depth)
                pred_depth_keypoints_uvd = self.GFM.offset2joint_weight(pred_depth_offset, hand_depth, 0.8)
            else:
                pred_depth_offset, pred_depth_feat = self.backbone_d(data_dict['depth'])    
            
            pred_depth_keypoints_xyz = self.GFM.uvd2xyz(pred_depth_keypoints_uvd, self.params)
            data_dict['pred_depth_keypoints_uvd'] = pred_depth_keypoints_uvd
            data_dict['pred_depth_keypoints_xyz'] = pred_depth_keypoints_xyz
    
        
        
        # compute loss
        loss_dict = self.compute_loss(data_dict, epoch)
        
        return True, data_dict, loss_dict
    
    
    def compute_loss(self, data_dict, epoch):
        loss_dict = {}
        if self.opt.model.keypoints.use_rgb:
            kpts_loss_rgb = self.L2Loss(data_dict['pred_rgb_keypoints_xyz'], data_dict['hand_kpts'].squeeze(1))
            error_rgb = self.L1Loss(data_dict['pred_rgb_keypoints_xyz'], data_dict['hand_kpts'].squeeze(1))
            x_error_rgb = self.L1Loss(data_dict['pred_rgb_keypoints_xyz'][:, :, 0], data_dict['hand_kpts'].squeeze(1)[:, :, 0])
            y_error_rgb = self.L1Loss(data_dict['pred_rgb_keypoints_xyz'][:, :, 1], data_dict['hand_kpts'].squeeze(1)[:, :, 1])
            z_error_rgb = self.L1Loss(data_dict['pred_rgb_keypoints_xyz'][:, :, 2], data_dict['hand_kpts'].squeeze(1)[:, :, 2])
            loss_net = kpts_loss_rgb
            loss_dict.update({
                'loss_net': loss_net,
                'error': error_rgb,
                'x_error': x_error_rgb,
                'y_error': y_error_rgb,
                'z_error': z_error_rgb,
                'z_error_rgb': z_error_rgb,
                'kpts_loss_rgb': kpts_loss_rgb,
            }
            )
            
        
        if self.opt.model.keypoints.use_depth:
            kpts_loss_depth = self.L2Loss(data_dict['pred_depth_keypoints_xyz'], data_dict['hand_kpts'].squeeze(1))
            loss_net = kpts_loss_depth
            error_depth = self.L1Loss(data_dict['pred_depth_keypoints_xyz'], data_dict['hand_kpts'].squeeze(1))
            x_error_depth = self.L1Loss(data_dict['pred_depth_keypoints_xyz'][:, :, 0], data_dict['hand_kpts'].squeeze(1)[:, :, 0])
            y_error_depth = self.L1Loss(data_dict['pred_depth_keypoints_xyz'][:, :, 1], data_dict['hand_kpts'].squeeze(1)[:, :, 1])
            z_error_depth = self.L1Loss(data_dict['pred_depth_keypoints_xyz'][:, :, 2], data_dict['hand_kpts'].squeeze(1)[:, :, 2])
            loss_dict = {
                'loss_net': loss_net,
                'error': error_depth,
                'x_error': x_error_depth,
                'y_error': y_error_depth,
                'z_error': z_error_depth,
                'kpts_loss_depth': kpts_loss_depth,
            }
        
        return loss_dict
    