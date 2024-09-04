import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.point_utils import get_surface_normal

def mse_loss(pred, gt, reduction='mean'):
    return F.mse_loss(pred, gt, reduction=reduction)

def l1_loss(pred, gt, reduction='mean'):
    return F.l1_loss(pred, gt, reduction=reduction)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, output, target):
        l1_loss = self.l1_loss(output, target)
        mse_loss = self.mse_loss(output, target)
        combined_loss = l1_loss + mse_loss
        return combined_loss

def masked_mse_loss(pred, gt, mask, reduction='mean'):
    ''' pred, gt, mask should be broadcastable, mask is 0-1 '''
    diff = (pred - gt)**2
    if reduction == 'mean':
        ele_num = torch.sum(mask)
        # avoid divide by 0
        if ele_num.item() == 0:
            ele_num += 1e-8
        return torch.sum(mask * diff) / ele_num
    else:
        return torch.sum(mask * diff)

def masked_l1_loss(pred, gt, mask, reduction='mean'):
    ''' pred, gt, mask should be broadcastable, mask is 0-1 '''
    diff = torch.abs(pred - gt)
    if reduction == 'mean':
        ele_num = torch.sum(mask)
        # avoid divide by 0
        if ele_num.item() == 0:
            ele_num += 1e-8
        return torch.sum(mask * diff) / ele_num
    else:
        return torch.sum(mask * diff)

def rmse_depth(pred, gt):
    '''pred, gt: (N,H,W) '''
    diff = (pred - gt)**2
    rmse_batch = torch.sqrt(torch.mean(diff, [1,2]))
    rmse_error = torch.mean(rmse_batch)
    return rmse_error

def masked_rmse_depth(pred, gt, mask):
    '''pred, gt, mask: (N,H,W) '''
    diff = (pred - gt)**2
    ele_num = torch.sum(mask, [1,2])
    rmse_batch = torch.sqrt(torch.sum(diff*mask, [1,2]) / (ele_num+1e-8))
    rmse_error = torch.mean(rmse_batch)
    return rmse_error

def get_metrics_depth_restoration_inference(gt, pred, width, height, seg_mask=None):
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
        pred = pred.unsqueeze(0)
    if seg_mask.dim() == 3:
        seg_mask = seg_mask.unsqueeze(0)
    B = gt.shape[0]
    gt = F.interpolate(gt, size=[height, width], mode="nearest")
    pred = F.interpolate(pred, size=[height, width], mode="nearest")

    gt = gt.detach().permute(0, 2, 3, 1).cpu().numpy().astype("float32")
    pred = pred.detach().permute(0, 2, 3, 1).cpu().numpy().astype("float32")
    if not seg_mask is None:
        seg_mask = seg_mask.float()
        seg_mask = F.interpolate(seg_mask, size=[height, width], mode="nearest")
        seg_mask = seg_mask.detach().cpu().permute(0, 2, 3, 1).numpy()

    gt_depth = gt
    pred_depth = pred
    gt_depth[np.isnan(gt_depth)] = 0
    gt_depth[np.isinf(gt_depth)] = 0
    mask_valid_region = (gt_depth > 0)

    if not seg_mask is None:
        seg_mask = seg_mask.astype(np.uint8)
        mask_valid_region = np.logical_and(mask_valid_region, seg_mask)

    gt = torch.from_numpy(gt_depth).float().cuda()
    pred = torch.from_numpy(pred_depth).float().cuda()
    mask = torch.from_numpy(mask_valid_region).bool().cuda()

    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    rmse = 0.0
    rmse_log = 0.0
    log10 = 0.0
    abs_rel = 0.0
    mae = 0.0
    sq_rel =0.0
    
    safe_log = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
    safe_log10 = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))

    num_valid = 0

    for i in range(B):
        gt_i = gt[i][mask[i]]
        pred_i = pred[i][mask[i]]

        if len(gt_i) > 0:
            num_valid += 1
            thresh = torch.max(gt_i / pred_i, pred_i / gt_i)

            a1_i = (thresh < 1.05).float().mean()
            a2_i = (thresh < 1.10).float().mean()
            a3_i = (thresh < 1.25).float().mean()

            rmse_i = ((gt_i - pred_i) ** 2).mean().sqrt()
            rmse_log_i = ((safe_log(gt_i) - safe_log(pred_i)) ** 2).mean().sqrt()
            log10_i = ((safe_log10(gt_i) - safe_log10(pred_i)) ** 2).mean().sqrt()
            abs_rel_i = ((gt_i - pred_i).abs() / gt_i).mean()
            mae_i = (gt_i - pred_i).abs().mean()
            sq_rel_i = (((gt_i - pred_i) ** 2) / gt_i).mean()
            
            a1 += a1_i
            a2 += a2_i
            a3 += a3_i
            rmse += rmse_i
            rmse_log += rmse_log_i
            log10 += log10_i
            abs_rel += abs_rel_i
            mae += mae_i
            sq_rel += sq_rel_i

    return a1, a2, a3, rmse, rmse_log, log10, abs_rel, abs_rel, mae, sq_rel, num_valid



class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        edge_ky = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        edge_k = np.stack((edge_kx, edge_ky))

        # edge_k = torch.from_numpy(edge_k).double().view(2, 1, 3, 3)
        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        n, c, h, w = x.shape
            x is n examples, each have h*w pixels, and each pixel contain c=1 channel value

        n, 2, h, w = out.shape
            2 channel: first represents dx, second represents dy
        """
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = Sobel().cuda()

    def forward(self, input, target):
        input_grad = self.sobel(input)
        target_grad = self.sobel(target)
        #  n, 2, h, w = out.shape
        #  2 channel: first represents dx, second represents dy

        input_grad_dx = input_grad[:, 0, :, :].contiguous().view_as(input)
        input_grad_dy = input_grad[:, 1, :, :].contiguous().view_as(input)
        target_grad_dx = target_grad[:, 0, :, :].contiguous().view_as(target)
        target_grad_dy = target_grad[:, 1, :, :].contiguous().view_as(target)

        # loss_dx = torch.abs(input_grad_dx - target_grad_dx).mean()#torch.log(torch.abs(input_grad_dx - target_grad_dx) + 0.5).mean()
        # loss_dy = torch.abs(input_grad_dy - target_grad_dy).mean()#torch.log(torch.abs(input_grad_dy - target_grad_dy) + 0.5).mean()

        loss_dx = torch.abs(input_grad_dx - target_grad_dx)  # torch.log(torch.abs(input_grad_dx - target_grad_dx) + 0.5).mean()
        loss_dy = torch.abs(input_grad_dy - target_grad_dy)  # torch.log(torch.abs(input_grad_dy - target_grad_dy) + 0.5).mean()

        # loss_dx = torch.log(torch.abs(input_grad_dx - target_grad_dx) + 0.5)
        # loss_dy = torch.log(torch.abs(input_grad_dy - target_grad_dy) + 0.5)     

        # loss_dx = torch.log(torch.abs(input_grad_dx - target_grad_dx) + 1.0)
        # loss_dy = torch.log(torch.abs(input_grad_dy - target_grad_dy) + 1.0)

        return loss_dx + loss_dy

# def compute_xyz(depth_img, camera_params):
#     """ Compute ordered point cloud from depth image and camera parameters.

#         If focal lengths fx,fy are stored in the camera_params dictionary, use that.
#         Else, assume camera_params contains parameters used to generate synthetic data (e.g. fov, near, far, etc)

#         @param depth_img: a [H x W] numpy array of depth values in meters
#         @param camera_params: a dictionary with parameters of the camera used
#     """
#     # Compute focal length from camera parameters
#     fx = camera_params['fx']
#     fy = camera_params['fy']
#     x_offset = camera_params['cx']
#     y_offset = camera_params['cy']
#     indices = np.indices((int(camera_params['yres']), int(camera_params['xres'])), dtype=np.float32).transpose(1, 2,
#                                                                                                                 0)
#     z_e = depth_img
#     x_e = (indices[..., 1] - x_offset) * z_e / fx
#     y_e = (indices[..., 0] - y_offset) * z_e / fy
#     xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
#     return xyz_img

# def depth_loss(syn_ds, ins_masks, pred_ds, pred_ds_initial, camera_params, depth_restoration_train_with_masks):
#     loss_instance_weight = 1.0
#     loss_background_weight = 0.4
#     loss_normal_weight = 0.1
#     loss_grad_weight = 0.6
#     loss_weight_d_initial = 0.0
#     l1 = nn.L1Loss(reduction='none')
#     huber_loss_fn = nn.SmoothL1Loss(reduction='none')
#     grad_loss = GradientLoss()
    
#     syn_xyz = compute_xyz(syn_ds, camera_params)
#     pred_xyz = compute_xyz(pred_ds, camera_params)
#     pred_xyz_inital = compute_xyz(pred_ds_initial, camera_params)
    
#     normal_labels, _, _ = get_surface_normal(syn_xyz)
#     normal_pred, _, _ = get_surface_normal(pred_xyz)
#     normal_pred_ori, _, _ = get_surface_normal(pred_xyz_inital)

#     loss_d_initial = l1(pred_ds_initial, syn_ds) + loss_normal_weight * torch.mean(l1(normal_pred_ori, normal_labels), dim=1,
#                                                                                    keepdim=True) + loss_grad_weight * grad_loss(pred_ds_initial, syn_ds)

#     loss_d = l1(pred_ds, syn_ds) + loss_normal_weight * torch.mean(l1(normal_pred, normal_labels), dim=1,
#                                                                                    keepdim=True) + loss_grad_weight * grad_loss(pred_ds, syn_ds)
#     loss = loss_d + loss_weight_d_initial * loss_d_initial

#     if depth_restoration_train_with_masks:
#         num_instance = torch.sum(ins_masks)
#         num_background = torch.sum(1 - ins_masks)
#         loss_instance = torch.sum(loss * ins_masks) / num_instance
#         loss_background = torch.sum(loss * (1 - ins_masks)) / num_background
#         loss = loss_instance_weight * loss_instance + loss_background_weight * loss_background
#     else:
#         loss = loss.mean()

#     return normal_labels, normal_pred, loss_d_initial, loss_d, loss, loss_instance, loss_background


def compute_xyz_from_depth(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters.

        If focal lengths fx, fy are stored in the camera_params dictionary, use that.
        Else, assume camera_params contains parameters used to generate synthetic data (e.g. fov, near, far, etc)

        @param depth_img: a [B x H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used
    """
    # Extract batch size, height, and width
    if depth_img.ndim == 4:
        depth_img = depth_img.squeeze(1)
    B, H, W = depth_img.shape
    
    # Initialize the output tensor
    xyz_img = torch.zeros((B, H, W, 3), dtype=torch.float32).to(depth_img.device)
    
    # Compute x_e, y_e, and z_e for each image in the batch
    for b in range(B):
        if isinstance(camera_params, tuple):
            fx = camera_params[0]
            fy = camera_params[1]
            x_offset = camera_params[2]
            y_offset = camera_params[3]
        else:
            fx = camera_params['fx'][b]
            fy = camera_params['fy'][b]
            x_offset = camera_params['cx'][b]
            y_offset = camera_params['cy'][b]
        
        indices = torch.stack(torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32)), dim=-1)
        indices = indices.to(depth_img.device)
        
        z_e = depth_img[b]
        x_e = (indices[..., 1] - x_offset) * z_e / fx
        y_e = (indices[..., 0] - y_offset) * z_e / fy
        xyz_img[b] = torch.stack([x_e, y_e, z_e], dim=-1)

    return xyz_img