import torch
import os
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import utils.vis_utils as vis_utils
import numpy as np

def visualize_test_data(data_dict, vis_bid, result_dir):
    rgb_img = data_dict['rgb_img'][vis_bid].detach().cpu().numpy() # (3,h,w), float, (0,1)
    rgb_img = np.transpose(rgb_img,(1,2,0))*255.0
    rgb_img = rgb_img.astype(np.uint8) # (h,w,3), uint8, (0,255)
    height, width = rgb_img.shape[0], rgb_img.shape[1]
    color = rgb_img.reshape(-1,3)
    item = data_dict['item'][vis_bid]

    # undo transform of pred mask
    corrupt_mask_img = data_dict['corrupt_mask'][vis_bid].cpu().numpy()
    corrupt_mask_img = (corrupt_mask_img*255).astype(np.uint8)
    # undo transform of gt mask
    valid_mask_img = data_dict['valid_mask'][vis_bid].cpu().numpy()
    valid_mask_img = (valid_mask_img*255).astype(np.uint8)
  
    # masks visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes = axes.flat
    for ax in axes:
        ax.axis("off")
    # Inp RGB
    axes[0].set_title("Inp RGB")
    axes[0].imshow(rgb_img)
    # Corrupt Mask
    axes[1].set_title("Corrupt Mask")
    axes[1].imshow(corrupt_mask_img, cmap='gray', vmin=0, vmax=255)
    # Valid Mask
    axes[2].set_title("Valid Mask")
    axes[2].imshow(valid_mask_img, cmap='gray', vmin=0, vmax=255)
    # Save figure
    plt.tight_layout()
    # name setting
    dst_path = osp.join(result_dir, f'{item}_segmentation.png')
    plt.savefig(dst_path)
    plt.close(fig)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.imshow(rgb_img)
    dst_path = osp.join(result_dir, f'{item}_rgb.png')
    plt.savefig(dst_path, bbox_inches='tight', pad_inches = 0)
    plt.close(fig)
    
    corrupt_depth = data_dict['depth_corrupt'][vis_bid].squeeze(0).detach().cpu().numpy() # (1,h,w), float, (0,1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.imshow(corrupt_depth, cmap='gray')
    dst_path = osp.join(result_dir, f'{item}_corrupt_depth.png')
    plt.savefig(dst_path, bbox_inches='tight', pad_inches = 0)
    plt.close(fig)
    
    if 'hand_kpts_uv' in data_dict.keys():
        pred_keypoint_uv = data_dict['hand_kpts_uv'][vis_bid].squeeze(0).detach().cpu().numpy()
        pred_keypoint_xyz = data_dict['hand_kpts'][vis_bid].squeeze(0).detach().cpu().numpy()
        connections = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], \
            [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20], [0, 1], [0, 5], [0, 9], [0, 13], [0, 17]]
        connections_color = ['red', 'red', 'red', 'green', 'green', 'green', 'blue', 'blue', 'blue',
                                'yellow', 'yellow', 'yellow', 'cyan', 'cyan', 'cyan', 'red', 'green', 'blue', 'yellow', 'cyan']
        points_color = ['purple', 'red', 'red', 'red', 'red', 'green', 'green', 'green', 'green',
                        'blue', 'blue', 'blue', 'blue', 'yellow', 'yellow', 'yellow', 'yellow', 'cyan', 'cyan', 'cyan', 'cyan']
        
        
        # Create the figure and subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        axes = axes.flat
        for ax in axes:
            ax.axis("off")
        # Plot 2D keypoints
        axes[0].set_title("Pred Keypoints uv")
        axes[0].imshow(rgb_img)
        kpts_uv = pred_keypoint_uv
        kpts_uv[:, 0] = np.clip(kpts_uv[:, 0], 0, width)
        kpts_uv[:, 1] = np.clip(kpts_uv[:, 1], 0, height)
        axes[0].scatter(kpts_uv[:, 0], kpts_uv[:, 1], color=points_color[:21])
        for i_c in range(len(connections)):
            i, j = connections[i_c]
            axes[0].plot([kpts_uv[:, 0][i], kpts_uv[:, 0][j]], 
                        [kpts_uv[:, 1][i], kpts_uv[:, 1][j]], color=connections_color[i_c])
        # for i, kpt in enumerate(kpts_uv):
        #     axes[0].text(kpt[0], kpt[1], str(i), color='g', fontsize=12)
        
        # Plot 3D keypoints
        axes[1] = fig.add_subplot(122, projection='3d')
        axes[1].set_title("Pred Keypoints xyz")
        kpts_xyz = pred_keypoint_xyz
        axes[1].scatter(kpts_xyz[:, 0], kpts_xyz[:, 1], kpts_xyz[:, 2], color=points_color[:21])
        for i_c in range(len(connections)):
            i, j = connections[i_c]
            axes[1].plot([kpts_xyz[:, 0][i], kpts_xyz[:, 0][j]], 
                        [kpts_xyz[:, 1][i], kpts_xyz[:, 1][j]], 
                        [kpts_xyz[:, 2][i], kpts_xyz[:, 2][j]], color=connections_color[i_c])
        # for i in range(21):
        #     axes[1].text(kpts_xyz[:, 0][i], kpts_xyz[:, 1][i], kpts_xyz[:, 2][i], str(i), color='black', fontsize=9)
        plt.tight_layout()
        dst_path = osp.join(result_dir, f'{item}_keypoints.png')
        plt.savefig(dst_path)
        plt.close(fig)
        
        # Create the figure and subplots
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.axis("off")
        # ax.scatter(kpts_xyz[:, 0], kpts_xyz[:, 1], kpts_xyz[:, 2], color=points_color[:21])
        # for i_c in range(len(connections)):
        #     i, j = connections[i_c]
        #     ax.plot([kpts_xyz[:, 0][i], kpts_xyz[:, 0][j]], 
        #             [kpts_xyz[:, 1][i], kpts_xyz[:, 1][j]], 
        #             [kpts_xyz[:, 2][i], kpts_xyz[:, 2][j]], color=connections_color[i_c])
        # dst_path = osp.join(result_dir, f'{item}_keypoints_1.png')
        # plt.savefig(dst_path)
        # plt.close(fig)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.scatter(kpts_uv[:, 0], kpts_uv[:, 1], color=points_color[:21])
        for i_c in range(len(connections)):
            i, j = connections[i_c]
            ax.plot([kpts_uv[:, 0][i], kpts_uv[:, 0][j]], 
                        [kpts_uv[:, 1][i], kpts_uv[:, 1][j]], color=connections_color[i_c], linewidth=2)
        dst_path = osp.join(result_dir, f'{item}_keypoints_1.png')
        plt.savefig(dst_path)
        plt.close(fig)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.imshow(rgb_img)
        ax.scatter(kpts_uv[:, 0], kpts_uv[:, 1], color=points_color[:21])
        for i_c in range(len(connections)):
            i, j = connections[i_c]
            ax.plot([kpts_uv[:, 0][i], kpts_uv[:, 0][j]], 
                        [kpts_uv[:, 1][i], kpts_uv[:, 1][j]], color=connections_color[i_c])
        dst_path = osp.join(result_dir, f'{item}_keypoints_rgb.png')
        plt.savefig(dst_path, bbox_inches='tight', pad_inches = 0)
        plt.close(fig)
    
    # save inp pcl
    vis_inp_pcl = data_dict['xyz_corrupt_flat'].clone()
    vis_inp_pcl = vis_inp_pcl[vis_bid].detach().cpu().numpy()
    dst_path = os.path.join(result_dir, f'{item}_inp.ply')
    vis_utils.save_point_cloud(vis_inp_pcl, color, dst_path)
    
    # save gt pcl
    vis_gt_pcl = data_dict['xyz_corrupt_flat'].clone()
    vis_gt_pcl[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['gt_pos']
    vis_gt_pcl = vis_gt_pcl[vis_bid].detach().cpu().numpy()
    dst_path = os.path.join(result_dir, f'{item}_gt.ply')
    vis_utils.save_point_cloud(vis_gt_pcl, color, dst_path)
    
    # save pred pcl
    vis_pred_pcl = data_dict['xyz_corrupt_flat'].clone()
    vis_pred_pcl[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['pred_pos']
    corrupt_mask_flat = data_dict['corrupt_mask'].view(data_dict['rgb_img'].shape[0],-1)
    miss_idx = torch.nonzero(corrupt_mask_flat, as_tuple=False)
    miss_bid = miss_idx[:,0]
    miss_flat_img_id = miss_idx[:,1]
    transparent_pred = vis_pred_pcl[miss_bid, miss_flat_img_id]
    vis_pred_pcl = vis_pred_pcl[vis_bid].detach().cpu().numpy()
    dst_path = os.path.join(result_dir, f'{item}_pred.ply')
    vis_utils.save_point_cloud(vis_pred_pcl, color, dst_path)
    # vis_transparent_pred_pcl = data_dict['xyz_corrupt_flat'].clone()
    # vis_transparent_pred_pcl[miss_bid, miss_flat_img_id] = transparent_pred
    # vis_transparent_pred_pcl = transparent_pred[vis_bid].detach().cpu().numpy()
    # dst_path = os.path.join(result_dir, f'{item}_pred_transparent.ply')
    # vis_utils.save_point_cloud(vis_transparent_pred_pcl, color, dst_path)

    raise NotImplementedError('debug stop')