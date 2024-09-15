import torch
import json
import os.path as osp
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib
import segmentation_models_pytorch as smp
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import models.keypoints.pipeline_keypoints as pipeline
from datasets.handdepth_datasets import HandDepthDataset, augs_train, input_only
import torch.optim.lr_scheduler as lr_scheduler
import utils.optimizer_utils as optimizer_utils
from utils.training_utils import *
from models.keypoints.loss import SmoothL1Loss


class TrainKeypoints(object):
    def __init__(self, opt):
        super(TrainKeypoints, self).__init__()
        self.opt = opt
        if self.opt.dist.ddp:
            print('Use GPU {} in Node {} for training'.format(self.opt.gpu_id, self.opt.dist.node_rank))
        # set device as local gpu id.
        torch.cuda.set_device(self.opt.gpu_id)
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_id))
        self.setup_misc()
        if self.opt.dist.ddp:
            dist.barrier()
        self.setup_model()
        self.setup_data()
        # sync all processes at the end of init
        if self.opt.dist.ddp:
            dist.barrier()
    
    def setup_misc(self):
        ''' Only process 0 will handle This part '''
        print('===> Setup miscs')
        
        # prepare directory
        self.ckpt_dir = osp.join(self.opt.base_log_dir, 'ckpt', self.opt.log_name)
        self.result_dir = osp.join(self.opt.base_log_dir, 'result', self.opt.log_name)
        if self.opt.gpu_id != 0:
            return
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # meters to record stats for validation and testing
        self.train_loss = AverageValueMeter()
        self.val_loss = AverageValueMeter()
        self.error = AverageValueMeter()
        self.x_error = AverageValueMeter()
        self.y_error = AverageValueMeter()
        self.z_error = AverageValueMeter()
        if self.opt.training.record_gradient:
            self.gradient_norm = AverageValueMeter()
            self.gradient_max = AverageValueMeter()
        
        # vis dir
        vis_dir = create_dir(osp.join(self.result_dir, '{}_vis'.format(self.opt.exp_type)))
        setattr(self, '{}_vis_dir'.format(self.opt.exp_type), vis_dir)
        # log path
        log_path = osp.join(self.result_dir, '{}_log.txt'.format(self.opt.exp_type))
        setattr(self, 'log_path', log_path)

        if self.opt.exp_type in ['train', 'valid']:
            self.valid_log_path = osp.join(self.result_dir, 'valid_log.txt')

        # write config to file
        if self.opt.exp_type == 'train':
            self.valid_vis_dir = create_dir(osp.join(self.result_dir, 'valid_vis'))
            all_keys = self.opt.get_all_keys(all_keys={}, dic=self.opt.dict, parent_key='')
            with open(osp.join(self.result_dir, 'config.txt'), 'w') as file:
                for k,v in all_keys.items():
                    file.write('{}: {}\n'.format(k,v))
            with open(osp.join(self.ckpt_dir, 'config.txt'), 'w') as file:
                for k,v in all_keys.items():
                    file.write('{}: {}\n'.format(k,v))
    
    def setup_model(self):
        print('===> Building models, GPU {}'.format(self.opt.gpu_id))
        self.model = pipeline.Pipeline(self.opt, self.device)
        
        # optimizer for training, valid also needs optimizer for saving ckpt
        if self.opt.exp_type in ['train', 'valid']:
            model_params = list()
            if self.opt.model.keypoints.use_rgb:
                model_params += list(self.model.backbone_rgb.parameters())
            if self.opt.model.keypoints.use_depth:
                model_params += list(self.model.backbone_d.parameters())
            self.optimizer = getattr(optimizer_utils, self.opt.training.optimizer_name)(model_params, lr=self.opt.training.lr)
            self.model_params = model_params
            # lr scheduler
            if self.opt.training.scheduler_name == 'StepLR':
                self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.opt.training.nepoch_decay, 
                                    gamma=self.opt.training.decay_gamma)
            else:
                raise NotImplementedError('Does not support scheduler type: {}'.format(self.opt.training.scheduler_name))
            print('optimizer at GPU {}'.format(self.opt.gpu_id))
        
        # load checkpoint
        if self.opt.checkpoint_path is not None and osp.isfile(self.opt.checkpoint_path):
            loc = 'cuda:{}'.format(self.opt.gpu_id)
            checkpoint = torch.load(self.opt.checkpoint_path, map_location=loc)
            if self.opt.model.keypoints.use_rgb:
                restore(self.model.backbone_rgb, checkpoint['backbone_rgb'])
            if self.opt.model.keypoints.use_depth:
                restore(self.model.backbone_d, checkpoint['backbone_d'])
            print('Loaded checkpoint at epoch {} from {}.'.format(checkpoint['epoch'], self.opt.checkpoint_path))
        
        if self.opt.exp_type in ['test'] and self.opt.checkpoint_path is None:
            raise ValueError('Should identify checkpoint_path for testing!')
        if self.opt.exp_type in ['test'] and self.opt.checkpoint_path is None:
            raise ValueError('Should identify checkpoint_path for testing!')

        if self.opt.exp_type in ['train', 'valid']:
            # only process 0 handles ckpt save, so only process 0 can have min_*_error
            if self.opt.gpu_id == 0:
                self.min_err, self.max_acc = 1e5, -1
                print("=> Setup min error for GPU {}".format(self.opt.gpu_id))
        
        # resume training
        if self.opt.exp_type == 'train':
            self.start_epoch = 0
            self.opt.resume = osp.join(self.ckpt_dir, self.opt.resume)
            if self.opt.resume is not None and osp.isfile(self.opt.resume):
                loc = 'cuda:{}'.format(self.opt.gpu_id)
                checkpoint = torch.load(self.opt.resume, map_location=loc)
                if self.opt.model.keypoints.use_rgb:
                    restore(self.model.backbone_rgb, checkpoint['backbone_rgb'])
                if self.opt.model.keypoints.use_depth:
                    restore(self.model.backbone_d, checkpoint['backbone_d'])
                if 'epoch' in checkpoint.keys():
                    self.start_epoch = checkpoint['epoch']+1
                if 'optimizer' in checkpoint.keys():
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                if self.opt.gpu_id == 0:
                    if 'min_err' in checkpoint.keys():
                        self.min_err = checkpoint['min_err']
                    if 'max_acc' in checkpoint.keys():
                        self.max_acc = checkpoint['max_acc']
                    if 'min_angle_err' in checkpoint.keys():
                        self.min_angle_err = checkpoint['min_angle_err']
                    print("=> Loaded min error for GPU {} at loc {}".format(self.opt.gpu_id, loc))
                print("=> Continue Training from '{}' for GPU {} at loc {}".format(self.opt.resume, self.opt.gpu_id, loc))

    def setup_data(self):
        print('===> Setup data loader')
        # prepare dataset and loader
        data_path_train = self.opt.DATA.TRAIN_SET_PATH
        data_path_val = self.opt.DATA.VAL_SET_PATH
        data_path_test = self.opt.DATA.TEST_SET_PATH
        
        if self.opt.dist.ddp:
            batch_size_train = self.opt.training.batch_size * self.opt.num_gpus
            batch_size_valid = self.opt.training.valid_batch_size * self.opt.num_gpus
        else:
            batch_size_train = self.opt.training.batch_size
            batch_size_valid = self.opt.training.valid_batch_size
            batch_size_test = self.opt.training.valid_batch_size
        
         # generate trainLoader
        datasets_train_path = [
                {'rgb': data_path_train,
                'depth': data_path_train,
                'hand_mask': data_path_train,
                'obj_mask': data_path_train,
                'meta': data_path_train, }
            ]
        dataset_train_list = []
        for dataset in datasets_train_path:
            dataset_train = HandDepthDataset(config=self.opt,
                                                fx=self.opt.DATA.SIM_CAMERA_PARAM_FX,
                                                fy=self.opt.DATA.SIM_CAMERA_PARAM_FY,
                                                rgb_dir=dataset["rgb"],
                                                depth_dir=dataset["depth"],
                                                hand_mask_dir=dataset["hand_mask"],
                                                obj_mask_dir=dataset["obj_mask"],
                                                meta_dir=dataset["meta"],
                                                ignore_obj=self.opt.DATA.IGNORE_TRAIN_OBJ,
                                                transform=augs_train,
                                                input_only=input_only,
                                                logger=None,
                                                mode='train',
                                                )
            dataset_train_list.append(dataset_train)
        dataset_train = torch.utils.data.ConcatDataset(dataset_train_list)
        train_size = int(self.opt.DATA.PERCENTAGE_DATA_FOR_TRAIN * len(dataset_train))
        train_choose, _ = torch.utils.data.random_split(dataset_train, (train_size, len(dataset_train) - train_size))
        self.train_data_loader = DataLoader(train_choose,
                                batch_size=batch_size_train,
                                shuffle=True,
                                num_workers=8,
                                drop_last=True,
                                pin_memory=True)
        
        # generate validationLoader
        datasets_val_path = [
                {'rgb': data_path_val,
                'depth': data_path_val,
                'hand_mask': data_path_val,
                'obj_mask': data_path_val,
                'meta': data_path_val, }
            ]
        dataset_val_list = []
        for dataset in datasets_val_path:
            dataset_val = HandDepthDataset(config=self.opt,
                                                fx=self.opt.DATA.SIM_CAMERA_PARAM_FX,
                                                fy=self.opt.DATA.SIM_CAMERA_PARAM_FY,
                                                rgb_dir=dataset["rgb"],
                                                depth_dir=dataset["depth"],
                                                hand_mask_dir=dataset["hand_mask"],
                                                obj_mask_dir=dataset["obj_mask"],
                                                meta_dir=dataset["meta"],
                                                ignore_obj=self.opt.DATA.IGNORE_VAL_OBJ,
                                                transform=augs_train,
                                                input_only=input_only,
                                                logger=None,
                                                mode='val',
                                                )
            dataset_val_list.append(dataset_val)
        dataset_val = torch.utils.data.ConcatDataset(dataset_val_list)
        val_size = int(self.opt.DATA.PERCENTAGE_DATA_FOR_VAL * len(dataset_val))
        val_choose, _ = torch.utils.data.random_split(dataset_val, (val_size, len(dataset_val) - val_size))
        self.val_data_loader = DataLoader(val_choose,
                                batch_size=batch_size_valid,
                                shuffle=True,
                                num_workers=8,
                                drop_last=True,
                                pin_memory=True)
        
        if self.opt.exp_type == 'test':
            # generate test loader
            datasets_test_path = [
                    {'rgb': data_path_test,
                    'depth': data_path_test,
                    'hand_mask': data_path_test,
                    'obj_mask': data_path_test,
                    'meta': data_path_test, }
                ]
            dataset_test_list = []
            for dataset in datasets_test_path:
                dataset_test = HandDepthDataset(config=self.opt,
                                                    fx=self.opt.DATA.SIM_CAMERA_PARAM_FX,
                                                    fy=self.opt.DATA.SIM_CAMERA_PARAM_FY,
                                                    rgb_dir=dataset["rgb"],
                                                    depth_dir=dataset["depth"],
                                                    hand_mask_dir=dataset["hand_mask"],
                                                    obj_mask_dir=dataset["obj_mask"],
                                                    meta_dir=dataset["meta"],
                                                    ignore_obj=self.opt.DATA.IGNORE_TEST_OBJ,
                                                    transform=augs_train,
                                                    input_only=input_only,
                                                    logger=None,
                                                    mode='test',
                                                    )
                dataset_test_list.append(dataset_test)
            dataset_test = torch.utils.data.ConcatDataset(dataset_test_list)
            test_size = int(self.opt.DATA.PERCENTAGE_DATA_FOR_TEST * len(dataset_test))
            test_choose, _ = torch.utils.data.random_split(dataset_test, (test_size, len(dataset_test) - test_size))
            self.test_data_loader = DataLoader(test_choose,
                                    batch_size=batch_size_test,
                                    shuffle=False,
                                    num_workers=8,
                                    drop_last=True,
                                    pin_memory=True)
    
    def train_epoch(self):
        for epoch in range(self.start_epoch, self.opt.training.nepochs):
            debug_print("=> Epoch {} for GPU {}".format(epoch, self.opt.gpu_id), self.opt.debug)
            # train
            self.train(self.train_data_loader, 'train', epoch)

            # valid and logging is only done by process 0
            if self.opt.gpu_id == 0:
                if self.opt.training.do_valid:
                    # valid
                    with open(self.valid_log_path, 'a') as file:
                        file.write('Epoch[{}]:\n'.format(epoch))
                    self.validate(self.val_data_loader, 'valid', epoch)

                # save ckpt and write log
                self.save_ckpt_and_log(epoch)
            # lr scheduler
            self.scheduler.step()
            
    def train(self, data_loader, exp_type, epoch):
        if self.opt.gpu_id == 0:
            self.train_loss.reset()

        self.model.train()
        for iteration, batch in enumerate(data_loader):
            debug_print("=> Iter {} for GPU {}".format(iteration, self.opt.gpu_id), self.opt.debug)
            self.run_iteration(epoch, iteration, len(data_loader), exp_type, 
                self.opt.training.train_vis_iter, batch)
            if self.opt.debug and iteration == 5:
                break
    
    def validate(self, data_loader, exp_type, epoch, dataset_name='valid'):
        with torch.no_grad():
            self.val_loss.reset()
            self.error.reset()
            self.x_error.reset()
            self.y_error.reset()
            self.z_error.reset()
            
            self.model.eval()
            for iteration, batch in enumerate(data_loader):
                self.run_iteration(epoch, iteration, len(data_loader), exp_type, 
                    self.opt.training.val_vis_iter, batch)
                if self.opt.debug and iteration == 5:
                    break
            
            err_log = '        {}: error: {:.6f}, x_error: {:.6f}, y_error: {:.6f}, z_error: {:.6f}\n'.format(
                    dataset_name, self.error.avg, self.x_error.avg, self.y_error.avg, self.z_error.avg)
            print(err_log)
            with open(self.valid_log_path, 'a') as file:
                file.write(err_log)
            
    def run_iteration(self, epoch, iteration, iter_len, exp_type, vis_iter, batch):
        pred_mask = None
        # Forward pass
        success_flag, data_dict, loss_dict = self.model(batch, exp_type, epoch, pred_mask)
        if exp_type == 'train' and self.opt.dist.ddp:
            # have to set barrier to wait for all processes finished forward pass
            dist.barrier()
            success_num = torch.Tensor([success_flag]).to(self.device)
            dist.all_reduce(success_num, op=dist.ReduceOp.SUM)
            # at least one gpu fails: clear grad buffer and return
            if success_num[0] < self.opt.dist.ngpus_per_node:
                print('gpu {}: {}'.format(self.opt.gpu_id, success_num[0]))
                self.optimizer.zero_grad()
                return
        elif not success_flag:
            if exp_type == 'train':
                self.optimizer.zero_grad()
            return
        
        # Backward pass
        if exp_type == 'train':
            self.optimizer.zero_grad()
            loss_dict['loss_net'].backward()
            
            # compute the gradient norm
            if self.opt.training.record_gradient:
                total_norm = 0
                max_grad = 0  # Initialize max gradient value
                for param in self.model_params:
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        # Record the maximum gradient value
                        max_grad = max(max_grad, param.grad.data.abs().max().item())
                total_norm = total_norm ** 0.5
                self.gradient_norm.update(total_norm)
                self.gradient_max.update(max_grad)  # Update the maximum gradient value

            self.optimizer.step()
            
        # Metrics update
        if exp_type != 'train':
            self.val_loss.update(loss_dict['loss_net'].item())
            self.error.update(loss_dict['error'].item())
            self.x_error.update(loss_dict['x_error'].item())
            self.y_error.update(loss_dict['y_error'].item())
            self.z_error.update(loss_dict['z_error'].item())
        elif self.opt.gpu_id == 0:
            self.train_loss.update(loss_dict['loss_net'].item())
            
        # Logging
        log_cond1 = (iteration % self.opt.training.log_interval == 0) and (exp_type != 'train')
        log_cond2 = (iteration % self.opt.training.log_interval == 0) and (self.opt.gpu_id == 0)
        if log_cond1:
            err_log = '===> Epoch[{}]({}/{}):  error: {}, x_error: {}, y_error: {}, z_error: {}'.format(
                    epoch, iteration, iter_len,
                    loss_dict['error'].item(), loss_dict['x_error'].item(), loss_dict['y_error'].item(), loss_dict['z_error'].item())
            print(err_log)
        elif log_cond2:
            loss_net_4log = loss_dict['loss_net'].item()
            log_info = '===> Epoch[{}]({}/{}):  Loss: {:.5f}'.format(
                    epoch, iteration, iter_len, loss_net_4log)
            print(log_info)
        
        # Visualization
        vis_bid = 0
        vis_cond1 = (exp_type != 'test') and (self.opt.gpu_id == 0) and (iteration % vis_iter == 0)
        vis_cond2 = (exp_type == 'test') and (iteration % vis_iter == 0)
        if vis_cond1 or vis_cond2:
            result_dir = getattr(self, '{}_vis_dir'.format(exp_type))
            self.visualize(data_dict, exp_type, epoch, iteration, vis_bid, result_dir)
    
    def save_ckpt_and_log(self, epoch):
        flag_best = False
        if self.error.avg < self.min_err:
            self.min_err = self.error.avg
            flag_best = True
        # dump stats in log file
        log_table = {
            'epoch' : epoch,
            'train_loss' : self.train_loss.avg,
            'val_loss' : self.val_loss.avg,
            'error': self.error.avg,
            'x_error': self.x_error.avg,
            'y_error': self.y_error.avg,
            'z_error': self.z_error.avg,
        }
        if self.opt.training.record_gradient:
            log_table.update({'gradient_norm': self.gradient_norm.val,
                              'gradient_max': self.gradient_max.val})
        with open(self.log_path, 'a') as file:
            if self.opt.exp_type == 'train':
                file.write(json.dumps(log_table))
                file.write('\n')
                
        ckpt_dict = log_table.copy()
        if self.opt.model.keypoints.use_rgb:
            ckpt_dict.update({'backbone_rgb': self.model.backbone_rgb.state_dict()})
        if self.opt.model.keypoints.use_depth:
            ckpt_dict.update({'backbone_d': self.model.backbone_d.state_dict()})
        ckpt_dict.update({
            'optimizer': self.optimizer.state_dict(),
        })
        torch.save(ckpt_dict, osp.join(self.ckpt_dir,'latest_network.pth'))
        if flag_best:
            torch.save(ckpt_dict, osp.join(self.ckpt_dir,'best_network.pth'))
        if epoch % self.opt.training.nepoch_ckpt == 0:
            torch.save(ckpt_dict, osp.join(self.ckpt_dir,'epoch{:03d}_network.pth'.format(epoch)))
        
            
    def visualize(self, data_dict, exp_type, epoch, iteration, vis_bid, result_dir):
        # untransform and save rgb
        rgb_img = data_dict['rgb_img'][vis_bid].detach().cpu().numpy() # (3,h,w), float, (0,1)
        rgb_img = np.transpose(rgb_img,(1,2,0))*255.0
        rgb_img = rgb_img.astype(np.uint8) # (h,w,3), uint8, (0,255)
        height, width = rgb_img.shape[0], rgb_img.shape[1]
        
        gt_keypoint_uv = data_dict['hand_kpts_uv'][vis_bid].squeeze(0).detach().cpu().numpy()
        gt_keypoint_xyz = data_dict['hand_kpts'][vis_bid].squeeze(0).detach().cpu().numpy()
        
        if self.opt.model.keypoints.use_depth:
            pred_depth_keypoint_uv = data_dict['pred_depth_keypoints_uvd'][vis_bid].detach().cpu().numpy()[:, :2]
            pred_depth_keypoint_xyz = data_dict['pred_depth_keypoints_xyz'][vis_bid].detach().cpu().numpy()
        
        if self.opt.model.keypoints.use_rgb:
            pred_rgb_keypoint_uv = data_dict['pred_rgb_keypoints_uvd'][vis_bid].detach().cpu().numpy()[:, :2]
            pred_rgb_keypoint_xyz = data_dict['pred_rgb_keypoints_xyz'][vis_bid].detach().cpu().numpy()
        
        # hand keypoints connection
        connections = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], \
        [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20], [0, 1], [0, 5], [0, 9], [0, 13], [0, 17]]
        
        ''' depth branch result '''
        if self.opt.model.keypoints.use_depth:
            fig, axes = plt.subplots(1, 2, figsize=(20, 7))
            axes = axes.flat
            for ax in axes:
                ax.axis("off")
            # GT kpts
            axes[0].set_title("GT Keypoints uv")
            kpts = gt_keypoint_uv
            kpts[:, 0] = np.clip(kpts[:, 0], 0, width)
            kpts[:, 1] = np.clip(kpts[:, 1], 0, height)
            axes[0].imshow(rgb_img)
            axes[0].plot(kpts[:, 0], kpts[:, 1], 'ro', markersize=5)
            for i, kpt in enumerate(kpts):
                axes[0].text(kpt[0], kpt[1], str(i), color='g', fontsize=12)
            # Predict kpts
            axes[1].set_title("Pred Keypoints uv")
            axes[1].imshow(rgb_img)
            kpts = pred_depth_keypoint_uv
            kpts[:, 0] = np.clip(kpts[:, 0], 0, width)
            kpts[:, 1] = np.clip(kpts[:, 1], 0, height)
            axes[1].plot(kpts[:, 0], kpts[:, 1], 'ro', markersize=5)
            for i, kpt in enumerate(kpts):
                axes[1].text(kpt[0], kpt[1], str(i), color='g', fontsize=12)
            # Save figure
            plt.tight_layout()
            # name setting
            dst_path = osp.join(result_dir, 'epo{:03d}_iter{:05d}_depth_kpts_2d.png'.format(epoch, iteration))
            plt.savefig(dst_path)
            plt.close(fig)
            
            
            fig, axes = plt.subplots(1, 2, figsize=(20, 7), subplot_kw={'projection': '3d'})
            axes = axes.flat
            # Inp RGB
            axes[0].set_title("GT Keypoints xyz")
            kpts = gt_keypoint_xyz
            axes[0].scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2])
            for connection_i in connections:
                i, j = connection_i
                axes[0].plot([kpts[:, 0][i], kpts[:, 0][j]], [kpts[:, 1][i], kpts[:, 1][j]], [kpts[:, 2][i], kpts[:, 2][j]], color='blue')
            for i in range(21):
                axes[0].text(kpts[:, 0][i], kpts[:, 1][i], kpts[:, 2][i], str(i), color="red", fontsize=9)
            
            axes[1].set_title("Pred Keypoints xyz")
            kpts = pred_depth_keypoint_xyz
            axes[1].scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2])
            for connection_i in connections:
                i, j = connection_i
                axes[1].plot([kpts[:, 0][i], kpts[:, 0][j]], [kpts[:, 1][i], kpts[:, 1][j]], [kpts[:, 2][i], kpts[:, 2][j]], color='blue')
            for i in range(21):
                axes[1].text(kpts[:, 0][i], kpts[:, 1][i], kpts[:, 2][i], str(i), color="red", fontsize=9)
            # Save figure
            plt.tight_layout()
            # name setting
            dst_path = osp.join(result_dir, 'epo{:03d}_iter{:05d}_depth_kpts_3d.png'.format(epoch, iteration))
            plt.savefig(dst_path)
            plt.close(fig)
            
        ''' rgb branch result '''
        if self.opt.model.keypoints.use_rgb:
            fig, axes = plt.subplots(1, 2, figsize=(20, 7))
            axes = axes.flat
            for ax in axes:
                ax.axis("off")
            # GT kpts
            axes[0].set_title("GT Keypoints uv")
            kpts = gt_keypoint_uv
            kpts[:, 0] = np.clip(kpts[:, 0], 0, width)
            kpts[:, 1] = np.clip(kpts[:, 1], 0, height)
            axes[0].imshow(rgb_img)
            axes[0].plot(kpts[:, 0], kpts[:, 1], 'ro', markersize=5)
            for i, kpt in enumerate(kpts):
                axes[0].text(kpt[0], kpt[1], str(i), color='g', fontsize=12)
            # Predict kpts
            axes[1].set_title("Pred Keypoints uv")
            axes[1].imshow(rgb_img)
            kpts = pred_rgb_keypoint_uv
            kpts[:, 0] = np.clip(kpts[:, 0], 0, width)
            kpts[:, 1] = np.clip(kpts[:, 1], 0, height)
            axes[1].plot(kpts[:, 0], kpts[:, 1], 'ro', markersize=5)
            for i, kpt in enumerate(kpts):
                axes[1].text(kpt[0], kpt[1], str(i), color='g', fontsize=12)
            # Save figure
            plt.tight_layout()
            # name setting
            dst_path = osp.join(result_dir, 'epo{:03d}_iter{:05d}_rgb_kpts_2d.png'.format(epoch, iteration))
            plt.savefig(dst_path)
            plt.close(fig)
            
            
            fig, axes = plt.subplots(1, 2, figsize=(20, 7), subplot_kw={'projection': '3d'})
            axes = axes.flat
            # Inp RGB
            axes[0].set_title("GT Keypoints xyz")
            kpts = gt_keypoint_xyz
            axes[0].scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2])
            for connection_i in connections:
                i, j = connection_i
                axes[0].plot([kpts[:, 0][i], kpts[:, 0][j]], [kpts[:, 1][i], kpts[:, 1][j]], [kpts[:, 2][i], kpts[:, 2][j]], color='blue')
            for i in range(21):
                axes[0].text(kpts[:, 0][i], kpts[:, 1][i], kpts[:, 2][i], str(i), color="red", fontsize=9)
            
            # Corrupt Mask
            axes[1].set_title("GT Keypoints xyz")
            kpts = pred_rgb_keypoint_xyz
            axes[1].scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2])
            for connection_i in connections:
                i, j = connection_i
                axes[1].plot([kpts[:, 0][i], kpts[:, 0][j]], [kpts[:, 1][i], kpts[:, 1][j]], [kpts[:, 2][i], kpts[:, 2][j]], color='blue')
            for i in range(21):
                axes[1].text(kpts[:, 0][i], kpts[:, 1][i], kpts[:, 2][i], str(i), color="red", fontsize=9)
            # Save figure
            plt.tight_layout()
            # name setting
            dst_path = osp.join(result_dir, 'epo{:03d}_iter{:05d}_rgb_kpts_3d.png'.format(epoch, iteration))
            plt.savefig(dst_path)
            plt.close(fig)

def createAndRunTrainer(gpu_id, opt):
    '''
        gpu_id: gpu id for current trainer. same as process local rank. handled by spawn for multiprocessing, or 
        manually passed in for single GPU training
        opt: option data
    '''

    # local gpu id for current process in residing node.
    opt.gpu_id = gpu_id
    if opt.dist.ddp:
        # global gpu id for current process 
        opt.dist.global_gpu_id = opt.dist.node_rank * opt.dist.ngpus_per_node + gpu_id
        # total GPU number
        opt.dist.world_size = opt.dist.ngpus_per_node * opt.dist.nodes_num
        dist.init_process_group(backend=opt.dist.dist_backend, init_method=opt.dist.dist_url,
                                world_size=opt.dist.world_size, rank=opt.dist.global_gpu_id)
    # init trainer
    trainer = TrainKeypoints(opt)

    # Training
    if opt.exp_type == 'train':
        print("=> Start Training")
        trainer.train_epoch()
    
    # Testing
    elif opt.exp_type == 'test':
        if opt.dist.ddp:
            raise ValueError('Testing should use single GPU')
        print("=> Start Testing")
        # vis dir
        trainer.test_vis_dir = create_dir(osp.join(trainer.result_dir, '{}_test_vis'.format(opt.dataset.type)))
        # log path
        trainer.log_path = osp.join(trainer.result_dir, '{}_test_log.txt'.format(opt.dataset.type))
        with open(trainer.log_path, 'a') as file:
            file.write('Mask Type: {}\n'.format(opt.mask_type))
            file.write('checkpoint path: {}\n'.format(opt.checkpoint_path.split('/')[-1]))
        if opt.dataset.type == 'cleargrasp':
            trainer.test(trainer.syn_known_test_data_loader, 'test', 'syn_known', epoch=0)
            trainer.test(trainer.syn_novel_test_data_loader, 'test', 'syn_novel', epoch=1)
            trainer.test(trainer.real_known_test_data_loader, 'test', 'real_known', epoch=2)
            trainer.test(trainer.real_novel_test_data_loader, 'test', 'real_novel', epoch=3)
        elif opt.dataset.type == 'transhand':
            trainer.test(trainer.test_data_loader, 'test', 'test', epoch=0)

    else:
        raise NotImplementedError('{} not supported'.format(opt.exp_type))