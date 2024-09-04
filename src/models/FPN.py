import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils.training_utils import *
from opt import *
from mmcv.cnn import ConvModule

from models.SwinTransformer import SwinTransformerSys

class FPN_Swin(nn.Module):
    def __init__(self, config, img_size=224, num_classes=3, logger=None, 
                 use_adp=True, fuse_conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(FPN_Swin, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.img_size = img_size
        self.logger = logger
        self.use_adp = use_adp
        self.num_outs = len(config.model.SWIN.DEPTHS)
        self.target_size = (img_size, img_size)
        self.outC = config.model.rgb_out
        self.embed_dim = config.model.SWIN.EMBED_DIM
        self.in_channles = [self.embed_dim, 2*self.embed_dim, 4*self.embed_dim, 8*self.embed_dim]
        self.out_channels = config.model.SWIN.EMBED_DIM

        self.backbone_rgb_branch = SwinTransformerSys(img_size=img_size,
                                                    patch_size=config.model.SWIN.PATCH_SIZE,
                                                    in_chans=3,
                                                    embed_dim=config.model.SWIN.EMBED_DIM,
                                                    depths=config.model.SWIN.DEPTHS,
                                                    num_heads=config.model.SWIN.NUM_HEADS,
                                                    window_size=config.model.SWIN.WINDOW_SIZE,
                                                    mlp_ratio=config.model.SWIN.MLP_RATIO,
                                                    qkv_bias=config.model.SWIN.QKV_BIAS,
                                                    qk_scale=None,
                                                    drop_rate=config.model.SWIN.DROP_RATE,
                                                    drop_path_rate=config.model.SWIN.DROP_PATH_RATE,
                                                    ape=config.model.SWIN.APE,
                                                    patch_norm=config.model.SWIN.PATCH_NORM,
                                                    use_checkpoint=False,
                                                    logger=logger)
        
        if use_adp:
            adp_list = []
            for i in range(self.num_outs):
                if i==0:
                    resize = nn.AdaptiveAvgPool2d(self.target_size)
                else:
                    resize = nn.Upsample(size = self.target_size, mode='bilinear', align_corners=True)
                adp = nn.Sequential(
                    resize,
                    ConvModule(
                        self.in_channles[i],
                        self.out_channels,
                        1,
                        padding=0,
                        conv_cfg=fuse_conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False),
                )
                adp_list.append(adp)
            self.adp = nn.ModuleList(adp_list)
        
        self.reduc_conv = ConvModule(
                self.out_channels * self.num_outs,
                config.model.rgb_out,
                3,
                padding=1,
                conv_cfg=fuse_conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
        
    def forward(self, rgb):
        outs = self.backbone_rgb_branch(rgb)
        
        if len(outs) > 1:
            resize_outs = []
            if self.use_adp:
                for i in range(len(outs)):
                    feature = self.adp[i](outs[i])
                    resize_outs.append(feature)
            else:
                target_size = self.target_size
                for i in range(len(outs)):
                    feature = outs[i]
                    if feature.shape[2:] != target_size:
                        feature = F.interpolate(feature, target_size,  mode='bilinear', align_corners=True)
                    resize_outs.append(feature)
            out = torch.cat(resize_outs, dim=1)
            out = self.reduc_conv(out)
        
        return out
    
    def init_weights(self):
        self.backbone_rgb_branch.init_weights(pretrained=self.config.model.SWIN.PRETRAIN_CKPT)
        # if self.use_adp:
        #     self.adp.init_weights()
        self.reduc_conv.init_weights()
        
class FPN_Swin2(nn.Module):
    def __init__(self, config, img_size=224, num_classes=3, logger=None, 
                 use_adp=True, fuse_conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(FPN_Swin2, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.img_size = img_size
        self.logger = logger
        self.use_adp = use_adp
        self.num_outs = len(config.model.SWIN.DEPTHS)
        self.target_size = (img_size // 4, img_size // 4)
        self.outC = config.model.rgb_out
        self.embed_dim = config.model.SWIN.EMBED_DIM
        self.in_channles = [self.embed_dim, 2*self.embed_dim, 4*self.embed_dim, 8*self.embed_dim]
        self.out_channels = config.model.SWIN.EMBED_DIM

        self.backbone_rgb_branch = SwinTransformerSys(img_size=img_size,
                                                    patch_size=config.model.SWIN.PATCH_SIZE,
                                                    in_chans=3,
                                                    embed_dim=config.model.SWIN.EMBED_DIM,
                                                    depths=config.model.SWIN.DEPTHS,
                                                    num_heads=config.model.SWIN.NUM_HEADS,
                                                    window_size=config.model.SWIN.WINDOW_SIZE,
                                                    mlp_ratio=config.model.SWIN.MLP_RATIO,
                                                    qkv_bias=config.model.SWIN.QKV_BIAS,
                                                    qk_scale=None,
                                                    drop_rate=config.model.SWIN.DROP_RATE,
                                                    drop_path_rate=config.model.SWIN.DROP_PATH_RATE,
                                                    ape=config.model.SWIN.APE,
                                                    patch_norm=config.model.SWIN.PATCH_NORM,
                                                    use_checkpoint=False,
                                                    logger=logger)
        
        if use_adp:
            adp_list = []
            for i in range(self.num_outs):
                if i==0:
                    resize = nn.AdaptiveAvgPool2d(self.target_size)
                else:
                    resize = nn.Upsample(size = self.target_size, mode='bilinear', align_corners=True)
                adp = nn.Sequential(
                    resize,
                    ConvModule(
                        self.in_channles[i],
                        self.out_channels,
                        1,
                        padding=0,
                        conv_cfg=fuse_conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False),
                )
                adp_list.append(adp)
            self.adp = nn.ModuleList(adp_list)
        
        self.reduc_conv = ConvModule(
                self.out_channels * self.num_outs,
                config.model.rgb_out,
                1,
                padding=1,
                conv_cfg=fuse_conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
        
    def forward(self, rgb):
        outs = self.backbone_rgb_branch(rgb)
        
        if len(outs) > 1:
            resize_outs = []
            if self.use_adp:
                for i in range(len(outs)):
                    feature = self.adp[i](outs[i])
                    resize_outs.append(feature)
            else:
                raise NotImplementedError('Not implemented yet')
            out = torch.cat(resize_outs, dim=1)
            out = self.reduc_conv(out)
        out = F.interpolate(out, self.img_size, mode='bilinear', align_corners=False)
        return out
    
    def init_weights(self):
        self.backbone_rgb_branch.init_weights(pretrained=self.config.model.SWIN.PRETRAIN_CKPT)
        self.reduc_conv.init_weights()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LIDF Training')
    parser.add_argument('--default_cfg_path', default = '/home/yuran/Projects/implicit_depth/src/experiments/implicit_depth/default_config.yaml', help='default config file')
    parser.add_argument("--cfg_path", type=str, default = '/home/yuran/Projects/implicit_depth/src/experiments/implicit_depth/train_lidf.yaml', help="List of updated config file")
    args = parser.parse_args()

    # setup opt
    if args.default_cfg_path is None:
        raise ValueError('default config path not found, should define one')
    opt = Params(args.default_cfg_path)
    if args.cfg_path is not None:
        opt.update(args.cfg_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.vis_gpu
    model = FPN_Swin2(opt)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    