#!/usr/bin/env python3

import os
import glob
import sys
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
import imageio
from tqdm import tqdm
import cv2
import copy
import json
from scipy.ndimage import gaussian_filter

augs_train = iaa.Sequential([
    # Geometric Augs
    iaa.Resize({
        "height": 224,
        "width": 224
    }, interpolation='cubic'),

    # Bright Patches
    iaa.Sometimes(
        0.1,
        iaa.blend.Alpha(factor=(0.2, 0.7),
                        first=iaa.blend.SimplexNoiseAlpha(first=iaa.Multiply((1.5, 3.0), per_channel=False),
                                                          upscale_method='cubic',
                                                          iterations=(1, 2)),
                        name="simplex-blend")),

    # Color Space Mods
    iaa.Sometimes(
        0.3,
        iaa.OneOf([
            iaa.Add((20, 20), per_channel=0.7, name="add"),
            iaa.Multiply((1.3, 1.3), per_channel=0.7, name="mul"),
            iaa.WithColorspace(to_colorspace="HSV",
                               from_colorspace="RGB",
                               children=iaa.WithChannels(0, iaa.Add((-200, 200))),
                               name="hue"),
            iaa.WithColorspace(to_colorspace="HSV",
                               from_colorspace="RGB",
                               children=iaa.WithChannels(1, iaa.Add((-20, 20))),
                               name="sat"),
            iaa.ContrastNormalization((0.5, 1.5), per_channel=0.2, name="norm"),
            iaa.Grayscale(alpha=(0.0, 1.0), name="gray"),
        ])),

    # Blur and Noise
    iaa.Sometimes(
        0.2,
        iaa.SomeOf((1, None), [
            iaa.OneOf([iaa.MotionBlur(k=3, name="motion-blur"),
                       iaa.GaussianBlur(sigma=(0.5, 1.0), name="gaus-blur")]),
            iaa.OneOf([
                iaa.AddElementwise((-5, 5), per_channel=0.5, name="add-element"),
                iaa.MultiplyElementwise((0.95, 1.05), per_channel=0.5, name="mul-element"),
                iaa.AdditiveGaussianNoise(scale=0.01 * 255, per_channel=0.5, name="guas-noise"),
                iaa.AdditiveLaplaceNoise(scale=(0, 0.01 * 255), per_channel=True, name="lap-noise"),
                iaa.Sometimes(1.0, iaa.Dropout(p=(0.003, 0.01), per_channel=0.5, name="dropout")),
            ]),
        ],
                   random_order=True)),

    # Colored Blocks
    iaa.Sometimes(0.2, iaa.CoarseDropout(0.02, size_px=(4, 16), per_channel=0.5, name="cdropout")),
])

# Validation Dataset
augs_test = iaa.Sequential([
    iaa.Resize({
        "height": 224,
        "width": 224
    }, interpolation='nearest'),
])

input_only = [
    "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
    "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout"
]


def load_meta(file_path):
    json_data = json.load(open(file_path))
    # meta = []
    return json_data

def load_kpt_data_from_txt(file_path):
    # Initialize an empty list to store the data
    data = []

    # Open the file for reading
    with open(file_path, 'r') as file:
        # Read each line in the file
        for line in file:
            # Split the line into two numbers and convert them to floats
            num1, num2 = map(float, line.split())
            # Append the numbers as a tuple to the data list
            data.append((num1, num2))
            
    return data

class KptsToHeatMap:
    def __init__(self, hand_kpts, rgb_image=None):
        if rgb_image is None:
            rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            self.rgb_image = rgb_image
        self.height, self.weight = rgb_image.shape[0], rgb_image.shape[1]
        self.sigma = self.height / 64 * 2
        
        self.hand_kpts = self._kpts_preprocess(hand_kpts)
        self.heatmap = self._generate_heatmap()
    
    def _kpts_preprocess(self, hand_kpts):
        new_hand_kpts = []
        for (x, y) in hand_kpts:
            x, y = int(x), int(y)
            x = np.clip(x, 0, self.weight - 1)
            y = np.clip(y, 0, self.height - 1)
            new_hand_kpts.append((x, y))
        return new_hand_kpts
    
    def _generate_heatmap(self):
        grid = np.zeros((self.height, self.weight))
        for (x, y) in self.hand_kpts:
            grid[y, x] += 1
        grid = gaussian_filter(grid, sigma=self.sigma)
        return grid

class HandDepthDataset(Dataset):
    """
    Dataset class for training model on estimation of surface normals.
    Uses imgaug for image augmentations.
    """

    def __init__(
            self,
            config=None,
            fx=446.31,
            fy=446.31,
            rgb_dir='',
            depth_dir='',
            hand_mask_dir='',
            obj_mask_dir='',
            meta_dir='',
            ignore_obj=None,
            transform=None,
            input_only=None,
            logger=None,
            mode=None,
    ):

        super().__init__()
        if ignore_obj is None:
            ignore_obj = ['']
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.hand_mask_dir = hand_mask_dir
        self.obj_mask_dir = obj_mask_dir
        self.meta_dir = meta_dir

        self.ignore_obj = ignore_obj

        self.config = config

        self.fx = fx
        self.fy = fy

        self.transform = transform
        self.input_only = input_only

        self.logger = logger
        self.mode = mode

        # Create list of filenames
        self._datalist_rgb = []
        self._datalist_depth = []  # Variable containing list of all input images filenames in dataset
        self._datalist_hand_mask = []
        self._datalist_obj_mask = []
        self._datalist_meta = []
        self._datalist_hand_kpt = []
        
        self._create_lists_filenames(self.rgb_dir,
                                     self.depth_dir,
                                     self.hand_mask_dir,
                                     self.obj_mask_dir,
                                     self.meta_dir,
                                     self.ignore_obj,
                                     self.logger,
                                     self.mode)

    def __len__(self):
        return len(self._datalist_rgb)
    
    def my_get_item(self, index):
        '''
        This method used to directly get data from dataset without any transformation.
        '''
        # Open rgb images
        rgb_path = self._datalist_rgb[index]
        _rgb = Image.open(rgb_path).convert('RGB')
        _rgb = np.array(_rgb)
        
        # Open depth images
        depth_path = self._datalist_depth[index]
        _depth = np.load(depth_path)
        if len(_depth.shape) == 3:
            _depth = _depth[:, :, 0]
        
        # Open hand mask images
        obj_mask_path = self._datalist_obj_mask[index]
        _obj_mask = cv2.imread(obj_mask_path, cv2.IMREAD_GRAYSCALE)

        # Open obj mask images
        hand_mask_path = self._datalist_hand_mask[index]
        _hand_mask = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Open hand kpts txt
        kpts_path = self._datalist_hand_kpt[index]
        _hand_kpts = load_kpt_data_from_txt(kpts_path)
        
        fx = self.fx
        fy = self.fy
        img_w = _depth.shape[1]
        img_h = _depth.shape[0]
        cx = img_w * 0.5 - 0.5
        cy = img_h * 0.5 - 0.5

        camera_params = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'yres': img_h,
            'xres': img_w,
        }
        
        # create one sample
        sample = {'rgb': _rgb,
                  'depth': _depth,
                  'mask_obj': _obj_mask,
                  'mask_hand': _hand_mask,
                  'camera_params': camera_params,
                  'hand_kpts': _hand_kpts}
        
        return sample

    def __getitem__(self, index):
        '''
        Returns an item from the dataset at the given index. 
        '''

        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

        # Open rgb images
        rgb_path = self._datalist_rgb[index]
        _rgb = Image.open(rgb_path).convert('RGB')
        _rgb = np.array(_rgb)

        # Open depth images
        depth_path = self._datalist_depth[index]
        _depth = np.load(depth_path)
        if len(_depth.shape) == 3:
            _depth = _depth[:, :, 0]
        _depth = _depth[np.newaxis, ...]

        # Open obj mask images
        obj_mask_path = self._datalist_obj_mask[index]
        _obj_mask = cv2.imread(obj_mask_path, cv2.IMREAD_GRAYSCALE)
        if self.config.DATA.DILATE_OBJECT_MASK:
            kernel = np.ones((5, 5), np.uint8)
            _obj_mask = cv2.dilate(_obj_mask, kernel, iterations=1)

        # Open hand mask images
        hand_mask_path = self._datalist_hand_mask[index]
        _hand_mask = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)

        # Open mate files
        meta_path = self._datalist_meta[index]
        _meta = load_meta(meta_path)
        
        # get hand kpts xyz
        _camera_point = np.array(_meta['camera point'])
        _camera_angle = np.array(_meta['camera angle'])
        _hand_kpts_world = np.array(list(_meta['hand_keypoint'].values()))
        if _hand_kpts_world.shape[1] == 3:
                _hand_kpts_world = np.hstack((_hand_kpts_world, np.ones((_hand_kpts_world.shape[0], 1))))
        
        # T_matrix_list = []
        _hand_kpts = []

        for j in range(21):
            T_matrix = cal_T(_camera_point, _camera_angle, _hand_kpts_world[j, :-1], np.array([0, 0, 0]))
            # T_matrix_list.append(T_matrix)
            point_cam = T_matrix.dot(np.array([0, 0, 0, 1]).T).T[:3]
            _hand_kpts.append(point_cam)

        # T_matrix_list = np.array(T_matrix_list).reshape((21, 4, 4))
        _hand_kpts = np.array(_hand_kpts).reshape((21, 3))
        
    
        # Open hand kpts txt
        _hand_kpts_uv = load_kpt_data_from_txt(self._datalist_hand_kpt[index])
        _hand_kpts_uv = np.array(_hand_kpts_uv)

        ori_h = _depth.shape[1]
        ori_w = _depth.shape[2]
        
        # find the item
        item_path = rgb_path
        item_path = item_path.split('.')[0]
        item_path_parts = item_path.split('/')
        item = f"{item_path_parts[-4]}_{item_path_parts[-1]}"

        # Apply image augmentations and convert to Tensor
        if self.transform:
            self.transform[0] = iaa.Resize({"height": self.config.DATA.PIC_RESIZED_H, "width": self.config.DATA.PIC_RESIZED_W}, interpolation='cubic')
            augs_test[0] = iaa.Resize({"height": self.config.DATA.PIC_RESIZED_H, "width": self.config.DATA.PIC_RESIZED_W}, interpolation='nearest')
            det_tf = self.transform.to_deterministic()
            det_tf_only_resize = augs_test.to_deterministic()

            _depth = _depth.transpose((1, 2, 0))  # To Shape: (H, W, 1)
            # transform to xyz_img
            img_h = _depth.shape[0]
            img_w = _depth.shape[1]

            _depth = det_tf_only_resize.augment_image(_depth, hooks=ia.HooksImages(activator=self._activator_masks))
            _depth = _depth.transpose((2, 0, 1))  # To Shape: (1, H, W)
            _depth[_depth <= 0] = 0.0
            _depth = _depth.squeeze(0)  # (H, W)

            fx = self.fx
            fy = self.fy
            cx = img_w * 0.5 - 0.5
            cy = img_h * 0.5 - 0.5

            camera_params = {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'yres': img_h,
                'xres': img_w,
            }

            # get image scale, (x_s, y_s)
            scale = (self.config.DATA.PIC_RESIZED_W / img_w, self.config.DATA.PIC_RESIZED_H / img_h)
            camera_params['fx'] *= scale[0]
            camera_params['fy'] *= scale[1]
            # camera_params['cx'] = self.config.DATA.PIC_RESIZED_W / 2 - 0.5
            # camera_params['cy'] = self.config.DATA.PIC_RESIZED_H / 2 - 0.5
            camera_params['cx'] *= scale[0]
            camera_params['cy'] *= scale[1]
            camera_params['xres'] *= scale[0]
            camera_params['yres'] *= scale[1]
            
            _hand_kpts_uv[:, 0] *= scale[0]
            _hand_kpts_uv[:, 1] *= scale[1]

            _rgb = det_tf.augment_image(_rgb)

            _hand_mask = det_tf_only_resize.augment_image(_hand_mask,
                                                          hooks=ia.HooksImages(activator=self._activator_masks))
            _obj_mask = det_tf_only_resize.augment_image(_obj_mask,
                                                         hooks=ia.HooksImages(activator=self._activator_masks))
            _depth_without_obj = (1. - _obj_mask / 255) * _depth
            _hand_depth = _hand_mask / 255 * _depth
            _mean_hand_depth = np.sum(_hand_depth) / (np.sum(_hand_mask) / 255)
            _depth_without_obj[_depth_without_obj == 0] = 0
            _seg_map = (_obj_mask / 255 + _hand_mask / 255 * 2).astype(np.uint8)
            
            # _heatmap = det_tf_only_resize.augment_image(_heatmap, hooks=ia.HooksImages(activator=self._activator_masks))
        
        # compute _xyz
        _xyz = self.compute_xyz(_depth, camera_params)
        _xyz_corrupt = self.compute_xyz(_depth_without_obj, camera_params)
        
        # Return Tensors
        _rgb_tensor = transforms.ToTensor()(_rgb)
        _xyz_tensor = transforms.ToTensor()(_xyz)
        _xyz_corrupt_tensor = transforms.ToTensor()(_xyz_corrupt)
        _depth_tensor = transforms.ToTensor()(_depth)
        _mask_obj_tensor = transforms.ToTensor()(_obj_mask)
        _mask_hand_tensor = transforms.ToTensor()(_hand_mask)
        _depth_without_obj_tensor = transforms.ToTensor()(_depth_without_obj)
        _hand_depth_tensor = transforms.ToTensor()(_hand_depth)
        _hand_kpts_tensor = transforms.ToTensor()(_hand_kpts)
        _hand_kpts_uv_tensor = transforms.ToTensor()(_hand_kpts_uv)
        _seg_map_tensor = transforms.ToTensor()(_seg_map) * 255
        # _heat_map_tensor = transforms.ToTensor()(_heatmap) * 255
        
        # sample = {'rgb': _rgb_tensor.to(torch.float32),
        #           'xyz': _xyz_tensor.to(torch.float32),
        #           'depth': _depth_tensor.to(torch.float32),
        #           'depth_without_obj': _depth_without_obj_tensor.to(torch.float32),
        #           'hand_depth': _hand_depth_tensor.to(torch.float32),
        #           'mask_obj': _mask_obj_tensor.to(torch.float32),
        #           'mask_hand': _mask_hand_tensor.to(torch.float32),
        #           'seg_map': _seg_map_tensor.to(torch.int64), 
        #           'heat_map': _heat_map_tensor.to(torch.float32),
        #           'camera_params': camera_params,
        #           'camera_internal': _meta['camera internal']}

        # for LIDF
        sample = {
            'rgb': _rgb_tensor.to(torch.float32),
            'corrupt_mask': _mask_obj_tensor.to(torch.float32),
            'xyz': _xyz_tensor.to(torch.float32),
            'xyz_corrupt': _xyz_corrupt_tensor.to(torch.float32),
            'depth': _depth_tensor.to(torch.float32),
            'depth_corrupt': _depth_without_obj_tensor.to(torch.float32),
            'hand_kpts': _hand_kpts_tensor.to(torch.float32),
            'hand_kpts_uv': _hand_kpts_uv_tensor.to(torch.float32),
            'hand_mask': _mask_hand_tensor.to(torch.float32),
            'fx': camera_params['fx'],
            'fy': camera_params['fy'],
            'cx': camera_params['cx'],
            'cy': camera_params['cy'],
            'xres': camera_params['xres'],
            'yres': camera_params['yres'],
            'item': item,
            # 'cam_param': (camera_params['fx'], camera_params['fy'], camera_params['cx'], camera_params['cy']),
        }
        
        
        return sample

    def _create_lists_filenames(self, rgb_dir, depth_dir, hand_mask_dir, obj_mask_dir, meta_dir, ignore_obj, logger, mode):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.

        Args:
            images_dir (str): Path to the dir where images are stored
            labels_dir (str): Path to the dir where labels are stored
            labels_dir (str): Path to the dir where masks are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        '''
        _obj_list_all = [os.path.split(x)[1] for x in sorted(glob.glob(os.path.join(rgb_dir, '*')))]
        _obj_list_choose = [x for x in _obj_list_all if x not in ignore_obj]
        if logger is not None:
            logger.info('Mode: {} --> Choose obj list: {}, ignore obj list: {}'.format(mode, _obj_list_choose, ignore_obj))
        self._datalist_rgb = sum([sorted(glob.glob(os.path.join(rgb_dir, obj_name, 'depth', 'rgb', '*'))) for obj_name in _obj_list_choose], [])
        self._datalist_depth = sum([sorted(glob.glob(os.path.join(depth_dir, obj_name, 'depth', 'depth_np', '*'))) for obj_name in _obj_list_choose], [])
        self._datalist_hand_mask = sum([sorted(glob.glob(os.path.join(hand_mask_dir, obj_name, 'depth', 'hand_mask', '*'))) for obj_name in _obj_list_choose], [])
        self._datalist_obj_mask = sum([sorted(glob.glob(os.path.join(obj_mask_dir, obj_name, 'depth', 'obj_mask', '*'))) for obj_name in _obj_list_choose], [])
        self._datalist_meta = sum([sorted(glob.glob(os.path.join(meta_dir, obj_name, 'depth', 'log_json', '*'))) for obj_name in _obj_list_choose], [])
        self._datalist_hand_kpt = sum([sorted(glob.glob(os.path.join(meta_dir, obj_name, 'depth', 'hand_kpt', '*'))) for obj_name in _obj_list_choose], [])
        
        # print(len(self._datalist_rgb), len(self._datalist_depth), len(self._datalist_hand_mask), len(self._datalist_obj_mask), len(self._datalist_meta), len(self._datalist_hand_kpt))
        
    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default

    def compute_xyz(self, depth_img, camera_params):
        """ Compute ordered point cloud from depth image and camera parameters.

            If focal lengths fx,fy are stored in the camera_params dictionary, use that.
            Else, assume camera_params contains parameters used to generate synthetic data (e.g. fov, near, far, etc)

            @param depth_img: a [H x W] numpy array of depth values in meters
            @param camera_params: a dictionary with parameters of the camera used
        """
        # Compute focal length from camera parameters
        fx = camera_params['fx']
        fy = camera_params['fy']
        x_offset = camera_params['cx']
        y_offset = camera_params['cy']
        indices = np.indices((int(camera_params['yres']), int(camera_params['xres'])), dtype=np.float32).transpose(1, 2,
                                                                                                                   0)
        z_e = depth_img
        x_e = (indices[..., 1] - x_offset) * z_e / fx
        y_e = (indices[..., 0] - y_offset) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
        return xyz_img
    
# 通过角度计算旋转矩阵
def AnglesTorotationMatrix(angle):
    x = angle[0] * np.pi / 180
    y = angle[1] * np.pi / 180
    z = angle[2] * np.pi / 180
    R = np.array([[np.cos(z) * np.cos(y), np.cos(z) * np.sin(y) * np.sin(x) - np.sin(z) * np.cos(x),
                   np.cos(z) * np.sin(y) * np.cos(x) + np.sin(z) * np.sin(x)],
                  [np.sin(z) * np.cos(y), np.sin(z) * np.sin(y) * np.sin(x) + np.cos(z) * np.cos(x),
                   np.sin(z) * np.sin(y) * np.cos(x) - np.cos(z) * np.sin(x)],
                  [-np.sin(y), np.cos(y) * np.sin(x), np.cos(y) * np.cos(x)]])
    return R


# 计算整体的旋转矩阵
def cal_T(camera_point, camera_angle, obj_position, obj_rotation):
    cal = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    R_cam_world = AnglesTorotationMatrix(camera_angle)
    t_cam_world = camera_point
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = R_cam_world
    T_cam_world[:3, 3] = t_cam_world
    # print(T_cam_world)
    R_obj_world = AnglesTorotationMatrix(obj_rotation)
    t_obj_world = obj_position
    T_obj_world = np.eye(4)
    T_obj_world[:3, :3] = R_obj_world
    T_obj_world[:3, 3] = t_obj_world
    # print(T_obj_world)

    T_matrix = np.linalg.inv(T_cam_world).dot(T_obj_world)
    T_matrix = cal.dot(T_matrix)

    return T_matrix
