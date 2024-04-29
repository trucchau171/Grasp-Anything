import glob
import os
import re
import numpy as np
import pickle
import torch
import random

import numpy as np
import torch
import torch.utils.data

from utils.dataset_processing import grasp, image, mask
from .grasp_data import GraspDatasetBase


class GraspAnythingPlusDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything++ dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnythingPlusDataset, self).__init__(**kwargs)

        file_names = os.listdir(os.path.join(file_path, 'image'))
        file_names = [file_name[:-4] for file_name in file_names]
        self.rgb_files = [os.path.join(file_path, 'image', file_name + '.jpg') for file_name in file_names]
        self.grasp_files = [os.path.join(file_path, 'grasp_label', file_name + '_0_0.pt') for file_name in file_names]
        self.prompt_files = [os.path.join(file_path, 'grasp_instructions', file_name + '_0_0.pkl') for file_name in file_names] 
        self.mask_files = [os.path.join(file_path, '_mask', file_name + '.npy') for file_name in file_names] 
        if os.path.exists(self.mask_files[0]):
            print("Mask path exists.")
        else:
            print("Mask path not exists.")

        self.grasp_files.sort()
        self.prompt_files.sort()
        self.rgb_files.sort()
        self.mask_files.sort()

        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
        else:
            print("Found " + str(self.length) + " files.")

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]


    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):       
        # Jacquard try
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))

        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_file = self.rgb_files[idx]
        rgb_img = image.Image.from_file(rgb_file)

        if os.path.exists(self.mask_files[idx]):
            mask_file = self.mask_files[idx]
            mask_img = mask.Mask.from_file(mask_file)
        else:
            mask_img = np.ones_like(np.asarray(rgb_img)[:,:,0])
        rgb_img = image.Image.mask_out_image(rgb_img, mask_img)

        # Jacquard try
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def preprocess_caption(self, caption: str) -> str:
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        prompt_file = self.prompt_files[idx]
        with open(prompt_file, 'rb') as ff:
            prompt = pickle.load(ff)
        prompt = self.preprocess_caption(caption=prompt)
        return x, (pos, cos, sin, width), idx, rot, zoom_factor, prompt
