from typing import Optional
import webdataset as wds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from torch.utils.data import Dataset
import os
import json
from PIL import Image
import torch
import cv2 as cv
import numpy as np


from sgm.data.read_write_model import read_model
from sgm.data.utils_camera import (
    read_intrinsics_colmap,
    read_extrinsics_colmap,
    read_intrinsics_nerfstudio,
    read_extrinsics_nerfstudio,
    opencv_to_opengl,
    colmap_to_nerfstudio,
    nerfstudio_to_colmap
)
from einops import rearrange, repeat 

import webdataset as wds
from torch.utils.data import DataLoader

# try: TODO: fix torchdata version
#     from sdata import create_dataset, create_dummy_dataset, create_loader
# except ImportError as e:
#     print("#" * 100)
#     print("Datasets not yet available")
#     print("to enable, we need to add stable-datasets as a submodule")
#     print("please use ``git submodule update --init --recursive``")
#     print("and do ``pip install -e stable-datasets/`` from the root of this repo")
#     print("#" * 100)
#     exit(1)


class StableDataModuleFromConfig(LightningDataModule):
    def __init__(
        self,
        train: DictConfig,
        validation: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        skip_val_loader: bool = False,
        dummy: bool = False,
    ):
        super().__init__()
        self.train_config = train
        assert (
            "datapipeline" in self.train_config and "loader" in self.train_config
        ), "train config requires the fields `datapipeline` and `loader`"

        self.val_config = validation
        if not skip_val_loader:
            if self.val_config is not None:
                assert (
                    "datapipeline" in self.val_config and "loader" in self.val_config
                ), "validation config requires the fields `datapipeline` and `loader`"
            else:
                print(
                    "Warning: No Validation datapipeline defined, using that one from training"
                )
                self.val_config = train

        self.test_config = test
        if self.test_config is not None:
            assert (
                "datapipeline" in self.test_config and "loader" in self.test_config
            ), "test config requires the fields `datapipeline` and `loader`"

        self.dummy = dummy
        if self.dummy:
            print("#" * 100)
            print("USING DUMMY DATASET: HOPE YOU'RE DEBUGGING ;)")
            print("#" * 100)

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        if self.dummy:
            data_fn = create_dummy_dataset
        else:
            data_fn = create_dataset

        self.train_datapipeline = data_fn(**self.train_config.datapipeline)
        if self.val_config:
            self.val_datapipeline = data_fn(**self.val_config.datapipeline)
        if self.test_config:
            self.test_datapipeline = data_fn(**self.test_config.datapipeline)

    def train_dataloader(self):
        loader = create_loader(self.train_datapipeline, **self.train_config.loader)
        return loader

    def val_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.val_datapipeline, **self.val_config.loader)

    def test_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.test_datapipeline, **self.test_config.loader)


def scale_cameras(c2ws, camera_scale=2.0):
    """
    Scales the camera-to-world matrices such that the first camera's distance is normalized.

    Parameters:
        c2ws (np.ndarray): Batch of camera-to-world matrices of shape (N, 4, 4).
        camera_scale (float): Target scale for the first camera's distance. Default is 2.0.

    Returns:
        np.ndarray: Scaled camera-to-world matrices.
    """
    camera_dists = c2ws[:, :3, 3]
    first_camera_dist = np.linalg.norm(camera_dists[0])
    translation_scaling_factor = (
        camera_scale if np.isclose(first_camera_dist, 0.0, atol=1e-5)
        else (camera_scale / first_camera_dist)
    )
    c2ws[:, :3, 3] *= translation_scaling_factor
    return c2ws


def normalize_first_to_identity(c2ws):
    """
    Normalize c2ws such that the first camera-to-world matrix becomes the identity matrix.

    Parameters:
        c2ws (np.ndarray): Batch of camera-to-world matrices of shape (N, 4, 4).

    Returns:
        np.ndarray: Updated c2ws with the first matrix as the identity.
    """
    first_c2w = c2ws[0]
    inv_first_c2w = np.linalg.inv(first_c2w)  # Invert the first matrix
    c2ws = np.einsum('ij,njk->nik', inv_first_c2w, c2ws)  # Apply the transformation
    return c2ws


class DL3DVDataset(Dataset):
    def __init__(self, dataset_dir, colmap_dir, num_images, transform=None, levels=None):
        self.dataset_dir = dataset_dir
        self.colmap_dir = colmap_dir
        self.transform = transform
        self.levels = levels if levels else []
        self.num_images = num_images
        self.scenes = self._load_scenes()
        self.adjacent_frame_sampling_prob = 0.2

        if "480P" in dataset_dir:
            self.image_shape = (270, 480)

    def _load_scenes(self):
        scenes = []
        for level in os.listdir(self.dataset_dir):
            if self.levels and level not in self.levels:
                continue
            level_path = os.path.join(self.dataset_dir, level)
            if os.path.isdir(level_path):
                for scene in os.listdir(level_path):
                    scene_path = os.path.join(level_path, scene)
                    if os.path.isdir(scene_path):
                        scenes.append(scene_path)
        return scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = self.scenes[idx]
        images_dir = os.path.join(scene_path, "images_8")

        # Load images
        frames = []
        images_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])

        # Sample frames indices
        if np.random.rand() <= self.adjacent_frame_sampling_prob:
            max_start_idx = max(0, len(images_files) - self.num_images)
            start_idx = np.random.randint(0, max_start_idx + 1)
            images_idxs = np.arange(start_idx, start_idx + self.num_images)
        else:
            images_idxs = np.random.choice(len(images_files), self.num_images, replace=False)

        images_files = [images_files[i] for i in images_idxs]
            
        frames = np.zeros((self.num_images, self.image_shape[0],  self.image_shape[1], 3))
        for i, img_file in enumerate(images_files):
            img_path = os.path.join(images_dir, img_file)
            image = cv.imread(img_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            if self.transform:
                frames[i] = image

        frames = frames.astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
        frames = frames * 2.0 - 1.0  # Normalize to [-1, 1]
        # TODO: reisze to target shape

        # Load colmap data
        colmap_scene_path = os.path.join(
            self.colmap_dir, os.path.relpath(scene_path, self.dataset_dir), "colmap", "sparse", "0"
        )
        cameras_metas, images_metas, _ = read_model(colmap_scene_path)
        images_metas = list(images_metas.items())[:self.num_images] 
        
        # Read extrinsics from COLMAP
        c2ws = np.array([read_extrinsics_colmap(image_meta, mode="c2w") for image_id, image_meta in images_metas])
        
        # Read intrinsics from COLMAP
        intrinsics = read_intrinsics_colmap(cameras_metas[1], normalize=True)
        Ks = repeat(intrinsics, 'd1 d2 -> n d1 d2', n=self.num_images)

        # Convert c2ws and Ks to torch tensors
        c2ws = torch.from_numpy(c2ws).float()
        Ks = torch.from_numpy(Ks).float()

        # TODO: verify and add normalization of cameras

        # Sample input and target frames
        num_input = np.random.randint(1, self.num_images)  # Randomly select number of input frames
        input_frames_indices = np.random.choice(self.num_images, num_input, replace=False)
        target_frames_indices = np.setdiff1d(np.arange(self.num_images), input_frames_indices)

        # Create masks
        cond_frames_mask = torch.zeros(self.num_images, dtype=bool)
        cond_frames_mask[input_frames_indices] = True

        camera_mask = torch.ones(self.num_images, dtype=bool)

        # Separate input and target frames
        cond_frames_without_noise = frames[input_frames_indices]
        cond_frames = frames[target_frames_indices]

        # Add optional noise to input frames
        # cond_frames = cond_frames_without_noise + 0.0 * np.random.randn(*cond_frames_without_noise.shape).astype(np.float32) TODO: in training code?


        output_dict = {
            "cond_frames_without_noise": cond_frames_without_noise,  # Input frames
            "cond_frames": cond_frames,  # Input frames with optional noise
            "cond_frames_mask": cond_frames_mask,  # Boolean mask for input frames
            "c2w": c2ws,  # Camera-to-world matrices
            "K": Ks,  # Intrinsic matrices
            "camera_mask": camera_mask,  # Boolean mask for input cameras
            # "plucker_coordinate": plucker_coordinate,  # Plücker coordinates
        }

        return output_dict



class DL3DVDataModuleFromConfig(LightningDataModule):
    def __init__(
            self,
            dataset_dir,
            colmap_dir, 
            batch_size, 
            num_workers=0, 
            shuffle=True):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.colmap_dir = colmap_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_dataset = DL3DVDataset(
            dataset_dir,
            colmap_dir,
            num_images=8,
        )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
