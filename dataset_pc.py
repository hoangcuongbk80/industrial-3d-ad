import pathlib
import glob
import os

from torch.utils.data import Dataset
import open3d as o3d
import numpy as np


def real3d_classes():
    return ['airplane','car','candybar','chicken',
            'diamond','duck','fish','gemstone',
            'seahorse','shell','starfish','toffees']


class Dataset3dad_train(Dataset):
    def __init__(self, dataset_dir, cls_name, num_points, if_norm=True, if_cut=False):
        self.num_points = num_points
        self.dataset_dir = dataset_dir
        self.if_norm = if_norm
        self.train_sample_list = glob.glob(
            os.path.join(dataset_dir, cls_name, 'train', '*template*.pcd')
        )

    def norm_pcd(self, point_cloud):
        center = np.mean(point_cloud, axis=0)
        return point_cloud - center

    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.train_sample_list[idx])
        pts = np.asarray(pcd.points)
        if self.if_norm:
            pts = self.norm_pcd(pts)
        mask = np.zeros(len(pts), dtype=np.float32)
        label = 0
        return pts, mask, label, self.train_sample_list[idx]

    def __len__(self):
        return len(self.train_sample_list)


class Dataset3dad_test(Dataset):
    def __init__(self, dataset_dir, cls_name, num_points, if_norm=True, if_cut=False):
        self.num_points = num_points
        self.dataset_dir = dataset_dir
        self.if_norm = if_norm

        all_test = glob.glob(os.path.join(dataset_dir, cls_name, 'test', '*.pcd'))
        # exclude temporary templates if any
        self.test_sample_list = [p for p in all_test if 'temp' not in p]
        self.gt_dir = os.path.join(dataset_dir, cls_name, 'gt')

    def norm_pcd(self, point_cloud):
        center = np.mean(point_cloud, axis=0)
        return point_cloud - center

    def __getitem__(self, idx):
        path = self.test_sample_list[idx]
        if 'good' in path:
            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(pcd.points)
            mask = np.zeros(len(pts), dtype=np.float32)
            label = 0
        else:
            stem = pathlib.Path(path).stem
            gt_txt = os.path.join(self.gt_dir, f"{stem}.txt")
            data = np.loadtxt(gt_txt)
            pts, mask = data[:, :3], data[:, 3].astype(np.float32)
            label = 1
        if self.if_norm:
            pts = self.norm_pcd(pts)
        return pts, mask, label, path

    def __len__(self):
        return len(self.test_sample_list)


def industrial3d_classes():
    return [
        'small blade',
        'big blade',
        'tea pot',
        'rabbit',
        'bowl',
        'bearing',
        'gear',
        'valve',
        'pump',
        'connector'
    ]

class DatasetIndustrial3dad_train(Dataset):

    def __init__(self, dataset_dir, cls_name, num_points, if_norm=True):
        self.dataset_dir = dataset_dir
        self.cls_name = cls_name
        self.num_points = num_points
        self.if_norm = if_norm
        # all .ply files in train/
        self.train_list = glob.glob(
            os.path.join(dataset_dir, cls_name, 'train', '*.ply')
        )

    def norm_pcd(self, pts):
        center = np.mean(pts, axis=0)
        return pts - center

    def __getitem__(self, idx):
        path = self.train_list[idx]
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        if self.if_norm:
            pts = self.norm_pcd(pts)
        mask = np.zeros(len(pts), dtype=np.float32)
        label = 0
        return pts, mask, label, path

    def __len__(self):
        return len(self.train_list)


class DatasetIndustrial3dad_test(Dataset):

    def __init__(self, dataset_dir, cls_name, num_points, if_norm=True):
        self.dataset_dir = dataset_dir
        self.cls_name = cls_name
        self.num_points = num_points
        self.if_norm = if_norm

        test_glob = glob.glob(
            os.path.join(dataset_dir, cls_name, 'test', '*.ply')
        )
        self.test_list = [p for p in test_glob if 'template' not in p]
        self.gt_dir = os.path.join(dataset_dir, cls_name, 'gt')

    def norm_pcd(self, pts):
        center = np.mean(pts, axis=0)
        return pts - center

    def __getitem__(self, idx):
        path = self.test_list[idx]
        if 'good' in path:
            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(pcd.points)
            mask = np.zeros(len(pts), dtype=np.float32)
            label = 0
        else:
            stem = pathlib.Path(path).stem
            txt_path = os.path.join(self.gt_dir, f"{stem}.txt")
            data = np.loadtxt(txt_path)
            pts, mask = data[:, :3], data[:, 3].astype(np.float32)
            label = 1
        if self.if_norm:
            pts = self.norm_pcd(pts)
        return pts, mask, label, path

    def __len__(self):
        return len(self.test_list)
