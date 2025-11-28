# --- imports 保持不变 ----------------------------------------------------------
from collections.abc import Callable, Sequence
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
import collections
import numpy as np
from monai.data import *
import pickle
from monai.transforms import *
from math import *

from torch.utils.data import Dataset
import os
import numpy as np
import torch
import random
import torch.nn.functional as F
from .data_aug import RandomGammaTransfer, tensor_z_score_clip, RandomHorizontalFlipCoord, RandomVerticalFlipCoord, \
    RandomRotateCoord, RandomRotateAgSynTline, RandomGaussianBlur, RandomNoise

# -----------------------------------------------------------------------------

class seis_dataset(Dataset):
    def __init__(self, args, shape, pretrain=False):
        """
        args.field_dir: directory with .npy seismic volumes
        shape: target output shape (T, H, W), e.g. (128,128,128)
        pretrain: if True, each file is sampled multiple times per epoch
        """
        self.pretrain = pretrain
        self.args = args

        # 1) collect all file paths, do not load
        self.field_paths = [
            os.path.join(root, fname)
            for root, _, files in os.walk(args.field_dir)
            for fname in files if fname.endswith('.npy')
        ]

        self.target_size = shape
        tt, th, tw = self.target_size
        assert th == tw, "Height and width must be equal"
        self.crop_range_wh = (th // 2, th * 2)
        self.crop_range_t  = (tt // 2, tt * 2)

        self.pos_aug_list = [
            RandomRotateCoord,
            RandomVerticalFlipCoord,
            RandomHorizontalFlipCoord,
            RandomRotateAgSynTline,
        ]
        self.vox_aug_list = [
            RandomNoise,
            RandomGammaTransfer,
            RandomGaussianBlur
        ]

        # instantiate VoCoAugmentation
        self.voco_aug = VoCoAugmentation(args, aug=False)

    def __len__(self):
        n = len(self.field_paths)
        return n * 100 if self.pretrain else n

    def __getitem__(self, index):
        # 2) determine file by index (cycle through)
        n = len(self.field_paths)
        file_idx = index % n
        path = self.field_paths[file_idx]

        # 3) mmap load
        full_seis = np.load(path, mmap_mode='r')
        T, H, W  = full_seis.shape

        # 4) random crop dims and start positions
        sample_t = random.randint(self.crop_range_t[0], min(self.crop_range_t[1], T))
        sample_h = random.randint(self.crop_range_wh[0], min(self.crop_range_wh[1], H))
        sample_w = random.randint(self.crop_range_wh[0], min(self.crop_range_wh[1], W))
        start_t  = random.randint(0, T - sample_t)
        start_h  = random.randint(0, H - sample_h)
        start_w  = random.randint(0, W - sample_w)

        # 5) slice and copy to memory
        seis_np = full_seis[
            start_t:start_t+sample_t,
            start_h:start_h+sample_h,
            start_w:start_w+sample_w
        ].astype(np.float32).copy()

        seis = torch.from_numpy(seis_np)[None]  # [1, D, H, W]

        # 6) resize to target
        seis = F.interpolate(
            seis[None],
            size=self.target_size,
            mode='trilinear',
            align_corners=True
        )[0]

        # 7) spatial and intensity augmentations
        random.shuffle(self.pos_aug_list)
        for aug in self.pos_aug_list:
            seis, _ = aug(seis, torch.zeros_like(seis))
        random.shuffle(self.vox_aug_list)
        for aug in self.vox_aug_list:
            seis = aug(seis)

        # 8) VoCoAugmentation → imgs, labels, crops
        raw_imgs, labels, raw_crops = self.voco_aug({'seis': seis})
        imgs  = [{'image': im} for im in raw_imgs]
        crops = [{'image': cr} for cr in raw_crops]
        return imgs, labels, crops

        # imgs, labels, crops = self.voco_aug({'seis': seis})
        # return imgs, labels, crops


# =============================================================================
class VoCoAugmentation():
    """
    3D 版本的 VoCo 预处理：
    - 生成 sw_batch_size 张大视图（random 3D ROI）
    - 基于 4×4×4（A 尺度）和 8×8×8（B 尺度）划分体积，共 64+512=576 个 block
    - 返回：
        imgs  : list[Tensor 1×d×h×w]，长度 = sw_batch_size
        labels: np.ndarray [sw_batch_size, 576]，按 [A(64), B(512)] 顺序
        crops : list[Tensor 1×d×h×w]，长度 = 576
    """
    def __init__(self, args, aug):
        self.args = args
        self.aug  = aug

    # -------------------------------------------------------------------------
    def __call__(self, x_in):
        # x_in['seis']: [1, D, H, W]
        _, D, H, W = x_in['seis'].shape
        max_size = int(min(D, H, W))

        roi_big   = min(int(self.args.roi_big),   max_size)   # 沿用 coarse 尺度的大视图裁剪边长
        roi_small = min(int(self.args.roi_small), max_size)   # resize 后边长

        # 双尺度网格：A = 4^3=64, B = 8^3=512
        n_A = 4
        n_B = 8

        # --- 随机大视图 + 对应重叠标签 [sw, 576] -----------------------
        vanilla_trans, labels = get_vanilla_transform_3d_dual(
            num=self.args.sw_batch_size,
            n_A=n_A,
            n_B=n_B,
            roi_view=roi_big,   # 作为 coarse 尺度的 k_A 尺寸
            max_size=max_size,
            aug=self.aug,
            roi_small=roi_small,
        )

        # --- 规则网格 base crops：64 + 512 = 576 个 --------------------
        crops_trans = get_crop_transform_3d_dual(
            n_A=n_A,
            n_B=n_B,
            max_size=max_size,
            roi_A=max_size // n_A,
            roi_B=max_size // n_B,
            roi_small=roi_small,
            aug=self.aug,
        )

        # 实际执行 transform 得到 Tensor
        imgs  = [trans(x_in)['seis'] for trans in vanilla_trans]  # list[sw] of [1,d,h,w]
        crops = [trans(x_in)['seis'] for trans in crops_trans]    # list[576] of [1,d,h,w]

        return imgs, labels, crops


# =============================================================================
def _sample_center_and_labels_3d(roi, max_size, n):
    """
    单尺度的重叠比例标签：
    - 体积被看成 [0, max_size]^3
    - 划分为 n^3 个 block
    - 随机采样一个边长 roi 的立方体，计算与各 block 的重叠体积 / roi^3
    返回：
        center_x, center_y, center_z, labels (np.ndarray [n^3])
    """
    half = roi // 2

    # 确保大视图完全落在 [0, max_size] 区间内
    low = half
    high = max_size - half
    if high <= low:
        center_x = center_y = center_z = max_size // 2
    else:
        center_x = np.random.randint(low=low, high=high)
        center_y = np.random.randint(low=low, high=high)
        center_z = np.random.randint(low=low, high=high)

    x_min, x_max = center_x - half, center_x + half
    y_min, y_max = center_y - half, center_y + half
    z_min, z_max = center_z - half, center_z + half

    roi_vol = float(roi * roi * roi)

    block = max_size / float(n)
    labels = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                bx_min = i * block
                bx_max = (i + 1) * block
                by_min = j * block
                by_max = (j + 1) * block
                bz_min = k * block
                bz_max = (k + 1) * block

                dx = min(bx_max, x_max) - max(bx_min, x_min)
                dy = min(by_max, y_max) - max(by_min, y_min)
                dz = min(bz_max, z_max) - max(bz_min, z_min)
                if dx <= 0 or dy <= 0 or dz <= 0:
                    vol = 0.0
                else:
                    vol = dx * dy * dz
                labels.append(vol / roi_vol)

    labels = np.asarray(labels, dtype=np.float32)  # [n^3]
    return int(center_x), int(center_y), int(center_z), labels


def get_vanilla_transform_3d_dual(
    num=2,
    n_A=4,
    n_B=8,
    roi_view=32,
    max_size=128,
    aug=False,
    roi_small=32,
):
    """
    3D 版本的大视图 + 双尺度标签（修改版）：

    - 尺度 A（coarse）：
        * base 网格：n_A^3（例如 4×4×4 = 64），block 尺寸 = max_size / n_A（例如 32）
        * k_A 立方体：边长 roi_A = roi_view（通常也设置为 max_size / n_A）
    - 尺度 B（fine）：
        * base 网格：n_B^3（例如 8×8×8 = 512），block 尺寸 = max_size / n_B（例如 16）
        * k_B 立方体：边长 roi_B = max_size / n_B

    逻辑：
        1. 先为 A 尺度随机采样一个中心，生成 labels_A（长度 64）
        2. 再为 B 尺度随机采样一个中心，生成 labels_B（长度 512）
        3. 拼成 label_vec = concat([labels_A, labels_B])，长度 576
        4. 生成 num 个大视图 transform：
            - 第 0 个视图：裁剪 k_A（32³）
            - 第 1 个视图：裁剪 k_B（16³）
            - 若 num > 2，其余视图简单复用 B 视图的裁剪策略
        5. labels_list 中的每一行都放同一个 label_vec，方便 DataLoader 叠成 [B, sw, 576]，
           训练时代码仍然只取 labels[:, 0, :]，即每个样本一条标签向量，
           但其中前 64 维对应 coarse 的 k_A，后 512 维对应 fine 的 k_B。
    """
    vanilla_trans = []
    labels_list = []

    # 尺度 A / B 的 k 尺寸
    roi_A = int(roi_view)             # 和原来 roi_view 一致，用作 coarse k_A 的边长（一般是 32）
    roi_B = max_size // n_B           # 用细尺度 block 尺寸作为 k_B（一般是 16）

    # 采样 A 尺度的位置标签（仅 A 网格）
    center_A_x, center_A_y, center_A_z, labels_A = _sample_center_and_labels_3d(
        roi=roi_A, max_size=max_size, n=n_A
    )
    # 采样 B 尺度的位置标签（仅 B 网格）
    center_B_x, center_B_y, center_B_z, labels_B = _sample_center_and_labels_3d(
        roi=roi_B, max_size=max_size, n=n_B
    )

    # 拼成 [64 + 512] = 576 维的位置标签
    label_vec = np.concatenate([labels_A, labels_B], axis=0)  # [576]

    # 生成 num 个大视图 transform
    # 第一个视图：使用 A 尺度的 k_A（32³）
    if aug:
        trans_A = Compose([
            SpatialCropd(
                keys=['seis'],
                roi_center=[center_A_x, center_A_y, center_A_z],
                roi_size=[roi_A, roi_A, roi_A],
            ),
            Resized(
                keys=['seis'],
                spatial_size=(roi_small,)*3,
                mode='bilinear',
                align_corners=True,
            ),
            RandFlipd(keys=['seis'], prob=0.2, spatial_axis=0),
            RandFlipd(keys=['seis'], prob=0.2, spatial_axis=1),
            RandFlipd(keys=['seis'], prob=0.2, spatial_axis=2),
            RandRotate90d(keys=['seis'], prob=0.2, max_k=3),
            RandShiftIntensityd(keys='seis', offsets=0.1, prob=0.1),
            ToTensord(keys=['seis']),
        ])
    else:
        trans_A = Compose([
            SpatialCropd(
                keys=['seis'],
                roi_center=[center_A_x, center_A_y, center_A_z],
                roi_size=[roi_A, roi_A, roi_A],
            ),
            Resized(
                keys=['seis'],
                spatial_size=(roi_small,)*3,
                mode='bilinear',
                align_corners=True,
            ),
            ToTensord(keys=['seis']),
        ])

    # 第二个视图：使用 B 尺度的 k_B（16³）
    if aug:
        trans_B = Compose([
            SpatialCropd(
                keys=['seis'],
                roi_center=[center_B_x, center_B_y, center_B_z],
                roi_size=[roi_B, roi_B, roi_B],
            ),
            Resized(
                keys=['seis'],
                spatial_size=(roi_small,)*3,
                mode='bilinear',
                align_corners=True,
            ),
            RandFlipd(keys=['seis'], prob=0.2, spatial_axis=0),
            RandFlipd(keys=['seis'], prob=0.2, spatial_axis=1),
            RandFlipd(keys=['seis'], prob=0.2, spatial_axis=2),
            RandRotate90d(keys=['seis'], prob=0.2, max_k=3),
            RandShiftIntensityd(keys='seis', offsets=0.1, prob=0.1),
            ToTensord(keys=['seis']),
        ])
    else:
        trans_B = Compose([
            SpatialCropd(
                keys=['seis'],
                roi_center=[center_B_x, center_B_y, center_B_z],
                roi_size=[roi_B, roi_B, roi_B],
            ),
            Resized(
                keys=['seis'],
                spatial_size=(roi_small,)*3,
                mode='bilinear',
                align_corners=True,
            ),
            ToTensord(keys=['seis']),
        ])

    # 组装前两个视图
    if num <= 0:
        num = 1
    if num >= 1:
        vanilla_trans.append(trans_A)
        labels_list.append(label_vec)
    if num >= 2:
        vanilla_trans.append(trans_B)
        labels_list.append(label_vec)

    # 若 sw_batch_size > 2，则简单复用 B 尺度的视图做额外 view（不会在 loss 中单独用到）
    for _ in range(2, num):
        vanilla_trans.append(trans_B)
        labels_list.append(label_vec)

    labels = np.stack(labels_list, axis=0).astype(np.float32)  # [num, 576]
    return vanilla_trans, labels


# =============================================================================
def get_position_label_3d_dual(
    roi=32,
    max_size=128,
    n_A=4,
    n_B=8,
):
    """
    （原始版本，当前未在 get_vanilla_transform_3d_dual 中使用，
     保留以便对比 / 调试）
    """
    half = roi // 2

    # 确保大视图完全落在 [0, max_size] 区间内
    low = half
    high = max_size - half
    if high <= low:
        center_x = center_y = center_z = max_size // 2
    else:
        center_x = np.random.randint(low=low, high=high)
        center_y = np.random.randint(low=low, high=high)
        center_z = np.random.randint(low=low, high=high)

    x_min, x_max = center_x - half, center_x + half
    y_min, y_max = center_y - half, center_y + half
    z_min, z_max = center_z - half, center_z + half

    roi_vol = float(roi * roi * roi)

    # ---------------- A 尺度：4×4×4 = 64 个 block -------------------
    block_A = max_size / float(n_A)
    labels_A = []
    for i in range(n_A):
        for j in range(n_A):
            for k in range(n_A):
                bx_min = i * block_A
                bx_max = (i + 1) * block_A
                by_min = j * block_A
                by_max = (j + 1) * block_A
                bz_min = k * block_A
                bz_max = (k + 1) * block_A

                dx = min(bx_max, x_max) - max(bx_min, x_min)
                dy = min(by_max, y_max) - max(by_min, y_min)
                dz = min(bz_max, z_max) - max(bz_min, z_min)
                if dx <= 0 or dy <= 0 or dz <= 0:
                    vol = 0.0
                else:
                    vol = dx * dy * dz
                labels_A.append(vol / roi_vol)

    # ---------------- B 尺度：8×8×8 = 512 个 block -------------------
    block_B = max_size / float(n_B)
    labels_B = []
    for i in range(n_B):
        for j in range(n_B):
            for k in range(n_B):
                bx_min = i * block_B
                bx_max = (i + 1) * block_B
                by_min = j * block_B
                by_max = (j + 1) * block_B
                bz_min = k * block_B
                bz_max = (k + 1) * block_B

                dx = min(bx_max, x_max) - max(bx_min, x_min)
                dy = min(by_max, y_max) - max(by_min, y_min)
                dz = min(bz_max, z_max) - max(bz_min, z_min)
                if dx <= 0 or dy <= 0 or dz <= 0:
                    vol = 0.0
                else:
                    vol = dx * dy * dz
                labels_B.append(vol / roi_vol)

    labels_A = np.asarray(labels_A, dtype=np.float32)  # [64]
    labels_B = np.asarray(labels_B, dtype=np.float32)  # [512]
    label_vec = np.concatenate([labels_A, labels_B], axis=0)  # [576]

    return int(center_x), int(center_y), int(center_z), label_vec


# =============================================================================
def get_crop_transform_3d_dual(
    n_A=4,
    n_B=8,
    max_size=128,
    roi_A=32,
    roi_B=16,
    roi_small=32,
    aug=False,
):
    """
    生成双尺度 base crops：
    - 先生成 A 尺度 4×4×4=64 个 3D crop
    - 再生成 B 尺度 8×8×8=512 个 3D crop
    顺序：先 A 再 B，对应 labels 的 [0:64] 和 [64:576]
    """
    voco_trans = []

    # ------- 尺度 A：4×4×4 -----------------------------------------
    block_A = max_size / float(n_A)
    for i in range(n_A):
        for j in range(n_A):
            for k in range(n_A):
                cx = i * block_A + block_A / 2.0
                cy = j * block_A + block_A / 2.0
                cz = k * block_A + block_A / 2.0
                center = [int(cx), int(cy), int(cz)]

                if aug:
                    trans = Compose([
                        SpatialCropd(
                            keys=['seis'],
                            roi_center=center,
                            roi_size=[roi_A, roi_A, roi_A],
                        ),
                        Resized(
                            keys=['seis'],
                            spatial_size=(roi_small,)*3,
                            mode='bilinear',
                            align_corners=True,
                        ),
                        RandFlipd(keys=['seis'], prob=0.2, spatial_axis=0),
                        RandFlipd(keys=['seis'], prob=0.2, spatial_axis=1),
                        RandFlipd(keys=['seis'], prob=0.2, spatial_axis=2),
                        RandRotate90d(keys=['seis'], prob=0.2, max_k=3),
                        RandShiftIntensityd(keys='seis', offsets=0.1, prob=0.1),
                        ToTensord(keys=['seis']),
                    ])
                else:
                    trans = Compose([
                        SpatialCropd(
                            keys=['seis'],
                            roi_center=center,
                            roi_size=[roi_A, roi_A, roi_A],
                        ),
                        Resized(
                            keys=['seis'],
                            spatial_size=(roi_small,)*3,
                            mode='bilinear',
                            align_corners=True,
                        ),
                        ToTensord(keys=['seis']),
                    ])
                voco_trans.append(trans)

    # ------- 尺度 B：8×8×8 -----------------------------------------
    block_B = max_size / float(n_B)
    for i in range(n_B):
        for j in range(n_B):
            for k in range(n_B):
                cx = i * block_B + block_B / 2.0
                cy = j * block_B + block_B / 2.0
                cz = k * block_B + block_B / 2.0
                center = [int(cx), int(cy), int(cz)]

                if aug:
                    trans = Compose([
                        SpatialCropd(
                            keys=['seis'],
                            roi_center=center,
                            roi_size=[roi_B, roi_B, roi_B],
                        ),
                        Resized(
                            keys=['seis'],
                            spatial_size=(roi_small,)*3,
                            mode='bilinear',
                            align_corners=True,
                        ),
                        RandFlipd(keys=['seis'], prob=0.2, spatial_axis=0),
                        RandFlipd(keys=['seis'], prob=0.2, spatial_axis=1),
                        RandFlipd(keys=['seis'], prob=0.2, spatial_axis=2),
                        RandRotate90d(keys=['seis'], prob=0.2, max_k=3),
                        RandShiftIntensityd(keys='seis', offsets=0.1, prob=0.1),
                        ToTensord(keys=['seis']),
                    ])
                else:
                    trans = Compose([
                        SpatialCropd(
                            keys=['seis'],
                            roi_center=center,
                            roi_size=[roi_B, roi_B, roi_B],
                        ),
                        Resized(
                            keys=['seis'],
                            spatial_size=(roi_small,)*3,
                            mode='bilinear',
                            align_corners=True,
                        ),
                        ToTensord(keys=['seis']),
                    ])
                voco_trans.append(trans)

    # 总数应该是 64 + 512 = 576
    return voco_trans


# =============================================================================
# =============================================================================
if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('--field_dir',    type=str, default='data')
    parser.add_argument('--roi_small', type=int, default=16, help='小视图裁剪后边长（32³）')
    parser.add_argument('--roi_big', type=int, default=32, help='大视图随机裁剪边长（64³）')
    parser.add_argument('--sw_batch_size', type=int, default=2, help='生成大视图的数量')

    args = parser.parse_args()

    dataset = seis_dataset(args, shape=(128,128,128), pretrain=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (imgs, labels, crops) in enumerate(dataloader):
        imgs_np = np.stack([img.numpy() for img in imgs], axis=0)
        # 如果 imgs 本来就是一个 Tensor（比如已经被 collate 成 [B,2,1,64,64,64]），
        # 那就直接：
        # imgs_np = imgs.numpy()

        # labels 如果是 Tensor：
        labels_np = labels.numpy()
        # 如果它已经是 ndarray，就不用再调用 .numpy()

        # crops 同 imgs，先把 list 转成 ndarray：
        crops_np = np.stack([c.numpy() for c in crops], axis=0)

        print(f'Batch {i+1}: imgs {imgs[0].shape}  labels {labels.shape}  crops {len(crops)}')
        break
