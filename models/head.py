"""
VoCoHead with HRNet-style FaultNet backbone and dual-scale pretraining:
- Per-scale: L^s = L_pred^s + lambda_reg * L_reg^s
- Total (in voco_train.py): L_total = L^A + beta_dual * L^B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .HRNet import HRNet


class FaultEncoder(nn.Module):
    """
    使用 HRNet/FaultNet 作为主干：
    - 输入:  [B, C, D, H, W]  (C 一般是 1)
    - FaultNet.forward 输出: [B, 1, D, H, W] 的 fault 概率图
    - 这里对输出做全局平均池化 → [B, 1]，再映射到 [B, F] 作为预训练特征
    """

    def __init__(self, in_channels: int, feat_dim: int = 1024, base_channels: int = 8):
        super().__init__()

        # FaultNet 写死是 1 通道输入，这里做个简单校验
        assert in_channels == 1, f"FaultNet expects 1 input channel, got {in_channels}"

        # HRNet 风格主干
        self.backbone = HRNet(base=base_channels)

        # FaultNet 输出通道数是 1，所以这里线性层输入维度=1，输出维度=feat_dim
        self.proj = nn.Linear(1, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, D, H, W]
        return: [B, feat_dim]
        """
        # segmentation 输出: [B, 1, D, H, W]
        seg = self.backbone(x)

        # 全局平均池化 → [B, 1, 1, 1, 1] → flatten → [B, 1]
        g = F.adaptive_avg_pool3d(seg, output_size=1).flatten(1)

        # 映射到预训练 embedding 维度 [B, F]
        feat = self.proj(g)
        return feat


def position_prediction_loss(
    p_feats: torch.Tensor, base_feats: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Position prediction loss L_pred.

    p_feats   : [B, F]      — random subvolume features p
    base_feats: [B, K, F]   — base grid features q_i（此处已是 teacher 分支可视为 stop-grad 后的 q）
    labels    : [B, K]      — overlap ratios y_i in [0,1]

    Implementation:
        l_i = cos(p, q_i) mapped to [0,1]
        d_i = |y_i - l_i|
        L_pred = - mean log(1 - d_i)
    """
    B, K = labels.shape

    p_norm = F.normalize(p_feats, dim=-1)       # [B, F]
    b_norm = F.normalize(base_feats, dim=-1)    # [B, K, F]

    # cosine similarities: [B, K]
    sims = torch.einsum("bf,bkf->bk", p_norm, b_norm)
    sims = (sims + 1.0) / 2.0  # map from [-1,1] → [0,1]

    d = torch.abs(labels - sims)  # [B, K]
    loss = -torch.log(1.0 - d + 1e-6).mean()
    return loss


def spatial_smoothness_reg_loss(
    base_feats: torch.Tensor, D: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """
    Spatial smoothness regularization loss L_reg.

    base_feats: [B, K, F] — features q_i for each base block（teacher 分支特征）
    D        : [K, K]    — Manhattan distances D_ij between blocks
    phi      : scalar learnable param; beta = log(1 + exp(phi))

    Implementation:
        s_ij = cos(q_i, q_j)
        beta = log(1 + e^phi)
        w_ij = 1 - exp(-beta * D_ij)
        L_reg = 2/(n(n-1)) * sum_{i<j} w_ij * |s_ij|
    """
    B, K, Fdim = base_feats.shape
    feats = F.normalize(base_feats, dim=-1)  # [B, K, F]

    # cosine similarity matrix per sample: [B, K, K]
    sims = torch.einsum("bkf,blf->bkl", feats, feats)
    sims_abs = sims.abs()

    beta = torch.log1p(torch.exp(phi))  # >= 0
    D = D.to(base_feats.device)
    w = 1.0 - torch.exp(-beta * D)  # [K, K]

    # only use upper triangle i<j
    mask = torch.triu(
        torch.ones(K, K, device=base_feats.device, dtype=torch.bool), diagonal=1
    )
    w_exp = w.unsqueeze(0).expand(B, -1, -1)  # [B, K, K]

    pair = w_exp * sims_abs  # [B, K, K]
    pair_upper = pair[:, mask]  # [B, K*(K-1)/2]

    factor = 2.0 / (K * (K - 1))
    loss_per_sample = factor * pair_upper.sum(dim=-1)  # [B]
    return loss_per_sample.mean()


class Head(nn.Module):
    """
    VoCoHead with HRNet(FaultNet) backbone and dual-scale pretraining:

    - For each scale s ∈ {A, B}:
        L^s = L_pred^s + lambda_reg * L_reg^s

    - Total loss used in training (in voco_train.py):
        L_total = L^A + beta_dual * L^B

    Input:
    - img   : [B*sw,  C, D, H, W]  — random subvolumes (sw views per sample)
              其中约定：
                * 第 0 个 view 对应 coarse 尺度的 k_A（例如 32³）
                * 第 1 个 view 对应 fine   尺度的 k_B（例如 16³）
                * 若 sw > 2，其余 view 只作为额外增强，目前在 loss 中不单独区分使用
    - crops : [B*K,   C, D, H, W]  — base crops; 顺序为 A 尺度的 64 个 + B 尺度的 512 个
    - labels: [B, K]               — overlap labels; 前 K_A 列 = coarse(A)，后 K_B 列 = fine(B)
    """

    def __init__(self, args):
        super().__init__()

        in_ch = args.in_channels
        feat_dim = getattr(args, "feature_size", 1024)
        base_channels = getattr(args, "hrnet_base", 8)

        # 使用 HRNet/FaultNet 作为编码器
        self.encoder = FaultEncoder(in_channels=in_ch, feat_dim=feat_dim, base_channels=base_channels)
        self.sw_batch_size = args.sw_batch_size

        # loss hyper-params
        self.lambda_reg = getattr(args, "lambda_reg", 1.0)
        self.beta_dual = getattr(args, "beta_dual", 0.7)

        # dual-scale block counts (for 4x4x4 + 8x8x8, you can set 64, 512)
        self.num_blocks_A = int(getattr(args, "num_blocks_A", 0))
        self.num_blocks_B = int(getattr(args, "num_blocks_B", 0))

        # learnable phi for each scale (L_reg)
        self.phi_A = nn.Parameter(torch.tensor(0.0))
        self.phi_B = nn.Parameter(torch.tensor(0.0))

        # if K_A, K_B known, prebuild distance matrices; otherwise build on the fly
        if self.num_blocks_A > 0:
            D_A = self._build_3d_distance(self.num_blocks_A)
            self.register_buffer("D_A", D_A, persistent=False)
        else:
            self.D_A = None

        if self.num_blocks_B > 0:
            D_B = self._build_3d_distance(self.num_blocks_B)
            self.register_buffer("D_B", D_B, persistent=False)
        else:
            self.D_B = None

    @staticmethod
    def _build_3d_distance(K: int) -> torch.Tensor:
        """
        Build a 3D Manhattan distance matrix for K blocks.
        If K is a perfect cube, arrange them on an n×n×n grid;
        otherwise, fall back to 1D index distance.
        """
        n = int(round(K ** (1.0 / 3.0)))
        if n ** 3 != K or K <= 0:
            # fallback: 1D layout
            idx = torch.arange(K, dtype=torch.float32)
            D = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
            return D

        coords = []
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    coords.append([x, y, z])
        coords = torch.tensor(coords, dtype=torch.float32)  # [K, 3]
        D = (coords.unsqueeze(0) - coords.unsqueeze(1)).abs().sum(-1)  # [K, K]
        return D

    def _single_scale_loss(self, p_feats, base_feats, labels, phi, D=None):
        """
        Compute L = L_pred + lambda_reg * L_reg for one scale.

        这里按照论文中的“prototype stop-grad”思想：
            - base_feats 先做 detach()，作为 teacher 分支 q
            - L_pred、L_reg 都基于 q 计算，对 encoder 不回传梯度
            - encoder 的更新来源是 p_feats 对应的分支
        """
        B, K, Fdim = base_feats.shape
        if D is None or D.shape[0] != K:
            # build distance matrix on the fly
            D = self._build_3d_distance(K).to(base_feats.device)

        # teacher 原型：阻断梯度
        base_teacher = base_feats.detach()

        L_pred = position_prediction_loss(p_feats, base_teacher, labels)
        L_reg = spatial_smoothness_reg_loss(base_teacher, D, phi)
        L_total = L_pred + self.lambda_reg * L_reg
        return L_total, L_pred, L_reg

    def forward(
        self, img: torch.Tensor, crops: torch.Tensor, labels: torch.Tensor
    ):
        """
        img   : [B*sw, C, D, H, W]
        crops : [B*K,  C, D, H, W]
        labels: [B, K]

        Returns:
            L_A, L_B  (if only single scale is available, L_B is 0)
        """
        B, K_total = labels.shape
        sw = self.sw_batch_size
        C = img.shape[1]

        assert img.size(0) == B * sw, f"img batch ({img.size(0)}) != B*sw ({B*sw})"
        assert crops.size(0) % B == 0, "crops batch must be divisible by B"

        # reshape into per-sample form
        img = img.view(B, sw, C, *img.shape[2:])          # [B, sw, C, D, H, W]
        crops = crops.view(B, -1, C, *crops.shape[2:])    # [B, K_total, C, D, H, W]
        K_total_from_crops = crops.size(1)
        assert (
            K_total_from_crops == K_total
        ), f"labels K={K_total}, crops K={K_total_from_crops} mismatch"

        # encode random subvolumes：得到所有 view 的特征
        img_flat = img.view(B * sw, C, *img.shape[3:])    # [B*sw, C, D, H, W]
        p_all = self.encoder(img_flat)                    # [B*sw, F]
        p_all = p_all.view(B, sw, -1)                     # [B, sw, F]

        # coarse 尺度 A 对应第 0 个 view；fine 尺度 B 对应第 1 个 view
        p_A = p_all[:, 0, :]                              # [B, F]
        if sw >= 2:
            p_B = p_all[:, 1, :]                          # [B, F]
        else:
            # 如果 sw=1，则没有单独的 fine 视图，退化为单尺度
            p_B = p_A

        # encode base crops
        crops_flat = crops.view(B * K_total, C, *crops.shape[3:])
        base_all = self.encoder(crops_flat)          # [B*K_total, F]
        base_feats = base_all.view(B, K_total, -1)   # [B, K_total, F]

        # decide whether dual-scale is active
        use_dual = (
            self.num_blocks_A > 0
            and self.num_blocks_B > 0
            and (self.num_blocks_A + self.num_blocks_B == K_total)
        )

        if use_dual:
            K_A = self.num_blocks_A
            K_B = self.num_blocks_B

            base_A = base_feats[:, :K_A, :]      # [B, K_A, F]
            base_B = base_feats[:, K_A:, :]      # [B, K_B, F]
            labels_A = labels[:, :K_A]           # [B, K_A]
            labels_B = labels[:, K_A:]           # [B, K_B]

            D_A = self.D_A.to(base_feats.device) if self.D_A is not None else None
            D_B = self.D_B.to(base_feats.device) if self.D_B is not None else None

            # coarse 用 p_A，fine 用 p_B；base 原型在 _single_scale_loss 内部 stop-grad
            L_A, _, _ = self._single_scale_loss(
                p_A, base_A, labels_A, self.phi_A, D_A
            )
            L_B, _, _ = self._single_scale_loss(
                p_B, base_B, labels_B, self.phi_B, D_B
            )
            return L_A, L_B
        else:
            # fallback: single-scale mode; treat everything as scale A
            L_single, _, _ = self._single_scale_loss(
                p_A, base_feats, labels, self.phi_A, None
            )
            L_dummy = torch.zeros_like(L_single)
            return L_single, L_dummy
