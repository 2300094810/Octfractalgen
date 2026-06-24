import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


_project_root = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
_octgpt_root = os.path.join(_project_root, 'octgpt')

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _octgpt_root not in sys.path:
    sys.path.insert(0, _octgpt_root)


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary occupancy classification.
    logits:  (N, 2)
    targets: (N,) in {0,1}
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)  # (N,2)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - pt).pow(self.gamma)
        ce = F.cross_entropy(logits, targets, reduction="none")
        return (focal_weight * ce).mean()


# ============================================================================
# Octant Position Embedding
# ============================================================================

class OctantPositionEmbedding(nn.Module):
    """
    Relative 3D position embedding for 8 octant children.
    """

    def __init__(self, dim: int):
        super().__init__()
        offsets = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ], dtype=torch.float32)
        self.register_buffer("offsets", offsets)
        self.proj = nn.Linear(3, dim)

    def forward(self) -> torch.Tensor:
        return self.proj(self.offsets)  # (8, dim)


# ============================================================================
# Parent -> 8 children feature expander
# ============================================================================

class LocalCrossAttentionExpander(nn.Module):
    """
    Parent feature -> 8 child features via local cross-attention.
    """

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim

        self.octant_queries = nn.Parameter(torch.zeros(8, dim))
        nn.init.normal_(self.octant_queries, std=0.02)

        self.octant_pos_emb = OctantPositionEmbedding(dim)

        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        hidden_dim = int(dim * mlp_ratio)
        self.norm_ffn = LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, parent_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            parent_features: (N, C)

        Returns:
            child_features: (N*8, C)
        """
        N = parent_features.shape[0]
        if N == 0:
            return torch.zeros(0, self.dim, device=parent_features.device)

        context = parent_features.unsqueeze(1)  # (N,1,C)

        q = self.octant_queries.unsqueeze(0).expand(N, -1, -1)  # (N,8,C)
        q = q + self.octant_pos_emb().unsqueeze(0)

        attn_out, _ = self.cross_attn(
            query=self.norm_q(q),
            key=self.norm_kv(context),
            value=self.norm_kv(context),
        )

        children = q + attn_out
        children = children + self.ffn(self.norm_ffn(children))

        return children.reshape(N * 8, self.dim)


# ============================================================================
# Occupancy-driven Fractal Generator v4.5 modified
# ============================================================================

class FractalGenerator(nn.Module):
    """
    v4.5 modified:
      - per-level occupancy MLP heads
      - global condition with random latent
      - context dropout during training
      - FiLM conditioning for stronger latent control
      - depth embedding
      - Fourier positional encoding
    """

    def __init__(
        self,
        feature_dim: int = 384,
        full_depth: int = 2,
        depth_stop: int = 5,
        expander_num_heads: int = 4,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        occ_threshold: float = 0.5,
        occ_band: float = 0.05,
        occ_weight: float = 1.0,
        alive_weight: float = 0.1,
        rate_weight: float = 0.1,
        # Fourier positional encoding
        num_freqs: int = 6,
        pos_emb_scale: float = 1.0,
        # Global condition
        latent_dim: int = 64,
        global_max_points: int = 2048,
        noise_std: float = 1.0,
        context_drop_prob: float = 0.5,
        film_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        assert depth_stop >= full_depth, (
            f"depth_stop must be >= full_depth, "
            f"got depth_stop={depth_stop}, full_depth={full_depth}"
        )

        self.feature_dim = feature_dim
        self.full_depth = full_depth
        self.depth_stop = depth_stop
        self.num_depths = depth_stop - full_depth + 1
        self.num_expand_stages = max(depth_stop - full_depth, 0)

        self.occ_threshold = occ_threshold
        self.occ_band = occ_band
        self.occ_weight = occ_weight
        self.alive_weight = alive_weight
        self.rate_weight = rate_weight

        self.num_freqs = num_freqs
        self.pos_emb_scale = pos_emb_scale

        self.latent_dim = latent_dim
        self.global_max_points = global_max_points
        self.noise_std = noise_std
        self.context_drop_prob = context_drop_prob
        self.film_scale = film_scale

        # Loss
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Root seed
        self.root_embedding = nn.Parameter(torch.zeros(1, feature_dim))
        nn.init.normal_(self.root_embedding, std=0.02)

        # Fourier positional encoding projection
        pos_in_dim = 3 * (1 + 2 * num_freqs)
        self.pos_proj = nn.Linear(pos_in_dim, feature_dim)

        # Depth embedding
        self.depth_emb = nn.Embedding(depth_stop + 1, feature_dim)

        # Global condition encoder from sample points
        self.global_encoder = nn.Sequential(
            nn.Linear(pos_in_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Noise latent -> feature
        self.latent_to_feat = nn.Sequential(
            nn.Linear(latent_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # FiLM conditioning.
        # global_ctx -> gamma, beta
        self.global_film = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )

        # Per-level expanders
        self.feature_expanders = nn.ModuleList([
            LocalCrossAttentionExpander(
                dim=feature_dim,
                num_heads=expander_num_heads,
            )
            for _ in range(self.num_expand_stages)
        ])

        # Shared child positional bias
        self.child_pos_emb = nn.Parameter(torch.zeros(8, feature_dim))
        nn.init.normal_(self.child_pos_emb, std=0.02)

        # Per-level occupancy heads.
        # 原来是 nn.Linear(feature_dim, 2)，现在改成更强一点的 MLP。
        self.occupancy_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, 2),
            )
            for _ in range(self.num_depths)
        ])

        self.apply(self._init_weights)

        # root_embedding 和 child_pos_emb 在 apply 后会被跳过，因为它们不是 module。
        # 这里再单独初始化一次。
        nn.init.normal_(self.root_embedding, std=0.02)
        nn.init.normal_(self.child_pos_emb, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def _safe_div(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        return numer / denom.clamp(min=1.0)

    # ------------------------------------------------------------------
    # Fourier features
    # ------------------------------------------------------------------

    def _fourier_encode(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (N,3), expected in roughly [-1,1]

        returns:
            (N, 3 * (1 + 2*num_freqs))
        """
        feats = [xyz]
        for k in range(self.num_freqs):
            freq = (2.0 ** k) * math.pi
            feats.append(torch.sin(freq * xyz))
            feats.append(torch.cos(freq * xyz))
        return torch.cat(feats, dim=-1)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _make_full_grid_coords(
        self,
        batch_size: int,
        depth: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Make full grid candidate coords at a given depth.

        Returns:
            coords: (B * 2^d * 2^d * 2^d, 4)
                    columns [x, y, z, b]
        """
        scale = 2 ** depth

        xs = torch.arange(scale, device=device)
        ys = torch.arange(scale, device=device)
        zs = torch.arange(scale, device=device)

        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
        xyz = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        coords_all = []
        for b in range(batch_size):
            bcol = torch.full(
                (xyz.shape[0], 1),
                b,
                device=device,
                dtype=torch.long,
            )
            coords_all.append(torch.cat([xyz, bcol], dim=1))

        return torch.cat(coords_all, dim=0).long()

    def _expand_child_coords(self, parent_coords: torch.Tensor) -> torch.Tensor:
        """
        Expand parent voxel coords at depth d into 8 child voxel coords at depth d+1.

        parent_coords: (N,4), columns [x,y,z,b]
        """
        if parent_coords.shape[0] == 0:
            return torch.zeros(
                0,
                4,
                device=parent_coords.device,
                dtype=torch.long,
            )

        xyz = parent_coords[:, :3]
        b = parent_coords[:, 3:4]

        offsets = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ], device=parent_coords.device, dtype=torch.long)

        child_xyz = xyz.unsqueeze(1) * 2 + offsets.unsqueeze(0)  # (N,8,3)
        child_b = b.unsqueeze(1).expand(-1, 8, -1)               # (N,8,1)

        child_coords = torch.cat([child_xyz, child_b], dim=-1)
        return child_coords.reshape(-1, 4).long()

    def _coords_to_pos_emb(self, coords: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Absolute coordinate embedding using Fourier features.

        coords: (N,4) [x,y,z,b]
        We use voxel centers mapped to [-1,1].
        """
        if coords.shape[0] == 0:
            return torch.zeros(0, self.feature_dim, device=coords.device)

        scale = float(2 ** depth)
        xyz = coords[:, :3].float()

        # voxel center in [0,1]
        xyz = (xyz + 0.5) / scale

        # map to [-1,1]
        xyz = xyz * 2.0 - 1.0
        xyz = xyz * self.pos_emb_scale

        enc = self._fourier_encode(xyz)
        return self.pos_proj(enc)

    # ------------------------------------------------------------------
    # Global condition
    # ------------------------------------------------------------------

    def _encode_global_condition(
        self,
        pos: torch.Tensor,
        sdf: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build per-sample global context.

        Training:
            global_ctx = point_ctx + noise_ctx

            但是加入 context dropout 后，训练时会随机丢掉 point_ctx，
            让模型不能只依赖真实点云条件。

        Generation:
            pos is None, so global_ctx = noise_ctx
        """

        # Random latent branch.
        # Used in both training and generation.
        noise = torch.randn(batch_size, self.latent_dim, device=device) * self.noise_std
        noise_ctx = self.latent_to_feat(noise)  # (B,C)

        # Generation mode: no input points available.
        if pos is None:
            return noise_ctx

        # Training mode: build point context.
        if sdf is not None:
            valid = (sdf <= self.occ_band)
            if valid.sum() == 0:
                valid = torch.ones_like(sdf, dtype=torch.bool)
        else:
            valid = torch.ones(pos.shape[0], device=device, dtype=torch.bool)

        xyz_all = pos[valid, :3]
        b_all = pos[valid, 3].long()

        ctx_list = []
        for b in range(batch_size):
            mask_b = (b_all == b)
            xyz_b = xyz_all[mask_b]

            if xyz_b.shape[0] == 0:
                ctx_b = torch.zeros(self.feature_dim, device=device)
            else:
                if xyz_b.shape[0] > self.global_max_points:
                    idx = torch.randperm(xyz_b.shape[0], device=device)[:self.global_max_points]
                    xyz_b = xyz_b[idx]

                enc_b = self._fourier_encode(xyz_b)
                feat_b = self.global_encoder(enc_b)
                ctx_b = feat_b.mean(dim=0)

            ctx_list.append(ctx_b)

        point_ctx = torch.stack(ctx_list, dim=0)  # (B,C)

        # --------------------------------------------------------------
        # Key modification:
        # context dropout
        # --------------------------------------------------------------
        # 训练时随机丢掉真实点云条件。
        # 这样生成时只有 noise_ctx，模型也不会完全不适应。
        if self.training and self.context_drop_prob > 0:
            drop_mask = (
                torch.rand(batch_size, 1, device=device) < self.context_drop_prob
            ).float()
            point_ctx = point_ctx * (1.0 - drop_mask)

        return point_ctx + noise_ctx

    # ------------------------------------------------------------------
    # GT occupancy from sampled points
    # ------------------------------------------------------------------

    def _build_gt_occupancy(
        self,
        coords: torch.Tensor,
        depth: int,
        pos: torch.Tensor,
        sdf: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Build GT occupancy from sampled points ONLY.

        A voxel is occupied if it contains at least one sample point.

        pos:
            (N, 4), columns [x, y, z, b]
            xyz coordinates are assumed to be in [-1, 1].

        sdf:
            kept for API compatibility, ignored here.
        """
        device = coords.device
        N = coords.shape[0]

        if N == 0:
            return torch.zeros(0, device=device, dtype=torch.long)

        scale = 2 ** depth

        cx = coords[:, 0].long()
        cy = coords[:, 1].long()
        cz = coords[:, 2].long()
        cb = coords[:, 3].long()

        cand_keys = (
            cb * (scale ** 3)
            + cx * (scale ** 2)
            + cy * scale
            + cz
        )

        qxyz = pos[:, :3]
        qb = pos[:, 3].long()

        # Map all sample points to voxel indices.
        cell = ((qxyz + 1.0) / 2.0 * scale).long().clamp(0, scale - 1)

        sample_keys = (
            qb * (scale ** 3)
            + cell[:, 0] * (scale ** 2)
            + cell[:, 1] * scale
            + cell[:, 2]
        )

        sample_keys = torch.unique(sample_keys)

        if sample_keys.numel() == 0:
            return torch.zeros(N, device=device, dtype=torch.long)

        sorted_keys, _ = torch.sort(sample_keys)
        idx = torch.searchsorted(sorted_keys, cand_keys)
        idx = idx.clamp(0, sorted_keys.numel() - 1)

        matched = (sorted_keys[idx] == cand_keys)
        return matched.long()

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _alive_loss(
        self,
        child_logits: torch.Tensor,
        num_parents: int,
    ) -> torch.Tensor:
        """
        For GT occupied parents, require at least one child occupied.

        Soft version:
            P_alive = 1 - prod_j (1 - p_j)
            L_alive = -log(P_alive)
        """
        if num_parents == 0 or child_logits.shape[0] == 0:
            return torch.tensor(0.0, device=child_logits.device)

        probs = F.softmax(child_logits, dim=-1)[:, 1]
        probs = probs.view(num_parents, 8)

        alive_prob = 1.0 - torch.prod(
            1.0 - probs.clamp(1e-6, 1.0 - 1e-6),
            dim=1,
        )

        return -torch.log(alive_prob.clamp(min=1e-6)).mean()

    def _rate_loss(
        self,
        logits: torch.Tensor,
        gt_occ: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)[:, 1]
        pred_rate = probs.mean()
        gt_rate = gt_occ.float().mean()
        return (pred_rate - gt_rate).pow(2)

    def _record_metrics(
        self,
        output: dict,
        logits: torch.Tensor,
        gt_occ: torch.Tensor,
        depth: int,
    ):
        pred_occ = logits.argmax(dim=-1)

        tp = ((pred_occ == 1) & (gt_occ == 1)).float().sum()
        fp = ((pred_occ == 1) & (gt_occ == 0)).float().sum()
        fn = ((pred_occ == 0) & (gt_occ == 1)).float().sum()

        gt_pos = (gt_occ == 1).float().sum()
        pred_pos = (pred_occ == 1).float().sum()
        total = torch.tensor(float(gt_occ.numel()), device=logits.device)

        acc = (pred_occ == gt_occ).float().mean()
        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)

        output[f"occ_acc_d{depth}"] = acc
        output[f"occ_precision_d{depth}"] = precision
        output[f"occ_recall_d{depth}"] = recall
        output[f"occ_gt_pos_rate_d{depth}"] = gt_pos / total
        output[f"occ_pred_pos_rate_d{depth}"] = pred_pos / total
        output[f"num_nodes_d{depth}"] = total
        output[f"num_gt_occ_d{depth}"] = gt_pos
        output[f"num_pred_occ_d{depth}"] = pred_pos

        return acc

    # ------------------------------------------------------------------
    # Feature init / expansion
    # ------------------------------------------------------------------

    def _add_context(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        depth: int,
        global_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add absolute position embedding + depth embedding,
        then use global context as FiLM modulation.

        global_ctx:
            (B,C), indexed by coords[:,3]
        """
        if feats.shape[0] == 0:
            return feats

        b = coords[:, 3].long()

        # Local node information.
        feats = feats + self._coords_to_pos_emb(coords, depth)
        feats = feats + self.depth_emb(
            torch.full(
                (feats.shape[0],),
                depth,
                device=feats.device,
                dtype=torch.long,
            )
        )

        # --------------------------------------------------------------
        # Key modification:
        # FiLM conditioning
        # --------------------------------------------------------------
        gamma_beta = self.global_film(global_ctx[b])
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        # Small gamma scale makes early training more stable.
        feats = feats * (1.0 + self.film_scale * gamma) + beta

        return feats

    def _init_features(
        self,
        coords: torch.Tensor,
        depth: int,
        global_ctx: torch.Tensor,
    ) -> torch.Tensor:
        N = coords.shape[0]
        feats = self.root_embedding.expand(N, -1).contiguous()
        feats = self._add_context(feats, coords, depth, global_ctx)
        return feats

    def _expand_features(
        self,
        parent_features: torch.Tensor,
        parent_coords: torch.Tensor,
        parent_depth: int,
        level_idx: int,
        global_ctx: torch.Tensor,
    ):
        """
        Expand GT/pred occupied parents into child features + child coords.
        """
        if parent_features.shape[0] == 0:
            empty_feats = torch.zeros(
                0,
                self.feature_dim,
                device=parent_features.device,
            )
            empty_coords = torch.zeros(
                0,
                4,
                device=parent_features.device,
                dtype=torch.long,
            )
            return empty_feats, empty_coords

        child_features = self.feature_expanders[level_idx](parent_features)

        N = parent_features.shape[0]
        child_features = child_features.view(N, 8, self.feature_dim)
        child_features = child_features + self.child_pos_emb.unsqueeze(0)
        child_features = child_features.reshape(N * 8, self.feature_dim)

        child_coords = self._expand_child_coords(parent_coords)

        child_features = self._add_context(
            child_features,
            child_coords,
            parent_depth + 1,
            global_ctx,
        )

        return child_features, child_coords

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, octree_gt=None, pos=None, sdf=None):
        """
        Training forward with teacher forcing.

        Process:
          - predict current-node occupancy with per-level head
          - if GT occupancy=1 and not last depth, expand to children
          - for occupied parents, enforce alive loss on preview child occupancy
        """
        assert pos is not None, "pos is required for occupancy supervision."

        device = pos.device
        batch_size = int(pos[:, 3].max().item()) + 1

        global_ctx = self._encode_global_condition(
            pos=pos,
            sdf=sdf,
            batch_size=batch_size,
            device=device,
        )

        output = {}

        total_occ_loss = torch.tensor(0.0, device=device)
        total_alive_loss = torch.tensor(0.0, device=device)
        total_rate_loss = torch.tensor(0.0, device=device)
        total_occ_acc = torch.tensor(0.0, device=device)

        # Start from full grid candidates at full_depth.
        current_coords = self._make_full_grid_coords(
            batch_size,
            self.full_depth,
            device,
        )
        current_features = self._init_features(
            current_coords,
            self.full_depth,
            global_ctx,
        )

        active_depths = 0
        expand_stages_used = 0

        for depth in range(self.full_depth, self.depth_stop + 1):
            head_idx = depth - self.full_depth

            logits = self.occupancy_heads[head_idx](current_features)
            gt_occ = self._build_gt_occupancy(
                current_coords,
                depth,
                pos,
                sdf,
            )

            occ_loss = self.focal_loss(logits, gt_occ)
            rate_loss = self._rate_loss(logits, gt_occ)

            total_occ_loss = total_occ_loss + occ_loss
            total_rate_loss = total_rate_loss + rate_loss
            active_depths += 1

            acc = self._record_metrics(output, logits, gt_occ, depth)
            total_occ_acc = total_occ_acc + acc

            # Final layer: output occupancy only, no further split.
            if depth == self.depth_stop:
                break

            # Teacher forcing: only GT occupied nodes continue.
            keep_mask = (gt_occ == 1)
            parent_features = current_features[keep_mask]
            parent_coords = current_coords[keep_mask]

            if parent_features.shape[0] == 0:
                z = torch.tensor(0.0, device=device)

                for rem_depth in range(depth + 1, self.depth_stop + 1):
                    output[f"occ_acc_d{rem_depth}"] = z
                    output[f"occ_precision_d{rem_depth}"] = z
                    output[f"occ_recall_d{rem_depth}"] = z
                    output[f"occ_gt_pos_rate_d{rem_depth}"] = z
                    output[f"occ_pred_pos_rate_d{rem_depth}"] = z
                    output[f"num_nodes_d{rem_depth}"] = z
                    output[f"num_gt_occ_d{rem_depth}"] = z
                    output[f"num_pred_occ_d{rem_depth}"] = z

                break

            level_idx = depth - self.full_depth

            child_features, child_coords = self._expand_features(
                parent_features,
                parent_coords,
                depth,
                level_idx,
                global_ctx,
            )

            # Preview next-depth child occupancy for alive loss.
            next_head_idx = head_idx + 1
            preview_child_logits = self.occupancy_heads[next_head_idx](child_features)

            alive_loss = self._alive_loss(
                preview_child_logits,
                parent_features.shape[0],
            )

            total_alive_loss = total_alive_loss + alive_loss
            expand_stages_used += 1

            current_features = child_features
            current_coords = child_coords

        output["occ_loss"] = total_occ_loss / max(active_depths, 1)
        output["alive_loss"] = total_alive_loss / max(expand_stages_used, 1)
        output["rate_loss"] = total_rate_loss / max(active_depths, 1)
        output["occ_accuracy"] = total_occ_acc / max(active_depths, 1)

        output["loss"] = (
            self.occ_weight * output["occ_loss"]
            + self.alive_weight * output["alive_loss"]
            + self.rate_weight * output["rate_loss"]
        )

        return output

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, batch_size: int = 1, device: str = "cuda"):
        """
        Free-running generation.

        Process:
          - current node predicts occupancy
          - occ=0 => prune
          - occ=1 => keep
          - if not last depth => expand kept nodes into children
          - if last depth => final occupied voxels

        Returns:
          stats: dict with per-depth counts
          final_coords: occupied coords at depth_stop
        """
        stats = {}

        if isinstance(device, str):
            device = torch.device(device)

        global_ctx = self._encode_global_condition(
            pos=None,
            sdf=None,
            batch_size=batch_size,
            device=device,
        )

        current_coords = self._make_full_grid_coords(
            batch_size,
            self.full_depth,
            device,
        )
        current_features = self._init_features(
            current_coords,
            self.full_depth,
            global_ctx,
        )

        final_coords = torch.zeros(
            0,
            4,
            device=device,
            dtype=torch.long,
        )

        for depth in range(self.full_depth, self.depth_stop + 1):
            head_idx = depth - self.full_depth

            logits = self.occupancy_heads[head_idx](current_features)
            probs = F.softmax(logits, dim=-1)[:, 1]

            pred_occ = (probs > self.occ_threshold).long()

            num_nodes = int(current_coords.shape[0])
            num_occ = int(pred_occ.sum().item())

            stats[f"depth_{depth}_num_nodes"] = num_nodes
            stats[f"depth_{depth}_num_occ"] = num_occ

            keep_mask = (pred_occ == 1)

            if depth == self.depth_stop:
                final_coords = current_coords[keep_mask]
                break

            parent_features = current_features[keep_mask]
            parent_coords = current_coords[keep_mask]

            if parent_features.shape[0] == 0:
                for rem_depth in range(depth + 1, self.depth_stop + 1):
                    stats[f"depth_{rem_depth}_num_nodes"] = 0
                    stats[f"depth_{rem_depth}_num_occ"] = 0

                final_coords = torch.zeros(
                    0,
                    4,
                    device=device,
                    dtype=torch.long,
                )
                break

            level_idx = depth - self.full_depth

            current_features, current_coords = self._expand_features(
                parent_features,
                parent_coords,
                depth,
                level_idx,
                global_ctx,
            )

        return stats, final_coords