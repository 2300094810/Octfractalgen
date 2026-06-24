import os
import numpy as np
import torch


def coords_to_points(final_coords: torch.Tensor, depth: int):
    """
    将最终 occupied voxel 的整数坐标转换为 [-1,1]^3 中的点云中心点。

    Args:
        final_coords: (N,4) tensor, columns [x, y, z, b]
        depth: final voxel depth

    Returns:
        points: (N,3) numpy array in [-1,1]
        batch_ids: (N,) numpy array
    """
    if final_coords is None or final_coords.numel() == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    coords = final_coords.detach().cpu().numpy()
    xyz = coords[:, :3].astype(np.float32)
    batch_ids = coords[:, 3].astype(np.int64)

    scale = float(2 ** depth)

    # voxel center in [0,1]
    pts = (xyz + 0.5) / scale

    # map to [-1,1]
    pts = pts * 2.0 - 1.0
    return pts, batch_ids


def save_pointcloud_ply(points: np.ndarray, save_path: str):
    """
    Save as ASCII PLY point cloud for MeshLab.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")