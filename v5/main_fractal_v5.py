"""
Training / evaluation entry-point for occupancy-driven FractalGenerator v4
只保存 MeshLab 可打开的 .ply 点云文件
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_ROOT, ".."))
_OCTGPT_ROOT = os.path.join(_PROJECT_ROOT, "octgpt")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _OCTGPT_ROOT not in sys.path:
    sys.path.insert(0, _OCTGPT_ROOT)

import torch
from tqdm import tqdm

from thsolver import Solver
from octgpt.utils import builder
from fractal_models.fractal_generator_v5 import FractalGenerator
from vis_occ_utils import coords_to_points, save_pointcloud_ply


class FractalSolver(Solver):
    def __init__(self, FLAGS, is_master=True):
        super().__init__(FLAGS, is_master)
        self.depth = FLAGS.MODEL.depth
        self.depth_stop = FLAGS.MODEL.depth_stop
        self.full_depth = FLAGS.MODEL.full_depth

    # ------------------------------------------------------------------
    # Model & dataset
    # ------------------------------------------------------------------

    def get_model(self, flags):
        model = FractalGenerator(**flags.FractalGen)
        model.cuda(device=self.device)
        self.model_module = model
        return model

    def get_dataset(self, flags):
        return builder.build_dataset(flags)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def batch_to_cuda(self, batch):
        for key in ["octree", "octree_in", "octree_gt",
                    "pos", "sdf", "grad", "weight", "occu", "color"]:
            if key in batch:
                batch[key] = batch[key].cuda()

    def model_forward(self, batch):
        self.batch_to_cuda(batch)

        output = self.model(
            octree_gt=batch.get("octree_gt", None),
            pos=batch.get("pos", None),
            sdf=batch.get("sdf", None),
        )
        return output

    # ------------------------------------------------------------------
    # Train / test steps
    # ------------------------------------------------------------------

    def train_step(self, batch):
        output = self.model_forward(batch)
        return {"train/" + k: v for k, v in output.items()}

    def test_step(self, batch):
        with torch.no_grad():
            output = self.model_forward(batch)
        return {"test/" + k: v for k, v in output.items()}

    def test_epoch(self, epoch):
        if epoch % 20 != 0:
            return
        super().test_epoch(epoch)
        if self.is_master:
            self.generate_step(epoch)

    # ------------------------------------------------------------------
    # Save PLY only
    # ------------------------------------------------------------------

    def _save_generated_pointcloud(self, final_coords, sample_index: int):
        save_dir = os.path.join(self.logdir, "visuals")
        os.makedirs(save_dir, exist_ok=True)

        points, batch_ids = coords_to_points(final_coords, self.depth_stop)
        ply_path = os.path.join(save_dir, f"sample_{sample_index:04d}_points.ply")
        save_pointcloud_ply(points, ply_path)
        print(f"Saved MeshLab point cloud: {ply_path}")

    # ------------------------------------------------------------------
    # Generation stats
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_step(self, index):
        model = self.model_module
        model.eval()

        with torch.autocast("cuda", enabled=self.use_amp):
            stats, final_coords = model.generate(batch_size=1, device=self.device)

        print("=" * 80)
        print(f"[Occupancy Generation v4] sample index = {index}")
        print(f"full_depth = {self.full_depth}, depth_stop = {self.depth_stop}")

        for depth in range(self.full_depth, self.depth_stop + 1):
            n = stats.get(f"depth_{depth}_num_nodes", 0)
            occ = stats.get(f"depth_{depth}_num_occ", 0)
            print(f"Depth {depth}: candidate nodes = {n}, predicted occupied = {occ}")

        final_num = int(final_coords.shape[0]) if final_coords is not None else 0
        print(f"Final depth {self.depth_stop} occupied voxels = {final_num}")

        self._save_generated_pointcloud(final_coords, index)
        print("=" * 80)

    # ------------------------------------------------------------------
    # Bulk generation
    # ------------------------------------------------------------------

    def generate(self):
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.load_checkpoint()
        self.model.eval()

        num_samples = self.FLAGS.get("num_generate", 20)
        for i in tqdm(range(num_samples), ncols=80):
            self.generate_step(i)


if __name__ == "__main__":
    FractalSolver.main()