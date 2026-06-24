## Example Workflow

```bash
# 1. Create conda environment
conda create -n fractal_v5 python=3.10 -y

# 2. Activate environment
conda activate fractal_v5

# 3. Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install requirements
pip install -r requirements.txt

(Please change the file path to your own path.)
# 5. Train v5
python v5/main_fractal_v5.py --config configs/shapenet_fractal_v5.yaml

# 6. Generate samples with checkpoint
python v5/main_fractal_v5.py --config configs/shapenet_fractal_v5.yaml SOLVER.run generate SOLVER.ckpt logs/fractal/v5/best_model.pth
```
