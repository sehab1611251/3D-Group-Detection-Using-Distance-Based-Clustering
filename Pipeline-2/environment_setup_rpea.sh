
set -e  # Exit on error

# Install Python 3.9 and system packages...
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev libsparsehash-dev ninja-build wget

# Create and activate Python 3.9 virtual environment...
python3.9 -m venv /content/py39_env
/content/py39_env/bin/python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 1.13.1 (CUDA 11.6)
/content/py39_env/bin/pip install torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Install build and runtime Python packages
/content/py39_env/bin/pip install cmake pybind11 shapely pyquaternion pillow pyvista open3d git+https://github.com/teepark/python-lzf.git

# Patch PyTorch CUDA mismatch before any builds
python3 <<'EOF'
import re
file_path = "/content/py39_env/lib/python3.9/site-packages/torch/utils/cpp_extension.py"
with open(file_path, "r") as f:
    lines = f.readlines()
with open(file_path, "w") as f:
    for line in lines:
        if "raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))" in line:
            indent = re.match(r'^(\s*)', line).group(1)
            f.write(f"{indent}# {line.lstrip()}")
            f.write(f"{indent}pass\n")
        else:
            f.write(line)
EOF

# Clone and install TorchSparse (only if not already present and non-empty)
if [ -d "/content/torchsparse" ] && [ "$(ls -A /content/torchsparse)" ]; then
    echo "torchsparse directory already exists and is non-empty; skipping clone/install."
else
    cd /content
    git clone https://github.com/mit-han-lab/torchsparse.git
    cd torchsparse
    git checkout v1.2.0
    /content/py39_env/bin/python setup.py build_ext --inplace
    /content/py39_env/bin/python setup.py install
fi

# Clone and install RPEA
cd /content
git clone https://github.com/jinzhengguang/RPEA.git
cd RPEA
/content/py39_env/bin/python setup.py develop

# Reinstall lidar_det and fix directory
/content/py39_env/bin/python -m pip uninstall -y lidar_det
/content/py39_env/bin/python -m pip install .
cp -r /content/RPEA/lidar_det/model/nets /content/py39_env/lib/python3.9/site-packages/lidar_det/model/

# Build IOU3D module
cd /content/RPEA/lib/iou3d
/content/py39_env/bin/python setup.py build_ext --inplace
/content/py39_env/bin/python setup.py install

# Downgrade NumPy for compatibility
/content/py39_env/bin/python -m pip install numpy==1.23.5 --force-reinstall

# Install pyvirtualdisplay in the virtual environment
/content/py39_env/bin/pip install pyvirtualdisplay

# Install system dependency (xvfb) for virtual display
apt-get update -qq
apt-get install -y xvfb

# Install OpenCV
/content/py39_env/bin/pip install opencv-python

# Download pretrained RPEA weights
wget -O /content/RPEA/RPEA_JRDB2022.pth https://github.com/jinzhengguang/RPEA/releases/download/v1.0/RPEA_JRDB2022.pth

echo " Environment setup complete. Ready to run RPEA."
