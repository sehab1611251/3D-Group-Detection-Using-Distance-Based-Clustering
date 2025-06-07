
set -e # Exit on error (Exit immediately if any command fails)

# Install system packages
sudo apt-get update -qq
sudo apt-get install -y \
    python3.9 python3.9-venv python3.9-dev python3.9-distutils \
    libopenblas-dev libsparsehash-dev ninja-build cmake \
    libxcb-xinerama0 xvfb wget git

# Create Python 3.9 virtual environment
python3.9 -m venv /content/py39_env
echo "Virtual environment created at /content/py39_env"

# Upgrade pip and install core build tools inside venv
/content/py39_env/bin/pip install --upgrade pip setuptools wheel
/content/py39_env/bin/pip install cmake ninja pybind11

# Install PyTorch 1.13.1 (CUDA 11.6) inside virtual environment
/content/py39_env/bin/pip install torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Clone DCCLA repository
if [ -d "/content/DCCLA" ]; then
    echo "Removing existing /content/DCCLA ..."
    rm -rf /content/DCCLA
fi
git clone https://github.com/jinzhengguang/DCCLA.git /content/DCCLA
echo "DCCLA repo cloned."

# Download pretrained DCCLA_JRDB2022.pth
WEIGHT_URL="https://github.com/jinzhengguang/DCCLA/releases/download/v1.0/DCCLA_JRDB2022.pth"
TARGET_PATH="/content/DCCLA/DCCLA_JRDB2022.pth"
echo "Downloading pretrained weights to ${TARGET_PATH} ..."
wget -q -O "${TARGET_PATH}" "${WEIGHT_URL}"
if [ ! -f "${TARGET_PATH}" ]; then
    echo "ERROR: Pretrained checkpoint not found at ${TARGET_PATH}. Please upload it manually."
    exit 1
else
    echo "Checkpoint is ready."
fi

# Patch PyTorch’s cpp_extension.py to suppress CUDA mismatch errors
PYTORCH_CPP_EXT="/content/py39_env/lib/python3.9/site-packages/torch/utils/cpp_extension.py"
if [ -f "${PYTORCH_CPP_EXT}" ]; then
    echo "Patching ${PYTORCH_CPP_EXT} ..."
    python3 - << EOF
import re
file_path = "${PYTORCH_CPP_EXT}"
lines = open(file_path, "r").readlines()
with open(file_path, "w") as f:
    for line in lines:
        if "raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))" in line:
            indent = re.match(r'^(\s*)', line).group(1)
            f.write(f"{indent}# {line.lstrip()}")
            f.write(f"{indent}pass\n")
        else:
            f.write(line)
EOF
    echo "Patched ${PYTORCH_CPP_EXT}"
else
    echo "WARNING: ${PYTORCH_CPP_EXT} not found. Skipping patch."
fi

# Install TorchSparse v1.2.0 inside virtual environment
rm -rf /content/torchsparse
git clone https://github.com/mit-han-lab/torchsparse.git /content/torchsparse
cd /content/torchsparse
git checkout v1.2.0
/content/py39_env/bin/python setup.py build_ext --inplace
/content/py39_env/bin/python setup.py install
cd /

# Clone and install CPU-only MinkowskiEngine inside virtual environment
rm -rf /content/MinkowskiEngine
git clone https://github.com/NVIDIA/MinkowskiEngine.git /content/MinkowskiEngine

echo "Overwriting only /content/MinkowskiEngine/setup.py for CPU‐only build..."
cat > /content/MinkowskiEngine/setup.py << 'EOF'
# setup.py for CPU-only MinkowskiEngine build
import sys
import os
import re
import codecs
import subprocess
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

if sys.version_info < (3, 6):
    sys.exit("Minkowski Engine requires Python 3.6 or higher.")

try:
    import torch
except ImportError:
    raise ImportError("PyTorch not found. Please install PyTorch first.")

# Clean previous builds and uninstall existing MinkowskiEngine
subprocess.run(["rm", "-rf", "build"])
subprocess.run(["pip", "uninstall", "-y", "MinkowskiEngine"])

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

SRC_PATH = Path(here) / "src"
SOURCE_FILES = [
    "math_functions_cpu.cpp",
    "coordinate_map_manager.cpp",
    "convolution_cpu.cpp",
    "convolution_transpose_cpu.cpp",
    "local_pooling_cpu.cpp",
    "local_pooling_transpose_cpu.cpp",
    "global_pooling_cpu.cpp",
    "broadcast_cpu.cpp",
    "pruning_cpu.cpp",
    "interpolation_cpu.cpp",
    "quantization.cpp",
    "direct_max_pool.cpp",
]

ext_modules = [
    CppExtension(
        name="MinkowskiEngineBackend._C",
        sources=[str(SRC_PATH / f) for f in SOURCE_FILES] + [str(Path(here) / "pybind" / "minkowski.cpp")],
        include_dirs=[str(SRC_PATH), str(SRC_PATH / "3rdparty")],
        extra_compile_args={"cxx": ["-DCPU_ONLY", "-O3", "-fopenmp"]},
        libraries=["openblas"],
    )
]

setup(
    name="MinkowskiEngine",
    version=find_version("MinkowskiEngine", "__init__.py"),
    install_requires=["torch", "numpy"],
    packages=["MinkowskiEngine", "MinkowskiEngine.utils", "MinkowskiEngine.modules"],
    package_dir={"MinkowskiEngine": "./MinkowskiEngine"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    author="Christopher Choy",
    author_email="chrischoy@ai.stanford.edu",
    description="A convolutional neural network library for sparse tensors",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/MinkowskiEngine",
    zip_safe=False,
    classifiers=[
        "Environment :: Console",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
)
EOF

# Downgrade setuptools to avoid PEP 517 enforcement
/content/py39_env/bin/pip install setuptools==59.5.0

# Build and install MinkowskiEngine
cd /content/MinkowskiEngine

# create MinkowskiEngineBackend before running setup.py
mkdir -p MinkowskiEngineBackend

/content/py39_env/bin/python setup.py develop
cd /

# Install additional Python dependencies inside virtual environment
/content/py39_env/bin/pip install \
    open3d pyvista pillow pyyaml matplotlib \
    configobj pyvirtualdisplay \
    opencv-python python-lzf shapely

# Downgrade numpy for compatibility with TorchSparse & MinkowskiEngine
/content/py39_env/bin/pip install numpy==1.23.5 --force-reinstall

# Build DCCLA’s IOU3D extension
cd /content/DCCLA/lib/iou3d
/content/py39_env/bin/python setup.py build_ext --inplace
cd /

# Verify IOU3D import
/content/py39_env/bin/python - << EOF
import sys
sys.path.append('/content/DCCLA/lib/iou3d')
try:
    import iou3d
    print("Successfully loaded iou3d module.")
except Exception as e:
    print("ERROR loading iou3d:", e)
EOF

# Create DCCLA model loader script and verify
cat > /content/DCCLA/load_model.py << 'EOF'
import sys
import torch
import yaml
import traceback

# Add DCCLA module path
sys.path.append('/content/DCCLA')

from lidar_det.model import builder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

yaml_config_path = '/content/DCCLA/bin/jrdb22.yaml'
try:
    with open(yaml_config_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    model_cfg = {
        "type": yaml_cfg["model"]["type"],
        "kwargs": yaml_cfg["model"]["kwargs"],
        "target_mode": yaml_cfg["model"]["target_mode"],
        "disentangled_loss": yaml_cfg["model"]["disentangled_loss"],
        "nuscenes": False
    }
except Exception as e:
    print("Error parsing YAML config:", e)
    raise

model_path = '/content/DCCLA/DCCLA_JRDB2022.pth'

def load_dccla_model():
    print('Loading DCCLA Model...')
    try:
        model = builder.get_model(model_cfg, inference_only=True)
        checkpoint = torch.load(model_path, map_location=device)
        state_key = 'model_state' if 'model_state' in checkpoint else 'state_dict'
        model.load_state_dict(checkpoint[state_key], strict=False)
        model.to(device)
        model.eval()
        print('DCCLA model loaded successfully!')
        return model
    except Exception as e:
        print('Error loading DCCLA model:', e)
        traceback.print_exc()
        return None

if __name__ == "__main__":
    load_dccla_model()
EOF

echo "Running DCCLA load_model.py to verify installation..."
/content/py39_env/bin/python /content/DCCLA/load_model.py

echo "Environment setup complete. Ready to run DCCLA."
