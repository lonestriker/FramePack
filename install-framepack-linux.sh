#!/bin/bash -x
#
# Install script for Framepack on Linux
# cu126 works for 3090 and 4090, but will not work for 5090
#

TARGET_5090=0

# Install uv (manages Python and modules)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ -f "$HOME/.local/bin/env" ]; then
    source $HOME/.local/bin/env
fi

uv venv --seed --python-preference=only-managed --python=3.10
source .venv/bin/activate

if [ "$TARGET_5090" -ne 1 ]; then
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
    pip install flash-attn
    uv pip install sageattention==1.0.6

else
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    # No pre-built xformers, sageattention or flash-attn for 5090 as of 2025-04-23
    pushd .
    # Build SageAttention from source at parallel directory level as this script
    cd ..
    git clone https://github.com/thu-ml/SageAttention.git
    cd SageAttention/
    pip install .
    popd

    # xformers does not seem to make a difference for 5090
    # From: https://github.com/facebookresearch/xformers/blob/main/README.md#installing-xformers
    # (Optional) Makes the build much faster
    # pip install ninja
    # Set TORCH_CUDA_ARCH_LIST if running and building on different GPU types
    # pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
    # (this can take dozens of minutes)
fi

uv pip install -r requirements.txt
