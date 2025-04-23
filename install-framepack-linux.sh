#!/bin/bash -x

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv venv --seed --python-preference=only-managed --python=3.10
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install flash-attn
uv pip install sageattention==1.0.6
uv pip install -r requirements.txt
