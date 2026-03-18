#!/bin/bash
set -e
uv pip install torch torchvision torchaudio
uv pip uninstall ninja && uv pip install ninja -U 
uv pip install packaging
uv pip install psutil
uv pip install -U xformers
uv pip install flash-attn --no-build-isolation
uv pip install rotary-emb --no-build-isolation
cd flash-attention/csrc/layer_norm
uv pip install . --no-build-isolation
cd ../../..

cd TinyLlama
uv pip install -r requirements.txt tokenizers sentencepiece
uv pip install lm-eval==0.4.4 numpy==1.25.0 bitsandbytes==0.43.1
uv pip install openai==0.28 fschat==0.2.34 anthropic
uv pip install -U bitsandbytes
uv pip install --upgrade scikit-learn --no-build-isolation
uv pip install "numpy<2.0"