FROM nvcr.io/nvidia/pytorch:23.12-py3

# 2. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# 3. Create the venv
RUN mkdir -p /venv
RUN uv venv --python=3.9 /venv/mdm-prime-v2-litgpt

# 4. CRITICAL FIX: "Activate" the environment for Docker
# This points 'python', 'pip', and 'uv' to your specific venv
ENV VIRTUAL_ENV=/venv/mdm-prime-v2-litgpt
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 5. Now install. uv will see $VIRTUAL_ENV and install here automatically.
RUN uv pip install torch torchvision torchaudio
RUN uv pip install --upgrade ninja packaging psutil

# # 5. Install xformers and generic flash-attention
RUN uv pip install -U xformers
RUN uv pip install flash-attn --no-build-isolation

# # 6. Install Rotary Embedding
RUN uv pip install rotary-emb --no-build-isolation

# # 7. Compile Flash Attention Kernels (xentropy & layer_norm)
# # 8.6 = RTX 3090, 9.0 = H100
ENV TORCH_CUDA_ARCH_LIST="8.6;9.0"
ENV MAX_JOBS=2
ADD ./flash-attention /tmp/flash-attention
RUN cd /tmp/flash-attention/csrc/xentropy \
    && uv pip install . --no-build-isolation
RUN cd /tmp/flash-attention/csrc/layer_norm \
    && uv pip install . --no-build-isolation
RUN cd /workspace \
    && rm -rf /tmp/flash-attention
    
# # 8. Install TinyLlama & Project Dependencies
# # We group these to optimize caching.
ADD ./TinyLlama /tmp/TinyLlama
RUN uv pip install -r /tmp/TinyLlama/requirements.txt \
    && rm -rf /tmp/flash-attention

RUN uv pip install \
    tokenizers \
    sentencepiece \
    "lm-eval==0.4.4" \
    "openai==0.28" \
    "fschat==0.2.34" \
    anthropic

# 6. (Optional) Visual prompt for interactive sessions
RUN echo "source /venv/mdm-prime-v2-litgpt/bin/activate" >> ~/.bashrc
RUN sed -i 's/^.*FLASH_VER_LAST =.*/    FLASH_VER_LAST = parse_version("2.8.3")  # last supported, inclusive/' \
    /venv/mdm-prime-v2-litgpt/lib/python3.9/site-packages/xformers/ops/fmha/flash.py
RUN uv pip install -U bitsandbytes
RUN uv pip install --upgrade scikit-learn --no-build-isolation
RUN uv pip install "numpy<2.0"
RUN uv pip install "datasets==2.19.0"
COPY . /workspace
