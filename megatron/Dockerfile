FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN pip install megatron-core[lts]
RUN pip install megatron-core[mlm]
RUN pip install datasets
RUN pip install wandb
RUN pip install --upgrade huggingface_hub
RUN pip install transformers -U

# Set the working directory inside the container
WORKDIR /workspace
COPY . /workspace