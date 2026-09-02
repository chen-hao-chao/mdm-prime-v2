<div align="center">
<br>
<img src="assets/mdm-prime-v2.png" width="600">
</div>

<br>
<p align="center">
<a href="https://arxiv.org/abs/2603.16077"><img src="https://img.shields.io/badge/arXiv-2603.16077-b31b1b.svg?logo=arxiv&logoColor=red" alt="MDM-Prime Paper on arXiv"/></a>
<a href="https://huggingface.co/collections/chen-hao-chao/mdm-prime"><img src="https://img.shields.io/badge/🤗_HuggingFace%20-MDM_Prime_v2%20-orange" alt="MDM-Prime-v2 on Hugging Face"/></a>
<a href="https://hub.docker.com/r/chenhaochao/mdm-prime-v2-megatron"><img src="https://img.shields.io/badge/docker-MDM_Prime_v2_Megatron-blue.svg?logo=docker" alt="MDM-Prime-v2 on Docker"/></a>
<a href="https://hub.docker.com/r/chenhaochao/mdm-prime-v2-litgpt"><img src="https://img.shields.io/badge/docker-MDM_Prime_v2_litgpt-blue.svg?logo=docker" alt="MDM-Prime-v2 on Docker"/></a>
<a href="https://x.com/chenhao_chao/status/2034647722947461489"><img src="https://img.shields.io/badge/Prime_v2-black.svg?logo=X" alt="MDM-Prime-v2 on X"/></a><br>
</p>


## News
- :rocket: **[Aug 20, 2026]** Our paper has been accepted to **EMNLP 2026 (Main)**.
- :pencil2: **[May 22, 2026]** Released a corrected [paper](https://arxiv.org/abs/2603.16077). Check out [mdm-prime](https://github.com/chen-hao-chao/mdm-prime) for perplexity evaluation on OWT.
- :notebook: **[May 1, 2026]** Released [errata note](https://chen-hao-chao.github.io/dependency-breaks-validity/). The current NLL evaluation has bugs. ([old preprint](https://arxiv.org/abs/2603.16077v2))


## What’s Inside

This repository contains the code implementation of the experiments presented in the paper [*MDM-Prime-v2: Binary Encoding and Index Shuffling Enable Scaling of Diffusion Language Models*](https://arxiv.org/abs/2603.16077).

- :whale: **Docker environments** for easy installation
- 🤗 **Pretrained weights** for inference and evaluation
- :chart_with_downwards_trend: **Weights and Biases logs** for enhanced reproducibility
- :microscope: **Code for all experiments** in our paper:
  - Scaling Analysis
  - Larger-scale Pretraining



## Overview

### Scaling Analysis
- **Folder**: [mdm-prime-v2/megatron](/megatron)
- **Dataset**: [allenai/c4](https://huggingface.co/datasets/allenai/c4)
- **Weights & Biases Logs**: [lance_chao/megatron-all-runs](https://wandb.ai/lance_chao/megatron-all-runs)
- **Best for**: (1) Studying the loss behavior; (2) Pretraining with advanced parallelism

### Larger-scale Pretraining
- **Folder**: [mdm-prime-v2/lit_gpt](/lit_gpt)
- **Dataset**: [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) (or [gmongaras/SlimPajama-627B_Reupload](https://huggingface.co/datasets/gmongaras/SlimPajama-627B_Reupload))
- **Best for**: (1) Pretraining 1.1B models; (2) Running inference and downstream applications

## Demo

- Download our docker image and launch `gradio_demo.py`:
```bash
# Pull and launch the docker image
docker pull chenhaochao/mdm-prime-v2-litgpt:latest
docker run -v $(pwd):/workspace --rm -it --gpus all --ipc=host -p 3000:3000 chenhaochao/mdm-prime-v2-litgpt:latest

# Install gradio and run gradio_demo.py
uv pip install gradio
/venv/mdm-prime-v2-litgpt/bin/python gradio_demo.py
```

- Loading the model's weights takes a few minutes. After running the commands, the demo website will be available at `http://localhost:3000/`.


| <div align="center"><img src="assets/demo.gif" width="100%"></div> |
| --- |


## License
This code implementation is developed based on the following repositories.

- [ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM) (at commit `1df2e12`), licensed under the `Apache-2.0` license.
- [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama) (at commit `bf12224`), licensed under the `Apache-2.0` license.
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (at commit `636179d`), licensed under the `Apache-2.0` license.
- [wmn-231314/diffusion-data-constraint](https://github.com/wmn-231314/diffusion-data-constraint) (at commit `61002b2`), licensed under the `Apache-2.0` license.

Further changes based on the code in this folder are licensed under the `Apache-2.0` license.

## Citation

If you find this code implementation useful, please consider citing our papers.

```bib
@inproceedings{chao2026mdmprimev2,
      title = {{MDM-Prime-v2: Binary Encoding and Index Shuffling Enable Scaling of Diffusion Language Models}}, 
      author = {Chen-Hao Chao and Wei-Fang Sun and Junwei Quan and Chun-Yi Lee and Rahul G. Krishnan},
      booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
      year = {2026},
}
@article{chao2026dependency,
      title   = {{Dependency Breaks Validity of Loss Functions in Masked Diffusion Models}},
      author  = {Chen-Hao Chao and Minkai Xu and Tomas Geffner and Arash Vahdat and Rahul G. Krishnan},
      journal = {chen-hao-chao.github.io},
      year    = {2026}
}
@inproceedings{chao2025mdmprime,
      title = {{Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking}}, 
      author = {Chen-Hao Chao and Wei-Fang Sun and Hanwen Liang and Chun-Yi Lee and Rahul G. Krishnan},
      booktitle = {Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)},
      year = {2025},
}
```
