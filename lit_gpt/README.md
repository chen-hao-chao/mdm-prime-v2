# Pretraining using Lit-GPT

<a href="https://arxiv.org/abs/2603.16077"><img src="https://img.shields.io/badge/arXiv-2603.16077-b31b1b.svg?logo=arxiv&logoColor=red" alt="MDM-Prime Paper on arXiv"/></a>
<a href="https://huggingface.co/chen-hao-chao/mdm-prime-v2-slimpajama"><img src="https://img.shields.io/badge/🤗_HuggingFace%20-MDM_Prime_v2_Slimpajama%20-orange" alt="MDM-Prime-v2 on Hugging Face"/></a>
<a href="https://hub.docker.com/r/chenhaochao/mdm-prime-v2-litgpt"><img src="https://img.shields.io/badge/dockerhub-MDM_Prime_v2_litgpt-blue.svg?logo=docker" alt="MDM-Prime-v2 on Docker"/></a>

This folder contains the code implementation of the scaling experiments presented in **Section 4.3** of [our paper](https://arxiv.org/abs/2603.16077). Our implementation is primarily based on [ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM) and [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama).

<img src="assets/table_demo.gif" width="100%">

## Install Dependencies

Lunch our pre-built [:whale: Docker](https://www.docker.com/) image or install dependencies using [uv](https://github.com/astral-sh/uv) .

### Install Dependencies via uv

1. Download uv using:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and launch it using the following command:
```bash
uv venv --python=3.9 venvs/mdm-prime-v2-litgpt
source venvs/mdm-prime-v2-litgpt/bin/activate
```

3. Clone [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama) and [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) under the current directory (`mdm-prime-v2-preview/lit_gpt`):
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
git clone https://github.com/jzhang38/TinyLlama.git
```

Install the dependencies via the following commands:
```bash
sh install.sh
```

For more installation guidance, please refer to [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md).

<details>
<summary><strong>Possible Error Messages & Solutions</strong></summary>

**Error**. When executing `sh install.sh`:
```
...
ImportError: Requires Flash-Attention version >=2.7.1,<=2.8.2 but got 2.8.3.                                                                                                                                                 
    raise ImportError(                                                                                                                                                                                                       
ImportError: Requires Flash-Attention version >=2.7.1,<=2.8.2 but got 2.8.3.                                                                                                                                                 
E0204 16:16:04.490637 2098044 torch/distributed/elastic/multiprocessing/api.py:874] failed (exitcode: 1) local_rank: 0 (pid: 2098107) of binary: /project/6101781/chchao0/venvs/mdm-prime-v2-litgpt/bin/python3   
...
```
**Solution**. Modify `/path_to_mdm-prime-v2-litgpt/lib/python3.9/site-packages/xformers/ops/fmha/flash.py` Line 77:
```diff
- FLASH_VER_LAST = parse_version("2.8.2") 
+ FLASH_VER_LAST = parse_version("2.8.4") 
```

</details>

### Launch our Pre-built Image

1. Pull our pre-built docker image:
```bash
docker pull chenhaochao/mdm-prime-v2-litgpt:latest
# or
apptainer pull mdm-prime-v2-litgpt.sif docker://chenhaochao/mdm-prime-v2-litgpt:latest
```

2. Launch the docker image at `mdm-prime-v2/lit_gpt` through the following commands:
```bash
docker run -v $(pwd):/workspace --rm -it --gpus all --ipc=host chenhaochao/mdm-prime-v2-litgpt:latest
# or
apptainer run --nv --bind "$(pwd)":/workspace --pwd /workspace mdm-prime-v2-litgpt.sif
```

---

## Data Preparation

The dataset and the tokenizer will be downloaded through the 🤗 Huggingface APIs. Please first login to Huggingface via the following command:
```bash
hf auth login
```

1. **Download Raw Dataset**. Download [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) (or [gmongaras/SlimPajama-627B_Reupload](https://huggingface.co/datasets/gmongaras/SlimPajama-627B_Reupload) if it fails) and using the following command (also see [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md#data-preparation)):
```bash
cd ${path_to_dataset}
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
```
The SlimPajama dataset takes around 893GB diskspace. If you hit the limits of the Huggingface API, consider using [dataset/download_ds.py](/lit_gpt/dataset/download_ds.py) to download the dataset chunk-by-chunk.

2. **Download Tokenizer**. Specify your target path (`path_to_tokenizer`) in [dataset/download_tokenizer.py](/lit_gpt/dataset/download_tokenizer.py) and run the following command.
```bash
python dataset/download_tokenizer.py
```

3. **Data Preprocessing**. Use the provided scripts to tokenize the Slimpajama dataset. Remember to specify `source_path`, `tokenizer_path`, and `destination_path`:
```bash
python scripts/prepare_slimpajama.py --source_path ${path_to_dataset} --tokenizer_path ${path_to_tokenizer}  --destination_path ${path_to_destination} --split validation --percentage 1.0
python scripts/prepare_slimpajama.py --source_path ${path_to_dataset} --tokenizer_path ${path_to_tokenizer} --destination_path ${path_to_destination} --split train --percentage 1.0
```

---

## Commands

:pushpin: **Quick Start**

Pretrain MDM-Prime-v2 with 170M non-embedding parameters and 10e18 training FLOPs on a node with 2 GPUs.
```bash
lightning run model --accelerator=cuda --devices=2 --num-nodes=1 \
    /workspace/pretrain/train_prime_rl.py \
    --nodes_num 1 --gpu_num 2 \
    --model 170 --flops 10. --ssl_ratio 0.01 \
    --eval_freq 5000 \
    --wandb_project mdm_prime_v2_170M \
    --result_path ${path_to_workdir} \
    --data_path ${path_to_destination} < /dev/null
```

- **Arguments:**
    - `--nodes_num`: number of nodes to be used for training. (default: `1`)
    - `--gpu_num`: number of GPUs to be used for training. (default: `8`)
    - `--eval_freq`: frequency of evaluation during training. (default: `1000`)
    - `--model`: size for the model parameters (in millions). (default: `1028`)
    - `--flops`: total FLOPs target, scaled by $10^{18}$. (default: `None`)
    - `--ssl_ratio`: stochastic sequence length ratio. (default: `None`)
    - `--result_path`: path where training results and logs will be saved. (default: `workdir`)
    - `--wandb_project`: name of the Weights & Biases project for logging. (default: `mdm_prime_v2_1028M`)
    - `--data_path`: file path to the training dataset. (default: `/download/slim_star_combined`)

<br>

:pushpin: **Single-node Pretraining**

Pretrain MDM-Prime-v2 with 1028M non-embedding parameters and 3300e18 training FLOPs on a node with 8 GPUs.
```bash
lightning run model --accelerator=cuda --devices=8 --num-nodes=1 \
    /workspace/pretrain/train_prime_rl.py \
    --nodes_num 1 --gpu_num 8 \
    --model 1028 --flops 3300. --ssl_ratio 0.01 \
    --eval_freq 5000 \
    --wandb_project mdm_prime_v2_1028M \
    --result_path ${path_to_workdir} \
    --data_path ${path_to_destination} < /dev/null
```

<br>

:pushpin: **Multi-node Pretraining**

We are using [slurm](https://slurm.schedmd.com/documentation.html) for multi-node pretraining. Modify the following files to specify the hardware setup and the paths to the dataset and working directories:
- The slurm script ([pretrain_mdm_prime_v2.slurm](/lit_gpt/pretrain_mdm_prime_v2.slurm)) 
- The training script ([pretrain_mdm_prime_v2.sh](/lit_gpt/pretrain_mdm_prime_v2.sh))

Start training using the following command:
```bash
sbatch pretrain_mdm_prime_v2.slurm
```

<details>
<summary><strong>Detailed Description </strong> (click to expand)</summary>

The slurm script ([pretrain_mdm_prime_v2.slurm](/lit_gpt/pretrain_mdm_prime_v2.slurm)) requests two nodes with 8 GPUs on each node, and exports `MASTER_ADDR` and `MASTER_PORT` for nodes to communicate: 
```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
```
It then performs `srun` with the corresponding number of nodes (`SLURM_NNODES`). [lightning](https://lightning.ai/docs/fabric/stable/fundamentals/launch.html) will handle the multi-node setup to properly perform the training script ([pretrain_mdm_prime_v2.sh](/lit_gpt/pretrain_mdm_prime_v2.sh)). If you are running on an image, remember to export the file via `--container-image` and specify which directory to be mounted using `--container-mounts`:
```bash
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    --export=ALL \
    --container-image mdm_prime_v2_image.sqsh \
    --container-writable \
    --container-mounts mdm-prime-v2/lit_gpt:/workspace \
    bash /workspace/pretrain_mdm_prime_v2.sh
```

</details>


<br>

:pushpin: **Zero-shot Q&A Evaluation**

We use [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for zero-shot Q&A evaluation. Evaluate our pretrained MDM-Prime-v2 using the following command (example: `sciq`) -- you should get accuracy `83.30`:
```bash
export TRUST_REMOTE_CODE=1
python evaluate_prime.py --tasks sciq \
                         --model prime \
                         --batch_size 32 \
                         --model_args use_hf=True,cfg=0.25,temp=0.5,mc_num=256,chunk="0.0_0.75/1_15"
```
- **Arguments:**
    - `--tasks`: name of the task. (available tasks: [EleutherAI/lm-evaluation-harness/lm_eval/tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks#tasks))
    - `--model`: fix to `prime` in this code base. (see [evaluate_prime.py](/lit_gpt/evaluate_prime.py) Line 36).
    - `--batch_size`: evaluation batch size.
    - `--model_args`: customized variables for inference.
        - `--use_hf`: whether to use our pretrained checkpoint from Huggingface. (default: `False`)
        - `--cfg`: classifier-free guidance scale. (default: `0`)
        - `--temp`: temperature for the scheduler, calculated as (1-temp) * t + temp. (default: `1.0`)
        - `--chunk`: the chunking strategy for noise, formatted as "points/sizes". (default: `"0.0/1"`)

The full evaluation commands are provided in [eval_prime.sh](/lit_gpt/eval_prime.sh). It takes a few hours to complete all of the evaluation on a single GPU.

<br>

:pushpin: **Sampling**

Generate 5 samples using 200 function evaluations using our pretrained MDM-Prime-v2 (1.1B) model. [pszemraj/qmsum-cleaned](https://huggingface.co/datasets/pszemraj/qmsum-cleaned) is used for context.
```bash
python sampling.py --model_name='chen-hao-chao/mdm-prime-v2-slimpajama' --seed=0 \
                   --num_samples 5 \
                   --nfe 200
```

- **Arguments:**
    - `--model_name`: name of the huggingface repository.
        - or use `--ckpt_path` to specify the path to your checkpoint.
    - `--num_samples`: number of generated samples. (default: `5`)
    - `--nfe`: number of function evaluations. (default: `200`)
    - `--seed`: random seed for reproducibility. (default: `42`)

<details>
<summary><strong>Example Results</strong> (MDM-Prime-v2)</summary>

```
=== Unique Sample 5 ===
PREFIX: Barry Hughes: I understand your point. There's no change whatsoever to the burden of proof, nor to the standard of proof.                                                                                                                     
Suzy Davies AM: Perhaps you can run us through it quickly.
Barry Hughes: If the defence argue that that act was--. So, we have to show that there's been an unlawful assault. So, if we remove the defence of reasonable chastisement, in a sense that alters some things but it doesn't alter the basic responsibility of the prosecution, which is to establish its case beyond a reasonable doubt. And if the defence raise an argument and say,'Well, look, that was a lawful act; I was only doing what I thought was reasonable in the circumstances', it's for the Crown to disprove that.
Suzy Davies AM: Okay. That's great.
Barry Hughes:


SUFFIX: Suzy Davies AM: Okay. That's really helpful for us to understand that. Obviously, when we're talking about CPS guidelines and all the rest, we've already come some distance down the process, haven't we? Have you got any views on what might be done to prevent cases even coming as far as arrest? Because one of the things that we have to consider is that once you're arrested, that is recorded somewhere and will appear in things like DBS checks in the future, even if it goes no further. Do you have any views on how intervention might work better earlier on, even at the point of the knock at the door?
Barry Hughes: From the perspective of the CPS, I'm not sure I can help you there.
Suzy Davies AM: That's fine. I was expecting that answer.



GENERATED: It makes it attractive for prosecution, but the other point, Part I, guilt a completely open nature, does is make it easier for admins to accused. It just says that the person has been convicted wrongfully charged.                     
Jonathan Bronson: Does the Crown understand that?
Barry Hughes: Well...
Control stream ions lar Suzy Davies AM: Okay. That's really helpful for us to understand that. Obviously, when we're talking about CPS guidelines and all the rest, we've already come some distance down the process, haven't we? Have you got any views on what might be done to prevent cases even coming as far as arrest? Because one of the things that we have to consider is that once you're arrested, that is recorded somewhere and will appear in things like DBS checks in the future, even if it goes no further. Do you have any views on how intervention might work better earlier on, even at the point of the knock at the door?
Barry Hughes: From the perspective of the CPS, I'm not sure I can help you there.
Suzy Davies AM: That's fine. I was expecting that answer.


REFERENCE: And we've got to disprove that to the criminal standard, which is beyond a reasonable doubt. So, you can certainly see--. I can see the potential for individuals who feel strongly about this to look to contest the matter, to not admit any wrongdoing at all and to take the matter to trial, and it would be our responsibility to disprove that.
============================== 
```

</details>


<br>

Generate 5 samples using 200 function evaluations using [EleutherAI/pythia-1.4b-deduped](https://huggingface.co/EleutherAI/pythia-1.4b-deduped).
```bash
python sampling.py --model_name='EleutherAI/pythia-1.4b-deduped' --seed=0
```


<details>
<summary><strong>Example Results</strong> (Pythia)</summary>

```
=== Unique Sample 5 ===
PREFIX: Barry Hughes: I understand your point. There's no change whatsoever to the burden of proof, nor to the standard of proof.                                                                                                                     Suzy Davies AM: Perhaps you can run us through it quickly.
Barry Hughes: If the defence argue that that act was--. So, we have to show that there's been an unlawful assault. So, if we remove the defence of reasonable chastisement, in a sense that alters some things but it doesn't alter the basic responsibility of the prosecution, which is to establish its case beyond a reasonable doubt. And if the defence raise an argument and say,'Well, look, that was a lawful act; I was only doing what I thought was reasonable in the circumstances', it's for the Crown to disprove that.
Suzy Davies AM: Okay. That's great.
Barry Hughes:


SUFFIX: Suzy Davies AM: Okay. That's really helpful for us to understand that. Obviously, when we're talking about CPS guidelines and all the rest, we've already come some distance down the process, haven't we? Have you got any views on what might be done to prevent cases even coming as far as arrest? Because one of the things that we have to consider is that once you're arrested, that is recorded somewhere and will appear in things like DBS checks in the future, even if it goes no further. Do you have any views on how intervention might work better earlier on, even at the point of the knock at the door?
Barry Hughes: From the perspective of the CPS, I'm not sure I can help you there.
Suzy Davies AM: That's fine. I was expecting that answer.



GENERATED: It depends on the circumstances as to whether that's a defence or not.                                                                                                                                                                     
Suzy Davies AM: Alright. So, that sounds much smoother?
Barry Hughes: Yeah, I think so.

Barry Hughes: Yeah. It's one of them that's really confusing and has happened to me but not a lot of people talking about


REFERENCE: And we've got to disprove that to the criminal standard, which is beyond a reasonable doubt. So, you can certainly see--. I can see the potential for individuals who feel strongly about this to look to contest the matter, to not admit any wrongdoing at all and to take the matter to trial, and it would be our responsibility to disprove that.
==============================      
```

</details>


<br>

---

## Implementation

### Training Scripts

The following files are modified to adapt the original [ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM) code base to our training code:

- `pretrain/train_prime_rl.py` is based on `pretrain/train_mdm_rl.py`: [[diff]](https://github.com/chen-hao-chao/mdm-prime-v2/compare/096214f5c3fdefe0aa8302881eb586e8e5eacc18...e491c9b26a6551b6d93936ae3241255202be681d#diff-eaa7bc8d76dee9957b55627532470f9e20b1639b713d77f480e67b8978164431)
- `lit_gpt/diffmodel.py` is modified to accept sub-token inputs: [[diff]](https://github.com/chen-hao-chao/mdm-prime-v2/compare/096214f5c3fdefe0aa8302881eb586e8e5eacc18...e491c9b26a6551b6d93936ae3241255202be681d#diff-dc4e1b21e60647885b4de82e823aa5c8ecd6c6bbf59b66fe23349eb4e3895dca)

### Subtokenizers

We make [subtokenizer](/lit_gpt/subtokenizer) a separate package. MDM-Prime-v2 uses `BasebShufflingLayer`, which is a `torch.nn.Module` object that can encode (or decode) tokens into sub-tokens. Example usage ($b=2$, $\ell=15$):
```python
from subtokenizer.layers import BasebShufflingLayer

base = 2
target_length = 15 # token granularity in our paper
random_ratio = 1.0
subtokenizer = BasebShufflingLayer(base=base, target_length=target_length, perm=None, vocab_size=32000)
```
The argument `perm` is a random permutation dictionary. The above example sets `perm=None` and `BasebShufflingLayer` will initiate a random permutation dictionary. If `perm=None`, remember to save `subtokenizer.perm` using `torch.save` to ensure reproducibility of the experiments.

In this code base, our pre-built permutation dictionary is automatically loaded by default: (Please refer to [train_prime_rl.py](/lit_gpt/pretrain/train_prime_rl.py#L521-L524) Lines 521-524.)
```python
fname = f"subtokenizer/perm/perm_{base**target_length}.pt"
if os.path.exists(fname):
    perm = torch.load(fname, map_location="cpu")
```
Our pre-built dictionary is save at [subtokenizer/perm](/lit_gpt/subtokenizer/perm):
```
subtokenizer /
    └── perm /
        └── perm_32768.pt
```

---

## License
This code implementation is developed based on the following repository.

- [ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM) (at commit `1df2e12`), licensed under the `Apache-2.0` license.
- [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama) (at commit `bf12224`), licensed under the `Apache-2.0` license.

Further changes based on the code in this folder are licensed under the `Apache-2.0` license.


---

## Citation

If you find this code implementation useful, please consider citing our papers.

```bib
@article{chao2026mdmprimev2,
      title = {{MDM-Prime-v2: Binary Encoding and Index Shuffling Enable Compute-optimal Scaling of Diffusion Language Models}}, 
      author = {Chen-Hao Chao, Wei-Fang Sun, Junwei Quan, Chun-Yi Lee, Rahul G. Krishnan},
      year = {2026},
}
@inproceedings{chao2025mdmprime,
      title = {{Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking}}, 
      author = {Chen-Hao Chao, Wei-Fang Sun, Hanwen Liang, Chun-Yi Lee, Rahul G. Krishnan},
      booktitle = {Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)},
      year = {2025},
}
```
