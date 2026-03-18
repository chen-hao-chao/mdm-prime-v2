from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from pathlib import Path

LOCAL_DIR_DATASET = "download/c4_en_snapshot"
LOCAL_DIR_TOKENIZER = "download/tokenizer/gpt2_tok"
Path(LOCAL_DIR_DATASET).mkdir(parents=True, exist_ok=True)
Path(LOCAL_DIR_TOKENIZER).mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="allenai/c4",
    repo_type="dataset",
    allow_patterns=[
        "en/c4-train.*.json.gz",
        "en/c4-validation.*.json.gz",
        "en/LICENSE*",
        "en/README*",
    ],
    local_dir=LOCAL_DIR_DATASET,
    local_dir_use_symlinks=False,
    resume_download=True,
    force_download=True,
    revision="main",
)
print("Downloaded dataset to:", LOCAL_DIR_DATASET)

tok = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
tok.save_pretrained(LOCAL_DIR_TOKENIZER)
print("Downloaded tokenizer to:", LOCAL_DIR_TOKENIZER)

