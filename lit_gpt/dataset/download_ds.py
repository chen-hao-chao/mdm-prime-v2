import time
from huggingface_hub import snapshot_download, login
from pathlib import Path

path_to_dataset = "/path_to_dataset"
REPO_ID = "cerebras/SlimPajama-627B"
Path(path_to_dataset).mkdir(parents=True, exist_ok=True)
print(f"Starting polite download to {path_to_dataset}...")

# 2. DOWNLOAD VALIDATION & TEST SETS FIRST (Smaller)
print("Downloading Validation and Test sets...")
for i in [1,2,3,4,5]:
    chunk_pattern = f"validation/chunk{i}/*.jsonl.zst"
    print(f"--------------------------------------------------")
    print(f"Downloading Chunk {i}/5...")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=[chunk_pattern],
            local_dir=path_to_dataset,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4  # Keep this low to avoid 429 errors
        )
        print(f"Chunk val complete.")
    except Exception as e:
        print(f"Error: {e}")
        print("Waiting 2 minutes before retrying this chunk...")
        time.sleep(120)
        # Simple retry logic (optional)
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=[chunk_pattern],
            local_dir=path_to_dataset,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=2  # Keep this low to avoid 429 errors
        )
    # 4. SLEEP TO RESET QUOTA
    # The limit is 1000 reqs / 5 mins. 
    # A 60s sleep usually keeps you safe between large batches.
    print("Sleeping for 60s to respect rate limits...")
    time.sleep(60)

print("Validation/Test done. Sleeping for 30s...")
time.sleep(30)

# 3. LOOP THROUGH TRAINING CHUNKS 1-10
for i in [10]:
    chunk_pattern = f"train/chunk{i}/*.jsonl.zst"
    print(f"--------------------------------------------------")
    print(f"Downloading Chunk {i}/10...")
    
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=[chunk_pattern],
            local_dir=path_to_dataset,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )
        print(f"Chunk {i} complete.")
    except Exception as e:
        print(f"Error on Chunk {i}: {e}")
        print("Waiting 2 minutes before retrying this chunk...")
        time.sleep(120)
        # Simple retry logic (optional)
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=[chunk_pattern],
            local_dir=path_to_dataset,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=2
        )

    # 4. SLEEP TO RESET QUOTA
    # The limit is 1000 reqs / 5 mins. 
    # A 60s sleep usually keeps you safe between large batches.
    print("Sleeping for 60s to respect rate limits...")
    time.sleep(60)

print("All downloads complete!")