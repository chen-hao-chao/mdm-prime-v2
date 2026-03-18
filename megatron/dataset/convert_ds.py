import os, json, gzip, glob
from pathlib import Path

IN_DIR  = "download/c4_en_snapshot/en"
OUT_ROOT = "download/c4_jsonl"
Path(OUT_ROOT, "train").mkdir(parents=True, exist_ok=True)
Path(OUT_ROOT, "validation").mkdir(parents=True, exist_ok=True)

def convert(split):
    files = sorted(glob.glob(os.path.join(IN_DIR, f"c4-{split}.*.json.gz")))
    for i, path in enumerate(files):
        out_path = os.path.join(OUT_ROOT, split, f"c4_en_{split}_{i:05d}.jsonl")
        n = 0
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                try:
                    rec = json.loads(line)
                    txt = (rec.get("text") or "").replace("\n", " ").strip()
                    if txt:
                        fout.write(json.dumps({"text": txt}) + "\n")
                        n += 1
                except Exception:
                    continue
        print(f"[{split}] {os.path.basename(path)} -> {out_path} ({n} lines)")

convert("train")
convert("validation")
print("JSONL written under:", OUT_ROOT)
