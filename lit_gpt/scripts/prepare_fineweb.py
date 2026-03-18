import os
from datatrove.pipeline.readers import ParquetReader
from transformers import AutoTokenizer
import pickle

fineweb_list = [
        "CC-MAIN-2024-18",
        "CC-MAIN-2024-10",
    ]
tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', use_fast=True)

total_tokens = 100 * 1024 * 2048 # same as pretrain
os.makedirs('data/fineweb', exist_ok=True)

for fineweb in fineweb_list:
    print(f"Downloading {fineweb}")
    pickle_filename = f'data/fineweb/{fineweb}.pkl'
    data_reader = ParquetReader(f"hf://datasets/HuggingFaceFW/fineweb/data/{fineweb}", limit=500000)

    tokens = []
    for document in data_reader():
        token = tokenizer.encode(document.text, add_special_tokens=True)
        tokens += token

        print(f'Finished: {fineweb}, {len(tokens) / total_tokens}')
        if len(tokens) > total_tokens:
            with open(pickle_filename, 'wb') as f:
                pickle.dump(tokens[:total_tokens], f)
            break