import polars as pl
import pandas as pd
import polars.selectors as cs
import numpy as np
import os
import json
import time
import random

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print
from typing import List

from module.utils import get_project_root, retry, timer
from module.bow import BagOfWordsEmbedding

def convert_np_arrays_to_lists(item):
    if isinstance(item, np.ndarray):
        return item.tolist()
    return item

def main():
    print("Initializing...")
    project_root = get_project_root()
    model_path = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data")
    chunk_path = os.path.join(data_path, "embed_chunks")
    pruned_path = os.path.join(data_path, 'pruned_embed')
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in os.listdir(data_path) if file.endswith('.parquet') and file not in non_data_list]
    data_list = [file for file in data_list if '_embed.parquet' not in file]

    with open(os.path.join(model_path, 'bow_token_map.json'), 'r') as f:
        token_map = json.load(f)
    
    inv_token_map = {str(v): k for k, v in token_map.items()}

    if 'pruned_token_map.json' not in os.listdir(model_path):
        print("Generating pruned token map...")
        with open(os.path.join(project_root, 'tf_idf.json'), 'r', encoding='utf-8') as f:
            tf_idf = json.load(f)
        score_threshhold = 30
        pruned_token_score = {k: v for k, v in tf_idf.items() if v > score_threshhold}
        dropped_token_score = {k: v for k, v in tf_idf.items() if v <= score_threshhold}
        pruned_tokens = pruned_token_score.keys()
        dropped_tokens = dropped_token_score.keys()
        print(f"Pruned tokens: {len(pruned_tokens)}")
        print(f"Dropped tokens: {len(dropped_tokens)}")
        pruned_tokens_id = [token_map[token] for token in pruned_tokens]
        dropped_tokens_id = [token_map[token] for token in dropped_tokens]
        print(type(pruned_tokens_id[0]))
        print(type(dropped_tokens_id[0]))
        pruned_token_map = {k:v for k, v in token_map.items() if v in pruned_tokens_id}
        dropped_token_map = {k:v for k, v in token_map.items() if v in dropped_tokens_id}
        print(f"Pruned tokens: {len(pruned_token_map)}")
        print(f"Dropped tokens: {len(dropped_token_map)}")

        with open(os.path.join(model_path, 'pruned_token_map.json'), 'w') as f:
            json.dump(pruned_token_map, f)
        
        with open(os.path.join(model_path, 'dropped_token_map.json'), 'w') as f:
            json.dump(dropped_token_map, f)
        
        del pruned_token_score, dropped_token_score, pruned_tokens, dropped_tokens, pruned_tokens_id, dropped_tokens_id, pruned_token_map, dropped_token_map
    
    print("Loading pruned token map...")
    with open(os.path.join(model_path, 'pruned_token_map.json'), 'r') as f:
        pruned_token_map = json.load(f)
    
    with open(os.path.join(model_path, 'dropped_token_map.json'), 'r') as f:
        dropped_token_map = json.load(f)

    for file in data_list:
        chunk_list = os.listdir(os.path.join(chunk_path, file))
        if not os.path.exists(os.path.join(pruned_path, file)):
            os.makedirs(os.path.join(pruned_path, file))
        for chunk in tqdm(chunk_list):
            if chunk in os.listdir(os.path.join(pruned_path, file)):
                continue
            # Read data
            data = pl.read_parquet(os.path.join(chunk_path, file, chunk))
            # print(data.columns)
            # Convert 'embedding' column to numpy array
            embeddings = np.array(data['embeddings'].to_numpy())
            embeddings = np.stack(embeddings)
            # Remove columns corresponding to dropped tokens
            pruned_embeddings = np.delete(embeddings, list(dropped_token_map.values()), axis=1)
            # Create a new DataFrame with pruned embeddings
            _data = pl.DataFrame({"id": data['id'], "embeddings": pruned_embeddings.tolist()})
            data = data.drop("embeddings")
            data = data.join(_data, on="id")
            # Save data
            data.write_parquet(os.path.join(pruned_path, file, chunk))
            del data, _data, embeddings, pruned_embeddings

if __name__ == "__main__":
    main()