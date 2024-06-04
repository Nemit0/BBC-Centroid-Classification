import polars as pl
import polars.selectors as cs
import numpy as np
import os
import json
import random

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print

from module.utils import get_project_root

def main():
    print("Initializing...")
    random.seed(42)
    project_root = get_project_root()
    model_path = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data")
    chunk_path = os.path.join(data_path, "data_chunks")
    embed_chunk_path = os.path.join(data_path, "embed_chunks")
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in os.listdir(data_path) if file.endswith('.parquet') and file not in non_data_list and 'embed' not in file]
    chunk_list = []
    for file in data_list:
        _chunk_path = os.path.join(embed_chunk_path, file)
        _file_list = os.listdir(_chunk_path)
        _file_list = [os.path.join(_chunk_path, _file) for _file in _file_list]
        chunk_list.extend(_file_list)
        del _file_list
    
    print(f"Total number of data chunks: {len(chunk_list)}")
    print("Sampling file for token analysis")
    ratio = 0.1
    sample_size = int(len(chunk_list) * ratio)
    sample_list = random.sample(chunk_list, sample_size)

    print("Reading sample data and summing token frequency")
    with open(os.path.join(model_path, "bow_token_map.json"), 'r') as f:
        token_map = json.load(f)
    
    inv_token_map = {str(v): k for k, v in token_map.items()}
    token_frequency = {t:0 for t in inv_token_map.keys()}

    for file in tqdm(sample_list):
        df = pl.read_parquet(file)
        _embedding_sum = np.sum(np.stack(df['embeddings'].to_list()), axis=0)
        token_frequency = {k: token_frequency[k] + v for k, v in zip(inv_token_map.keys(), _embedding_sum)}
        del df, _embedding_sum
    
    print("Sorting token frequency")
    sorted_token_frequency = dict(sorted(token_frequency.items(), key=lambda item: item[1], reverse=True))
    sorted_token_frequency = {k: v for k, v in sorted_token_frequency.items() if v > 0}

    print("Writing token frequency to file")
    with open(os.path.join(model_path, "bow_token_frequency.json"), 'w') as f:
        json.dump(sorted_token_frequency, f)
    
    print("Converting token id in token frequency to token")
    token_frequency = {inv_token_map[k]: v for k, v in sorted_token_frequency.items()}
    with open(os.path.join(model_path, "bow_token_frequency_token.json"), 'w') as f:
        json.dump(token_frequency, f)
    
    print("Done")

if __name__ == "__main__":
    main()