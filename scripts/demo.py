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
from operator import call

from module.utils import get_project_root, retry
from module.bow import BagOfWordsEmbedding

#@retry(5)
def main():
    print("Initializing...")
    project_root = get_project_root()
    model_path = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data")
    chunk_path = os.path.join(data_path, "data_chunks")

    @dict
    @sorted
    @call
    def chunks():
        for path in os.listdir(chunk_path):
            embed_list = [file for file in os.listdir(os.path.join(chunk_path, path)) if file.endswith('_embed.parquet')]
            print(len(embed_list))
            chunk_list = [file for file in os.listdir(os.path.join(chunk_path, path)) if not file.endswith('_embed.parquet') and file.endswith('.parquet')]
            embed_list = [os.path.join(chunk_path, path, file) for file in embed_list]
            chunk_list = [os.path.join(chunk_path, path, file) for file in chunk_list]
            for i in range(len(embed_list)):
                yield embed_list[i], chunk_list[i]
    
    data_rows = 0
    embed_rows = 0
    sampled_keys = random.sample(list(chunks.keys()), k=max(1, int(len(chunks) * 0.1)))
    sampled_chunks = {key: chunks[key] for key in sampled_keys}
    chunks = sampled_chunks
    del sampled_keys, sampled_chunks

    def count_items(kfile, vfile):
        data = pl.read_parquet(kfile)
        embed = pl.read_parquet(vfile)
        _data_rows = data.height
        _embed_rows = embed.height
        del data, embed
        return _data_rows, _embed_rows
    
    for k, v in chunks.items():
        _data_rows, _embed_rows = count_items(k, v)
        data_rows += _data_rows
        embed_rows += _embed_rows
        print(f"Embed rows: {_data_rows}, Data rows: {_embed_rows}, Total Percent: {data_rows/embed_rows*100:.2f}%")
    
    # This output that approx 33% of the data is embedded

if __name__ == "__main__":
    main()
