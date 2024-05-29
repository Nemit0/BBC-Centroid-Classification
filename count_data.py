import polars as pl
import polars.selectors as cs
import numpy as np
import os
import json

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print

from module.utils import get_project_root, retry
from module.bow import BagOfWordsEmbedding

def main():
    print("Initializing...")
    project_root = get_project_root()
    model_path = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data")
    chunk_path = os.path.join(data_path, "data_chunks")
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in os.listdir(data_path) if file.endswith('.parquet') and file not in non_data_list]

    chunk_data = 0
    embeded_data = 0
    for file in os.listdir(chunk_path):
        _chunk_path = os.path.join(chunk_path, file)
        chunk_list = [chunk for chunk in os.listdir(_chunk_path) if chunk.endswith('.parquet') and f"embedding" not in chunk]
        embeddings_list = [chunk for chunk in os.listdir(_chunk_path) if chunk.endswith('.parquet') and f"embed" in chunk]
        chunk_data += len(chunk_list)
        embeded_data += len(embeddings_list)
    
    print(f"Total data chunks: {chunk_data}")
    print(f"Total embedded data: {embeded_data}")


if __name__ == "__main__":
    main()