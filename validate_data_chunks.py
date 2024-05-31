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

@retry(5)
def main():
    print("Initializing...")
    project_root = get_project_root()
    model_path = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data")
    chunk_path = os.path.join(data_path, "data_chunks")
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in os.listdir(data_path) if file.endswith('.parquet') and file not in non_data_list]

    data_chunk_list = []
    for file in data_list:
        file_path = os.path.join(chunk_path, file)
        data_chunk_list.extend([os.path.join(file_path, _file) for _file in os.listdir(file_path)])
    
    data_chunk_list = [chunk for chunk in data_chunk_list if '_embed.parquet' in chunk]
    removed_file_list = []
    for chunk in tqdm(data_chunk_list):
        try:
            df = pl.read_parquet(chunk)
            del df
        except Exception as e:
            print(f"Failed to scan file {chunk}")
            removed_file_list.append(chunk)
            os.remove(chunk)
        
    print(f"Removed file list: {removed_file_list}")

    with open(os.path.join(project_root, 'removed_files.json'), 'w') as f:
        json.dump(removed_file_list, f, indent=4)


    print(data_chunk_list)
            
if __name__ == "__main__":
    main()
