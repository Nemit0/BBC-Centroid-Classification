import polars as pl
import numpy as np
import os
import sys
import json
import tiktoken

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from module.utils import get_embedding, get_project_root

def main():
    load_dotenv()
    encoder = tiktoken.encoding_for_model('gpt-4')
    openai = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
    print('initialized')

    data_path = os.path.join(get_project_root(), "data")
    chunk_path = os.path.join(data_path, "chunk")
    log_path = os.path.join(chunk_path, "log")
    with open(os.path.join(data_path, "category_list.json"), "r") as f:
        categories = json.load(f)
    print('category loaded')
    # _category_dict = {
    #     'categories': categories,
    #     'embeddings': [None] * len(categories)
    # }

    def apply_category_embedding(category):
        embedding = get_embedding(category, openai, encoder)
        return (category, embedding)

    print('start embedding')

    # First divide the categories into n evenly sized chunks
    n = 1000
    category_size = len(categories)
    chunk_size = int(np.floor(category_size / n) + 1)
    chunk_num = int(np.floor(category_size // chunk_size)+1)
    category_chunks = [categories[i:i+chunk_size] for i in range(0, category_size-chunk_size, chunk_size)]
    category_chunks.append(categories[chunk_size*(chunk_num-1):])
    # Ensure that the chunk is perfectly divided without any loss
    assert sum([len(chunk) for chunk in category_chunks]) == len(categories)
    
    for i, chunk in enumerate(category_chunks):
        with open(os.path.join(chunk_path, f"chunk_{i}_categories.json"), "w") as f:
            json.dump(chunk, f)
    
    chunk_list = [os.path.join(chunk_path, f"chunk_{i}_categories.json") for i in range(chunk_num)]
    chunk_list = [_chunk for _chunk in chunk_list if not os.path.exists(f"{_chunk}_embeddings.json")]
    # Process per chunks
    for i, chunk_filename in enumerate(tqdm(chunk_list)):
        with open(chunk_filename, "r") as f:
            chunk = json.load(f)
        # strip chunk_filename to get only filename
        _chunk_filename = os.path.basename(chunk_filename)
        _category_dict = {
                    'categories': chunk,
                    'embeddings': [None] * len(chunk)
        }
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(apply_category_embedding, category) for category in chunk]
            for future in as_completed(futures):
                category, embedding = future.result()
                chunk_index = chunk.index(category)
                _category_dict['embeddings'][chunk_index] = embedding
        with open(os.path.join(chunk_path, f"{_chunk_filename}_embeddings.json"), "w") as f:
            json.dump(_category_dict, f)
        del _category_dict

    print('embedding done')
    
    category_df = pl.DataFrame(_category_dict)
    # save to parquet
    category_df.write_parquet(os.path.join(data_path, "category_embeddings.parquet"))
    
    return 0

if __name__ == "__main__":
    main()