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
    with open(os.path.join(data_path, "category_list.json"), "r") as f:
        categories = json.load(f)
    print('category loaded')
    _category_dict = {
        'categories': categories,
        'embeddings': [None] * len(categories)
    }
    # category_df = pl.DataFrame(_category_dict)

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

    # # Process per chunk using multiprocessing
    # with Pool(cpu_count()) as pool:
    #     results = pool.map(apply_category_embedding, categories)
    #     for category, embedding in results:
    #         idx = categories.index(category)
    #         _category_dict['embeddings'][idx] = embedding
    # print('embedding done')

    # Process per chunks
    for chunk in tqdm(category_chunks):
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(apply_category_embedding, category) for category in chunk]
            for future in as_completed(futures):
                category, embedding = future.result()
                # print(category, embedding)
                # idx = categories.index(category)
                # print(idx)
                #category_df[idx, 'embeddings'] = embedding
                _category_dict['embeddings'][categories.index(category)] = embedding
    print('embedding done')
    
    category_df = pl.DataFrame(_category_dict)
    # save to parquet
    category_df.write_parquet(os.path.join(data_path, "category_embeddings.parquet"))
    
    return 0

if __name__ == "__main__":
    main()