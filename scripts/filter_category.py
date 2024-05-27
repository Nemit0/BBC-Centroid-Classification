import polars as pl
import polars.selectors as cs
import numpy as np
import os
import json

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from rich import print

from module.utils import get_embedding, get_project_root, cosine_similarity

def process_file(file_path, similarity_threshold, categories):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pl.DataFrame(data).lazy()
    filter_conditions = reduce(lambda acc, cat: acc | (pl.col(cat) > similarity_threshold), categories, pl.lit(False))
    query = df.filter(filter_conditions)
    # Trigger computation and return result
    return list(query.collect().select(cs.by_name('categories')))

def main():
    print("Initializing...")
    project_root = get_project_root()
    data_path = os.path.join(project_root, "data")
    chunk_path = os.path.join(data_path, "chunk")
    category_in_interest = ['history', 'science', 'art', 'literature']

    similarity_threshold = 0.215
    files_list = [file for file in os.listdir(chunk_path) if file.endswith("_with_similarities.json")]
    print(f"Found {len(files_list)} files.")
    with open(os.path.join(chunk_path, files_list[0]), 'r') as f:
        data = json.load(f)
    
    results = []
    for index in tqdm(range(0, 1000)): 
        file_name = f"chunk_{index}_categories.json_embeddings.json_with_similarities.json"
        file_path = os.path.join(chunk_path, file_name)
        result_list = list(process_file(file_path, similarity_threshold, category_in_interest))
        results.extend(list(result_list[0]))
    
    results = list(set(results))
    with open(os.path.join(data_path, "category_list.json"), 'r') as f:
        category_list = json.load(f)
    
    print(results[:5])
    filtered_percentage = 100 - len(results) / len(category_list) * 100
    
    print(f"Filtered {filtered_percentage:.2f}% of the data.")

    with open(os.path.join(data_path, "filtered_category_list.json"), 'w') as f:
        json.dump(results, f)
    
    return filtered_percentage

if __name__ == "__main__":
    main()