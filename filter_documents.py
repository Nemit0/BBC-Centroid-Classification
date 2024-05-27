import polars as pl
import polars.selectors as cs
import numpy as np
import os
import json
import sys

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from rich import print

from module.utils import get_embedding, get_project_root, cosine_similarity

def main():
    print("Initializing...")
    project_root = get_project_root()
    data_path = os.path.join(project_root, "data")
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = os.listdir(data_path)
    data_list = [file for file in data_list if file.endswith('.parquet') and file not in non_data_list]

    with open(os.path.join(data_path, "filtered_category_list.json"), 'r') as f:
        filtered_category_list = json.load(f)
    
    print(f"Found {len(data_list)} files.")
    def filter_dataframe(df: pl.DataFrame | pl.LazyFrame, categories: list) -> pl.LazyFrame:
        if isinstance(df, pl.DataFrame):
            df = df.lazy()
        # Filter rows where the column 'categories' contains any of the categories
        category_filter = reduce(lambda acc, cat: acc | pl.col('categories').arr.contains(cat), categories, pl.lit(False))
        query = df.filter(category_filter)
        return query
    
    print(f"Filtering documents with categories: {filtered_category_list[:10]} and {len(filtered_category_list) - 10} more.")
    # filter documents in each file and save as parquet in different name
    for file in tqdm(data_list):
        df = pl.scan_parquet(os.path.join(data_path, file))
        filtered_df = filter_dataframe(df, filtered_category_list)
        filtered_df.sink_parquet(os.path.join(data_path, f"filtered_{file}"))
        del df
    
    return 0

if __name__ == "__main__":
    main()