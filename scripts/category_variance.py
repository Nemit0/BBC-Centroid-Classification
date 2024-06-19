import os
import json
import time

import numpy as np
import polars as pl

from collections import Counter
from typing import Iterable
from operator import add
from tqdm import tqdm

from module.utils import get_project_root

def distance(a:Iterable, b:Iterable) -> float:
    return np.linalg.norm(np.array(a)-np.array(b))

def main():
    project_root = get_project_root()
    data_path = os.path.join(project_root, 'data')
    category_count_list = [file for file in os.listdir(data_path) if file.endswith("_category_counts.json")]
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in os.listdir(data_path) if file.endswith('.parquet') and file not in non_data_list]
    data_list = [file for file in data_list if '_embed.parquet' not in file]
    category_path = os.path.join(data_path, 'category_grouped')

    if 'category_count_sorted' in os.listdir(data_path):
        with open(os.path.join(data_path, 'category_counts_new.json'), 'r', encoding='utf-8') as f:
            category = json.load(f)
        
        # Sort by key descending
        category = dict(sorted(category.items(), key=lambda x: x[1], reverse=False))
        
        print(list(Counter(category))[:100])

        with open(os.path.join(data_path, 'category_count_sorted.json'), 'w') as f:
            json.dump(category, f, indent=4)
        
        del category

    reduced_path = os.path.join(data_path, 'cur_applied')
    train_test_ratio = 0.85
    
    with open(os.path.join(category_path, 'category_info_centroid.json'), 'r') as f:
        category_info = json.load(f)
    
    print(len(category_info))

    # Calculate the centroid
    for file in tqdm(data_list):
        chunk_list = os.listdir(os.path.join(reduced_path, file))
        for chunk in tqdm(chunk_list):
            df = pl.read_parquet(os.path.join(reduced_path, file, chunk))
            df = df.slice(0, int(df.height*train_test_ratio))
            for i in range(df.height):
                _doc_category = df['categories'][i]
                for _category in _doc_category:
                    if _category in category_info.keys():
                        category_info[_category]['id'].append(df['id'][i])
                        if 'distance' not in category_info[_category].keys():
                            category_info[_category]['distance'] = []
                        category_info[_category]['distance'].append(distance(df['embeddings'][i].to_list(), category_info[_category]['centroid']))
            del df
    
    
    # Calculate the centroid
    for _category in category_info.keys():
        if len(category_info[_category]['id']) == 0:
            print(f'Category {_category} has no document')
        category_info[_category]['variance'] = np.var(category_info[_category]['distance'])
    
    # Remvove distance key and value for each category
    for _category in category_info.keys():
        del category_info[_category]['distance']
    
    # Save the category info
    with open(os.path.join(category_path, 'category_info_centroid_variance.json'), 'w') as f:
        json.dump(category_info, f, indent=4)
    
if __name__ == "__main__":
    main()