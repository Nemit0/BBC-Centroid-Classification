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
from module.hnsw import HNSWGraph

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

    with open(os.path.join(category_path, 'category_info_centroid_variance.json'), 'r') as f:
        category_info = json.load(f)
        max_element = len(category_info)
        del category_info
    
    hnsw = HNSWGraph(dim=1500, max_elements=max_element, M=50, ef=10, level=3, dict_path=os.path.join(category_path, 'category_info_centroid_variance.json'))

    hnsw.save_model(os.path.join(category_path, 'hnsw_model.json'))

    

if __name__ == '__main__':
    main()