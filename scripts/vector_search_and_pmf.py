import os
import json
import time

import numpy as np
import polars as pl

from collections import Counter
from typing import Iterable, List
from operator import add
from tqdm import tqdm

from module.utils import get_project_root
from module.hnsw import HNSWGraph

def gaussian(distance, sigma):
    """
    Calculate the pmf for each document with respect to each cluster using Gaussian distribution
    """
    value = np.exp(-0.5 * (distance / (sigma+0.0001)) ** 2)
    pmf = value / (np.sum(value)+0.0001)
    return pmf

def distance(a:Iterable, b:Iterable) -> float:
    return np.linalg.norm(np.array(a)-np.array(b))

def loss(a:List[str], b:List[str], l1:int=1, l2:int=1) -> float:
    J = (l1*len(set(a)-set(b))) + (l2*len(set(b)-set(a)))/len(set(a).union(set(b)))
    return J

def main():
    project_root = get_project_root()
    data_path = os.path.join(project_root, 'data')
    category_count_list = [file for file in os.listdir(data_path) if file.endswith("_category_counts.json")]
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in os.listdir(data_path) if file.endswith('.parquet') and file not in non_data_list]
    data_list = [file for file in data_list if '_embed.parquet' not in file]
    category_path = os.path.join(data_path, 'category_grouped')
    reduced_path = os.path.join(data_path, 'cur_applied')

    with open(os.path.join(category_path, 'category_info_centroid_variance.json'), 'r') as f:
        category_info = json.load(f)
    
    for category in category_info.keys():
        category_info[category]['threshold'] = 0.0
    
    train_test_ratio = 0.85
    learning_step = 0.05
    iteration = 15

    for file in data_list:
        chunk_list = os.listdir(os.path.join(reduced_path, file))
        for chunk in tqdm(chunk_list):
            df = pl.read_parquet(os.path.join(reduced_path, file, chunk))
            df = df.slice(0, int(df.height*train_test_ratio))
            for i in range(df.height):
                _doc_category = df['categories'][i]
                _temp = {k:distance(df['embeddings'][i], category_info[k]['centroid']) for k in category_info.keys()}
                _temp = dict(sorted(_temp.items(), key=lambda x: x[1]))
                # Get top 10, keeping dict structure
                _temp = dict(list(_temp.items())[:500])
                print(_temp.keys())
                print(_doc_category)
                print(f"Difference:{set(_doc_category)-set(_temp.keys())}")
                print(f"Intersection:{set(_doc_category).intersection(set(_temp.keys()))}, that is {len(set(_doc_category).intersection(set(_temp.keys())))} out of {len(_doc_category)}")

                for i in range(iteration):
                    for _category in _temp.keys():
                        if i == 0 and category_info[_category]['threshold'] == 0.0:
                            J_prev = loss(_doc_category, _temp.keys())
                            category_info[_category]['threshold'] += learning_step
                        predicted_categories = [k for k in _temp.keys() if gaussian(distance(df['embeddings'][i], category_info[k]['centroid']), category_info[k]['variance']) > category_info[k]['threshold']]
                        J = loss(_doc_category, predicted_categories)

                        if J < J_prev:
                            if category_info[_category]['threshold'] <= 0.7:
                                category_info[_category]['threshold'] += learning_step
                        J_prev = J
                        # print(f'J: {J}')
            del df
    
    with open(os.path.join(category_path, 'category_info_centroid_variance_threshold.json'), 'w') as f:
        json.dump(category_info, f, indent=4)
if __name__ == '__main__':
    main()