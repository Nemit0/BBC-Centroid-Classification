import os
import json
import time

import numpy as np
import polars as pl
import tiktoken

from polars import col
from collections import Counter
from typing import Iterable, List
from operator import add
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from module.utils import get_project_root, get_embedding, cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed

def euclidean_distance(a:Iterable, b:Iterable) -> float:
    return np.linalg.norm(np.array(a)-np.array(b))

def normalze_vector(vector:List[float]) -> List[float]:
    norm = np.linalg.norm(vector)
    return [i/norm for i in vector]

def loss(a:List[str], b:List[str], l1:int=1, l2:int=1) -> float:
    J = (l1*len(set(a).difference(set(b)))) + (l2*len(set(b).difference(set(a))))/len(set(a).union(set(b)))
    return J

def main():
    project_root_path = get_project_root()
    data_path = os.path.join(project_root_path, 'data')
    working_path = os.path.join(project_root_path, 'data', 'z_analysis')
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    encoder = tiktoken.encoding_for_model('gpt-4')
    batch_path = os.path.join(working_path, 'batch')
    print(batch_path)

    with open(os.path.join(data_path, 'filtered_category_list.json'), 'r') as f:
        category_list = json.load(f)
    
    print(len(category_list))
    print(type(category_list))

    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
        # Load dataset
        df = pl.read_parquet(os.path.join(data_path, 'o.parquet'))
        print(df.columns)
        print(df.height)

        # First filter
        filter_mask = df['categories'].apply(lambda x: any(item in category_list for item in x))
        df = df.filter(filter_mask)
        print(df.height)
        batch_size = 1500
        
        for i in tqdm(range(0, df.height, batch_size)):
            print(os.path.join(batch_path, f'batch_{i}.parquet'))
            df.slice(i, i+batch_size).write_parquet(os.path.join(batch_path, f'batch_{i}.parquet'))
        # Write data
        df.write_parquet(os.path.join(working_path, 'o_filtered.parquet'))
        del df

    batch_list = [os.path.join(batch_path, file) for file in os.listdir(batch_path) if file.endswith('.parquet')]
    def process_row(row:pl.Series):
        return row[0], get_embedding(row[2], client, encoder)
    if not os.listdir(os.path.join(working_path, 'embed_batch')):
        for batch in batch_list:
            df = pl.read_parquet(batch)
            # Initialize empty column
            df = df.with_columns(pl.Series('embeddings', [[] for _ in range(df.height)], dtype=pl.List))
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_row, row) for row in df.iter_rows()]
                for future in tqdm(as_completed(futures)):
                    result = future.result()
                    df = df.with_columns(
                        pl.when(df['id'] == result[0]).then(result[1])
                        .otherwise(df['embeddings'])
                        .alias('embeddings')
                    )
            df.drop_in_place('text')
            df.write_parquet(os.path.join(working_path, 'embed_batch', os.path.basename(batch)))

    # Get unique categories in the filtered dataset
    category_info = {k:{'centroid':[], 'variance':[], 'vector_sum':[], 'distance':[], 'count':0, 'threshold':0.5, 'loss':0} for k in category_list}
    train_test_ratio = 0.85
    # print(category_info)
    batch_list = [os.path.join(working_path, 'embed_batch', file) for file in os.listdir(os.path.join(working_path, 'embed_batch')) if file.endswith('.parquet')]
    # First calculate centroid
    for batch in batch_list:
        df = pl.read_parquet(batch)
        df = df.slice(0, int(df.height*train_test_ratio))
        for row in tqdm(df.iter_rows()):
            for category in row[2]:
                if category in category_info.keys():
                    category_info[category]['vector_sum'] = list(map(add, category_info[category]['vector_sum'], row[3]))
                    category_info[category]['count'] += 1
                else:
                    category_info[category] = {
                        'centroid':[],
                        'variance':[],
                        'vector_sum':row[3],
                        'distance':[],
                        'count':0,
                        'threshold':0.5
                    }
        del df
    
    print("Calculating centroid")
    category_info = {k:v for k,v in category_info.items() if v['count'] > 5 and len(v['vector_sum']) > 0}
    for category in category_info.keys():
        category_info[category]['centroid'] = normalze_vector([v/(len(category_info[category]['vector_sum'])+1) for v in category_info[category]['vector_sum']])
    
    # Calculate varince
    for batch in batch_list:
        df = pl.read_parquet(batch)
        df = df.slice(0, int(df.height*train_test_ratio))
        for row in tqdm(df.iter_rows()):
            # print(row)
            for category in row[2]:
                if category in category_info.keys():
                    category_info[category]['distance'].append(euclidean_distance(category_info[category]['centroid'], row[3]))
                else:
                    pass
        del df
    
    print("calculating variance")
    for category in category_info.keys():
        category_info[category]['variance'] = np.var(category_info[category]['distance'])

    # print(category_info)
    print(f"Total categories:{len(category_info)}")
    with open(os.path.join(working_path, 'category_info_centroid_variance.json'), 'w') as f:
        json.dump(category_info, f)
    
    # Model Evaluation
    batch_list = [os.path.join(working_path, 'embed_batch', file) for file in os.listdir(os.path.join(working_path, 'embed_batch')) if file.endswith('.parquet')]
    for batch in batch_list:
        df = pl.read_parquet(batch)
        df = df.slice(int(df.height*train_test_ratio), df.height)
        for row in tqdm(df.iter_rows()):
            _doc_category = row[2]
            _temp = {k:euclidean_distance(row[3], category_info[k]['centroid']) for k in category_info.keys()}
            _temp = dict(sorted(_temp.items(), key=lambda x: x[1]))
            # Get top 10, keeping dict structure
            _temp = dict(list(_temp.items())[:1000])
            print(_temp.keys())
            print(_doc_category)
            print(f"Difference:{set(_doc_category)-set(_temp.keys())}")
            print(f"Intersection:{set(_doc_category).intersection(set(_temp.keys()))}, that is {len(set(_doc_category).intersection(set(_temp.keys())))} out of {len(_doc_category)}")
            for _category in _temp.keys():
                predicted_categories = [k for k in _temp.keys() if euclidean_distance(row[3], category_info[k]['centroid']) > category_info[k]['threshold']]
                J = loss(_doc_category, predicted_categories)
                category_info[_category]['loss'] = J
        del df
    
    with open(os.path.join(working_path, 'category_info_centroid_variance.json'), 'w') as f:
        json.dump(category_info, f)
    

if __name__ == '__main__':
    main()