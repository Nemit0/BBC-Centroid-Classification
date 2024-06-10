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
    category_info = {k:{'centroid':[], 'variance':[], 'vector_sum':[], 'distance':[]} for k in category_list}
    print(category_info)
    batch_list = [os.path.join(working_path, 'embed_batch', file) for file in os.listdir(os.path.join(working_path, 'embed_batch')) if file.endswith('.parquet')]
    for batch in batch_list:
        df = pl.read_parquet(batch)
        for row in tqdm(df.iter_rows()):
            for category in row[2]:
                if category in category_info.keys():
                    category_info[category]['vector_sum'] = list(map(add, category_info[category]['vector_sum'], row[3]))
                    category_info[category]['distance'] = cosine_similarity(category_info[category]['centroid'], row[3])
                else:
                    category_info[category] = {'centroid':[], 'variance':[], 'vector_sum':[], 'distance':[]}
                    category_info[category]['vector_sum'] = list(map(add, category_info[category]['vector_sum'], row[3]))
                    category_info[category]['distance'] = cosine_similarity(category_info[category]['centroid'], row[3])
        del df
    
    for category in category_info.keys():
        category_info[category]['centroid'] = [i/(len(category_info[category]['vector_sum'])+1) for i in category_info[category]['vector_sum']]
        category_info[category]['variance'] = np.var(category_info[category]['distance'])
    
    with open(os.path.join(working_path, 'category_info_centroid_variance.json'), 'w') as f:
        json.dump(category_info, f)

if __name__ == '__main__':
    main()