import polars as pl
import numpy as np
import os
import sys
import json
from module.utils import get_project_root
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    data_path = os.path.join(get_project_root(), 'data')
    # Load the category list
    with open(os.path.join(data_path, 'category_list.json'), 'r') as f:
        category_list = json.load(f)
    # create a dataaset list
    data_list = os.listdir(data_path)
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in data_list if file.endswith('.parquet') and file not in non_data_list]

    for file in tqdm(data_list):
        file_path = os.path.join(data_path, file)
        df = pl.read_parquet(file_path)
        _category_id_frame = {category:[] for category in category_list}
        def get_all_category_id(category):
            _category_id_frame[category].append(list(df.filter(df['categories'] == category)['id']))
        with ThreadPoolExecutor() as executor:
            future_to_category = {executor.submit(get_all_category_id, category): category for category in category_list}
            progress = tqdm(as_completed(future_to_category), total=len(category_list), desc="Processing Categories", file=sys.stdout)
            for future in progress:
                future.result()
        
        with open(os.path.join(data_path, f'{file}_category_id.json'), 'w') as f:
            json.dump(_category_id_frame, f)
        
        del df
        del _category_id_frame

if __name__ == '__main__':
    main()