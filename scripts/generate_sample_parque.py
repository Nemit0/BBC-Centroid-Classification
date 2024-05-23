import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from module.utils import get_project_root

def sample_data(data_list:list, n:int=100, output_path:str='sample.parquet'):
    # print(index_df)

    # Sample 100 random rows from data_list
    sample_df = pd.DataFrame()
    for file in tqdm(data_list):
        df = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'data', file))
        sample_df = pd.concat([sample_df, df.sample(n=100, random_state=42)], axis=0)
        del df  # Remove the sampled data file from memory

    # Save the sampled data as sample.parquet in the same directory
    sample_df.to_parquet(os.path.join(os.path.dirname(__file__), 'data', output_path))
    print(f'Sampled data saved as {output_path}')
    return sample_df

def main():
    project_root = get_project_root()
    data_list = [file for file in os.listdir(os.path.join(project_root, 'data')) if file.endswith('.parquet')]
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in data_list if file not in non_data_list]
    index_df = pd.read_parquet(os.path.join(project_root, 'data', 'wiki_2023_index.parquet'))

    # Sample 100 random rows from data_list
    print(os.listdir(os.path.join(project_root, 'data')))
    if 'sample.parquet' in os.listdir(os.path.join(project_root, 'data')):
        print('Sampled data already exists')
        sample_df = pd.read_parquet(os.path.join(project_root, 'data', 'sample.parquet'))
    else:
        sample_df = sample_data(data_list, n=100, output_path=os.path.join(project_root, 'data', 'sample.parquet'))
    
    # Randomize the sample data in random order
    sample_df = sample_df.sample(frac=1, random_state=42)
    print(sample_df.head())
    print(sample_df.shape)

    return 0

if __name__ == '__main__':
    main()
