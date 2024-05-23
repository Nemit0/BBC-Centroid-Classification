import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm

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
    data_list = [file for file in os.listdir(os.path.join(os.path.dirname(__file__), 'data')) if file.endswith('.parquet') and len(file) == 9]
    index_df = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'data', 'wiki_2023_index.parquet'))

    # Sample 100 random rows from data_list
    print(os.listdir(os.path.join(os.path.dirname(__file__), 'data')))
    if 'sample.parquet' in os.listdir(os.path.join(os.path.dirname(__file__), 'data')):
        print('Sampled data already exists')
        sample_df = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'data', 'sample.parquet'))
    else:
        sample_df = sample_data(data_list, n=100, output_path='./data/sample.parquet')
    
    # Randomize the sample data in random order
    sample_df = sample_df.sample(frac=1, random_state=42)
    print(sample_df.head())
    print(sample_df.shape)

    return 0

if __name__ == '__main__':
    main()
