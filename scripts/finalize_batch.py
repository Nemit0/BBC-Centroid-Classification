import polars as pl
import polars.selectors as cs
import numpy as np
import os
import json

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print

from module.utils import get_project_root, retry
from module.bow import BagOfWordsEmbedding

def main():
    print("Initializing...")
    project_root = get_project_root()
    model_path = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data")
    chunk_path = os.path.join(data_path, "data_chunks")
    embed_chunk_path = os.path.join(data_path, "embed_chunks")
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in os.listdir(data_path) if file.endswith('.parquet') and file not in non_data_list and 'embed' not in file]


    print('combining data')
    for file in tqdm(data_list):
        if not os.path.exists(os.path.join(embed_chunk_path, file)):
            print("Creating directory for", file)
            os.mkdir(os.path.join(embed_chunk_path, file))
        chunk_list = [file for file in os.listdir(os.path.join(chunk_path, file)) if file.endswith('_embed.parquet')]
        # print(chunk_list) Debug
        # This should split and save per 1500 rows
        # If the chunk is less then 1500, load the next data and concatenate and process.
        chunk_index = 0
        print('Processing', file)
        try:
            for i, chunk in tqdm(enumerate(chunk_list)):
                if i == 0:
                    df:pl.DataFrame = pl.read_parquet(os.path.join(chunk_path, file, chunk))
                else:
                    _df = pl.read_parquet(os.path.join(chunk_path, file, chunk))
                    df = pl.concat([df, _df])
                    del _df
                
                while df.height >= 1500:
                    # get the first 1500 rows and save it
                    _df = df.slice(0, 1500)
                    df = df.tail(df.height - 1500)
                    _df.write_parquet(os.path.join(embed_chunk_path, file, f'{chunk_index}_embed.parquet'))
                    chunk_index += 1
                    del _df
                
                if i == len(chunk_list) - 1:
                    print("Last chunk")
                    # Save the remaining data
                    df.write_parquet(os.path.join(embed_chunk_path, file, f'{chunk_index}_embed.parquet'))
                    del df
                
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            # Delete all files in the embed chunk path file directory so that we can restart the file and avoid data corruption
            for file in os.listdir(os.path.join(embed_chunk_path, file)):
                os.remove(os.path.join(embed_chunk_path, file))
            break
        except Exception as e:
            print(e)
            break
    
    print('done')

if __name__ == "__main__":
    main()
    