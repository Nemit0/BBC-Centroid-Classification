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

@retry(5)
def main():
    print("Initializing...")
    project_root = get_project_root()
    model_path = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data")
    chunk_path = os.path.join(data_path, "data_chunks")
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in os.listdir(data_path) if file.endswith('.parquet') and file not in non_data_list]

    print("Splitting data into chunks...")
    chunk_rows = 5000
    
    if not os.listdir(chunk_path):
        for file in tqdm(data_list):
            file_path = os.path.join(data_path, file)
            df = pl.read_parquet(file_path)
            total_rows = df.height
            chunk_num = int(np.floor(total_rows / chunk_rows)) + 1
            # Create directory for chunks if not exists
            file_chunk_dir = os.path.join(chunk_path, file)
            if not os.path.exists(file_chunk_dir):
                os.makedirs(file_chunk_dir)

            # Split data into chunks
            for i in tqdm(range(chunk_num)):
                start_row = i * chunk_rows
                end_row = min(start_row + chunk_rows, total_rows)
                df_chunk = df.slice(start_row, end_row - start_row)
                chunk_filename = os.path.join(file_chunk_dir, f"chunk_{i+1}.parquet")
                df_chunk.write_parquet(chunk_filename)

                print(f"Chunk {i+1} of {file} saved.")
            # Split data and save each chunk

    print("Generating BOW Encoding")
    # Convert data_list to append the path
    data_list = [os.path.join(data_path, file) for file in data_list]
    if "bow_token_map.json" not in os.listdir(model_path):
        print("Training BOW Encoder...")
        bow = BagOfWordsEmbedding(data_list)
        bow.train(max_workers=1, k=20)
        bow.save_words(os.path.join(model_path, "bow_token_map.json"))
        del bow
        
    bow = BagOfWordsEmbedding(data_list, unique_word_data=os.path.join(model_path, "bow_token_map.json"))
    print("BOW Encoder ready.")

    # Get category list
    with open(os.path.join(data_path, 'filtered_category_list.json'), 'r') as f:
        category_list = json.load(f)
    
    for file in tqdm(data_list):
        print(f"Processing {file}...")
        file_name = file.split('/')[-1]
        file_chunk_dir = os.path.join(chunk_path, file_name)
        chunk_list = [os.path.join(file_chunk_dir, chunk) for chunk in os.listdir(file_chunk_dir) if chunk.endswith('.parquet') and not chunk.endswith('embed.parquet')]
        embedded_list = [chunk for chunk in os.listdir(file_chunk_dir) if chunk.endswith('embed.parquet')]
        chunk_list = [chunk for chunk in chunk_list if os.path.basename(chunk).replace('.parquet', '_embed.parquet') not in embedded_list]
        
        def process_chunk(chunk_path):
            chunk_df = pl.read_parquet(chunk_path)
            print(f"Processing {os.path.basename(chunk_path)}...")
            # Filter rows where the 'categories' column contains any category in category_list
            chunk_df = chunk_df.filter(chunk_df["categories"].apply(lambda cats: any(cat in category_list for cat in cats)))
            print(f"Filtered {os.path.basename(chunk_path)}...")
            # Apply the embedding to the 'text' column
            if not chunk_df.is_empty():
                chunk_df = bow.embed_dataframe(chunk_df)

                # Drop the 'text' column after embedding
                chunk_df = chunk_df.drop('text')
                # Save the modified DataFrame back to a Parquet file
                chunk_df.write_parquet(os.path.join(os.path.dirname(chunk_path), f"{os.path.basename(chunk_path).replace('.parquet', '_embed.parquet')}"))
                print(f"Processed and saved embedded data for {os.path.basename(chunk_path)}")
            else:
                print(f"No data to process for {os.path.basename(chunk_path)}")
                # Write as empty file
                chunk_df.write_parquet(os.path.join(os.path.dirname(chunk_path), f"{os.path.basename(chunk_path).replace('.parquet', '_embed.parquet')}"))
            
            del chunk_df
            return 0
        
        chunk_chunk_size = 10
        chunk_chunk_list = [chunk_list[i:i+chunk_chunk_size] for i in range(0, len(chunk_list)-1, chunk_chunk_size)]
        chunk_chunk_list.append(chunk_list[(len(chunk_list) // chunk_chunk_size) * chunk_chunk_size:])
        for chunk_chunk in tqdm(chunk_chunk_list):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(process_chunk, chunk) for chunk in chunk_chunk]
                for future in as_completed(futures):
                    future.result()

    print("All chunks saved.")
    print("All processes completed.")
    return 0
            
if __name__ == "__main__":
    main()
