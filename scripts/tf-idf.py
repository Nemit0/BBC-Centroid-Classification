import polars as pl
import pandas as pd
import polars.selectors as cs
import numpy as np
import os
import json
import time
import random

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print
from typing import List

from module.utils import get_project_root, retry, timer
from module.bow import BagOfWordsEmbedding

def convert_np_arrays_to_lists(item):
    if isinstance(item, np.ndarray):
        return item.tolist()
    return item

def main():
    print("Initializing...")
    project_root = get_project_root()
    model_path = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data")
    chunk_path = os.path.join(data_path, "embed_chunks")
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in os.listdir(data_path) if file.endswith('.parquet') and file not in non_data_list]
    data_list = [file for file in data_list if '_embed.parquet' not in file]

    with open(os.path.join(model_path, 'bow_token_map.json'), 'r') as f:
        token_map = json.load(f)

    inv_token_map = {str(v): k for k, v in token_map.items()}
    total_token_frequency = {t:0 for t in inv_token_map.keys()}
    token_frequency_per_doc = {t:[] for t in inv_token_map.keys()}
    total_documents = 0
    inverse_document_frequency = {t:0 for t in inv_token_map.keys()}
    chunk_list = []
    for file in data_list:
        _chunk_list = os.listdir(os.path.join(chunk_path, file))
        _chunk_list = [os.path.join(chunk_path, file, chunk) for chunk in _chunk_list]
        chunk_list.extend(_chunk_list)

    print(f"Total number of chunks: {len(chunk_list)}")
    # randomly sample
    sample_ratio = 0.1
    number_of_samples = int(len(chunk_list) * sample_ratio)  # Calculate the number of samples to take
    sampled_chunk_list = random.sample(chunk_list, number_of_samples)  # Randomly sample chunks
    print(f"Number of samples: {number_of_samples}")

    for file in tqdm(sampled_chunk_list):
        data = pl.read_parquet(file)
        total_documents += data.height
        _embedding_sum = np.sum(np.stack(data['embeddings'].to_list()), axis=0)
        for token in inv_token_map.keys():
            token_frequency_per_doc[token].append(_embedding_sum[int(token)])
            total_token_frequency[token] += _embedding_sum[int(token)]
            inverse_document_frequency[token] += 1 if _embedding_sum[int(token)] > 0 else 0
        del data, _embedding_sum
    
    print("Finished counting token frequency")
    print("Calculating tf, idf, and tf-idf")
    tf_score = {t:0 for t in inv_token_map.keys()}
    idf_score = {t:0 for t in inv_token_map.keys()}
    tf_idf_score = {t:0 for t in inv_token_map.keys()}

    for token in inv_token_map.keys():
        max_frequency = max(token_frequency_per_doc[token]) if token_frequency_per_doc[token] else 1
        sum_frequencies = sum([frequency / max_frequency for frequency in token_frequency_per_doc[token] if frequency > 0])
        tf_score[token] = 0.5 + 0.5 * sum_frequencies

        # Inverse Document Frequency (IDF) Calculation: Smoothing by adding 1
        doc_frequency = inverse_document_frequency[token] if token in inverse_document_frequency else 0
        idf_score[token] = np.log((total_documents + 1) / (doc_frequency + 1))

        # TF-IDF Calculation
        tf_idf_score[token] = tf_score[token] * idf_score[token]
    
    tf_idf_score = {inv_token_map[k]:v for k, v in tf_idf_score.items()}
    
    print("Writing tf, idf, and tf-idf to file")
    with open(os.path.join(model_path, 'tf.json'), 'w') as f:
        json.dump(tf_score, f)
    
    with open(os.path.join(model_path, 'idf.json'), 'w') as f:
        json.dump(idf_score, f)
    
    with open(os.path.join(model_path, 'tf_idf.json'), 'w') as f:
        json.dump(tf_idf_score, f)

if __name__ == "__main__":
    main()