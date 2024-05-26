import polars as pl
import numpy as np
import os
import json
import tiktoken

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from module.utils import get_embedding, get_project_root, cosine_similarity

def main():
    print("Initializing...")
    load_dotenv()
    encoder = tiktoken.encoding_for_model('gpt-4')
    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    project_root = get_project_root()
    data_path = os.path.join(project_root, "data")
    chunk_path = os.path.join(data_path, "chunk")
    category_in_interest = ['history', 'science', 'art', 'literature']
    interest_embedding = {_category: np.array(get_embedding(_category, openai, encoder)) for _category in category_in_interest}

    print("Loading categories...")
    chunk_list = [file for file in os.listdir(chunk_path) if file.endswith("_embeddings.json")]
    print(f"Found {len(chunk_list)} chunks.")
    # category_embeddings = {'categories': [], 'embeddings': []}

    print("Loading and processing embeddings...")
    for chunk in tqdm(chunk_list):
        with open(os.path.join(chunk_path, chunk), "r") as f:
            embeddings_data = json.load(f)
            embeddings_data['embeddings'] = [np.array(embedding) for embedding in embeddings_data['embeddings']]
            for _category in category_in_interest:
                embeddings_data[_category] = [cosine_similarity(embedding, interest_embedding[_category]) for embedding in embeddings_data['embeddings']]
            
            embeddings_data['embeddings'] = [embedding.tolist() for embedding in embeddings_data['embeddings']]
            
            with open(os.path.join(chunk_path, f"{chunk}_with_similarities.json"), "w") as f:
                json.dump(embeddings_data, f)
            
            del embeddings_data
    
    print("Done.")

    return
    
if __name__ == "__main__":
    main()