import pandas as pd
import polars as pl
import numpy as np
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from module.utils import get_project_root, get_embedding

# Base variables/objects
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
encoder = tiktoken.encoding_for_model('gpt-4')

def process_row(row):
    """ Extract categories from the row. """
    return row[1]['categories']

def process_file(file_path):
    """ Read the Parquet file and process each row using multithreading, returning the categories. """
    df = pd.read_parquet(file_path)
    print(df.columns)
    print(df.head())

    categories = set()
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_row, df.iterrows())
        for result in results:
            categories.update(result)

    del df  # Clean up DataFrame from memory
    return categories

def get_unique_categories(data_list:list, write:bool=True) -> list:
    project_root = get_project_root()
    data_path = os.path.join(project_root, 'data')
    all_categories = set()
    for file in data_list:
        print(f"Processing {file}")
        file_path = os.path.join(data_path, file)
        file_categories = process_file(file_path)
        all_categories.update(file_categories)

    # Write the unique categories to a JSON file
    if write:
        with open(os.path.join(data_path, 'category_list.json'), 'w') as f:
            json.dump(list(all_categories), f)
    return list(all_categories)

def count_category_occurrence(data_list: list, category_list: list, write: bool = True) -> pl.DataFrame:
    project_root = get_project_root()
    data_path = os.path.join(project_root, 'data')
    category_count = {category: 0 for category in category_list}

    # Process each file sequentially
    for file in tqdm(data_list, desc="Processing Files"):
        print(f"Processing {file}")
        file_path = os.path.join(data_path, file)
        lf = pl.read_parquet(file_path)

        print("Start processing")
        # First, concatenate all categories from the 3rd column (0-indexed) into a single list
        categories = lf.get_column('categories').to_list()
        categories = [item for sublist in categories for item in sublist]  # Flatten the list of lists
        print(len(categories))

        print("Start counting")
        # Use ThreadPoolExecutor to count occurrences of each category
        def count_category(category):
            #print(category)
            return category, categories.count(category)
        
        print("Start mapping")
        with ThreadPoolExecutor(max_workers=64) as executor:
            # Wrap the executor.map call with tqdm for progress tracking
            results = list(tqdm(executor.map(count_category, category_list),
                                total=len(category_list),
                                desc="Counting Categories"))

            for category, count in results:
                category_count[category] += count

        del lf  # Clean up LazyFrame from memory

    # Save the category counts to a JSON file
    if write:
        with open(os.path.join(data_path, 'final_category_counts.json'), 'w') as f:
            json.dump(category_count, f)

    # Return results as a DataFrame
    return pl.DataFrame([category_count])

def process_category(category, category_embeddings):
    # Calculate the embedding for the category and similarity scores for interest categories
    embedding = get_embedding(category, client=client, encoder=encoder)
    scores = {
        cat: np.dot(embedding, category_embeddings[cat])
        for cat in category_embeddings
    }
    return category, embedding, scores

def main():
    base_path = os.path.join(os.getcwd(), 'data')
    data_list = os.listdir(base_path)
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in data_list if file.endswith('.parquet') and file not in non_data_list]

    category_in_interest = ['history', 'science', 'art', 'literature']
    with open(os.path.join(os.getcwd(), 'data', 'category_list.json'), 'r') as f:
        category_list = json.load(f)
    
    print(type(category_list))
    print(len(category_list))
    
    base_path = os.path.join(os.getcwd(), 'data')
    with open(os.path.join(base_path, 'category_list.json'), 'r') as f:
        category_list = json.load(f)

    category_frequency = count_category_occurrence(data_list, category_list)
    print(category_frequency)

    return 0

    category_in_interest = ['history', 'science', 'art', 'literature']
    category_embeddings = {
        'history': get_embedding('history'),
        'science': get_embedding('science'),
        'art': get_embedding('art'),
        'literature': get_embedding('literature')
    }

    # Dictionary to hold all the data
    category_list_embed = {
        'category': [],
        'embedding': [],
        'history': [],
        'science': [],
        'art': [],
        'literature': []
    }

    # Use ThreadPoolExecutor to process categories in parallel
    with ThreadPoolExecutor() as executor:
        # Submit all tasks to the executor
        future_to_category = {executor.submit(process_category, cat, category_embeddings): cat for cat in category_list}
        # Progress bar setup
        progress = tqdm(as_completed(future_to_category), total=len(category_list), desc="Processing Categories", file=sys.stdout)
        
        for future in progress:
            category, embedding, scores = future.result()
            category_list_embed['category'].append(category)
            category_list_embed['embedding'].append(embedding)
            for interest in category_in_interest:
                category_list_embed[interest].append(scores[interest])

    # Optional: Save or handle the category_list_embed dictionary as needed
    with open(os.path.join(base_path, 'category_list_embed.json'), 'w') as f:
        json.dump(category_list_embed, f)
    return 0
    # category_count = {category:0 for category in category_list}
    
    
    # # Process each file sequentially
    # for file in data_list:
    #     print(f"Processing {file}")
    #     file_path = os.path.join(base_path, file)
    #     df = pd.read_parquet(file_path)
        
    #     # Count occurrences for each category in the file
    #     for index, row in df.iterrows():
    #         categories_in_row = row['categories'].tolist()  # This is a list of categories
    #         for category in category_list:
    #             category_count[category] += categories_in_row.count(category)
        
    #     del df  # Clean up DataFrame from memory


    # # Save the category counts to a JSON file
    # with open(os.path.join(base_path, 'final_category_counts.json'), 'w') as f:
    #     json.dump(category_count, f)

    # return 0

if __name__ == '__main__':
    main()
