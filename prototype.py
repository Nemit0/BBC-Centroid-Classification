import pandas as pd
import numpy as np
import os
import tiktoken
import concurrent.futures
import warnings
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from module.utils import get_project_root, get_embedding

# Load environment variables
load_dotenv()

# Initialize OpenAI API
openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize the encoder for TikToken
encoder = tiktoken.encoding_for_model('gpt-4')

def generate_sample_embedding():

    # Get the project root directory
    project_root = get_project_root()

    # Load sample data
    sample_data = pd.read_parquet(os.path.join(project_root, 'data', 'sample.parquet'))

    # Sample DataFrame with an 'embedding' column initialized to hold lists
    df = pd.DataFrame({
        'id': [id for id in sample_data['id']],
        'title': [title for title in sample_data['title']],
        'text': [text for text in sample_data['text']],
        'categories': [categories for categories in sample_data['categories']],
        'embedding': [None] * 2600  # Initialize with None or np.nan
    })
    
    for i in tqdm(range(len(df))):
        # Correct way to assign a list to a specific cell in 'embedding' column
        df.at[i, 'embedding'] = get_embedding(df.iloc[i]['text'])

    # Save the updated DataFrame with embeddings
    df.to_parquet(os.path.join(project_root, 'data', 'sample_embedding.parquet'))

    return df

def main():
    project_root = get_project_root()
    sample_data = pd.read_parquet(os.path.join(project_root, 'data', 'sample_embedding.parquet'))
    print(sample_data.head())
    print(sample_data.columns)
    print(len(sample_data['embedding'][0]))
    del(variable)

if __name__ == '__main__':
    main()
