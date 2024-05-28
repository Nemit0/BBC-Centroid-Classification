import os
import warnings
import numpy as np

from tiktoken import Encoding
from openai import OpenAI
from typing import Callable

def get_project_root() -> str:
    file_path = os.path.abspath(__file__)
    while os.path.basename(file_path) != "mlgroup1":
        file_path = os.path.dirname(file_path)
    return file_path

def get_embedding(text:str, client:OpenAI, encoder:Encoding, model="text-embedding-3-small") -> list:
    # Ensure the text is a single line string
    text = text.replace("\n", " ")

    # Encode the text and truncate if necessary
    encoded_text = encoder.encode(text)
    if len(encoded_text) > 8000:
        warnings.warn('Text is too long, truncating to 8000 tokens.')
        text = encoder.decode(encoded_text[:8000])

    # Request embedding from OpenAI API
    response = client.embeddings.create(input=text, model=model)
    
    # Return the first embedding
    return response.data[0].embedding

def cosine_similarity(a:np.array, b:np.array) -> float:
    """
    Returns the cosine similarity between two vectors.
    """
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        warnings.warn("Input vectors are not numpy arrays. Trying to convert to numpy arrays.")
        try:
            a = np.array(a)
            b = np.array(b)
        except:
            raise ValueError("Input vectors cannot be converted to numpy arrays.")
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retry(func:Callable, max_try:int=5) -> Callable:
    def wrapper(*args, **kwargs):
        for i in range(max_try):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(e)
                if i == max_try - 1:
                    raise e
    return wrapper

data_list = [file for file in os.listdir(os.path.join(get_project_root(), "data")) if file.endswith(".parquet") and 'sample' not in file]