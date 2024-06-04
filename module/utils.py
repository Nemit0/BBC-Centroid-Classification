import os
import warnings
import numpy as np

from tiktoken import Encoding
from openai import OpenAI
from typing import Callable
from datetime import datetime

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

def retry(times, exceptions=Exception):
    """
    Retry Decorator
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param Exceptions: Lists of exceptions that trigger a retry attempt
    :type Exceptions: Tuple of Exceptions
    """
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f'Exception thrown: {e}, attempt {attempt + 1}/{times}')
                    attempt += 1
            return func(*args, **kwargs)
        return newfn
    return decorator

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        print(f"Function {func.__name__} executed in {end_time - start_time}")
        return result
    return wrapper


data_list = [file for file in os.listdir(os.path.join(get_project_root(), "data")) if file.endswith(".parquet") and 'sample' not in file]