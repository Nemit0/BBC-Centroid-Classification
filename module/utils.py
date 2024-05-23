import os
from openai import OpenAI
import warnings
from tiktoken import Encoding

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