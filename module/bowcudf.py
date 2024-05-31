import polars as pl
import numpy as np
import cudf as cd
import os
import re
import nltk
import json
import warnings
import sys
import pytorch

from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
from rich import print

try:
    from utils import get_project_root
except ModuleNotFoundError:
    from module.utils import get_project_root

def clean_and_split_words(text: cudf.Series, use_stemming: bool = True) -> cudf.Series:
    """
    Clean and split words from a cudf.Series containing text data.
    - Converts text to lowercase.
    - Removes special characters and numbers.
    - Removes stopwords and optionally applies stemming.
    Returns a flattened cudf.Series of words.
    """
    # Normalize text: lowercase and remove non-alphabet characters
    text = text.str.lower().str.replace(r'[^a-z\s]', '', regex=True)

    # Tokenize: split text into words
    words = text.str.split()

    # Load English stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords
    words = words.apply(lambda x: [word for word in x if word not in stop_words])

    if use_stemming:
        stemmer = PorterStemmer()
        words = words.apply(lambda x: [stemmer.stem(word) for word in x])

    # Flatten the list of lists into a single series of words
    words = words.list.flatten()
    return words

class BagOfWordsEmbedding:
    def __init__(self, data_list:list=None, unique_word_data:str=None, data_type:str='parquet') -> list:
        """
        Dataframe is expected to have a 'text' column
        where it should contain all the text data from documents.
        If unique_word_data is provided, it will be used to get the unique words.
        unique_word should be a string to the path of the data. Data should be json file of dict.
        """
        # Ensure that the nltk stopwords are downloaded
        nltk.download('stopwords')
        self.data_list = data_list
        self.data_type = data_type
        self.unique_words = []
        self.token_map = {}
        self.output_dim = 0
        if unique_word_data:
            with open(unique_word_data, 'r') as f:
                self.token_map = json.load(f)
                self.unique_words = list(self.token_map.keys())
                self.output_dim = len(self.unique_words)
            if not isinstance(self.token_map, dict):
                warnings.warn("Invalid data type for unique words. Ignoring the data.")
                self.unique_words = []
                self.token_map = {}
                self.output_dim = 0
        return None

    def load_data(self, path:str, type:str) -> pl.DataFrame:
        """
        Simple wrapper for dynamic loading of different types of data
        """
        match type:
            case 'parquet':
                return cd.read_parquet(path)
            case 'csv':
                return cd.read_csv(path)
            case _:
                raise ValueError(f"Invalid data type: {type}")

    def get_unique_words(self, data:cd.DataFrame) -> list:
        """
        Returns a list of unique words from the text data using cudf.DataFrame.
        This method assumes there is a 'text' column inside the DataFrame.
        """
        # Ensure the text column is treated as a cudf.Series
        text_series = data['text'].astype(str)

        # Clean and tokenize text
        words = clean_and_split_words(text_series)

        # Get unique words
        unique_words = words.drop_duplicates().to_arrow().to_pylist()
        unique_words.sort()

        return unique_words
    
    def load_and_process_data(self, data:str) -> list[str]:
        """
        Helper function to load and process each data item.
        This function is designed to be used with ThreadPoolExecutor.
        """
        data_frame = self.load_data(data, self.data_type)
        return self.get_unique_words(data_frame)
    
    def train(self, k:int=0) -> None:
        """
        Read and process the data to get the unique words, which will be used to create BoW embedding.
        The function returns nothing, but it's not executed on __init__ to allow for more flexibility, and potential model loading.
        Uses ThreadPoolExecutor to handle multiple data files.
        The argument k is min threshhold for the word to be included in the unique words of occurence.
        """
        _token_map = {}
        # cudf accelerated data
        for data in tqdm(self.data_list):
            _unique_word = set(self.get_unique_words(self.load_data(data, self.data_type, lazy=False)))
            for word in _unique_word:
                if word in _token_map.keys():
                    _token_map[word] += 1
                else:
                    _token_map[word] = 1
            _unique_word.update(self.unique_words)
        self.unique_words.extend(list(_unique_word))
        del _unique_word

        # Remove duplicates from the list of unique words just to make sure
        self.unique_words = list(set(self.unique_words))
        self.token_map = {word: i for i, word in enumerate(self.unique_words)}
        if k > 0:
            print(f"Removing words with occurence less than {k}. Current unique words: {len(self.unique_words)}")
            self.unique_words = [word for word in self.unique_words if _token_map[word] >= k]
            self.token_map = {word: i for i, word in enumerate(self.unique_words)}
            print(f"New unique words: {len(self.unique_words)}")

        self.output_dim = len(self.unique_words)
        return None

    def save_words(self, path:str) -> None:
        """
        Save the unique words to a json file.
        This file basically act as a vocabulary for the BoW embedding.
        """
        with open(path, 'w') as f:
            json.dump(self.token_map, f)

        return None
        
    def embed(self, document: str) -> np.array:
        """
        Embeds the document using the unique words after cleaning the text similarly to the training process.
        """
        # Clean the text as per the training preprocessing
        word_list = clean_and_split_words(document)
        vector = np.zeros(self.output_dim)

        for word in word_list:
            token_id = self.token_map.get(word, None)
            if token_id:
                vector[token_id] += 1
        
        return vector

    def embed_dataframe(self, data:cd.DataFrame) -> cd.DataFrame:
        """
        Embeds the text data in the dataframe using the unique words.
        """
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(self.embed, text) for text in data['text']]
            embeddings = [future.result() for future in as_completed(futures)]
        
        
        
        # Ensure all embeddings are numpy arrays of the same shape
        embeddings_array = np.array(embeddings)

        # Convert embeddings list to a Series and add to the dataframe
        embeddings_series = pl.Series("embeddings", embeddings_array.tolist())
        data = data.with_columns(embeddings_series)
        return data
    
if __name__ == "__main__":
    text = 'This is a sample text. It is used to test the Bag of Words embedding.'
    text_split = clean_and_split_words(text)
    print(text_split)

    # Uses sample data from the data folder
    data_path = os.path.join(get_project_root(), "data")
    model_path = os.path.join(get_project_root(), "models")
    data_list = [os.path.join(data_path, "sample.parquet")]
    if 'sample_data_unique_words.json' not in os.listdir(model_path):
        BowEncoder = BagOfWordsEmbedding(data_list, unique_word_data=None, data_type='parquet')
        BowEncoder.train()

        # Save the unique words
        save_path = os.path.join(model_path, 'sample_data_unique_words.json')
        BowEncoder.save_words(save_path)
        del BowEncoder

    # Load the unique words
    BowEncoder = BagOfWordsEmbedding(data_list, unique_word_data=os.path.join(model_path, 'sample_data_unique_words.json'), data_type='parquet')
    # test embeddings
    sample_embedding = BowEncoder.embed(text)
    print(sample_embedding)
    # Generate some sample embeddings
    sample_documents = pl.read_parquet(os.path.join(data_path, "sample.parquet"))
    embeddings = BowEncoder.embed_dataframe(sample_documents)
    print(embeddings.head())
    # sum up all embedding vectors into one
    embedding_sum = np.sum(np.stack(embeddings['embeddings'].to_list()), axis=0)

    embedding_sum_dict = {str(id): val for id, val in enumerate(embedding_sum)}

    print(embedding_sum)
    word_occurance = {word:embedding_sum_dict[str(id)] for word, id in BowEncoder.token_map.items()}

    # Sort by value descending
    word_occurance = dict(sorted(word_occurance.items(), key=lambda item: item[1], reverse=True))

    print(list(word_occurance.values())[:10], list(word_occurance.keys())[:10])

    sys.exit(0)

    # Visualize the embeddings using dimension reduction to 2D
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    # Convert the column directly to a NumPy array
    pca_embeddings = pca.fit_transform(np.stack(embeddings['embeddings'].to_list()))

    plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Document Embeddings')
    plt.show()