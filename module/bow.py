import polars as pl
import numpy as np
import os
import re
import nltk
import json
import warnings

from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
from rich import print

from .utils import get_project_root

def clean_and_split_words(text: str, use_stemming: bool = True) -> List[str]:
    """
    Clean and split words from text.
    - Converts text to lowercase.
    - Removes special characters and numbers.
    - Removes stopwords.
    - Optionally applies stemming.
    """
    # Convert text to lowercase and remove all non-letter characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    # Split text into words
    words = text.split()

    # Load English stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Optionally apply stemming
    if use_stemming:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]

    return words

class BagOfWordsEmbedding:
    def __init__(self, data_list:list=None, unique_word_data:str=None, data_type:str='parquet') -> list:
        """
        dataframe is expected to have a 'text' column
        where it should contain all the text data from documents.
        If unique_word_data is provided, it will be used to get the unique words.
        unique_word should be a string to the path of the data. Data should be json file of list.
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

    def load_data(self, path:str, type:str, lazy:bool=False) -> pl.DataFrame:
        """
        Simple wrapper for dynamic loading of different types of data
        """
        match type:
            case 'parquet':
                if lazy:
                    return pl.scan_parquet(path)
                return pl.read_parquet(path)
            case 'csv':
                if lazy:
                    return pl.scan_csv(path)
                return pl.read_csv(path)
            case _:
                raise ValueError(f"Invalid data type: {type}")

    def get_unique_words(self, data:pl.LazyFrame|pl.DataFrame) -> list:
        """
        Returns a list of unique words from the text data.
        This assumes there is a 'text' column inside the dataframe.
        Accelerated using multiprocessing.
        """
        word_list = []

        if isinstance(data, pl.LazyFrame):
            # Collect data if LazyFrame to handle with multiprocessing
            data = data.collect()

        # Now handle as a DataFrame using multiprocessing for potentially large data
        with Pool(cpu_count()) as pool:
           results = pool.map(clean_and_split_words, list(data['text']))

        word_list = [word for sublist in results for word in sublist] # Flatten list of lists
        unique_words = list(set(word_list))
        unique_words.sort()
        
        return unique_words
    
    def load_and_process_data(self, data:str) -> list[str]:
        """
        Helper function to load and process each data item.
        This function is designed to be used with ThreadPoolExecutor.
        """
        data_frame = self.load_data(data, self.data_type, lazy=False)
        return self.get_unique_words(data_frame)
    
    def train(self, max_workers:int=3, k:int=0) -> None:
        """
        Read and process the data to get the unique words, which will be used to create BoW embedding.
        The function returns nothing, but it's not executed on __init__ to allow for more flexibility, and potential model loading.
        Uses ThreadPoolExecutor to handle multiple data files.
        The argument k is min threshhold for the word to be included in the unique words of occurence.
        """
        _token_map = {}
        if max_workers > 1:
            # Initialize ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks to the executor
                futures = {executor.submit(self.load_and_process_data, data): data for data in self.data_list}

                for future in as_completed(futures):
                    _unique_word = set(future.result())
                    for word in _unique_word:
                        _token_map[word] += 1
                    _unique_word.update(self.unique_words)
                self.unique_words.extend(list(_unique_word))
                del _unique_word    

        elif max_workers == 1:
            # Single-threaded processing
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
        else:
            raise ValueError("Invalid value for max_workers. Must be greater than 0.")

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

    def embed_dataframe(self, data: pl.DataFrame) -> pl.DataFrame:
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
    # sample 100 rows
    sample_documents = sample_documents.head(100)
    embeddings = BowEncoder.embed_dataframe(sample_documents)
    print(embeddings.head())

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