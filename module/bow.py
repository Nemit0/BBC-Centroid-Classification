import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os

def clean_and_split_words(text: str) -> list:
    """
    clean and split words from text.
    It should return lower-case words without any punctuation.
    """



class BagOfWord:
    def __init__(data: pd.DataFrame):
        """
        dataframe is expected to have a 'text' column
        where it should contain all the text data from documents.
        """

    def get_unique_words(self):
        """
        Returns a list of unique words from the text data.
        """
        word_list = []
        for list in data.text:
            word_list.extend(list.split())
