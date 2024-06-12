import os
import json
import json
import nltk
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from module.utils import get_project_root
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List
from rich import print

from sklearn.feature_selection import chi2

def gaussian_pdf(x, variance:np.float128, mean=0):
    if not isinstance(x, np.float128) or not isinstance(variance, np.float128) or not isinstance(mean, np.float128):
        x = np.float128(x)
        variance = np.float128(variance)
        mean = np.float128(mean)
    return np.float128((1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * ((x - mean) ** 2) / (variance+0.00001)))

def clean_and_split_words(text: str, use_stemming: bool = False) -> list:
    """
    Clean and split words from text.
    - Converts text to lowercase.
    - Removes special characters and numbers.
    - Removes stopwords.
    - Optionally applies stemming.
    """
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word.strip() for word in words if word not in stop_words]
    if use_stemming:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words

def generate_ngrams(words, ngram_range):
    ngrams_list = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(words) - n + 1):
            ngrams_list.append(' '.join(words[i:i + n]))
    return ngrams_list

class TfIdfVectorizer:
    def __init__(self, sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_word_lang='english'):
        nltk.download('stopwords')
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.norm = norm
        self.ngram_range = ngram_range
        self.stop_words = stopwords.words(stop_word_lang)
        self.token_map = {}
    
    def fit_transform(self, documents: pd.Series) -> np.ndarray:
        if self.ngram_range == (1,1):
            docs_tokens = [clean_and_split_words(doc) for doc in documents]
        else:
            docs_tokens = [generate_ngrams(clean_and_split_words(doc), self.ngram_range) for doc in documents]
        vocabulary = set(word for doc in docs_tokens for word in doc)
        vocab_index = {word: idx for idx, word in enumerate(vocabulary)}

        df = {word: 0 for word in vocabulary}
        for tokens in docs_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        total_docs = len(documents)
        idf = {word: np.log(total_docs / (df[word] + 1)) + 1 for word in vocabulary}

        tf = {word: 0 for word in vocabulary}
        for tokens in docs_tokens:
            for token in tokens:
                tf[token] += 1
        
        vocab_index = {word: idx for idx, word in enumerate(vocabulary) if tf[word] >= self.min_df}
        vocabulary = set(vocab_index.keys())

        tfidf_matrix = np.zeros((total_docs, len(vocabulary)))
        for doc_idx, tokens in enumerate(docs_tokens):
            doc_freq = {word: 0 for word in vocab_index.keys()}
            for token in tokens:
                if token in vocab_index:
                    doc_freq[token] += 1
            vector = np.array([doc_freq[word] * idf[word] for word in vocab_index.keys()])
            if self.sublinear_tf:
                vector = np.log(vector + 1)
            tfidf_matrix[doc_idx] = vector

        if self.norm == 'l2':
            norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
            tfidf_matrix = tfidf_matrix / norms
        elif self.norm == 'l1':
            norms = np.linalg.norm(tfidf_matrix, ord=1, axis=1, keepdims=True)
            tfidf_matrix = tfidf_matrix / norms
        elif self.norm == None:
            pass
        
        self.token_map = vocab_index

        return tfidf_matrix
    

def main():
    root_path = os.path.join(get_project_root(), 'data', 'bbc')
    text_path = os.path.join(root_path, 'raw_text')

    class_id_map = {
        'business': 0,
        'entertainment': 1,
        'politics': 2,
        'sport': 3,
        'tech': 4
    }

    df_dict = {
        'class': [],
        'text': [],
        'classid' : [], 
        'title': [],
        'filename': []
    }
    # Load the data into dataframe
    for _class in class_id_map.keys():
        _path = os.path.join(text_path, _class)
        text_list = os.listdir(_path)
        for _text in text_list:
            with open(os.path.join(_path, _text), 'r') as f:
                text = f.read()
            title = text.split('\n')[0]
            text = text.replace(title, '')
            df_dict['class'].append(_class)
            df_dict['text'].append(text)
            df_dict['classid'].append(class_id_map[_class])
            df_dict['title'].append(title)
            df_dict['filename'].append(_text)

    df = pd.DataFrame(df_dict)
    vectorizer = TfIdfVectorizer(norm=None, ngram_range=(1,2))
    texts = df['text']
    features = vectorizer.fit_transform(texts)
    features.shape
    print(features)
    print(features.shape)
    print(len(vectorizer.token_map))

    N = 5
    for category, category_id in sorted(class_id_map.items()):
        features_chi2 = chi2(features, df['classid'] == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(list(vectorizer.token_map.keys()))[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(category))
        print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

    df['embeddings'] = list(features)
    df.set_index('filename', inplace=True)
    print(df.head())

    # Start calculating centroind and variance from test dataset
    if isinstance(df.iloc[0]['embeddings'], list):
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(x, dtype=np.float128))

    train_test_ratio = 0.8
    train_size = int(len(df) * train_test_ratio)
    # Shuffle dataset
    df = df.sample(frac=1)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    centroid_df = train_df.groupby('classid')['embeddings'].apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index()

    centroid_df['embeddings'] = centroid_df['embeddings'].apply(lambda x: np.array(x, dtype=np.float128))

    centroid_df.columns = ['classid', 'centroid']
    print(centroid_df.head())

    df_test = df.copy()

    df['distance_to_centroid_0'] = df['embeddings'].apply(lambda x: np.float128(np.linalg.norm(x - centroid_df.loc[0]['centroid'])))
    df['distance_to_centroid_1'] = df['embeddings'].apply(lambda x: np.float128(np.linalg.norm(x - centroid_df.loc[1]['centroid'])))
    df['distance_to_centroid_2'] = df['embeddings'].apply(lambda x: np.float128(np.linalg.norm(x - centroid_df.loc[2]['centroid'])))
    df['distance_to_centroid_3'] = df['embeddings'].apply(lambda x: np.float128(np.linalg.norm(x - centroid_df.loc[3]['centroid'])))
    df['distance_to_centroid_4'] = df['embeddings'].apply(lambda x: np.float128(np.linalg.norm(x - centroid_df.loc[4]['centroid'])))


    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    print(df.head())

    for i in range(5):
        # Filter df for the current category
        category_mask = train_df['classid'] == i
        
        # Calculate variance of 'distance_to_centroid_i' for this category
        variance = np.var(train_df.loc[category_mask, f"distance_to_centroid_{i}"])
        
        # Assign calculated variance to the correct entry in centroid_df
        centroid_df.loc[centroid_df['classid'] == i, 'variance'] = np.var(train_df.loc[category_mask, f"distance_to_centroid_{i}"])

    df['pmf_cat0'] = df['distance_to_centroid_0'].apply(lambda x: float(gaussian_pdf(x, centroid_df.iloc[0]['variance'])))
    df['pmf_cat1'] = df['distance_to_centroid_1'].apply(lambda x: float(gaussian_pdf(x, centroid_df.iloc[1]['variance'])))
    df['pmf_cat2'] = df['distance_to_centroid_2'].apply(lambda x: float(gaussian_pdf(x, centroid_df.iloc[2]['variance'])))
    df['pmf_cat3'] = df['distance_to_centroid_3'].apply(lambda x: float(gaussian_pdf(x, centroid_df.iloc[3]['variance'])))
    df['pmf_cat4'] = df['distance_to_centroid_4'].apply(lambda x: float(gaussian_pdf(x, centroid_df.iloc[4]['variance'])))

    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    data = df.copy().iloc[train_size:]
    distance_cols = [col for col in data.columns if "distance_to_centroid_" in col]
    pmf_cols = [col for col in data.columns if "pmf_cat" in col]

    # Task 1: Finding the closest centroid
    data['closest_centroid'] = data[distance_cols].idxmin(axis=1).str.extract('(\d+)').astype(int)

    # Task 2: Finding the PMF category with the highest probability
    data['pmf_predict'] = data[pmf_cols].idxmax(axis=1).str.extract('(\d+)').astype(int)

    # Task 3: Comparison Columns
    data['distance_correct'] = (data['closest_centroid'] == data['classid']).astype(int)
    data['pmf_correct'] = (data['pmf_predict'] == data['classid']).astype(int)

    # Displaying the updated DataFrame with the new columns
    data[['classid', 'closest_centroid', 'pmf_predict', 'distance_correct', 'pmf_correct']].head()

    distance_accuracy = data['distance_correct'].mean()
    pmf_col_accuracy = data['pmf_correct'].mean()
    print(f"Accuracy of distance-based classification: {distance_accuracy:.2f}")
    print(f"Accuracy of PMF-based classification: {pmf_col_accuracy:.2f}")

    data['top2_distance'] = data[distance_cols].apply(lambda x: list(np.argsort(x.values)[:2]), axis=1)

    # Task 2: Top 2 categories based on PMF
    data['top2_pmf'] = data[pmf_cols].apply(lambda x: list(np.argsort(-x.values)[:2]), axis=1)

if __name__ == "__main__":
    main()