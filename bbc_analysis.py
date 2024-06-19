import os
import json
import nltk
import re
import warnings
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from module.utils import get_project_root
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List, Iterable
from rich import print
from tqdm import tqdm

from sklearn.feature_selection import chi2

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
    return np.divide(np.dot(a, b), (np.linalg.norm(a) * np.linalg.norm(b)), dtype=np.float128)

def euclidean_distance(a:np.array, b:np.array) -> float:
    """
    Returns the euclidean distance between two vectors.
    """
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        warnings.warn("Input vectors are not numpy arrays. Trying to convert to numpy arrays.")
        try:
            a = np.array(a)
            b = np.array(b)
        except:
            raise ValueError("Input vectors cannot be converted to numpy arrays.")
    return np.linalg.norm(a - b)

def gaussian_pdf(x, variance, mean=0):
    if variance < 1e-10:  # handle small variance to avoid division by zero
        variance += 1e-10
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * ((x - mean) ** 2) / variance)

def cost(a, b) -> float:
    return float(len(set(a).difference(b)) + len(set(b).difference(a))/len(set(a).union(b)))

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

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    return np.divide(vector, np.linalg.norm(vector))

def find_candidates(row, pmf_cols:Iterable, thresholds:dict) -> List[int]:
    candidates = []
    for idx, pmf_col in enumerate(pmf_cols):
        if row[pmf_col] >= thresholds[idx]:  # Check if PMF exceeds the threshold for this category
            candidates.append(idx)  # Use the category index as candidate
    return candidates if candidates else [-1]

class TfIdfVectorizer(object):
    def __init__(self, sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_word_lang='english'):
        # nltk.download('stopwords')
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.norm = norm
        self.ngram_range = ngram_range
        self.stop_words = stopwords.words(stop_word_lang)
        self.token_map = {}
        self.tf_value = {}
        self.idf_value = {}
    
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
        self.idf = idf
        tf = {word: 0 for word in vocabulary}
        for tokens in docs_tokens:
            for token in tokens:
                tf[token] += 1
        self.tf = tf
        
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

    def embed_doc(self, doc:str) -> np.array:
        raise NotImplementedError
        
    

def main(*args, **kwargs) -> tuple:
    root_path = os.path.join(get_project_root(), 'data', 'bbc')
    text_path = os.path.join(root_path, 'raw_text')

    warnings.filterwarnings("ignore")
    class_id_map = {
        'business': 0,
        'entertainment': 1,
        'politics': 2,
        'sport': 3,
        'tech': 4
    }

    inv_class_id_map = {v: k for k, v in class_id_map.items()}

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
    print(f"{len(df)} Articles loaded")
    df_norm = df.copy()

    vectorizer = TfIdfVectorizer(norm=None, ngram_range=(1,2))
    vectorizernorm = TfIdfVectorizer(norm='l2', ngram_range=(1,2))

    texts = df['text']
    features = vectorizer.fit_transform(texts)
    features_norm = vectorizernorm.fit_transform(texts)
    print(f"Shape of the feature matrix: {features.shape}")

    print(f"Calculating Chi2 for feature-category correlation analysis")
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
    print(f"Calculating Chi2 for feature-category correlation analysis (Normalized)")
    N = 5
    for category, category_id in sorted(class_id_map.items()):
        features_chi2 = chi2(features_norm, df['classid'] == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(list(vectorizernorm.token_map.keys()))[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(category))
        print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

    df['embeddings'] = list(features)
    df_norm['embeddings'] = list(features_norm)
    df.set_index('filename', inplace=True)
    df_norm.set_index('filename', inplace=True)
    print("All dataframe loaded with embeddings")
    print(f"Embedding Dimension:{df['embeddings'][0].shape}")

    # Start calculating centroind and variance from test dataset
    if isinstance(df.iloc[0]['embeddings'], list):
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(x, dtype=np.float128))
        df_norm['embeddings'] = df_norm['embeddings'].apply(lambda x: np.array(x, dtype=np.float128))

    train_test_ratio = 0.7
    df = df.sample(frac=1, **kwargs)
    df_norm = df_norm.sample(frac=1, random_state=1)
    train_df = df.iloc[:int(np.floor(train_test_ratio * len(df)))]
    test_df = df.iloc[int(np.floor(train_test_ratio * len(df))+1):]
    train_df_norm = df_norm.iloc[:int(np.floor(train_test_ratio * len(df_norm)))]
    test_df_norm = df_norm.iloc[int(np.floor(train_test_ratio * len(df_norm))+1):]
    print(f"Training dataset size: {train_df.shape[0]}, Testing dataset size: {test_df.shape[0]}")

    centroid_df = train_df.groupby('classid')['embeddings'].apply(lambda x: np.mean(np.stack(x, dtype=np.float128), axis=0, dtype=np.float128)).reset_index()
    centroid_df['embeddings'] = centroid_df['embeddings'].apply(lambda x: np.array(x, dtype=np.float128))
    centroid_df.columns = ['classid', 'centroid']
    centroid_df_norm = train_df_norm.groupby('classid')['embeddings'].apply(lambda x: np.mean(np.stack(x, dtype=np.float128), axis=0, dtype=np.float128)).reset_index()
    centroid_df_norm['embeddings'] = centroid_df_norm['embeddings'].apply(lambda x: normalize_vector(np.array(x, dtype=np.float128)))
    centroid_df_norm.columns = ['classid', 'centroid']
    print("Centroid calculated")

    for i in range(5):
        train_df[f"distance_to_centroid_{i}"] = train_df['embeddings'].apply(lambda x: euclidean_distance(x, centroid_df.iloc[i]['centroid']))
        test_df[f"distance_to_centroid_{i}"] = test_df['embeddings'].apply(lambda x: euclidean_distance(x, centroid_df.iloc[i]['centroid']))
        train_df_norm[f"distance_to_centroid_{i}"] = train_df_norm['embeddings'].apply(lambda x: np.divide(1, cosine_similarity(x, centroid_df_norm.iloc[i]['centroid']), dtype=np.float128))
        test_df_norm[f"distance_to_centroid_{i}"] = test_df_norm['embeddings'].apply(lambda x: np.divide(1, cosine_similarity(x, centroid_df_norm.iloc[i]['centroid']), dtype=np.float128))

    print(test_df_norm.head())
    for i in range(5):
        category_mask = train_df['classid'] == i
        centroid_df.loc[centroid_df['classid'] == i, 'variance'] = np.var(train_df.loc[category_mask, f"distance_to_centroid_{i}"])
        category_mark_norm = train_df_norm['classid'] == i
        centroid_df_norm.loc[centroid_df_norm['classid'] == i, 'variance'] = np.var(train_df_norm.loc[category_mark_norm, f"distance_to_centroid_{i}"])
    
    for i in range(5):
        train_df[f"pmf_cat{i}"] = train_df[f"distance_to_centroid_{i}"].apply(lambda x: gaussian_pdf(x, centroid_df.iloc[i]['variance']))
        test_df[f"pmf_cat{i}"] = test_df[f"distance_to_centroid_{i}"].apply(lambda x: gaussian_pdf(x, centroid_df.iloc[i]['variance']))
        train_df_norm[f"pmf_cat{i}"] = train_df_norm[f"distance_to_centroid_{i}"].apply(lambda x: gaussian_pdf(x, centroid_df_norm.iloc[i]['variance']))
        test_df_norm[f"pmf_cat{i}"] = test_df_norm[f"distance_to_centroid_{i}"].apply(lambda x: gaussian_pdf(x, centroid_df_norm.iloc[i]['variance']))

    distance_cols = [col for col in train_df.columns if "distance_to_centroid_" in col]
    pmf_cols = [col for col in train_df.columns if "pmf_cat" in col]
    data = train_df.copy()
    # Assess using the distance-based classification
    test_df['closest_centroid'] = test_df[distance_cols].idxmin(axis=1).str.extract('(\d+)').astype(int)
    test_df_norm['closest_centroid'] = test_df_norm[distance_cols].idxmin(axis=1).str.extract('(\d+)').astype(int)
    test_df['distance_correct'] = (test_df['closest_centroid'] == test_df['classid']).astype(int)
    test_df_norm['distance_correct'] = (test_df_norm['closest_centroid'] == test_df_norm['classid']).astype(int)
    print(f"Accuracy of distance-based classification: {test_df['distance_correct'].mean():.2f}")
    print(f"Accuracy of distance-based classification (Normalized): {test_df_norm['distance_correct'].mean():.2f}")
    # Assess using the PMF-based classification
    test_df['pmf_predict'] = test_df[pmf_cols].idxmax(axis=1).str.extract('(\d+)').astype(int)
    test_df_norm['pmf_predict'] = test_df_norm[pmf_cols].idxmax(axis=1).str.extract('(\d+)').astype(int)
    test_df['pmf_correct'] = (test_df['pmf_predict'] == test_df['classid']).astype(int)
    test_df_norm['pmf_correct'] = (test_df_norm['pmf_predict'] == test_df_norm['classid']).astype(int)
    print(f"Accuracy of PMF-based classification: {test_df['pmf_correct'].mean():.2f}")
    print(f"Accuracy of PMF-based classification (Normalized): {test_df_norm['pmf_correct'].mean():.2f}")

    category_threshold = {i : 0 for i in range(5)}
    category_threshold_norm = {i : 0 for i in range(5)}
    # Assess train_df to find the threshold for each category
    train_df['pmf_predict'] = train_df[pmf_cols].idxmax(axis=1).str.extract('(\d+)').astype(int)
    train_df['pmf_correct'] = (train_df['pmf_predict'] == train_df['classid']).astype(int)
    train_df_norm['pmf_predict'] = train_df_norm[pmf_cols].idxmax(axis=1).str.extract('(\d+)').astype(int)
    train_df_norm['pmf_correct'] = (train_df_norm['pmf_predict'] == train_df_norm['classid']).astype(int)

    learning_rate = 1e-14
    learning_step = 1000
    for i in range(5): # Category Loop
        for j in range(learning_step):
            if j == 0:
                prior_loss = 10000 # Some large number to initialize loss
            loss_list = []
            for k in range(len(test_df)):
                loss = cost(set([test_df.iloc[k]['classid']]), set([i]) if test_df[f'pmf_cat{i}'][k] > category_threshold[i] else set([]))
                loss_list.append(loss)
            
            loss_mean = np.mean(loss_list)  
            if loss_mean < prior_loss:
                category_threshold[i] += learning_rate
            elif loss_mean > prior_loss:
                category_threshold[i] -= learning_rate
            else:
                pass
            prior_loss = loss_mean
        
    # Repeat for normalized data
    for i in range(5): # Category Loop
        for j in tqdm(range(learning_step)) :
            if j == 0:
                prior_loss = 10000
            loss_list = []
            for k in range(len(test_df_norm)):
                loss = cost(set([test_df_norm.iloc[k]['classid']]), set([i]) if test_df_norm[f'pmf_cat{i}'][k] > category_threshold_norm[i] else set([]))
                loss_list.append(loss)
            
            loss_mean = np.mean(loss_list)
            if loss_mean < prior_loss:
                category_threshold_norm[i] += learning_rate
            elif loss_mean > prior_loss:
                category_threshold_norm[i] -= learning_rate
            else:
                pass
            prior_loss = loss_mean

    print(category_threshold)

    test_df['candidate_category'] = test_df.apply(lambda x: x['pmf_predict'] if x[f"pmf_cat{x['pmf_predict']}"] >= category_threshold[x['pmf_predict']] else x['classid'], axis=1)
    test_df_norm['candidate_category'] = test_df_norm.apply(lambda x: x['pmf_predict'] if x[f"pmf_cat{x['pmf_predict']}"] >= category_threshold_norm[x['pmf_predict']] else x['classid'], axis=1)

    # PMF columns in your DataFrame
    pmf_columns = [f'pmf_cat{i}' for i in range(5)]

    # Apply the function to find candidates for each document
    test_df['candidate_categories'] = test_df.apply(find_candidates, axis=1, pmf_cols=pmf_columns, thresholds=category_threshold)
    test_df_norm['candidate_categories'] = test_df_norm.apply(find_candidates, axis=1, pmf_cols=pmf_columns, thresholds=category_threshold)

    # Savezx
    test_df.to_csv(os.path.join(root_path, 'test_df.csv'))
    test_df_norm.to_csv(os.path.join(root_path, 'test_df_norm.csv'))
    return test_df['pmf_correct'].mean(), test_df_norm['pmf_correct'].mean()

if __name__ == "__main__":
    # test_size = 20
    # accuracy = []
    # accuracy_norm = []
    # test_rand_array = [random.randint(0, 100) for i in range(test_size)]
    # for seed in tqdm(test_rand_array):
    #     _acc, _acc_norm = main(rand_seed=seed)
    #     accuracy.append(_acc)
    #     accuracy_norm.append(_acc_norm)
    
    # _df = pd.DataFrame({
    #     'accuracy':accuracy,
    #     'accuracy_norm':accuracy_norm
    # })

    # df_melt = _df.melt(var_name='Accuracy Type', value_name='Accuracy Value')
    # sns.boxplot(x='Accuracy Type', y='Accuracy Value', data=df_melt)
    # plt.show()
    main(random_state=0)