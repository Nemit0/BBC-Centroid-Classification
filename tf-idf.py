import polars as pl
import polars.selectors as cs
import numpy as np
import os
import json

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print

from module.utils import get_project_root, retry
from module.bow import BagOfWordsEmbedding

class TfIdf(BagOfWordsEmbedding):
    def __init__(self, data_list, unique_word_data=None):
        super().__init__(data_list, unique_word_data)
        self.idf = None
        self.tfidf = None

    def train(self, max_workers=1, k=20):
        self.idf = self._compute_idf(max_workers)
        self.tfidf = self._compute_tfidf(k)

    def _compute_idf(self, max_workers=1):
        print("Computing IDF...")
        idf = {}
        total_docs = len(self.data_list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._compute_idf_worker, doc, total_docs) for doc in self.data_list]
            for future in as_completed(futures):
                idf.update(future.result())
        return idf

    def _compute_idf_worker(self, doc, total_docs):
        df = pl.read_parquet(doc)
        total_rows = df.height
        idf = {}
        for word in self.unique_words:
            word_count = df.select(cs.count(cs.col("text").contains(word))).collect().to_pandas().iloc[0, 0]
            if word_count > 0:
                idf[word] = np.log(total_docs / word_count)
        return idf

    def _compute_tfidf(self, k):
        print("Computing TF-IDF...")
        tfidf = {}
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(self._compute_tfidf_worker, doc, self.idf, k) for doc in self.data_list]
            for future in as_completed(futures):
                tfidf.update(future.result())
        return tfidf

    def _compute_tfidf_worker(self, doc, idf, k):
        df = pl.read_parquet(doc)
        total_rows = df.height
        tfidf = {}
        for i in tqdm(range(total_rows)):
            row = df.slice(i, 1)
            row_tfidf = {}
            for word in self.unique_words:
                word_count = row.select(cs.count(cs.col("text").contains(word))).collect().to_pandas().iloc[0, 0]
                if word_count > 0:
                    tf = word_count
                    tfidf_val = tf * idf[word]
                    row_tfidf[word] = tfidf_val
            row_tfidf = dict