import os
import json

import polars as pl
import pandas as pd
import numpy as np

import numpy.linalg as nla
import numpy.random as nr

from operator import add
from tqdm import tqdm
from typing import Iterable, List

from module.utils import get_project_root
from module.bow import BagOfWordsEmbedding
from concurrent.futures import ThreadPoolExecutor, as_completed

def gaussian(distance, sigma):
    """
    Calculate the pmf for each document with respect to each cluster using Gaussian distribution
    """
    value = np.exp(-0.5 * (distance / (sigma+0.0001)) ** 2)
    pmf = value / (np.sum(value)+0.0001)
    return pmf

def euclidean_distance(a:Iterable, b:Iterable) -> float:
    return np.linalg.norm(np.array(a)-np.array(b))

def loss(a:List[str], b:List[str], l1:int=1, l2:int=1) -> float:
    J = (l1*len(set(a).difference(set(b)))) + (l2*len(set(b).difference(set(a))))/len(set(a).union(set(b)))
    return J

def svd(A):  # to decompose W=X Sigma Y.T when CUR decomposition
  AT=np.transpose(A)
  M=np.dot(AT,A)
  eigval_M,eigvec_M=nla.eig(M)
  idx=np.argsort(eigval_M)
  idx=idx[::-1]
  eigval_M=eigval_M[idx]
  eigvec_M=eigvec_M[:,idx]

  V=np.copy(eigvec_M)

  M=np.dot(A,AT)
  eigval_M,eigvec_M=nla.eig(M)
  idx=np.argsort(eigval_M)
  idx=idx[::-1]
  eigval_M=eigval_M[idx]
  eigvec_M=eigvec_M[:,idx]

  U=np.copy(eigvec_M)

  Sigma=np.dot(np.transpose(U),A)
  Sigma=np.dot(Sigma,V)

  return U,Sigma,V

def delete_imaginary_part(U):  # to delete imaginary part when numbers real 
    real_U = np.empty(U.shape, dtype=object)
    for i in range(len(U)):
        for j in range(len(U[0])):
            if U[i, j].imag == 0:
                real_U[i, j] = U[i, j].real
            else:
                real_U[i, j] = U[i, j]
                print('complex derived')
                
    return real_U

def cur(A,r=2):
  p=np.sum(A**2,axis=0)  # Set prob to be selected
  q=np.sum(A**2,axis=1)
  p/=np.sum(p)
  q/=np.sum(q)

  col_list=nr.choice(np.array(range(len(p))),r,replace=False,p=p)
  row_list=nr.choice(np.array(range(len(q))),r,replace=False,p=q)


  C=np.zeros([len(A),r],float)
  R=np.zeros([r,len(A[0])],float)
  for i in range(r):
    C[:,i]=A[:,col_list[i]]
    C[:,i]/=((r*p[col_list[i]])**0.5)
    R[i,:]=A[row_list[i],:]
    R[i,:]/=((r*q[row_list[i]])**0.5)

  W=np.zeros([r,r],float)     # W=r*r, C X R 
  for i in range(r):
    for j in range(r):
      W[i,j]=A[row_list[i],col_list[j]]

  X,Sigma,Y=svd(W)

  # Sigma.+
  InvSig=np.zeros_like(np.transpose(Sigma))
  for i in range(len(Sigma)):
    if abs(Sigma[i,i])<=1.0e-6:    # if value is 0 or small, print 0
      InvSig[i,i]=0.0
    else:
      InvSig[i,i]=1/Sigma[i,i]     # take reverse 

  InvSig2=np.dot(InvSig,InvSig)

  U=np.dot(Y,InvSig2)
  U=np.dot(U,np.transpose(X))
  real_U = delete_imaginary_part(U)

  return C,real_U,R

def main():
    project_root = get_project_root()
    data_path = os.path.join(project_root, 'data')
    category_count_list = [file for file in os.listdir(data_path) if file.endswith("_category_counts.json")]
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in os.listdir(data_path) if file.endswith('.parquet') and file not in non_data_list]
    data_list = [file for file in data_list if '_embed.parquet' not in file]
    category_path = os.path.join(data_path, 'category_grouped')
    non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
    data_list = [file for file in os.listdir(data_path) if file.endswith('.parquet') and file not in non_data_list]
    data_list = [file for file in data_list if '_embed.parquet' not in file]

    data_list = ['o.parquet', 'z.parquet', 's.parquet']
    df = None
    # Get unique word list to use bag of words methods
    for data in data_list:
        if df is None:
          df = pl.read_parquet(os.path.join(data_path, data))
        else:
          df = df.vstack(pl.read_parquet(os.path.join(data_path, data)))
    
    chunk_path = os.path.join(data_path, 'analyze_embed_chunks')
    # Chunk in 1500 rows and save
    chunk_size = 1500
    for i in range(0, df.height, chunk_size):
        _df = df.slice(i, i+chunk_size).write_parquet(os.path.join(data_path, f'chunk_{i}.parquet'))
        _df.write_parquet(os.path.join(chunk_path, f'chunk_{i}.parquet'))
        del _df
    
    chunk_list = [os.path.join(chunk_path, file) for file in os.listdir(chunk_path) if file.endswith('.parquet')]
    # Randomly sample 10% of the chunks
    sample_ratio = 0.1
    sampled_chunks = np.random.choice(chunk_list, int(len(chunk_list)*sample_ratio), replace=False)
    bow = BagOfWordsEmbedding(data_list=sample_ratio, data_type='parquet')
    bow.train(max_workers=1, k=3, tf_udf=3)
    bow.save_words(os.path.join(data_path, 'bow_token_map_new.json'))
    reduced_path = os.path.join(project_root, 'data', 'reduced_chunks_new')
    if not os.path.exists(reduced_path):
        os.makedirs(reduced_path)
    
    for batch in tqdm(chunk_list):
        df = pl.read_parquet(batch)
        # Initialize empty column
        df = df.with_columns(pl.Series('embeddings', [[] for _ in range(df.height)], dtype=pl.List))
        for row in df.iter_rows():
            embeddings = bow.embed(row[2])
            df = df.with_columns(
                pl.when(df['id'] == row[0]).then(embeddings)
                .otherwise(df['embeddings'])
                .alias('embeddings')
            )
        df.drop_in_place('text')
        try:
            embeddings = df['embeddings'].to_list()
            embeddings = np.array(embeddings, float)
            C, U, R = cur(embeddings, r=1500)
            _df = pl.DataFrame({"id": df['id'], "embeddings": U.tolist()})
            df = df.drop("embeddings")
            df = df.join(_df, on="id")
            df.write_parquet(os.path.join(reduced_path, os.path.basename(batch)))
            del df, _df
        except:
            print(f"Error in {batch}")
            continue
    with open(os.path.join(data_path, 'filtered_category_list.json'), 'r') as f:
        category_list = json.load(f)

    chunk_list = [os.path.join(reduced_path, file) for file in os.listdir(reduced_path) if file.endswith('.parquet')]
    category_info = {k:{'centroid':[], 'variance':0, 'vector_sum':[], 'distance':[], 'count':0, 'threshold':0.5} for k in category_list}

    # First calculate centroid
    for batch in chunk_list:
        df = pl.read_parquet(batch)
        for row in tqdm(df.iter_rows()):
            for category in row[2]:
                if category in category_info.keys():
                    category_info[category]['vector_sum'] = list(map(add, category_info[category]['vector_sum'], row[3]))
                    category_info[category]['count'] += 1
                else:
                    category_info[category]['vector_sum'] = row[3]
                    category_info[category]['count'] = 1
    
    category_info = {k:v for k,v in category_info.items() if v['count'] > 5}
    for category in category_info.keys():
        category_info[category]['centroid'] = [v/(len(category_info[category]['vector_sum'])+1) for v in category_info[category]['vector_sum']]
    
    # Then calculate variance
    for batch in chunk_list:
        df = pl.read_parquet(batch)
        for row in tqdm(df.iter_rows()):
            for category in row[2]:
                if category in category_info.keys():
                    category_info[category]['distance'].append(euclidean_distance(category_info[category]['centroid'], row[3]))
                else:
                    category_info[category]['distance'].append(euclidean_distance(row[3], [0]*len(row[3])))
    
    for category in category_info.keys():
        category_info[category]['variance'] = np.var(category_info[category]['distance'])
    
    print(category_info)

    with open(os.path.join(data_path, 'category_info_centroid_variance_new_bow.json'), 'w') as f:
        json.dump(category_info, f)    
    
    # Model Evaluation
    loss_list = []
    train_test_ratio = 0.85
    for batch in chunk_list:
       df = pl.read_parquet(batch)
       df = df.slice(int(df.height*train_test_ratio), df.height)
       for row in tqdm(df.iter_rows()):
            embedding_vector = row[3]
            _temp = {k:euclidean_distance(embedding_vector, category_info[k]['centroid']) for k in category_info.keys()}
            _temp = dict(sorted(_temp.items(), key=lambda x: x[1]))
            _temp = dict(list(_temp.items())[:50])
            for category in category_info:
                _temp[category]['probability'] = gaussian(euclidean_distance(embedding_vector, category_info[category]['centroid']), category_info[category]['variance'])
            _temp = dict(sorted(_temp.items(), key=lambda x: x[1]))
            _temp = {k:v for k,v in _temp.items() if v['probability'] > category_info[k]['threshold']}
            print(_temp.keys())
            _loss = loss(row[2], _temp.keys())
            accuracy = 100 - _loss
            loss_list.append(_loss)
            print(f"Loss: {_loss}, Accuracy: {accuracy}")
            print(f"Average loss so far: {np.mean(loss_list)}")

    print(f"Average loss: {np.mean(loss_list)}")

if __name__ == '__main__':
    main()