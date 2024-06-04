import os
import json

import numpy as np
import numpy.linalg as nla
import numpy.random as nr
import polars as pl

from tqdm import tqdm
from rich import print

from module.utils import get_project_root

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
  print("Initializing...")
  project_root = get_project_root()
  model_path = os.path.join(project_root, "models")
  data_path = os.path.join(project_root, "data")
  chunk_path = os.path.join(data_path, "embed_chunks")
  pruned_path = os.path.join(data_path, 'pruned_embed')
  reduced_path = os.path.join(data_path, 'cur_applied')
  non_data_list = ['sample_embedding.parquet', 'sample.parquet', 'wiki_2023_index.parquet']
  data_list = [file for file in os.listdir(data_path) if file.endswith('.parquet') and file not in non_data_list]
  data_list = [file for file in data_list if '_embed.parquet' not in file]

  for file in tqdm(data_list):
    print(f"Processing {file}")
    if not os.path.exists(os.path.join(reduced_path, file)):
      os.mkdir(os.path.join(reduced_path, file))
    chunk_list = os.listdir(os.path.join(pruned_path, file))
    for chunk in tqdm(chunk_list):
      if chunk in os.listdir(os.path.join(reduced_path, file)):
        continue
      try:
        df = pl.read_parquet(os.path.join(pruned_path, file, chunk))
        embeddings = df['embeddings'].to_list()
        embeddings = np.array(embeddings, float)
        C, U, R = cur(embeddings, r=1500)
        _df = pl.DataFrame({"id": df['id'], "embeddings": U.tolist()})
        df = df.drop("embeddings")
        df = df.join(_df, on="id")
        # print(df.head())
        df.write_parquet(os.path.join(reduced_path, file, chunk))
        del df, _df, embeddings, C, U, R
      except Exception as e:
        print(e)
        print(f"Error occurred in {file}/{chunk}, likely due to mismatch in dimension: {df.height} rows, {len(embeddings)} columns.")
        continue

if __name__ == "__main__":
  main()