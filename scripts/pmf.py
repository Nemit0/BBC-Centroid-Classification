import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

def gaussian(distances, sigma):
    """
    Calculate the pmf for each document with respect to each cluster using Gaussian distribution
    """
    values=np.exp(-0.5*(distances/sigma)**2)
    pmf=values/np.sum(values, axis=1, keepdims=True)
    return pmf

def cost_func(Cd, Cd_prime, lambda1, lambda2): #set of categories = C
    J=lambda1*len(Cd_prime-Cd) + lambda2*len(Cd-Cd_prime)
    return J

def normalized_cost_func(Cd, Cd_prime, lambda1, lambda2):
    J=cost_funct(Cd, Cd_prime, lambda1, lambda2)
    return J/len(Cd|Cd_prime)


#dataset splitting
X_train, X_temp, y_train, _temp=train_split(embeddings, categories, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test=train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

categories=np.arange(cateogries.shape[1])
sigma=1 #example
pmf_list=[]

for cateogory in categories:
    category_indices=np.where(y_train[:, category]==1)[0]
    if len(category_indices)==0:
        continue

    category_documents = X_train[category_indices]
    distances = cdist(X_train, category_documents, 'euclidean')
    pmf = gaussian(distances, sigma)
    pmf_list.append(pmf)

pmf_array = np.hstack(pmf_list)
