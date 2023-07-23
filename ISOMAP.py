from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.graph import graph_shortest_path

from scipy.linalg import eigh
import scipy.sparse as sparse

from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

IRIS = True
MNIST = False

if IRIS:
    titre1 = 'myISOMAP - IRIS'
    titre2 = 'ISOMAP - IRIS'
    iris = datasets.load_iris()
    X = iris.data  
    y = iris.target
    
    N = X.shape[0]
    n_components = 2
    n_neighbors = 40
    
elif MNIST:
    titre1 = 'myISOMAP - MNIST'
    titre2 = 'ISOMAP - MNIST'
    digits = datasets.load_digits(n_class=6)
    X = digits.data
    y = digits.target
    
    N = X.shape[0]
    n_components = 2
    n_neighbors = 10

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('labels: %s' % np.unique(y))
print('Class distribution: %s' % np.bincount(y))

#----------------------------------------------------------------------
# Isomap projection of the digits dataset

# Step 1 
t0=time()
kng = kneighbors_graph(X[0:N], n_neighbors,mode='distance')
# Par construction une matrice creuse au format CSR 
# print( "K-neighboors :" + str(kng))

## on visualise le haut gauche de la matrice "densifiée" 
#plt.figure()
#plt.imshow(kng.todense()[0:20,0:20])#[0:20,0:20])
##
#### on symmétrise le graphe
kng = 0.5*(kng + kng.T)

#plt.figure()
#plt.imshow(kng.todense()[0:20,0:20])

#
### Step 2

D = graph_shortest_path(kng,directed=False)
#plt.figure(figsize=(12,12))
#plt.imshow(D)

## Step 3 :
# on calcule la matrice de similarité = -.5*(dist_matrix **2)
centrage = np.eye(N)-np.ones((N,N))/N
B =-0.5 * centrage @ (D**2) @ centrage

plt.figure(figsize=(12,12))
plt.imshow(B)
#
###valeurs propres de la matrice des similarités
[v,V] = eigh(B) # elles sont rangées dans l'ordre croissant
#
# print('valeurs propres',v)

# print(V[0])
#
### prendre les 2 premiers vecteurs propres
Y_iso  = np.fliplr(V[:,-2:] @ np.diag(v[-2:]**0.5))

##
fig, ax = plt.subplots(figsize=(8,8))
scatter = ax.scatter(Y_iso[:,0], Y_iso[:,1], c=y[0:N], cmap=plt.cm.Set1)
legend1 = ax.legend(*scatter.legend_elements(),loc="upper right", title=titre1)
ax.add_artist(legend1) 
#
#vérification avec scikitlearn
print("Computing Isomap embedding")

X_iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components,eigen_solver='dense',path_method='D').fit_transform(X)

fig, ax = plt.subplots(figsize=(8,8))
scatter = ax.scatter(X_iso[:,0], X_iso[:,1], c=y[0:N], cmap=plt.cm.Set1)
legend1 = ax.legend(*scatter.legend_elements(),loc="upper right", title=titre2)
ax.add_artist(legend1) 

