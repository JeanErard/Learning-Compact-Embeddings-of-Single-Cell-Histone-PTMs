import numpy as np
from sklearn.cluster import KMeans
import scanpy as sc

def clustering(anndata, Kmeans=True, leiden=True, visualization=False, random_state=42, type_obs='X_lsi', resolution=1, cell_type='cell_type') :
    labels = anndata.obs[cell_type]
    nbre_type = len(np.unique(labels))
    colors = [cell_type]
    if Kmeans:
        kmeans = KMeans(n_clusters=nbre_type, random_state=42)
        anndata.obs['Kmeans'] = kmeans.fit_predict(anndata.obsm[type_obs])
        colors.append('Kmeans')
    if leiden :
        sc.pp.neighbors(anndata, use_rep=type_obs, key_added='nn_lsi')
        sc.tl.leiden(anndata, resolution=resolution, key_added='leiden', neighbors_key='nn_lsi')
        colors.append('leiden')
    if visualization :
        sc.tl.umap(anndata, neighbors_key='nn_lsi')
        sc.pl.umap(anndata, color=colors)