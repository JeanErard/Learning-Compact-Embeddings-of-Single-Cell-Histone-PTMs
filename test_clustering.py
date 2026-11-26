from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scib_metrics import silhouette_label
import harmonypy as hm
import numpy as np

def NMI_ARI(anndata, clust='Kmeans', cell_type='cell_type') :
    true = anndata.obs[cell_type]
    pred = anndata.obs[clust]
    
    NMI = normalized_mutual_info_score(true, pred)
    ARI = adjusted_rand_score(true, pred)
    
    return NMI, ARI

def ASW_LISI(anndata, clust='Kmeans_label', type_obs = 'X_lsi') :
    ASW = silhouette_label(X=anndata.obsm[type_obs], labels=anndata.obs[clust])
    LISI = hm.compute_lisi(X=anndata.obsm[type_obs], metadata=anndata.obs, label_colnames=[clust])
    mean_LISI = np.mean(LISI)
    return ASW, mean_LISI

def clust_score(anndata, nmi_ari=True, asw_lisi=True, clust_name='Kmeans', type_obs='X_lsi', cell_type='cell_type', show=True) :
    if show :
        print(f'Clustering score for {clust_name} :')
    result = []
    if nmi_ari :
        NMI, ARI = NMI_ARI(anndata, clust=clust_name, cell_type=cell_type)
        if show :
            print(f' - NMI score : {NMI:.3f}, ARI score : {ARI:.3f}')
        result.append(float(NMI))
        result.append(float(ARI))
    if asw_lisi :
        ASW, LISI = ASW_LISI(anndata, clust=clust_name, type_obs=type_obs)
        if show :
            print(f' - ASW score : {ASW:.3f}, LISI score (mean) : {LISI.mean():.3f}')
        result.append(float(ASW))
        result.append(float(LISI))
    return result