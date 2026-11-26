import muon
from preprocessing import *


def LSI(anndata, n_components=50) :
    #TF-IDF transformation
    muon.atac.pp.tfidf(anndata, to_layer='tfidf') #anndata.X is now the tf-idf transformed data
    
    #switch X and layers['tfidf'] because I cannot have the attribute 'layer' in lsi()
    counts = anndata.X
    anndata.X = anndata.layers['tfidf'] #we set the tf-idf transformed data as the main data of the anndata object, so that it can be used for the SVD decomposition
    #SVD decomposition
    try:
        muon.atac.tl.lsi(anndata, n_comps=n_components)
        
    finally:
        anndata.X = counts
    remove_comp_corr_with_depth(anndata)
    

def remove_comp_corr_with_depth(anndata, threshold = 0.75) :
    depth = np.asarray(anndata.X.sum(axis=1)).flatten()
    lsi = anndata.obsm['X_lsi']
    remove = []
    for i in range(lsi.shape[1]) :
        corr = np.corrcoef(lsi[:, i], depth)[0, 1]
        if abs(corr) > threshold :
            remove.append(i)
    anndata.obsm['X_lsi'] = np.delete(lsi, remove, axis=1)
        
    

def dim_red_pipeline(anndata, increase_bin = False, filtering=True, nbre_cells = 50, factor=1, n_components=50) :
    anndata = anndata.copy()
    if increase_bin :
        anndata = increase_bin_size(anndata, factor)
    if filtering :
        anndata = prevalence_filtering(anndata, nbre_cells)
        
    LSI(anndata, n_components=(n_components+1))
    anndata.obsm['X_lsi'] = anndata.obsm['X_lsi'][:, 1:(n_components+1)]
    return anndata