import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad

def prevalence_filtering(anndata, threshold=50) :
    count_col = anndata.X.getnnz(axis=0)
    keep = count_col >= threshold
    print(f'Number of bins with more than {threshold} cells having non-zero values: {keep.sum()}')
    
    anndata_filtered = anndata[:, keep]
    return anndata_filtered

'''def prevalence_filtering_fit(anndata, threshold=50):
    count_col = anndata.X.getnnz(axis=0)
    keep = count_col >= threshold

    print(f"Number of bins kept: {keep.sum()}")

    return keep


def prevalence_filtering_transform(anndata, keep):
    return anndata[:, keep].copy()'''


def increase_bin_size(anndata, factor, size=500) :
    chrom = np.asarray(anndata.var['chrom'])
    start = np.asarray(anndata.var['start'])
    end = np.asarray(anndata.var['end'])
    n = len(start)
    width = size * factor #max size of the new bins
    
    group = np.zeros(n, dtype=np.int64) #where each bin belongs to a group of bins that will be merged together
    new_start = [start[0]]
    new_end = [end[0]]
    new_chrom = [chrom[0]]
    g = 0 #counter for the group number
    
    for i in range(1, n) :
        if chrom[i] == new_chrom[-1] and start[i] < new_start[-1] + width : #if the bin is on the same chromosome and its start is smaller than the max size of the new bin, it belongs to the same group
            new_end[-1] = end[i]
        else :
            g += 1 #a new group is created
            new_start.append(start[i])
            new_end.append(end[i])
            new_chrom.append(chrom[i])
        group[i] = g
    
    new_chrom = np.array(new_chrom)
    new_start = np.array(new_start)
    new_end = np.maximum(np.array(new_end), new_start + width) #if the last bin of a group is smaller than the width, we extend it to the width (like adding empty bins)
    
    M = sparse.csr_matrix( #matrix that has only one non-zero value per row, which is the group number of the bin, and it will be used to sum the bins in a group together
        (np.ones(n, dtype=anndata.X.dtype), (np.arange(n), group)),
        shape=(n, g + 1),
    )
    new_data = (anndata.X.tocsr() @ M).tocsr() #all the bins in a group are summed together to form a new bin
    
    return ad.AnnData(X=new_data, obs=anndata.obs, var=pd.DataFrame({'chrom': new_chrom, 'start': new_start, 'end': new_end}))

'''def preprocessing(adata_train, adata_val, factor, threshold):

    # Increase bin size
    train = increase_bin_size(adata_train, factor=factor)
    val = increase_bin_size(adata_val, factor=factor)

    # Prevalence filtering based only on train
    keep = train.X.getnnz(axis=0) >= threshold

    X_train = train[:, keep].X
    X_val = val[:, keep].X

    return X_train, X_val'''