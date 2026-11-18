from pathlib import Path
import numpy as np
import pandas as pd
from sympy import root
from scipy import sparse
import anndata as ad

def load_data_in_AnnData() :
    root = Path("Data")
    meta = pd.read_csv(root / "metadata.csv")

    #load H3K27ac
    data_H3K27ac = sparse.load_npz(root / "H3K27ac" / "counts.npz") 
    bins_ac = np.load(root / "H3K27ac" / "bins.npz")
    chrom_ac = bins_ac["chrom_levels"][bins_ac["chrom_codes"]]
    start_ac, end_ac = bins_ac["start"], bins_ac["end"]
    
    H3K27ac_ad = ad.AnnData(X=data_H3K27ac, obs=meta ,var=pd.DataFrame({'chrom': chrom_ac, 'start': start_ac, 'end': end_ac}))

    #load H3K27me3
    data_H3K27me3 = sparse.load_npz(root / "H3K27me3" / "counts.npz")
    bins_me3 = np.load(root / "H3K27me3" / "bins.npz")
    chrom_me3 = bins_me3["chrom_levels"][bins_me3["chrom_codes"]]
    start_me3, end_me3 = bins_me3["start"], bins_me3["end"]

    H3K27me3_ad = ad.AnnData(X=data_H3K27me3, obs=meta ,var=pd.DataFrame({'chrom': chrom_me3, 'start': start_me3, 'end': end_me3}))

    return H3K27ac_ad, H3K27me3_ad

