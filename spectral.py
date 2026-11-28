import snapatac2 as snap
from anndata import AnnData
import numpy as np


def spectral_embedding(anndata, n_components=30, random_state=42):
    """
    Compute spectral embedding using SnapATAC2 and store
    the result directly in the AnnData object.

    Parameters
    ----------
    anndata : AnnData
        Preprocessed count matrix (cells x genomic bins).
    n_components : int
        Number of spectral embedding dimensions.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        AnnData with spectral embedding in obsm["X_spectral"].
    """
    
    # Convert sparse matrix to CSR format with float64 values
    anndata.X = anndata.X.tocsr().astype(np.float64)

    # Clean sparse matrix
    anndata.X.eliminate_zeros()
    anndata.X.sort_indices()
    
    snap.tl.spectral(
        anndata,
        n_comps=n_components,
        features=None,
        random_state=random_state,
        sample_size=None,
        distance_metric="cosine",
        weighted_by_sd=True,
        feature_weights=None,
        inplace=True
    )

    print(
        f"Spectral embedding shape: "
        f"{anndata.obsm['X_spectral'].shape}"
    )

    return anndata