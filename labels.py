import pandas as pd

# Neuron subtypes too rare to be predicted on their own are merged:
#   EXC1-4 (23 + 14 + 2 + 1 = 40 cells)  -> EXC
#   INH1-4 (197 + 46 + 3 + 1 = 247 cells) -> INH
# All other classes have >= 42 cells and are kept as they are.
DEFAULT_GROUPS = {
    "EXC1": "EXC", "EXC2": "EXC", "EXC3": "EXC", "EXC4": "EXC",
    "INH1": "INH", "INH2": "INH", "INH3": "INH", "INH4": "INH",
}


def group_cell_types(adata, groups=DEFAULT_GROUPS, label_key="cell_type",
                     new_key="cell_type_grouped", verbose=True):
    """Add adata.obs[new_key] with the grouped classes (the original column is kept)."""
    labels = adata.obs[label_key].astype(str)
    grouped = labels.replace(groups)
    adata.obs[new_key] = pd.Categorical(grouped)

    if verbose:
        counts = adata.obs[new_key].value_counts()
        print(f"{labels.nunique()} classes -> {len(counts)} classes "
              f"(smallest: {counts.idxmin()}, {counts.min()} cells)")
    return adata
