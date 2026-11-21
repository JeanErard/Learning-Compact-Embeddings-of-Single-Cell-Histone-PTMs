from dim_reduction import dim_red_pipeline


def add_embeddings_LSI(H3K27ac_ad, H3K27me3_ad):

    H3K27ac_ad = dim_red_pipeline(
        H3K27ac_ad,
        increase_bin=True,
        filtering=True,
        nbre_cells=50,
        factor=100,
        n_components=50
    )

    H3K27me3_ad = dim_red_pipeline(
        H3K27me3_ad,
        increase_bin=True,
        filtering=True,
        nbre_cells=50,
        factor=100,
        n_components=50
    )

    return H3K27ac_ad, H3K27me3_ad