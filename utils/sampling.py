import anndata as ad
import numpy as np
from sklearn.model_selection import train_test_split

def sample_adata(
    adata: ad.AnnData, 
    sample_size: int = 2000, 
    stratify_by: str = None,
    random_state: int = 42
) -> ad.AnnData:
    """
    Sample a subset of cells from an AnnData object.

    Args:
        adata (AnnData): The full AnnData object to sample from.
        sample_size (int): Number of cells to sample.
        stratify_by (str, optional): Column name in adata.obs to perform stratified sampling.
        random_state (int): Seed for reproducibility.

    Returns:
        AnnData: A sampled AnnData object.
    """
    n_cells = adata.n_obs
    sample_size = min(sample_size, n_cells)
    rng = np.random.default_rng(random_state)

    if stratify_by and stratify_by in adata.obs.columns:
        from sklearn.model_selection import train_test_split
        stratify_labels = adata.obs[stratify_by]
        sample_idx, _ = train_test_split(
            np.arange(n_cells),
            train_size=sample_size,
            stratify=stratify_labels,
            random_state=random_state
        )
    else:
        sample_idx = rng.choice(n_cells, size=sample_size, replace=False)

    return adata[sample_idx].copy()