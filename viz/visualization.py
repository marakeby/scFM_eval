from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import anndata as ad
import scanpy as sc
import logging
logger = logging.getLogger('ml_logger')
from utils.logs_ import get_logger

from utils.sampling import sample_adata
logger = get_logger()

class EmbeddingVisualizer:
    """Class for visualizing embeddings with optional batch and label coloring."""
    def __init__(self, embedding, obs, save_dir=".", auto_subsample=True):
        
        self.auto_subsample=auto_subsample
        self.embedding = embedding
            
        self.obs = obs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot(self):
        adata = ad.AnnData(X=self.embedding)
        adata.obs = self.obs.copy()
        logger.info(f'Visualizing embeddings {adata.X.shape}')
        logger.info(f'Calc neighbors using X')

        if self.auto_subsample:
            if adata.shape[0]>10000:
                adata = sample_adata(adata, sample_size=5000, stratify_by=None)
    
            
        sc.pp.neighbors(adata, use_rep='X')
        logger.info(f'Calc UMAP')
        sc.tl.umap(adata)

        if 'batch' in adata.obs.columns:
            embeddings_fig = sc.pl.umap(adata, color='batch', show=False, wspace=0.4, frameon=False, return_fig=True)
            embeddings_fig.savefig(self.save_dir / 'embedding_batch.png', dpi=200, bbox_inches='tight')
            logger.info("Saved UMAP plot colored by batch to 'figures/umap_batch.png'")

        if 'label' in adata.obs.columns:
            embeddings_fig = sc.pl.umap(adata, color='label', show=False, wspace=0.4, frameon=False, return_fig=True)
            embeddings_fig.savefig(self.save_dir / 'embedding_label.png', dpi=200, bbox_inches='tight')
            logger.info("Saved UMAP plot colored by label to 'figures/umap_label.png'")

