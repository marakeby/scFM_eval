import os
import scanpy as sc
import numpy as np
from scimilarity.cell_embedding import CellEmbedding
from scimilarity.utils import align_dataset, lognorm_counts
from utils.logs_ import get_logger
from features.extractor import EmbeddingExtractor

class SCimilarityExtractor(EmbeddingExtractor):
    """
    Extracts single-cell embeddings using SCimilarity.

    Usage:
        extractor = SCimilarityExtractor(model_path="path/to/model_dir", use_gpu=False)
        adata = extractor.preprocess_h5ad("path/to/data.h5ad")
        embeddings = extractor.get_embeddings(adata)
    """

    def __init__(self, params):
        super().__init__(params)
        self.log = get_logger()
        self.log.info(f'SCimilarityExtractor ({self.params})')
        

        model_path = self.params['model_path']
        use_gpu = self.params.get('use_gpu', False) 
       
        self.ce = CellEmbedding(model_path=model_path, use_gpu=use_gpu)

    def preprocess_h5ad(self, h5ad_path_data , reorder: bool = True, norm: bool = True):
        """
        Loads and preprocesses AnnData according to SCimilarity requirements:
          - Aligns gene ordering
          - Applies log-normalization (TP10K)

        Returns:
            AnnData object with processed .X
        """
        if type(h5ad_path_data) ==str:
            adata = sc.read_h5ad(h5ad_path_data)
        else:
            adata = h5ad_path_data

        if reorder:
            adata = align_dataset(adata, self.ce.gene_order)
        if norm:
            adata = lognorm_counts(adata)

        return adata

    def get_embeddings(self, adata, num_cells: int = -1, buffer_size: int = 10000):
        """
        Computes embeddings using the SCimilarity model.

        Returns:
            numpy.ndarray of shape [num_cells, latent_dim]
        """
        X = adata.X
        embeddings = self.ce.get_embeddings(X, num_cells=num_cells, buffer_size=buffer_size)
        return embeddings
    
    def fit_transform(self, data_loader):
        self.data_loader = data_loader
        adata = data_loader.adata
        adata = self.preprocess_h5ad( adata , reorder = True, norm = True)
        embeddings = self.get_embeddings(adata, num_cells = -1, buffer_size = 10000)
        self.data_loader.adata.obsm['X_scimilarity'] = embeddings
            
        return embeddings

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract embeddings using SCimilarity")
    parser.add_argument("--h5ad", required=True, help="Path to .h5ad input file")
    parser.add_argument("--model", required=True, help="Path to SCimilarity model directory")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--output", default="scimilarity_embeddings.npy", help="Output numpy file")
    args = parser.parse_args()

    extractor = SCimilarityExtractor(model_path=args.model, use_gpu=args.use_gpu)
    adata = extractor.preprocess_h5ad(args.h5ad)
    embeddings = extractor.get_embeddings(adata)
    np.save(args.output, embeddings)
    print(f"Saved embeddings of shape {embeddings.shape} to {args.output}")