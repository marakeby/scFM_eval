"""
data_loader.py

Data loading utilities for single-cell data, supporting CSV and H5AD formats with preprocessing and filtering.
"""

import os
import json
import logging

import pandas as pd
import anndata as ad
import scanpy as sc

from utils.logs_ import get_logger
from setup_path import BASE_PATH
import numpy as np

class DataLoader:
    """Base class for loading and preprocessing single-cell data."""
    def __init__(self, params):
        """
        Args:
            params (dict): Configuration parameters for data loading.
        """
        self.params = params
        self.path = params['path']
        self.dataset_name = os.path.basename(self.path).split(".")[0]
        self.layer = params['layer_name']
        self.label_key = params['label_key']
        self.batch_key = params['batch_key']
        self.train_test_split = params['train_test_split']
        self.cv_splits = params['cv_splits']
        self.log = get_logger()
        self.adata = None

    @staticmethod
    def validate_config(params):
        """Validate that required parameters are present.

        Args:
            params (dict): Configuration parameters.
        Raises:
            AssertionError: If required parameters are missing.
        """
        assert 'path' in params, "Missing required parameter: 'path'"

    def prepare_data(self, process_dir):
        """Prepare data for downstream analysis. To be implemented in subclasses.

        Args:
            process_dir (str): Directory for processing outputs.
        """
        pass

    def load(self):
        """Load data and return AnnData object. To be implemented in subclasses.

        Returns:
            ad.AnnData: Loaded data object.
        """
        print(f"Loading data from {self.path}")
        adata = ad.AnnData()
        self.adata = adata
        self.log.info(f'Data Loaded, {self.adata.X.shape}')
        self.log.info(f'min {np.min(self.adata.X)}, max {np.max(self.adata.X)}')
        return adata

    def _filter(self, adata, filter_dict):
        """Filter AnnData object based on filter_dict.

        Args:
            adata (ad.AnnData): AnnData object to filter.
            filter_dict (dict): Dictionary of {column: [values]} to filter by.
        Returns:
            ad.AnnData: Filtered AnnData object.
        """
        for col, values in filter_dict.items():
            adata = adata[adata.obs[col].isin(values)]
        return adata

    def qc(self, min_genes, min_cells):
        """Apply quality control filters to the data.

        Args:
            min_genes (int): Minimum number of genes per cell.
            min_cells (int): Minimum number of cells per gene.
        """
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.log.info(f"Applying QC with min_genes={self.min_genes}, min_cells={self.min_cells}")
        self.log.info(f'obs shape before filtering, {self.adata.X.shape}')

        sc.pp.calculate_qc_metrics(self.adata, percent_top=None, log1p=False, inplace=True)
        sc.pp.filter_cells(self.adata, min_genes=int(self.min_genes))
        sc.pp.filter_genes(self.adata, min_cells=int(self.min_cells))
        self.log.info(f'obs shape after filtering, {self.adata.X.shape}')

    def scale(self, normalize, target_sum, apply_log1p):
        """Normalize and/or log-transform the data.

        Args:
            normalize (bool): Whether to normalize total counts per cell.
            target_sum (float): Target sum for normalization.
            apply_log1p (bool): Whether to apply log1p transformation.
        """
        
        # if layer_key == "X":
        #     adata.layers["counts"] = adata.X
        # elif layer_key != "counts":
        #     adata.layers["counts"] = adata.layers[layer_key]
    
        self.normalize = normalize
        self.target_sum = target_sum
        self.apply_log1p = apply_log1p

        if self.normalize:
            sc.pp.normalize_total(self.adata, target_sum=self.target_sum)
        if self.apply_log1p:
            sc.pp.log1p(self.adata)
        self.log.info(f"Applied LogScalerPreprocessor normalization to data.X, apply_log1p = {self.apply_log1p}, target_sum = {self.target_sum}")
        self.log.info(f'obs shape after scaling, {self.adata.X.shape}')

    def hvg(self, n_top_genes, flavor, batch_key=None):
        """Select highly variable genes (HVGs).

        Args:
            n_top_genes (int): Number of top HVGs to select.
            flavor (str): Method for HVG selection (e.g., 'seurat').
            batch_key (str, optional): Batch key for batch-aware HVG selection.
        """
        sc.pp.highly_variable_genes(self.adata, batch_key=batch_key, flavor=flavor, subset=True, n_top_genes=n_top_genes)
        self.log.info(f"Applied HVG to data.X, n_top_genes = {n_top_genes}, flavor = {flavor}, batch_key = {batch_key}")
        self.log.info(f'Data shape after HVG, {self.adata.X.shape}')


class CSVDataLoader(DataLoader):
    """Loader for CSV-formatted single-cell data."""
    def __init__(self, params):
        """
        Args:
            params (dict): Configuration parameters for CSV loading.
        """
        super().__init__(params)
        self.path = params['path']

    def load(self):
        """Load data from a CSV file.

        Returns:
            ad.AnnData: Loaded data object.
        """
        self.log.info(f"Loading CSV data from {self.path}")
        df = pd.read_csv(self.path)
        labels = df['label'].values
        features = df.drop(columns=['label']).values
        adata = ad.AnnData(X=features, obs=pd.DataFrame({'label': labels}))
        self.adata = adata
        return adata


class H5ADLoader(DataLoader):
    """Loader for H5AD-formatted single-cell data."""
    def __init__(self, params):
        """
        Args:
            params (dict): Configuration parameters for H5AD loading.
        """
        super().__init__(params)
        self.path = params['path']
        self.load_raw = bool(params['load_raw'])
        self.label_key = params['label_key']
        self.batch_key = params['batch_key']
        self.filter = params.get('filter', None)

    def load(self):
        """Load data from an H5AD file, with optional filtering and splitting.

        Returns:
            ad.AnnData: Loaded data object.
        """
        logging.info(f"Loading H5AD data from {self.path}")
        adata = ad.read_h5ad(self.path)
        adata.obs['label'] = adata.obs[self.label_key]
        adata.obs['batch'] = adata.obs[self.batch_key]
        if self.load_raw:
            adata = adata.raw.to_adata()
        if self.layer != 'X':
            adata.layers['original_X'] = adata.X
            adata.X = adata.layers[self.layer]
            
        # if self.layer == "X" and 'counts' not in adata.layers:
        #     adata.layers["counts"] = adata.X
        
        self.log.info(f'Data Loaded, {adata.X.shape}')
        self.log.info(f'X min {np.min(adata.X)}, X max {np.max(adata.X)}')

        if self.filter is not None:
            self.log.info(self.filter)
            filter_dict = {entry["column"]: entry["values"] for entry in self.filter}
            adata = self._filter(adata, filter_dict)

        self.adata = adata

        if self.train_test_split:
            fname = os.path.join(BASE_PATH, self.train_test_split)
            self.train_test_split_dict = json.load(open(fname))
            fname = os.path.join(BASE_PATH, self.cv_splits)
            self.cv_split_dict = json.load(open(fname))

        return adata


