"""
Main experiment runner for scFM_eval.
Handles configuration, data loading, preprocessing, feature extraction, model training, and evaluation.
"""

import os
import sys
import shutil
import logging
import random
import time
from functools import wraps
from pathlib import Path
from os.path import dirname, abspath, join, basename, exists

import yaml
import numpy as np
import torch
import pandas as pd

from viz.visualization import EmbeddingVisualizer
from evaluation.eval import EmbeddingEvaluator
from utils.logs_ import set_logging, get_logger
from setup_path import BASE_PATH, OUTPUT_PATH, PARAMS_PATH, DATA_PATH

# Mapping of embedding method names to their corresponding keys in AnnData
embedding_method_map = dict(PCA='X_pca', HVG='X_hvg', scVI='X_scVI', geneformer='X_geneformer', scgpt='X_scGPT')

# List to store timing records
def _get_timing_log():
    """Return the global timing log list."""
    global _timing_log
    if '_timing_log' not in globals():
        _timing_log = []
    return _timing_log

_timing_log = _get_timing_log()

def timing(func):
    """
    Decorator to measure and store execution time of a function.
    Appends timing info to the global _timing_log list.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        _timing_log.append({
            'function': func.__name__,
            'time_seconds': elapsed
        })
        return res
    return wrapper

def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
        deterministic (bool): If True, sets PyTorch to deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")

def get_configs(config_path):
    """
    Load experiment configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Tuple containing run_id, data_config, qc_config, preproc_config, hvg_config, feat_config, classification_config.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    run_id = config['run_id']
    data_config = config['dataset']
    qc_config =  config['qc']
    preproc_config = config['preprocessing']
    feat_config = config['embedding']
    classification_config = config['classification']
    hvg_config = None
    if 'hvg' in config:
        hvg_config = config['hvg']
    return run_id, data_config, qc_config, preproc_config, hvg_config, feat_config, classification_config

class Experiment:
    """
    Handles the execution of a machine learning experiment defined by a YAML config.
    Handles loading data, preprocessing, feature extraction, model training, and evaluation.
    """
    def __init__(self, config_path):
        """
        Initialize the Experiment with a given config path.
        Sets up directories, logging, and loads configuration.
        """
        self.config_path = join(PARAMS_PATH, config_path)
        self.run_id, self.data_config, self.qc_config, self.preproc_config, self.hvg, self.feat_config, self.classification_config = get_configs(self.config_path)

        self.vis_embedding = bool(self.feat_config['viz'])
        self.eval_embedding = bool(self.feat_config['eval'])

        # Prepare saving dir
        relative_sav_dir = os.path.splitext(config_path)[0]
        config_filename = os.path.basename(config_path)
        save_dir = join(OUTPUT_PATH, relative_sav_dir)
        self.save_dir = save_dir + f'_{self.run_id}' if self.run_id else save_dir

        if not exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Copy config_file.yaml to the saving directory
        shutil.copyfile(self.config_path, join(self.save_dir, config_filename))

        # Set logging format
        self.log = set_logging(self.save_dir)

        # Placeholders for data, embeddings, model, and results
        self.data = None
        self.embedding = None
        self.embedding_key = None
        self.model = None
        self.results = {}

    @timing
    def run(self):
        """
        Run the complete experiment workflow: data loading, QC, preprocessing, feature extraction, visualization, evaluation, and classification.
        """
        self.load_data()
        self.qc_data()
        self.preprocess_data()
        if self.hvg:
            self.filter_hvg()
        self.extract_embeddings()
        if self.vis_embedding:
            self.visualize_embedding()
        if self.eval_embedding:
            self.evaluate_embedding()
        self.train_classifier()

    def load_class(self, module_path, class_name):
        """
        Dynamically import and return a class by module and name.

        Args:
            module_path (str): Python module path.
            class_name (str): Name of the class to import.

        Returns:
            type: The class object.
        """
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @timing
    def load_data(self):
        """
        Load the dataset using the loader specified in the config.
        """
        loader_config = self.data_config
        loader_config['path'] = join(DATA_PATH, loader_config['path'])
        LoaderClass = self.load_class(loader_config['module'], loader_config['class'])
        self.log.info(f'Data Loader config: {loader_config}')
        self.loader = LoaderClass(loader_config)
        self.data = self.loader.load()

    @timing
    def qc_data(self):
        """
        Apply quality control filters to the dataset.
        """
        qc_config = self.qc_config
        if ('skip' in qc_config) and (qc_config['skip'] is True):
            return
        self.loader.qc(**qc_config)

    @timing
    def preprocess_data(self):
        """
        Apply preprocessing steps such as normalization.
        """
        preproc_config = self.preproc_config
        if ('skip' in preproc_config) and (preproc_config['skip'] is True):
            return
        self.loader.scale(**preproc_config)

    @timing
    def filter_hvg(self):
        """
        Select High Variant Genes only, if specified in config.
        """
        hvg_config = self.hvg
        if ('skip' in hvg_config) and (hvg_config['skip'] is True):
            return
        self.loader.hvg(**hvg_config)

    @timing
    def extract_embeddings(self):
        """
        Extract features or embeddings from the dataset using the specified extractor.
        """
        self.log.info('Extract features or embeddings from the dataset')
        feat_config = self.feat_config
        if ('skip' in feat_config) and (feat_config['skip'] is True):
            return
        self.embedding_key = embedding_method_map[feat_config['method']]
        feat_config['params']['save_dir'] = self.save_dir
        ExtractorClass = self.load_class(feat_config['module'], feat_config['class'])
        extractor = ExtractorClass(feat_config)
        self.embedding = extractor.fit_transform(self.loader)
        self.loader.adata.write_h5ad(join(self.save_dir, 'data.h5ad'), compression='gzip')

    @timing
    def train_classifier(self):
        """
        Train the machine learning model as specified in the config.
        """
        if ('skip' in self.classification_config) and (self.classification_config['skip'] is True):
            return
        clf_config = self.classification_config
        clf_config['params']['save_dir'] = self.save_dir
        viz = bool(self.classification_config['viz'])
        eval_ = bool(self.classification_config['eval'])
        clf_config['viz'] = viz
        clf_config['eval'] = eval_
        clf_config['params']['embedding_col'] = self.embedding_key
        ClfClass = self.load_class(clf_config['module'], clf_config['class'])
        clf = ClfClass(clf_config)
        clf.train(self.loader)

    def evaluate_model(self):
        """
        Run additional evaluation modules as specified in the config (if any).
        """
        evaluations = self.config.get('evaluations', [])
        for eval_cfg in evaluations:
            EvaluatorClass = self.load_class(eval_cfg['module'], eval_cfg['class'])
            evaluator = EvaluatorClass(eval_cfg)
            result = evaluator.evaluate(self.model, self.embedding, self.data)
            self.results[eval_cfg['name']] = result
        print("Evaluation Results:", self.results)

    def visualize_embedding(self):
        """
        Generate and save embedding visualizations using EmbeddingVisualizer.
        """
        visualizer = EmbeddingVisualizer(self.embedding, self.loader.adata.obs, save_dir=self.save_dir)
        visualizer.plot()

    def evaluate_embedding(self):
        """
        Evaluate the quality of the learned embeddings using EmbeddingEvaluator.
        """
        evaluator = EmbeddingEvaluator(self.loader.adata, embedding_key=self.embedding_key, save_dir=self.save_dir)
        evaluator.evaluate()


def main():
    """
    Main entry point for running an experiment from the command line.
    Sets random seed, parses config path, runs experiment, and saves timing log.
    """
    set_random_seed(42)
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'experiment.yaml'
    experiment = Experiment(config_path)
    experiment.run()
    timing_df = pd.DataFrame(_timing_log)
    timing_df.to_csv(join(experiment.save_dir, 'timing.csv'))

if __name__ == '__main__':
    main()