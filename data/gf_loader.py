import os
import pickle
from os.path import join
from data.data_loader import H5ADLoader


def get_gene_name_ensembl_map(dict_dir):

    ENSEMBL_DICTIONARY_FILE = join(dict_dir, 'gene_name_id_dict.pkl')

    def invert_dict(dict_obj):
        return {v: k for k, v in dict_obj.items()}

    with open(ENSEMBL_DICTIONARY_FILE, "rb") as f:
        gene_to_ensembl_dict  = pickle.load(f)
        ensembl_to_gene_dict = invert_dict(gene_to_ensembl_dict)

    return ensembl_to_gene_dict, gene_to_ensembl_dict

class GFLoader(H5ADLoader):
    def __init__(self, params):
        super().__init__(params)
        self.log.info(f'H5ADLoader {params}')
        
    def map_ensembl(self, gene_to_ensembl_dict ):
        # get mapping dictionaries [gene names <> ensembl ids]
        if type(gene_to_ensembl_dict) =='str': # you need to load a dict from path
            ensembl_to_gene_dict, gene_to_ensembl_dict = get_gene_name_ensembl_map(gene_to_ensembl_dict)
        else: # dict is already loaded for you
            gene_to_ensembl_dict = gene_to_ensembl_dict
        
        #convert gene name to ensembl ids
        self.adata.var['ensembl_id'] = self.adata.var.index.map(gene_to_ensembl_dict)
        nan_idx = self.adata.var.ensembl_id.isna()
        self.adata = self.adata[:,~nan_idx]
        n = sum(nan_idx)
        self.log.warning(f'warning: genes dont have ensembl IDs {n}. Genes without ensembl ID are REMOVED')
        self.log.info(self.adata.var.head())
        self.log.info(self.adata.shape)


    def prepare_data( self, save_dir = None, save_ext = "loom"):
        '''
        save adata in loom | h5ad formats. This saved file will be loaded again by the geneformer tokenizer

        :param processed_dir: directory used to save generated loom or H5ad file
        :param save_ext: extension used ot save processed data {loom | h5ad}
        :return: None
        '''
        self.processed_dir = join(save_dir, 'processed_data')
        self.adata.obs['n_counts'] = self.adata.obs['total_counts']
        self.adata.obs['adata_order'] = self.adata.obs.index.tolist()
        self.log.info(self.adata.shape)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        if save_ext == "loom":
            self.procssed_file = os.path.join(self.processed_dir,  f"{self.dataset_name}.loom")
            self.adata.write_loom(self.procssed_file)
            self.log.info(f'saving loom file to {self.procssed_file}')
        elif save_ext == "h5ad":
            self.procssed_file = os.path.join(self.processed_dir, f"{self.dataset_name}.h5ad")
            self.adata.write_h5ad(self.procssed_file)
            self.log.info(f'saving h5ad file to {self.procssed_file}')


