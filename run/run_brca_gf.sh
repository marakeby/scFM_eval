# #!/bin/sh
# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 

# --------------------------------cohort chemo vs naive( Cancer cells)--------------------------------------------

python run_exp.py brca_full/chemo/gf-6L-30M-i2048.yaml
python run_exp.py brca_full/chemo/Geneformer-V2-104M.yaml
python run_exp.py brca_full/chemo/Geneformer-V2-104M_CLcancer.yaml
python run_exp.py brca_full/chemo/Geneformer-V2-316M.yaml

python run_exp.py brca_full/chemo/scfoundation.yaml
python run_exp.py brca_full/chemo/scimilarity.yaml
python run_exp.py brca_full/chemo/cellplm.yaml

# #finetune
python run_exp.py brca_full/chemo/gf-6L-30M-i2048_finetune.yaml
python run_exp.py brca_full/chemo/Geneformer-V2-104M_finetune.yaml


# --------------------------------Pre vs Post ---------------------------------------------

python run_exp.py brca_full/pre_post/gf-6L-30M-i2048.yaml
python run_exp.py brca_full/pre_post/Geneformer-V2-104M.yaml
python run_exp.py brca_full/pre_post/Geneformer-V2-104M_CLcancer.yaml
python run_exp.py brca_full/pre_post/Geneformer-V2-316M.yaml

python run_exp.py brca_full/pre_post/scfoundation.yaml
python run_exp.py brca_full/pre_post/scimilarity.yaml
python run_exp.py brca_full/pre_post/cellplm.yaml

# finetune
python run_exp.py brca_full/pre_post/gf-6L-30M-i2048_finetune.yaml
python run_exp.py brca_full/pre_post/Geneformer-V2-104M_finetune.yaml


# --------------------------------outcome E vs NE (Tcells)---------------------------------------------


python run_exp.py brca_full/outcome/gf-6L-30M-i2048.yaml
python run_exp.py brca_full/outcome/Geneformer-V2-104M.yaml
python run_exp.py brca_full/outcome/Geneformer-V2-104M_CLcancer.yaml
python run_exp.py brca_full/outcome/Geneformer-V2-316M.yaml

python run_exp.py brca_full/outcome/scfoundation.yaml
python run_exp.py brca_full/outcome/scimilarity.yaml
python run_exp.py brca_full/outcome/cellplm.yaml

# finetune
python run_exp.py brca_full/outcome/gf-6L-30M-i2048_finetune.yaml
python run_exp.py brca_full/outcome/Geneformer-V2-104M_finetune.yaml


# --------------------------------subtype (ER+ vs TNBC)---------------------------------------------


python run_exp.py brca_full/subtype/gf-6L-30M-i2048.yaml
python run_exp.py brca_full/subtype/Geneformer-V2-104M.yaml
python run_exp.py brca_full/subtype/Geneformer-V2-104M_CLcancer.yaml
python run_exp.py brca_full/subtype/Geneformer-V2-316M.yaml

python run_exp.py brca_full/subtype/scfoundation.yaml
python run_exp.py brca_full/subtype/scimilarity.yaml
python run_exp.py brca_full/subtype/cellplm.yaml

# # finetune
python run_exp.py brca_full/outcome/gf-6L-30M-i2048_finetune.yaml
python run_exp.py brca_full/outcome/Geneformer-V2-104M_finetune.yaml



# -------------------------------- cell types ---------------------------------------------

python run_exp.py brca_full/cell_type/hvg.yaml
python run_exp.py brca_full/cell_type/pca.yaml
python run_exp.py brca_full/cell_type/scvi.yaml

python run_exp.py brca_full/cell_type/scgpt.yaml
python run_exp.py brca_full/cell_type/scgpt_cancer.yaml

python run_exp.py brca_full/cell_type/gf-6L-30M-i2048.yaml
python run_exp.py brca_full/cell_type/Geneformer-V2-104M.yaml
python run_exp.py brca_full/cell_type/Geneformer-V2-104M_CLcancer.yaml
python run_exp.py brca_full/cell_type/Geneformer-V2-316M.yaml


python run_exp.py brca_full/cell_type/scfoundation.yaml
python run_exp.py brca_full/cell_type/scimilarity.yaml
python run_exp.py brca_full/cell_type/cellplm.yaml

# # Continual Training
python run_exp.py brca_full/cell_type/gf-6L-30M-i2048_continue.yaml
python run_exp.py brca_full/cell_type/Geneformer-V2-104M_continue.yaml
