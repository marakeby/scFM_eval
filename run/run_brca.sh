# #!/bin/sh
# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 
#-------------------------------- LUAD 2 ---------------------------------------------
# python run_exp.py luad2_a100/gf-6L-30M-i2048_finetune.yaml

#--------------------------------Pre vs Post ---------------------------------------------

# python run_exp.py brca_full/pre_post/hvg.yaml
# python run_exp.py brca_full/pre_post/pca.yaml
# python run_exp.py brca_full/pre_post/scvi.yaml
# python run_exp.py brca_full/pre_post/scgpt.yaml
# python run_exp.py brca_full/pre_post/scgpt_cancer.yaml

# python run_exp.py brca_full/pre_post/gf-6L-30M-i2048.yaml
# python run_exp.py brca_full/pre_post/Geneformer-V2-104M.yaml
# python run_exp.py brca_full/pre_post/Geneformer-V2-104M_CLcancer.yaml
# python run_exp.py brca_full/pre_post/Geneformer-V2-316M.yaml


#finetune

# python run_exp.py brca_full/pre_post/gf-6L-30M-i2048_finetune.yaml
# python run_exp.py brca_full/pre_post/Geneformer-V2-104M_finetune.yaml


#continual Training
# python run_exp.py brca_full/pre_post/gf-6L-30M-i2048_continue.yaml

#go from here
# python run_exp.py brca_full/pre_post/Geneformer-V2-104M_continue.yaml
# python run_exp.py brca_full/pre_post/Geneformer-V2-104M_CLcancer_continue.yaml
# python run_exp.py brca_full/pre_post/Geneformer-V2-316M_continue.yaml

# python run_exp.py brca_full/pre_post/gf-6L-30M-i2048_4k.yaml
# python run_exp.py brca_full/pre_post/gf-6L-30M-i2048_8k.yaml
# python run_exp.py brca_full/pre_post/gf-6L-30M-i2048_10k.yaml
# python run_exp.py brca_full/pre_post/gf-6L-30M-i2048_all.yaml
#--------------------------------subtype (ER+ vs TNBC)---------------------------------------------

# python run_exp.py brca_full/subtype/hvg.yaml
# python run_exp.py brca_full/subtype/pca.yaml
# python run_exp.py brca_full/subtype/scvi.yaml
# python run_exp.py brca_full/subtype/scgpt.yaml
# python run_exp.py brca_full/subtype/scgpt_cancer.yaml

# python run_exp.py brca_full/subtype/gf-6L-30M-i2048.yaml
# python run_exp.py brca_full/subtype/Geneformer-V2-104M.yaml
# python run_exp.py brca_full/subtype/Geneformer-V2-104M_CLcancer.yaml
# python run_exp.py brca_full/subtype/Geneformer-V2-316M.yaml

#finetune
# python run_exp.py brca_full/subtype/gf-6L-30M-i2048_finetune.yaml
# python run_exp.py brca_full/subtype/Geneformer-V2-104M_finetune.yaml

#go from here

#continual Training
# python run_exp.py brca_full/subtype/gf-6L-30M-i2048_continue.yaml


# number of genes
# python run_exp.py brca_full/subtype/gf-6L-30M-i2048_4k.yaml
# python run_exp.py brca_full/subtype/gf-6L-30M-i2048_8k.yaml
# python run_exp.py brca_full/subtype/gf-6L-30M-i2048_10k.yaml
# python run_exp.py brca_full/subtype/gf-6L-30M-i2048_all.yaml


#--------------------------------outcome E vs NE (Tcells)---------------------------------------------

# python run_exp.py brca_full/outcome/hvg.yaml
# python run_exp.py brca_full/outcome/pca.yaml
# python run_exp.py brca_full/outcome/scvi.yaml
# python run_exp.py brca_full/outcome/scgpt.yaml
# python run_exp.py brca_full/outcome/scgpt_cancer.yaml


# python run_exp.py brca_full/outcome/gf-6L-30M-i2048.yaml
# python run_exp.py brca_full/outcome/Geneformer-V2-104M.yaml
# python run_exp.py brca_full/outcome/Geneformer-V2-104M_CLcancer.yaml
# python run_exp.py brca_full/outcome/Geneformer-V2-316M.yaml

#finetune
# python run_exp.py brca_full/outcome/gf-6L-30M-i2048_finetune.yaml
# python run_exp.py brca_full/outcome/Geneformer-V2-104M_finetune.yaml

#go from here

#continual Training
# python run_exp.py brca_full/outcome/gf-6L-30M-i2048_continue.yaml

#--------------------------------cohort chemo vs naive(Tcells)---------------------------------------------

# python run_exp.py brca_full/chemo/hvg.yaml
# python run_exp.py brca_full/chemo/pca.yaml
# python run_exp.py brca_full/chemo/scvi.yaml
# python run_exp.py brca_full/chemo/scgpt.yaml
# python run_exp.py brca_full/chemo/scgpt_cancer.yaml

# python run_exp.py brca_full/chemo/gf-6L-30M-i2048.yaml
# python run_exp.py brca_full/chemo/Geneformer-V2-104M.yaml
# python run_exp.py brca_full/chemo/Geneformer-V2-104M_CLcancer.yaml
# python run_exp.py brca_full/chemo/Geneformer-V2-316M.yaml

#finetune
# python run_exp.py brca_full/chemo/gf-6L-30M-i2048_finetune.yaml
# python run_exp.py brca_full/chemo/Geneformer-V2-104M_finetune.yaml


#go from here

#continual Training
# python run_exp.py brca_full/chemo/gf-6L-30M-i2048_continue.yaml

#-------------------------------- cell types ---------------------------------------------
#go from here
# python run_exp.py brca_full/cell_type/hvg.yaml
# python run_exp.py brca_full/cell_type/pca.yaml
# python run_exp.py brca_full/cell_type/scvi.yaml
# python run_exp.py brca_full/cell_type/scgpt.yaml
# python run_exp.py brca_full/cell_type/scgpt_cancer.yaml

# python run_exp.py brca_full/cell_type/gf-6L-30M-i2048.yaml
# python run_exp.py brca_full/cell_type/Geneformer-V2-104M.yaml
# python run_exp.py brca_full/cell_type/Geneformer-V2-104M_CLcancer.yaml
# python run_exp.py brca_full/cell_type/Geneformer-V2-316M.yaml

# python run_exp.py brca_full/cell_type/gf-6L-30M-i2048_continue.yaml
# python run_exp.py brca_full/cell_type/Geneformer-V2-104M_continue.yaml


#-------------------------------- ALL Cells ---------------------------------------------

# python run_exp.py brca_full/all_cells/gf-6L-30M-i2048.yaml
# python run_exp.py brca_full/all_cells/Geneformer-V2-104M.yaml
# python run_exp.py brca_full/all_cells/Geneformer-V2-104M_CLcancer.yaml
# python run_exp.py brca_full/all_cells/Geneformer-V2-316M.yaml

# python run_exp.py brca_full/all_cells/gf-6L-30M-i2048_continue.yaml

#-------------------------------- Cancer Cells ---------------------------------------------
# python run_exp.py brca_full/cancer_cells/gf-6L-30M-i2048.yaml
# python run_exp.py brca_full/cancer_cells/Geneformer-V2-104M.yaml
# python run_exp.py brca_full/cancer_cells/Geneformer-V2-104M_CLcancer.yaml
# python run_exp.py brca_full/cancer_cells/Geneformer-V2-316M.yaml

#-------------------------------- T Cells ---------------------------------------------
# python run_exp.py brca_full/tcells/gf-6L-30M-i2048.yaml
# python run_exp.py brca_full/tcells/Geneformer-V2-104M.yaml
# python run_exp.py brca_full/tcells/Geneformer-V2-104M_CLcancer.yaml
# python run_exp.py brca_full/tcells/Geneformer-V2-316M.yaml
