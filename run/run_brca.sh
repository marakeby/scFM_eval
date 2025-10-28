# #!/bin/sh
# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 

#--------------------------------Pre vs Post ---------------------------------------------

python run_exp.py brca_full/pre_post/hvg.yaml
python run_exp.py brca_full/pre_post/pca.yaml
python run_exp.py brca_full/pre_post/scvi.yaml
python run_exp.py brca_full/pre_post/scgpt.yaml
python run_exp.py brca_full/pre_post/scgpt_cancer.yaml


#--------------------------------subtype (ER+ vs TNBC)---------------------------------------------

python run_exp.py brca_full/subtype/hvg.yaml
python run_exp.py brca_full/subtype/pca.yaml
python run_exp.py brca_full/subtype/scvi.yaml
python run_exp.py brca_full/subtype/scgpt.yaml
python run_exp.py brca_full/subtype/scgpt_cancer.yaml


#--------------------------------outcome E vs NE (Tcells)---------------------------------------------

python run_exp.py brca_full/outcome/hvg.yaml
python run_exp.py brca_full/outcome/pca.yaml
python run_exp.py brca_full/outcome/scvi.yaml
python run_exp.py brca_full/outcome/scgpt.yaml
python run_exp.py brca_full/outcome/scgpt_cancer.yaml


#--------------------------------cohort chemo vs naive( Cancer cells)---------------------------------------------

python run_exp.py brca_full/chemo/hvg.yaml
python run_exp.py brca_full/chemo/pca.yaml
python run_exp.py brca_full/chemo/scvi.yaml
python run_exp.py brca_full/chemo/scgpt.yaml
python run_exp.py brca_full/chemo/scgpt_cancer.yaml


#-------------------------------- cell types ---------------------------------------------
#go from here
python run_exp.py brca_full/cell_type/hvg.yaml
python run_exp.py brca_full/cell_type/pca.yaml
python run_exp.py brca_full/cell_type/scvi.yaml
python run_exp.py brca_full/cell_type/scgpt.yaml
python run_exp.py brca_full/cell_type/scgpt_cancer.yaml

#-------------------------------- ALL Cells ---------------------------------------------

python run_exp.py brca_full/all_cells/hvg.yaml
python run_exp.py brca_full/all_cells/pca.yaml
python run_exp.py brca_full/all_cells/scvi.yaml
python run_exp.py brca_full/all_cells/scgpt.yaml
python run_exp.py brca_full/all_cells/scgpt_cancer.yaml

