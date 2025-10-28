# #!/bin/sh
# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 

# --------------------------------LUAD1---------------------------------------------

python run_exp.py luad1/hvg.yaml
python run_exp.py luad1/pca.yaml
python run_exp.py luad1/scgpt.yaml
python run_exp.py luad1/scgpt_cancer.yaml
python run_exp.py luad1/scvi.yaml


# --------------------------------LUAD2---------------------------------------------

python run_exp.py luad2/hvg.yaml
python run_exp.py luad2/pca.yaml
python run_exp.py luad2/scvi.yaml
python run_exp.py luad2/scgpt.yaml
python run_exp.py luad2/scgpt_cancer.yaml

