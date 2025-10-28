# #!/bin/sh
# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 

# --------------------------------LUAD1---------------------------------------------

#GF
python run_exp.py luad1/gf-6L-30M-i2048.yaml
python run_exp.py luad1/gf-6L-30M-i2048_no_batch.yaml
python run_exp.py luad1/Geneformer-V2-104M.yaml
python run_exp.py luad1/Geneformer-V2-104M_CLcancer.yaml
python run_exp.py luad1/Geneformer-V2-316M.yaml

#Other
python run_exp.py luad1/scfoundation.yaml
python run_exp.py luad1/scimilarity.yaml
python run_exp.py luad1/cellplm.yaml


# finetune
python run_exp.py luad1/gf-6L-30M-i2048_finetune.yaml
python run_exp.py luad1/Geneformer-V2-104M_finetune.yaml




# --------------------------------LUAD2---------------------------------------------

python run_exp.py luad2/gf-6L-30M-i2048.yaml
python run_exp.py luad2/Geneformer-V2-104M.yaml
python run_exp.py luad2/Geneformer-V2-104M_CLcancer.yaml
python run_exp.py luad2/Geneformer-V2-316M.yaml

#Other
python run_exp.py luad2/scfoundation.yaml
python run_exp.py luad2/scimilarity.yaml
python run_exp.py luad2/cellplm.yaml

# finetune
python run_exp.py luad2/gf-6L-30M-i2048_finetune.yaml
python run_exp.py luad2/Geneformer-V2-104M_finetune.yaml