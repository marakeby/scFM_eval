from os.path import join, realpath, dirname

BASE_PATH = dirname(realpath(__file__))
PARAMS_PATH = join(BASE_PATH, 'yaml')
RUN_PATH = join(BASE_PATH, 'run')
DATA_PATH = '/home/jupyter'
OUTPUT_PATH = '/home/jupyter/mnt/__output_clean'#join(BASE_PATH, '__output')