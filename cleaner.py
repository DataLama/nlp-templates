import shutil
from glob import glob

# remove all .ipynb_checkpoints in subdirectories.
for path in list(glob('**/.ipynb_checkpoints', recursive=True)):
    shutil.rmtree(path)
    
for path in list(glob('**/__pycache__', recursive=True)):
    shutil.rmtree(path)