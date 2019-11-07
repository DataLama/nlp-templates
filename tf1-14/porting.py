"""
Porting the model that you gonna serve.

Just pass the dir of your model and destination.

>>> model_porting('experiments/base_model', '../serve')

"""
import os
import shutil
import argparse
import subprocess

def _mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok = True)

def main(args):
    model_path = args.model_path
    serve_path = f'{args.serve_path}/model'
    
    _mkdir(serve_path)
    
    subprocess.run([f'cp -r {model_path}/saved_model {serve_path}'], shell = True)    # tensorflow serving model.
    subprocess.run([f'cp -r utils {serve_path}'], shell = True)                       # tokenizer and basic utilities.
    subprocess.run([f'cp {model_path}/hyper_params.json {serve_path}'], shell = True) # hyperparameter of the model.
    subprocess.run([f'cp data/config.json {serve_path}'], shell = True)               # index of token and label.
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='experiments/base_model', help="Directory of the saved_model which is going to be serving.")
    parser.add_argument('--serve_path', default='../serve', help="Directory where the serving(deploying) code is created.")
    args = parser.parse_args()
    main(args)