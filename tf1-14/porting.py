"""
You can Porting trained model 2 ways.

1) model_farm

- model_farm is the directory where you can stack your trained models with it's dependencies.(index, tokenizere, function etc)

- You can analyze the models and HARVEST(select) the best model from the farm.

>>> python porting.py --porting 0 --model_path experiments # porting all the models from experiments

>>> python porting.py --porting 0 --model_path experiments/base_model # porting one model from experiments

2) serve

- After find your best model in the farm, you can port the model for deployment.

>>> python porting.py --porting 1 --model_path experiments/best_model

"""
import os
import shutil
import argparse
import subprocess

def _mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok = True)

def main():
    #### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--porting', default = '0', help = "Porting to model_farm : 0 / Porting to serve : 1")
    parser.add_argument('--model_path', default = 'experiments/base_model', help = "Directory of the saved_model which is going to be serving.")
    parser.add_argument('--serve_path', default = '../serve', help = "Directory where the serving(deploying) code is created.")
    args = parser.parse_args()
    
    #### no
    
    
    model_path = args.model_path
    serve_path = f'{args.serve_path}/model'
    
    _mkdir(serve_path)
    
    subprocess.run([f'cp -r {model_path}/saved_model {serve_path}'], shell = True)    # tensorflow serving model.
    subprocess.run([f'cp -r utils {serve_path}'], shell = True)                       # tokenizer and basic utilities.
    subprocess.run([f'cp {model_path}/hyper_params.json {serve_path}'], shell = True) # hyperparameter of the model.
    subprocess.run([f'cp data/config.json {serve_path}'], shell = True)               # index of token and label.
    
if __name__=="__main__":
    main()