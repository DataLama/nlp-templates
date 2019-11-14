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
import numpy as np

def _mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok = True)
    
def _port2model_farm(sin_model_path):
    model_farm_path = '/root/.model_farm'
    model_name = sin_model_path.split('/')[-1]
    
    if not os.path.exists(model_farm_path):
        _mkdir({model_farm_path})
 
    while True:
        if os.path.exists(f'{model_farm_path}/{model_name}'):
            model_name = f'{model_name}-{np.random.randint(1000,10000)}'
            continue
        else:
            break
            
    _mkdir(f'{model_farm_path}/{model_name}')
    
    subprocess.run([f'cp {sin_model_path}/best_model_cpu.h5 {model_farm_path}/{model_name}'], shell = True)    # best_cpu_model.h5
    subprocess.run([f'cp {sin_model_path}/history.json {model_farm_path}/{model_name}'], shell = True)         # history.json file
    subprocess.run([f'cp {sin_model_path}/hyper_params.json {model_farm_path}/{model_name}'], shell = True)    # hyper_params.json
    subprocess.run([f'cp -r {sin_model_path}/checkpoints {model_farm_path}/{model_name}'], shell = True)       # checkpoints
    subprocess.run([f'cp data/config.json {model_farm_path}/{model_name}'], shell = True)                      # index of token and label.
    subprocess.run([f'cp -r utils {model_farm_path}/{model_name}'], shell = True)                              # utils

def main():
    #### argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--porting', default = '0', help = "Porting to model_farm : 0 / Porting to serve : 1")
    parser.add_argument('--model_path', default = 'experiments', help = "Full models in experiments or directory of the best model.")
    args = parser.parse_args()
    
    #### 
    if int(args.porting) == 0:
        if args.model_path == 'experiments':
            for mn in os.listdir(args.model_path):
                if os.path.exists(f'{args.model_path}/{mn}/history.json'):
                    _port2model_farm(f'{args.model_path}/{mn}')
        else:
            _port2model_farm(args.model_path)

        
        
    elif int(args.porting) == 1:
        serve_path = '../serve/model'
        model_path = args.model_path
        
        _mkdir(serve_path)
        
        subprocess.run([f'cp -r {model_path}/saved_model {serve_path}'], shell = True)    # tensorflow serving model.
        subprocess.run([f'cp -r utils {serve_path}'], shell = True)                       # tokenizer and basic utilities.
        subprocess.run([f'cp {model_path}/hyper_params.json {serve_path}'], shell = True) # hyperparameter of the model.
        subprocess.run([f'cp data/config.json {serve_path}'], shell = True)               # index of token and label.
        
        
    else:
        print("Wrong porting number!!! Please put 0 or 1.")
    
if __name__=="__main__":
    main()