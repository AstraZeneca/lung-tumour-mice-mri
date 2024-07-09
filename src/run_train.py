'''script for executing training scripts for the experiments'''
import argparse
from src.deepmeta.utils import pprint
from src.nnunet import utils as nn_utils

def get_args() -> argparse.Namespace:
    """Argument Parser
    Returns:
        argparse.Namespace: args objects containing user inputs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Experiment to conduct; if none all exps will be run",
        choices=[
            "exp1",
            "exp2",
            "exp3"
            ],
    )
    args = parser.parse_args()
    pprint.print_bold_red(args)
    return args

def train_model(train_params:dict,
                paths:dict,
                exp:str)-> None:
    """Train models for specified experiment using subprocesses

    Args:
        train_params (dict): dictionary with training parameters
        paths (dict): dictionary with pathways to data directory and save directory
        exp (str): experiment for which models should be trained (exp1 or exp2)
    Returns:
        None
    """
    total_dict={'exp1':{
                    'model':['unet','unet3p','deepmeta'],
                    'loss':['weighted_ce','unet3p_loss','fusion'],
                    'classes':3,
                    'bs': [64],
                    'dim':['2d']},
                'exp2':{
                    'model':['unet','unet3p','deepmeta'],
                    'loss':['weighted_ce','unet3p_loss','fusion'],
                    'classes':2,
                    'bs': [64]
                    'dim':['2d']},
                'exp3':{
                    'model':['unet',
                             'unet3p',
                             'deepmeta'],
                    'loss':['weighted_ce',
                            'unet3p_loss','fusion'],
                    'classes':2,
                    'bs': [64, 2],
                    'dim':['2d',
                           '3d']}
                }
    exp_dict = total_dict[exp]
    for dim in exp_dict['dim']:
        bs_id=0
        for i in range(3):
            subcommand0 = f'python -m src.deepmeta.train --data_path {paths["data_dir"]}/train{dim} --save_dir {paths["save_dir"]} '
            subcommand1 = f'--model {exp_dict["model"][i]} --loss {exp_dict["loss"][i]} -classes {exp_dict["classes"]} --dim {dim} '
            subcommand2 = f'--epochs {train_params["n_epochs"]} -nworkers {train_params["n_workers"]} -bs {exp_dict["bs"][bs_id]}'
            command = subcommand0 + subcommand1 + subcommand2
            nn_utils.run_subprocess(command)
        bs_id=bs_id+1
    return

def train_nnunet(params:dict,
                 exp:str) -> None:
    """Train nnunet for specified experiment using subprocesses
    Args:
        params (dict): dictionary with training parameters
        exp (str): experiment for which models should be trained (exp1 or exp2)
    Returns:
        None
    """
    no_epochs = {'1':'-tr nnUNetTrainer_1epoch',
                 '5':'-tr nnUNetTrainer_5epochs',
                 '100':'-tr nnUNetTrainer_100epochs',
                 '250':'-tr nnUNetTrainer_250epochs',
                 '500':'-tr nnUNetTrainer_500epochs',
                 '1000':''}
    total_dict={'exp1': {'dataset':1, 'dim':['2d'], 'fold':[0,1,2,3,4]},
                'exp2': {'dataset':1, 'dim':['2d'], 'fold':[0,1,2,3,4]},
                'exp3': {'dataset':1, 'dim':['2d','3d_fullres'], 'fold':[0,1,2,3,4]}}
    exp_dict = total_dict[exp]
    command_preprocess = 'nnUNetv2_plan_and_preprocess -d 1'
    nn_utils.run_subprocess(command_preprocess)
    for fold in exp_dict['fold']:
        for dim in exp_dict['dim']:
            command_fold = f'nnUNetv2_train 1 {dim} {fold} {no_epochs[str(params["n_epochs"])]}'
            nn_utils.run_subprocess(command_fold)
    return

def main():
    args=get_args()
    train_params = nn_utils.get_params("train_env")
    nnunet_params = nn_utils.get_params("nnunet_env")
    if args.exp:
        exp = args.exp
        paths = nn_utils.get_params(exp)
        #train_model(train_params, paths,exp)
        nn_utils.set_environment_variables(paths['main_dir'])
        train_nnunet(nnunet_params, exp)
    else:
        for exp in ['exp1','exp2','exp3']:
            paths = nn_utils.get_params(exp)
            train_model(train_params, paths,exp)
            nn_utils.set_environment_variables(paths['main_dir'])
            train_nnunet(nnunet_params, exp)

if __name__=='__main__':
    main()
