'''script for executing evaluation scripts for the experiments'''
import os
import argparse


from src.nnunet import utils as nn_utils
from src.deepmeta.utils import pprint

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

def eval_model(exp_params:dict,
               exp:str) -> None:
    """Evaluate performance of the trained models as a part of the specifed experiment
    Args:
        params (dict): params with specified pathways from params.yaml file
        exp (str): experiment for which to evaluate the models
    Return:
        None
    """
    total_dict={'exp1': {'model':['unet','unet3p','deepmeta'],
                         'model_path':f'{exp_params["save_dir"]}',
                         'dim':['2d'],
                         'data_dir':f'{exp_params["data_dir"]}/test',
                         'classes':3},
                'exp2': {'model':['unet','unet3p','deepmeta'],
                         'model_path':f'{exp_params["save_dir"]}',
                         'dim':['2d'],
                         'data_dir':f'{exp_params["data_dir"]}/test',
                         'classes':2},
                'exp3': {'model':['unet','unet3p','deepmeta'],
                         'model_path':f'{exp_params["save_dir"]}',
                         'dim':['2d','3d'],
                         'data_dir':f'{exp_params["data_dir"]}/test',
                         'classes':2}}
    exp_dict = total_dict[exp]
    for dim in exp_dict['dim']:
        print(dim)
        for model in exp_dict['model']:
            model_pathway = f'{exp_dict["model_path"]}/models/best_model_{model}_{dim}'
            output_pathway = f'{exp_dict["model_path"]}/outputs/{model}_{dim}'
            os.makedirs(output_pathway, exist_ok=True)
            command = f'python -m src.deepmeta.predict --save --model {model} --model_path {model_pathway} -classes {exp_dict["classes"]} --data_dir {exp_dict["data_dir"]} --dim {dim} --save_dir {output_pathway}'
            nn_utils.run_subprocess(command)
    return

def eval_nnunet(train_params:dict,
                exp_params,
                exp:str) -> None:
    """Evaluate performance of the trained nnunet as a part of the specifed experiment
    Args:
        params (dict): dictionary with training parameters
        exp (str): experiment for which to evaluate the model
        exp_params,
    Return:
        None
    """
    no_epochs = {'1':'-tr nnUNetTrainer_1epoch',
                 '5':'-tr nnUNetTrainer_5epochs',
                 '100':'-tr nnUNetTrainer_100epochs',
                 '250':'-tr nnUNetTrainer_250epochs',
                 '500':'-tr nnUNetTrainer_500epochs',
                 '1000':''}
    total_dict={'exp1': {'dataset':1,
                         'dim':['2d'],
                         'input':f'{exp_params["data_dir"]}/nnunet_test/images',
                         'labels':f'{exp_params["data_dir"]}/nnunet_test/labels',
                         'output':[f'{exp_params["save_dir"]}/outputs/nnunet_2d'],
                         'classes':3},
                'exp2': {'dataset':1,
                         'dim':['2d'],
                         'input':f'{exp_params["data_dir"]}/nnunet_test/images',
                         'labels':f'{exp_params["data_dir"]}/nnunet_test/labels',
                         'output':[f'{exp_params["save_dir"]}/outputs/nnunet_2d'],
                         'classes':2},
                'exp3': {'dataset':1,
                         'dim':['2d','3d_fullres'],
                         'input':f'{exp_params["data_dir"]}/nnunet_test/images',
                         'labels':f'{exp_params["data_dir"]}/nnunet_test/labels',
                         'output':[f'{exp_params["save_dir"]}/outputs/nnunet_2d',
                                   f'{exp_params["save_dir"]}/outputs/nnunet_3d'],
                         'classes':2}}
    exp_dict = total_dict[exp]
    i=0
    for dim in exp_dict["dim"]:
        os.makedirs(exp_dict["output"][i], exist_ok=True)
        print('running an inference')
        infer_command = f'nnUNetv2_predict -d 1 -c {dim} -i {exp_dict["input"]} -o {exp_dict["output"][i]} {no_epochs[str(train_params["n_epochs"])]}'
        nn_utils.run_subprocess(infer_command)
        print('running evaluation')
        eval_command = f'python -m src.nnunet.eval --gt_path {exp_dict["labels"]} --pt_path {exp_dict["output"][i]} -classes {exp_dict["classes"]} --save_path {exp_dict["output"][i]}'
        nn_utils.run_subprocess(eval_command)
        i=i+1
    return

def main():
    args=get_args()
    nnunet_params = nn_utils.get_params("nnunet_env")
    if args.exp:
        exp = args.exp
        exp_params= nn_utils.get_params(args.exp)
        paths = nn_utils.get_params(exp)
        eval_model(exp_params, exp)
        nn_utils.set_environment_variables(paths['main_dir'])
        eval_nnunet(nnunet_params, exp_params, exp)
    else:
        for exp in ['exp1','exp2','exp3']:
            exp_params= nn_utils.get_params(exp)
            paths = nn_utils.get_params(exp)
            eval_model(exp_params, exp)
            nn_utils.set_environment_variables(paths['main_dir'])
            eval_nnunet(nnunet_params, exp_params, exp)

if __name__=='__main__':
    main()
