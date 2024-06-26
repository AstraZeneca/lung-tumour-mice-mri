""""misc utilities for running subprocesses and nnunet"""
import os
import subprocess
import yaml

def set_environment_variables(path:str)-> None:
    '''
    Set environmental variables needed to run nnUNet, command depends on OS

    Args:
        path (str) - path where to set environmental variables to
    Returns:
        None
    '''
    for dir in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
        path_to_dir = os.path.join(path, dir)
        print(path_to_dir)
        os.environ[dir] = path_to_dir
    return

def get_params(key:str) -> dict:
    """function for getting params from yaml file
    Args:
        key (str): key to access appropriate dict from params.yaml
    Returns:
        dict:
    """
    params=yaml.safe_load(open("params.yaml"))[key]
    return params

def realtime_output(process:subprocess.CompletedProcess) -> None:
    '''
    Prints output of scripts executed from python

    Args:
        process - output of subprocess.Popen
    Returns:
        None
    '''
    while True:
        rt_output = process.stdout.readline()
        if rt_output == '' and process.poll() is not None:
            break
        if rt_output:
            print(rt_output.strip(), flush=True)
    return

def run_subprocess(command:str) -> None:
    """ Execute respective script via command in subprocess

    Args:
        command (str): command to be executed via subprocess
    Return:
        None
    """
    print(command)
    process=subprocess.Popen(command.split(' '),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             encoding='utf-8')
    realtime_output(process)
    return
