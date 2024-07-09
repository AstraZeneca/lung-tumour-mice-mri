'''setting up directories, downloading and unzipping data'''
import os
import urllib.request
import zipfile
import src.nnunet.utils as nn_utils

def create_nnunets() -> None:
    '''
    Create directories needed for nnUNet framework. 
    Args:
        None
    Return:
        None
    '''
    for exp in ['exp1','exp2','exp3']:
        params = nn_utils.get_params(exp)
        for nn_dir in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
            path = os.path.join(params['main_dir'], nn_dir)
            print('Working on ',path)
            os.makedirs(path, exist_ok=True)
    return

def run_subprocesses() -> None:
    '''
    run subprocesses related to nnunet framework and original data preparation/
    as specified in the paper. 
    Args:
        None
    Return:
        None
    '''
    #install nnunet
    command = 'pip install -e src/nnunet/nnUNet/.'
    nn_utils.run_subprocess(command)
    os.chdir('data/deepmeta_dataset/')
    command = 'python prepare_dataset_multi_threads.py'
    nn_utils.run_subprocess(command)
    os.chdir('../..')
    print('Subprocesses finished')
    return

def get_data(remove_original:bool=True) -> None:
    '''
    Download and unzip the deepmeta_dataset file
     
    Args:
        remove_original(bool, optional) - indicates whether the original data dir should be removed
    Return:
        None
    '''
    ZIP_PATH= 'deepmeta_dataset.zip'
    URL_LINK= 'https://zenodo.org/records/7014776/files/deepmeta_dataset.zip'
    urllib.request.urlretrieve(URL_LINK, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('data')
    if remove_original:
        os.remove(ZIP_PATH)
    print('Unzipped')
    return

def main():
    create_nnunets()
    get_data(remove_original=False)
    run_subprocesses()
    print('Set-up done')

if __name__=='__main__':
    main()
