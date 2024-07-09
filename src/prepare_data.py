"""create data variants used for experiments"""
import os
import argparse
import shutil
import src.data_variants._2d as dv_2d
import src.data_variants._3d as dv_3d
import src.nnunet.utils as nn_utils
from src.deepmeta.utils import pprint

def get_args() -> argparse.Namespace:
    """Argument Parser

    Returns:
        argparse.Namespace: args objects containing user inputs. 
        IF none, all experiments will be run
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Experiment to evaluate",
        choices=[
            "exp1",
            "exp2",
            "exp3"
            ],
    )
    args = parser.parse_args()
    pprint.print_bold_red(args)
    return args

def prepare_nnunet(name:str,
                   source_path,
                   classes:int=3,
                   preprocess:str="noNorm",
                   filetype:str='.tif',
                   dim='2d') -> None:
    """
    Prepare a dataset suitable for nnUNet framework. 
    Based on dataset variants generated for remainder models.
    Args:
        name (str): Name of the dataset.
        source_path (_type_): Path to the data variant that will be converted into nnunet format.
        classes (int, optional): Number of classes to predict;
                                 3 (default) for lung and lung tumour segmentation;
                                 2 for only lung tumour segmentation
        preprocess (str, optional): Preprocessing strategy.
                                    Can be noNorm (default) when no normalization is needed, 
                                    zscore for z-score normalization.
        filetype (str, optional): Image filetypes, only .tiff (default) or .nii.gz are supported.
        dim (str): dimension of the data to prepare
    Returns:
        None
    """
    dataset_name = dv_2d.create_nnunet_dir(name)
    if filetype=='.nii.gz':
        dv_2d.move_nnunet_imgs(dataset_name,
                               source_path,
                               filetype=filetype,
                               convert_to_nifti=True, dim=dim)
    else:
        dv_2d.move_nnunet_imgs(dataset_name,
                               source_path,
                               filetype=filetype,
                               dim=dim)
    dv_2d.create_json(dataset_name,
                      source_path,
                      classes=classes,
                      preprocess=preprocess,
                      filetype=filetype)
    print('nnUNet variant prepared')
    return

def prep_variant(source_path:str,
                 name:str,
                 test_names:str,
                 binary:bool=False,
                 dim:str='2d') -> None:
    """Prepare a data variant from the original data 
    Args:
        source_path (str): pathway to the original deepmeta directory
        name (str): name of the dataset variant
        test_names (str): list of test names to move
        binary (bool, optional): indicate if the masks are supposed to be binary
                                 (True for only lung tumour detection,
                                 False (default) for lung and lung tumour detection)
        dim (str): dimension of the data to prepare
    Return: 
        None
    """
    if binary:
        classes=2
    else:
        classes=3
    data_path = os.path.join(name, 'data')
    img_path = os.path.join(source_path, '3classes/Images')
    label_path = os.path.join(source_path, '3classes/Labels')
    test_path = os.path.join(source_path, 'Test')
    dv_2d.create_dirs(data_path)
    print(f'preparing {name} variant of the data...')
    print('copying imgs and labels...')
    shutil.copytree(img_path, f'{data_path}/train2d/images')
    dv_2d.move_labels(label_path,
                      f'{data_path}/train2d/labels',
                      binary=binary)
    print('preparing test sets...')
    dv_2d.move_test(test_path,
                    f'{data_path}/test',
                    test_names,
                    binary=binary)
    prepare_nnunet(name=name, source_path=data_path, classes=classes, dim=dim)
    print('done!')
    return

def prep_3d_variant(source_path:str,
                    name:str,
                    test_names:str) -> None:
    """Prepares a 3D dataset variant and also a sliced version of this variant (for exp3).
       It requiries metadata to correctly organize the mask slices
    Args:
        source_path (str): pathway to the original deepmeta directory
        name (str): name for the dataset directory
        test_names (str): list of test names to move
    Return: 
        None
    """
    data_path = os.path.join(name,'data')
    labels_pathway = os.path.join(source_path,'Metastases','Labels')
    metas_paths = os.listdir(labels_pathway)
    test_path = os.path.join(source_path,'Test')
    raw_data_path = os.path.join(source_path,'Raw_data')
    generated_masks_tif=dv_3d.create_3d_masks(metas_paths,
                                             labels_pathway)
    dv_3d.save_3d_masks(generated_masks_tif,
                       data_path)
    dv_3d.move_3d_imgs(raw_data_path,
                      data_path)
    dv_3d.run_data_aug(os.path.join(data_path,'train3d/images'),
                      os.path.join(data_path,'train3d/labels'))
    dv_3d.slice_3ds(data_path)
    #TESTDATA
    test_dictionary1 = dv_3d.stack_test_slices(test_path, test_names)
    for test in test_dictionary1.keys():
        dv_3d.save_3d_test_masks(test_dictionary1[test], test, f'{data_path}/test/labels')
    dv_3d.move_3d_test_imgs(test_path, test_names, os.path.join(data_path,'test'))
    prepare_nnunet(name=name,
                   source_path=data_path,
                   classes=2,
                   preprocess='zscore',
                   filetype='.nii.gz',
                   dim='3d')
    #prepare 2D variant of the data
    dv_2d.move_test(test_path,
                     f'{data_path}/test',
                     test_names,
                     binary=True)
    return

def main():
    args = get_args()
    TEST_NAMES = ['8_m2PL_day25','28_2Pc_day50','56_2PLc_day106','m2Pc_c1_10Corr_1']
    if args.exp:
        print(args.exp)
        paths = nn_utils.get_params(args.exp)
        if args.exp=='exp1':
            binary=False
        else:
            binary=True
        if args.exp!='exp3':
            prep_variant('data/deepmeta_dataset',
                         paths['main_dir'],
                         test_names=TEST_NAMES,
                         binary=binary,
                         dim='2d')
        else:
            prep_3d_variant('data/deepmeta_dataset',
                        paths['main_dir'],
                        test_names = TEST_NAMES)
    else:
        print('All experiments')
        paths1 = nn_utils.get_params('exp1')
        paths2 = nn_utils.get_params('exp2')
        paths3 = nn_utils.get_params('exp3')
        prep_variant('data/deepmeta_dataset',
                     paths1['main_dir'],
                     test_names=TEST_NAMES,
                     binary=False,
                     dim='2d')
        prep_variant('data/deepmeta_dataset',
                     paths2['main_dir'],
                     test_names=TEST_NAMES,
                     binary=True,
                     dim='2d')
        prep_3d_variant('data/deepmeta_dataset',
                        paths3['main_dir'],
                        test_names = TEST_NAMES)

if __name__=='__main__':
    main()
