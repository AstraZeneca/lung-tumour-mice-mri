"""utils for preapring 2D MRI data variants"""
import os
import json
import shutil
import numpy as np
import skimage
import nibabel
from tqdm import tqdm
from skimage import io

def create_dirs(path:str)-> None:
    """generate directories

    Args:
        path (str): pathway to where the directories are to be created
    Return:
        None
    """
    print('creating directories...')
    train_path = os.path.join(path, 'train2d')
    test_path = os.path.join(path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(os.path.join(test_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_path, 'labels'), exist_ok=True)
    return

def modify_labels(pathway:str,
                  input_img=None) -> np.array:
    """transform a mask from multi-class (lung and nodule) to binary class (only nodule)
    Can input pathway so that an image is read or directly input np.array of image
    Args:
        pathway (str): pathway to read an image
        input_img (np.array): array of image to be transforrmed.
    Returns:
        np.array: transformed mask
    """
    if pathway:
        img = io.imread(pathway)
    else:
        img = input_img
    img[img<=1.0]=0.0
    img[img>1.0]=1.0
    return img


def modify_labels_wrapper(source_path:str,
                          dest_path:str)-> None:
    """ wrapper around function to binarize masks, loops through a list
    
    Args:
        source_path (str): pathway to directory where masks are stored
        dest_path (str): pathway to dir where masks should be saved
    Return:
        None
    """  
    mask_list = os.listdir(source_path)
    os.makedirs(dest_path)
    print('Modifying Masks...')
    for mask in tqdm(mask_list):
        pathway = os.path.join(source_path,mask)
        label_output =modify_labels(pathway)
        io.imsave(os.path.join(dest_path, mask),
                  label_output,
                  check_contrast=False)
    return

def move_labels(source_path:str,
                dest_path:str,
                binary:bool=False) -> None:
    """ move masks from raw dataset directory to specific variant directory; 
    binarize for variant 2 and 3

    Args:
        source_path (str): path where masks are stored in the original dataset
        dest_path (str): path where masks should be saved 
        binary (bool, optional): choose True if you want to transform the mask into binary
                                mask
    Returns
        None
    """
    if binary:
        modify_labels_wrapper(source_path, dest_path)
    else:
        shutil.copytree(source_path, dest_path)
    return

def move_test(source_path:str,
              dest_path:str,
              test_names:list,
              binary:bool=False,
              dim:str='2d')->None:
    """move test images from orgiinal dataset to specific variant dataset

    Args:
        source_path (str): directory where images are stored
        dest_path (str): directory where to save the images
        test_names (list): names of the test sets
        binary (bool, false): indicate if masks should be binary (only tumour) 
                            or multiclass (lung and tumour)
    Returns:
        None
    """
    j=0
    for testset in test_names:
        print('moving', testset)
        shutil.copy(os.path.join(source_path, f'{testset}.tif'),
                    os.path.join(dest_path, f'{testset}.tif'))
        img = io.imread(os.path.join(source_path, f'{testset}.tif'))
        shutil.copytree(os.path.join(source_path, testset),
                        os.path.join(dest_path, testset))
        slice_and_save(img, j, testset, source_path, dest_path, binary)
        j=j+1
    return

def slice_and_save(img:np.array,
                   j:int,
                   testset:str,
                   source_path:str,
                   dest_path:str,
                   binary:bool=False) -> None:
    for i in tqdm(range(len(img))):
        io.imsave(os.path.join(dest_path, 'images', f'{j}_{i}.tif'), img[i],check_contrast=False)
        if binary:
            label = io.imread(os.path.join(source_path,testset,'3classes',f'{i}.tif'))
            io.imsave(os.path.join(dest_path,'labels',f'{j}_{i}.tif'),
                      modify_labels(input_img=label, pathway=None),
                      check_contrast=False)
        else:
            shutil.copy(os.path.join(source_path, testset,'3classes',f'{i}.tif'),
                        os.path.join(dest_path,'labels',f'{j}_{i}.tif'))

def enhance_contrast(image:np.array) -> np.array:
    """enhance contrast of an image
    (images come in 2 batches,one has lower brightness)
    Args:
        image (np.array) - image in np.array
    Returns:
        np.array - img with enhanced contrast
    """
    image = skimage.exposure.equalize_adapthist(image/255, clip_limit=0.03)
    return (image * 255).astype(np.uint8)

###########################################
########## NNUNET VARIANT #################
###########################################

def create_nnunet_dir(name:str) -> str:
    """create suitable dataset directory and json for nnunet

    Args:
        name (str): name to be used for dataset

    Returns:
        dataset name in a string format
    """
    dataset_name = f'Dataset001_{name}'
    dataset_path=os.path.join(f'{name}/nnUNet_raw', dataset_name)
    os.makedirs(os.path.join(dataset_path,'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path,'labelsTr'), exist_ok=True)
    return dataset_name

def create_json(dataset_name:str,
                source_path:str,
                classes:int=3,
                preprocess:str="noNorm",
                filetype:str='.tif') -> None:
    """create json file describing dataset based on filetype, no. of images and classes

    Args:
        dataset_name (str): dataset name, generated by create_nnunet_dir
        source_path (str): path to where the data is stored
        classes (int, optional): no classes (3 for lung and tumour, 2 for oly tumour).
        preprocess (str, optional): preprocessing strategy for nnUNet. 
                                    for variant 1 and 2 its "noNorm", for 3 - zscore.
        filetype (str, optional): filetype of images. ".tiff" for var. 1 and 2, ".nii.gz" for var.3.
    """
    main_dir = os.path.split(source_path)[0]
    print(main_dir)
    if classes==3:
        labels_dict={"background": 0,"lung": 1,"nodule": 2}
    else:
        labels_dict={"background": 0,"nodule": 1}
    no_img = len(os.listdir(os.path.join(main_dir,'nnUNet_raw',dataset_name, 'imagesTr')))
    data = {
        "channel_names": {"0": preprocess},
        "labels": labels_dict,
        "numTraining": no_img,
        "file_ending": filetype
    }
    with open(f'{main_dir}/nnUNet_raw/{dataset_name}/dataset.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    return

def transform_to_nii(source_path:str,
                     dest_path:str) -> None:
    """transforms an image from tif to nifti format and saves it in specified pathway
    Args:
        source_path (str): path to where imgs are stored
        source_path (str): path to where imgs are should be saved
    """
    img=io.imread(source_path)
    nifti = nibabel.Nifti1Image(img, None)
    nibabel.save(nifti, dest_path)
    return

def move_nnunet_imgs(dataset_name:str,
                     source_path:str,
                     filetype:str='.tif',
                     convert_to_nifti:bool=False,
                     dim:str='2d')->None:
    """ prepare the data for nnunet framework

    Args:
        dataset_name (str): name of the dataset (eg LungPlusNodule)
        source_path (str): directory where images are stored
        filetype (str, optional): filetype of the images to be moved.
                                ".tiff" for variant 1/2, ".nii.gz" for 3.
        convert_to_nifti (bool, optional): indicate if output in nifti format
        dim (str, dim): specify dimension (for dir name)
    """
    #train
    img_list = os.listdir(os.path.join(source_path, f'train{dim}/images'))
    main_dir = os.path.split(source_path)[0]
    imagesTr_path = os.path.join(main_dir,'nnUNet_raw',dataset_name,'imagesTr')
    labelsTr_path = os.path.join(main_dir,'nnUNet_raw',dataset_name,'labelsTr')
    nnunet_test_path = os.path.join(source_path, 'nnunet_test')
    test_images_path = os.path.join(source_path,'test/images')
    test_labels_path = os.path.join(source_path,'test/labels')
    print('moving train imgs...')
    for img in tqdm(img_list):
        if img.startswith(('t_', '90_', '180_', '270_')): #skip DA
            continue
        train_path = os.path.join(source_path, f'train{dim}/images', img)
        labels_path = os.path.join(source_path, f'train{dim}/labels', img)
        if convert_to_nifti:
            transform_to_nii(train_path,
                             f"{imagesTr_path}/{img.split('.')[0]}_0000.nii.gz")
            transform_to_nii(labels_path,
                             f"{labelsTr_path}/{img.split('.')[0]}.nii.gz")
        else:
            shutil.copy(train_path,
                        f"{imagesTr_path}/{img.split('.')[0]}_0000{filetype}")
            shutil.copy(labels_path,
                        f"{labelsTr_path}/{img}")
    #test
    img_list = os.listdir(test_images_path)
    os.makedirs(os.path.join(nnunet_test_path,'images'), exist_ok=True)
    os.makedirs(os.path.join(nnunet_test_path,'labels'), exist_ok=True)
    print('moving test imgs...')
    for img in tqdm(img_list):
        test_path = os.path.join(test_images_path, img)
        labels_path = os.path.join(test_labels_path, img)
        shutil.copy(test_path, f"{nnunet_test_path}/images/{img.split('.')[0]}_0000{filetype}")
        shutil.copy(labels_path, f"{nnunet_test_path}/labels/{img}")
    return
