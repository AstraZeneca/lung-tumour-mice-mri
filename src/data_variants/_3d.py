"""utils for preapring 3D MRI"""
import os
import shutil
import numpy as np
import pandas as pd
import nibabel
from skimage import io
from tqdm import tqdm
from volumentations import Compose, Rotate, ElasticTransform
from src.data_variants._2d import enhance_contrast, modify_labels

def assign_path_to_mice(meta_paths:list,
                        size:int=128) -> dict:
    """
        Function assigning specific 2D images with metastases to therespective mouse IDs
        to correctly reconstruct the masks. The function accepts a list of 2D images pathways 
        and assigns the slice pathways to respective mice (e.g. slices 0-128 to mouse with ID 0, 
        slices 129-256 to mouse with ID 1 etc). We need this function to 'decode' authors' 
        way of labelling the mice and then correctly stack the masks.
        
        Description: the function will loop over the range of 185 mice (all mice) and create a key for
        each mouse id  (except for 8, 28, 56) which are test. Then through division, it will decide the id
        of the slice wrt the whole 3D stack - if remainder of division is 0, its the first slice for this mouse;
        eg 256 % 128 = 0 -> slice named 256.tif is a 0 slice for mouse of id 1
        if remainder is not 0, the remainder corresponds to slice number
        eg 257 % 128 = 1 -> slice named 257.tif is a 1 slice for mouse of id 1
        Finally we simply assign a pathway for each slice in each mouse

    Args:
        meta_paths (dict): list of 2D slices which contain metastases (as noted in Metastases directory in org dataset)
        size (int, optional): length of a 3D mice i.e. no. of slices per one 3D image (128 in Deepmeta).

    Returns:
        dict - dictionary with keys being mice IDs and values being slices with their respective pathways
    """
    slice_dict={}
    for i in range(0,185): 
        if i in [8,28,56]: 
            continue
        else:
            slice_dict[i]={}
    for path in meta_paths:
        term = int(path.split('.')[0])
        if term % 128 ==0:
            slice=0
            mouse_id=(term//128)-1
        else:
            slice= term % 128
            mouse_id =(term - slice)//128
        slice_dict[int(mouse_id)][int(slice)]=path
    return slice_dict

def get_healthy_mice(path:str) -> list:
    """get a list of mice with no metastases
    Args:
        path (dict): pathway to the metadata csv file  
    Returns:
        list - list of ids of healthy mice
    """
    metadata=pd.read_csv(path)
    healthy_list = list(metadata.loc[metadata['Saine/Metas'] == 's'].index.values)
    return healthy_list

def create_3d_masks(metas_paths:dict,
                    pathway:str) -> dict:
    """ stack 2d masks into 3D masks. As each 3D MRI is 128x128x128, we need to first create 128x128x128 zero matrix;
    we will then loop through 128 masks along the z axis and replace every mask which (according to metas_directory)
    should contain metastases. This way we will reproduce the 3D masks. If mouse is healthy, produce 128x128x128 zero matrix.
    
    Description: function loops through all dictionary keys; then if the key has no metastases,its length will be 0
    (as there will be no slices with metastases in the dict). the function will then double check if the mouse 
    is healthy (if not healthy, it indicates missing data) so will be excluded from the dataset. if the key has metasteses,
    its length will be >0 and we will loop through the stack and replace respective slices with metastases slices
    Args:
        metas_dictionary (dict): dictionary indicating which slices in a specific 3D stack have tumours
        pathway (str): pathway to where the metastases slices are stored

    Returns:
        dict: dictionary with 3d masks tasks for each 3d stack
    """
    img_dict = {}
    metas_dictionary = assign_path_to_mice(metas_paths)
    healthy_mice_list = get_healthy_mice('data/deepmeta_dataset/mouse_ID.csv')
    for mice in metas_dictionary.keys():
        if len(metas_dictionary[mice])==0: 
            if mice in healthy_mice_list:
                img_dict[mice]=np.zeros((128,128,128))
        else:
            img_dict[mice]=np.zeros((128,128,128))
            for slice in metas_dictionary[mice]:
                img_dict[mice][slice,:,:] = io.imread(os.path.join(pathway, metas_dictionary[mice][slice]))/255
    return img_dict

def save_3d_masks(imgs:dict,
                  save_pathway:str) -> None:
    """save re-stacked 3D masks under specified save_pathway in a .nii.gz format.
    specifies affine as None so that a default affine is used instead.

    Args:
        imgs (dict): dictionary containing 3D masks (values) per mouse ID (keys)
        save_pathway (str): pathway to the directory where masks should be saved

    Returns:
        None
    """
    path = os.path.join(save_pathway, 'train3d/labels')
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
    for mice in imgs.keys():
        io.imsave(f'{path}/{mice}.tif', imgs[mice], check_contrast=False)
    return

def save_3d_test_masks(img:np.array,
                       save_name:str,
                       save_pathway:str) -> None:
    """save re-stacked 3D masks from the test set under
    specified save_pathway in a .nii.gz format, 
    under specified name. Modified save_3d_masks function
    
    Args:
        imgs (np.array): 3D mask in the np.array format to be saved
        save_name (str): name under which it should be saved
        save_pathway (str): pathway to the directory where masks should be saved

    Returns:
        None
    """
    if not os.path.exists(save_pathway):
        os.makedirs(save_pathway,exist_ok=True)
    nifti = nibabel.Nifti1Image(img, None)
    nibabel.save(nifti, f'{save_pathway}/{save_name}.nii.gz')
    return
            
def move_3d_imgs(source_pathway:str,
                 save_pathway:str) -> None:
    """Move 3D images from raw data directory,
    enhance contrast between batches and load as nifti

    Args:
        source_pathway (str): pathway to the directory where raw 3D data is stored
        save_pathway (str): directory where the 3D images in nifti format should be saved
    Returns:
        None        
    """
    path = os.path.join(save_pathway, 'train3d/images')
    if not os.path.exists(path):
        shutil.copytree(source_pathway,path)
    files_list = os.listdir(path)
    files_list = [file for file in files_list if file.endswith('.tif')]
    labels_list = os.listdir(os.path.join(save_pathway, 'train3d/labels'))
    labels_list = [label.split('.')[0] for label in labels_list]
    print('Saving imgs')
    for file in tqdm(files_list):
        file_pathway = os.path.join(path,file)
        id_number=file.split('_')[0]
        id_number=id_number.split('.')[0]
        if id_number in labels_list:
            img=io.imread(file_pathway)
            if int(id_number)<87:
                img_tosave = enhance_contrast(img)
            else:
                img_tosave = img
            io.imsave(f'{path}/{id_number}.tif', img_tosave)
        os.remove(file_pathway)
    return
    
def run_data_aug(img_dir_path:str,
                 label_dir_path:str) -> None:
    """runs data augmentation for all images in the specified pathway
    note: very resource intensive
    Args:
        img_dir_path (str) - pathway to the directory where imgs are saved
        label_dir_path (str) - pathway to the directory where imgs are to be stored
    Returns: 
        None
    """
    print('running data augmentation')
    img_list = [img for img in os.listdir(img_dir_path) if img.endswith('.tif')]
    for img_name in tqdm(img_list):
        label_name=img_name
        img_pathway=os.path.join(img_dir_path,img_name)
        label_pathway=os.path.join(label_dir_path,label_name)
        label = io.imread(label_pathway)
        img = io.imread(img_pathway)
        _, _ = elastic_wrapper(img, label, f'{img_dir_path}/t_{img_name}', f'{label_dir_path}/t_{img_name}')
        for i in [90,180,270]:
            r_img, r_label = rotate_wrapper(i, img, label, f'{img_dir_path}/{i}_{img_name}', f'{label_dir_path}/{i}_{img_name}')
            _, _ = elastic_wrapper(r_img, r_label, f'{img_dir_path}/t_{i}_{img_name}', f'{label_dir_path}/t_{i}_{img_name}')
    return
    
def stack_test_slices(source_pathway:str,
                      list_tests:list)->dict:
    """Stack 2D masks from test sets onto 3D images
    Args:
        source_pathway (str): pathway to the directory where the 2D test masks are stored
        save_pathway (str): directory where the 3D test masks should be saved
    Returns:
        dict - dictionary with saved 3D MRI masks        
    """
    test_dictionary = {}
    for test in list_tests:
        path = os.path.join(source_pathway, test)
        mask_list = []
        for i in range(128):
            img = io.imread(f'{path}/3classes/{i}.tif')
            img = modify_labels(input_img=img, pathway=None)
            mask_list.append(img)
            mask_stack = np.stack(mask_list, axis=0)
        test_dictionary[test]=mask_stack
    return test_dictionary

def move_3d_test_imgs(path_to_dir:str,
                      test_set_list:list,
                      save_test_pathway:str) -> None:
    """Move 3D test images from raw data directory, enhance contrast between batches and load as nifti

    Args:
        source_pathway (str): pathway to the directory where raw 3D data is stored
        test_set_list (list): list of test mice
        save_pathway (str): directory where the 3D images in nifti format should be saved
    Returns:
        None        
    """
    for test_set in tqdm(test_set_list):
        path = os.path.join(path_to_dir, test_set)
        os.makedirs(f'{save_test_pathway}/images/', exist_ok=True)
        img = io.imread(f'{path}.tif')
        if test_set == "m2Pc_c1_10Corr_1": #m2Pc_c1_10Corr_1 test set has natively enhanced contrast, no need to enhance it again
            img= img
        else: 
            img = enhance_contrast(img)
        io.imsave(f'{save_test_pathway}/images/{test_set}.nii.gz', img)
    return


def rotate_wrapper(angle:int,
                   img:np.array,
                   label:np.array,
                   img_path:str,
                   label_path:str) -> tuple:
    """wrapper for rotating images, based on volumentations package. conducts DA and
    then saves the images
    Args:
        angle(int) - angle by which should rotate
        img(np.array) - array of image to be rotated
        label(np.array) - array of mask to be rotated
        img_path(str) - pathway to the image
        label_path(str) - pathway to the label
    Returns: 
        None
    """
    rot_pipe = rotate_x_degrees(angle)
    data = {'image': img, 'mask': label}
    rot_data = rot_pipe(**data)
    r_img, r_lbl =rot_data['image'], rot_data['mask']
    io.imsave(img_path,r_img, check_contrast=False)
    io.imsave(label_path,r_lbl, check_contrast=False)
    return r_img, r_lbl
    
def elastic_wrapper(img:np.array,
                    label:np.array,
                    img_path:str,
                    label_path:str) -> tuple:
    """wrapper for elastic transformation of images, based on volumentations package. conducts DA and
    then saves the images
    Args:
        img(np.array) - array of image to be rotated
        label(np.array) - array of mask to be rotated
        img_path(str) - pathway to the image
        label_path(str) - pathway to the label
    Returns: 
        None
    """
    ela_pipe = elastic_transform()
    data = {'image': img, 'mask': label}
    ela_data = ela_pipe(**data)
    e_img, e_lbl = ela_data['image'], ela_data['mask']
    io.imsave(img_path,e_img, check_contrast=False)
    io.imsave(label_path,e_lbl, check_contrast=False)
    return e_img, e_lbl

def rotate_x_degrees(x:int) -> Compose:
    """Rotates 3D image by x degrees. Note we are only rotating along one axis.
    Part of volumentations.Compose pipe
    Args
        x - angle by which you should rotate
    """
    return Compose([
        Rotate((x, x), (0, 0), (0, 0), p=1)
    ], p=1.0)

def elastic_transform() -> Compose:
    """Apply elastic transformation to the images. 
    Part of volumentations.Compose pipe
    """
    return Compose([ElasticTransform((0.08, 0.25), interpolation=2, p=1)], p=1.0)

def slice_img(img:np.array,
              save_dir_path:str,
              save_name:str) -> None:
    """slice 3d image into 2d
    Args
        img(np.array) - image to be sliced
        save_dir_path(str) - pathway to where the slices should be saved
        save_name(str) - img name to use for saving
    """
    for i in range(len(img)):
        pathway = os.path.join(save_dir_path, f"{i}_{save_name}.tif")
        io.imsave(pathway,img[i], check_contrast=False)
    return

def slice_3ds(data_path:str) -> None:
    """
    slice images in a 3d format in a train3d directory into
    2d slices and store them in 2d directory
    Args:
        data_path - pathway to where the data is stored
    """
    img_path=os.path.join(data_path,'train3d', 'images')
    lbl_path=os.path.join(data_path,'train3d','labels')
    new_img_path = os.path.join(data_path,'train2d', 'images')
    new_lbl_path = os.path.join(data_path,'train2d', 'labels')
    os.makedirs(new_img_path,exist_ok=True)
    os.makedirs(new_lbl_path,exist_ok=True)
    img_list = os.listdir(img_path)
    img_list = [img for img in img_list if img.endswith('.tif')]
    print('slicing 3d imgs into 2d')
    for img_name in tqdm(img_list):
        img = io.imread(os.path.join(img_path,img_name))
        lbl = io.imread(os.path.join(lbl_path,img_name))
        slice_img(img, new_img_path, save_name=img_name.split('.')[0])
        slice_img(lbl, new_lbl_path, save_name=img_name.split('.')[0])
    return
    
    
