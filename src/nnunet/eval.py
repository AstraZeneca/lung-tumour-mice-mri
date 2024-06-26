"""evaluate nnunet performance for specified exp"""
import os
import json
import argparse
import torch
import numpy as np
import nibabel
from skimage import io
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score

def get_args() -> argparse.Namespace:
    """
    Argument parser.
    Args
        None
    Return
        argparse.Namespace storing inputs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Path to the data dir with ground truth."
    )
    parser.add_argument(
        "--pt_path",
        type=str,
        default=None,
        help="Path to the data dir with predictions."
    )
    parser.add_argument(
        "-classes",
        type=int,
        default=3,
        help="No. classes."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="path where to save json.results"
    )
    args = parser.parse_args()
    print(args)
    return args

def get_img_list(path:str) -> list:
    """list all .tif or .nii.gz files specified in the pathway
    Args:
        path (str): pathway to where all the files are stored

    Returns:
        list: list of all .tif or .nii.gz files
    """
    img_list = os.listdir(path)
    img_list = [img for img in img_list if img.endswith(('.tif','.nii.gz'))]
    return img_list

def read_imgs(img_name:str,
              gt_dir:str,
              pt_dir:str) -> tuple:
    """reading mask images from specified directories

    Args:
        img_name (str): filename
        gt_dir (str): directory with ground truth masks
        pt_dir (str): directory with predicted truth masks

    Raises:
        ValueError: If neither tif or nii.gz are specified, valueerror is raised

    Returns:
        tuple: ground truth mask and predicted mask 
    """
    gt_path = os.path.join(gt_dir, img_name)
    pt_path = os.path.join(pt_dir, img_name)
    if img_name.endswith('.tif'):
        gt = io.imread(gt_path)
        pt = io.imread(pt_path)
    elif img_name.endswith('.nii.gz'):
        gt = nibabel.load(gt_path).get_fdata()
        pt = nibabel.load(pt_path).get_fdata()
    else:
        raise ValueError('Wrong filetype')
    return gt, pt

def get_labels(classes:int) -> list:
    """get list of labels for metric calculation, based on the number of classes present
    Args:
        classes (int): no. classes to get labels for, 2 for binary, 3 for lung and lung tumour
    Raises:
        ValueError: if neither 2 or 3 classes are specified value error is raised
    Returns:
        list: list of labels to use for calculating f1_score
    """
    if classes==3:
        label_list=[0,1,2]
    elif classes == 2:
        label_list=[0,1]
    else:
        raise ValueError('Wrong no. classes')
    return label_list

def stats(gt:np.array,
          pt:np.array,
          labels:np.array) -> tuple:
    """
    calculates f1 and iou score for provided ground truth mask and prediction
    Args:
        gt (np.array): ground truth mask
        pt (np.array): predicted mask
        labels (np.array): list of labels to use for metric calculation
    Returns:
        tuple: IoU and F1 score for the provided image
    """
    iou=jaccard_score(gt.flatten(), pt.flatten(), labels=labels, average=None, zero_division=1)[1:]
    f1=f1_score(gt.flatten(), pt.flatten(), labels=labels, average=None, zero_division=1)[1:]
    return iou, f1

def stats_wrapper2D(img_list:list,
                    gt_path:str,
                    pt_path:str,
                    classes:int=3) ->None:
    """
    wrapper for calculating scores from gt and pt, only for 2D
    Args:
        img_list (list): list of images in a directory
        gt_path (str): pathway to ground truth mask
        pt_path (str): pathway to predicted mask
        classes (int): no. classes to calculate the scores for

    Returns:
        None
    """
    label_metrics = get_labels(classes)
    iou_list, f1_list = [], []
    for img in tqdm(img_list):
        gt, pt = read_imgs(img,gt_path, pt_path)
        iou, f1 = stats(gt, pt, label_metrics)
        iou_list.append(iou)
        f1_list.append(f1)
    print_results(iou_list,f1_list,classes, save_path = pt_path)
    return

def stats_wrapper3D(img_list:list,
                    gt_path:str,
                    pt_path:str,
                    classes:int=3) -> None:
    """
    wrapper for calculating scores from gt and pt, for 3D images
    Args:
        img_list (list): list of images in a directory
        gt_path (str): pathway to ground truth mask
        pt_path (str): pathway to predicted mask
        classes (int): no. classes to calculate the scores for
    Returns:
        None
    """
    label_metrics = get_labels(classes)
    iou_list2d, f1_list2d = [], []
    #2D
    print('2D', '\n')
    for img in tqdm(img_list):
        gt, pt = read_imgs(img,gt_path, pt_path)
        iou_list, f1_list = [], []
        for i in range(len(pt)):
            iou, f1 = stats(gt[i], pt[i], label_metrics)
            iou_list.append(iou)
            f1_list.append(f1)
        for i in iou_list:
            iou_list2d.append(i)
        for i in f1_list:
            f1_list2d.append(i)
    #3D
    print('3D', '\n')
    iou_list3d, f1_list3d = [], []
    for img in tqdm(img_list):
        gt, pt = read_imgs(img,gt_path, pt_path)
        iou, f1 = stats(gt, pt, label_metrics)
        iou_list3d.append(iou)
        f1_list3d.append(f1)
    print('3D')
    json_path = os.path.join(pt_path,'results_3d.json')
    print_results(iou_list3d,f1_list3d,classes, save_pathway = json_path)
    print('2D')
    json_path = os.path.join(pt_path,'results_2d.json')
    print_results(iou_list2d,f1_list2d,classes, save_pathway = json_path)
    
def print_results(iou_list:list,
                  f1_list:list,
                  classes:int,
                  save_pathway:str=None)-> None:
    """Wrapper for printing and saving model performance based in json file
    Args:
        iou_list (list): list of IoU scores obtained for all test images
        f1_list (list): list of F1 scores obtained for all test images
        classes (int): number of classes, 2 or 3 
        save_path (str, optional): pathway where to save json file with results,
    Returns
        None
    """
    iou_array = np.array(iou_list)
    f1_array = np.array(f1_list)
    if classes == 3:
        results = {
            "Lung IoU Mean": iou_array[:, 0].mean(),
            "Lung IoU Std": iou_array[:, 0].std(),
            "Lung F1 Mean": f1_array[:, 0].mean(),
            "Lung F1 Std": f1_array[:, 0].std(),
            "Nodule IoU Mean": iou_array[:, -1].mean(),
            "Nodule IoU Std": iou_array[:, -1].std(),
            "Nodule F1 Mean": f1_array[:, -1].mean(),
            "Nodule F1 Std": f1_array[:, -1].std()}
    else:
        results = {
            "Nodule IoU Mean": iou_array.mean(), "Nodule IoU Std": iou_array.std(),
            "Nodule F1 Mean": f1_array.mean(), "Nodule F1 Std": f1_array.std()}

    for key, value in results.items():
        print(f'{key}: {value}')

    if save_pathway:
        with open(save_pathway, 'w', encoding='utf-8') as f:
            json.dump(results, f)
    return

def main():
    args=get_args()
    img_list = get_img_list(args.gt_path)
    stats_wrapper3D(img_list, args.gt_path, args.pt_path, args.classes)
    torch.cuda.empty_cache()

if __name__=='__main__':
    main()
