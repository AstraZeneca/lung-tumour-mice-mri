"""evaluate nnunet performance for specified exp w postprocessing; note that this script
is different from others as it was designed to compare with DeepMeta prediction script"""
import os
import argparse
import json
import numpy as np
import nibabel
from skimage import io
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
import src.deepmeta.utils.postprocessing as pp

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
        "--img_path", type=str, default=None, help="Path to the data dir with imgs."
    )
    parser.add_argument(
        "--gt_path", type=str, default=None, help="Path to the data dir with ground truth."
    )
    parser.add_argument(
        "--pt_path", type=str, default=None, help="Path to the data dir with predictions."
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="path where to save json.results post-processed"
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

def get_img_list(path:str,
                 id:int) -> list:
    """list all .tif or .nii.gz files specified in the pathway and starting with id
    Args:
        path (str): pathway to where all the files are stored
        id (int):  integer indicating a mouse

    Returns:
        list: list of all .tif or .nii.gz files
    """
    img_list = os.listdir(path)
    img_list = [img for img in img_list if img.endswith(('.tif','.nii.gz'))]
    img_list = [img for img in img_list if img[9]==str(id)]
    return img_list

def re_stack(path:str,
             slice_list:list) -> np.array:
    """restacking the 2D annotation slices onto 3D annotation slices 

    Args:
        slice_list (list): list of slice filenames
    Returns
        np.array of 3D stacked annotations
    """
    plain_stack = np.zeros((128,128,128))
    for slice_name in slice_list:
        id = int(slice_name.split('_')[1].split('.')[0][1:])
        print(id)
        pathway = os.path.join(path, slice_name)
        plain_stack[id] = io.imread(pathway)
    return plain_stack
    
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

def stats(output_stack:list,
          mouse_labels:list,
          metric:str):
    """
    calculates f1 and iou score for provided ground truth mask and prediction
    Args:
        output_stack (np.array): ground truth mask
        mouse_labels (np.array): predicted mask
        metric (str): metric to be used for calculations
    Returns:
        tuple: mean score and associated std for the stack, as done in the deepmeta paper
    """
    if mouse_labels is not None:
        res = []
        for i, output in enumerate(output_stack):
            img_f = output.flatten()
            label_f = mouse_labels[i].flatten()
            if metric=='f1':
                res.append(
                    f1_score(
                        label_f, img_f, average=None, labels=[0,1,2], zero_division=1
                    )
                )
            else:
                res.append(
                    jaccard_score(
                        label_f, img_f, average=None, labels=[0,1,2], zero_division=1
                    )
                )
        res_mean = np.array(res).mean(0)
        print('mean')
        print(res_mean)
        res_std = np.array(res).std(0)
        print('std')
        print(res_std)
        return res_mean, res_std

def main():
    args=get_args()
    iou_list_mean = []
    iou_list_std = []
    f1_list_mean = []
    f1_list_std = []
    for id in [0,1,2,3]:
        print('Mouse ID ',id)
        img_list = get_img_list(args.img_path,id)
        img = re_stack(args.img_path, img_list)
        gt_list = get_img_list(args.gt_path,id)
        gt = re_stack(args.gt_path, gt_list)
        pt_list = get_img_list(args.pt_path,id)
        pt = re_stack(args.pt_path, pt_list)
        pp_gt = pp.postprocess(img,gt)
        pp_pt = pp.postprocess(img,pt)
        iou_list_mean.append(stats(pp_pt, pp_gt,'iou')[0][:])
        iou_list_std.append(stats(pp_pt, pp_gt,'iou')[1][:])
        f1_list_mean.append(stats(pp_pt, pp_gt,'f1')[0][:])
        f1_list_std.append(stats(pp_pt, pp_gt,'f1')[1][:])
        io.imsave(f'pp_gt{id}.tif', pp_gt)
        io.imsave(f'pp_pt{id}.tif', pp_pt)
    print("Total stats IoU:")
    print('Mean')
    mean_results = np.array(iou_list_mean).mean(0)
    print(mean_results)
    print('STD')
    std_results = np.array(iou_list_std).std(0)
    print(std_results)
    print("Total stats f1:")
    print('Mean')
    mean_results = np.array(f1_list_mean).mean(0)
    print(mean_results)
    print('STD')
    std_results = np.array(f1_list_std).std(0)
    print(std_results)
    
if __name__=='__main__':
    main()
