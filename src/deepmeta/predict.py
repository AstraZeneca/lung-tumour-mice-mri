"""modfied predict script from https://github.com/cbib/DeepMeta """
import argparse
import os
import json
from typing import List
import numpy as np
import torch
from skimage import io, transform
from torch import nn
from tqdm import tqdm
from src.deepmeta.utils import data
from src.deepmeta.utils import postprocessing as pp
from src.deepmeta.utils import pprint
from src.deepmeta.utils import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args() -> argparse.Namespace:
    """
    Argument parser.

    :return: Object containing all the parameters needed to train a model
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-classes",
        type=int,
        default=3,
        help="Number of classes to predict."
    )
    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        help="If flag, save predictions."
    )
    parser.set_defaults(save=False)
    parser.add_argument(
        "--postprocess",
        dest="postprocess",
        action="store_true",
        help="If flag, apply postprocess on predictions.",
    )
    parser.set_defaults(postprocess=False)
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        help="Model name. unet or unet_res",
        choices=[
            "unet",
            "unet3p",
            "deepmeta",
            "vanilla_unet",
        ],
    )
    parser.add_argument(
        "--model_path", type=str,
        default="Unet_full",
        help="Weights file name."
    )
    parser.add_argument(
        "--dim",
        type=str,
        default="2d",
        help="2d or 3d"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="iou",
        help="Metric for stats. iou or f1",
        choices=["iou", "f1"],
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the data directory "
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='results',
        help="Path to dir where saving models and output."
    )
    args = parser.parse_args()
    pprint.print_bold_red(args)
    return args


def load_model(config: argparse.Namespace,
               device: str = "cuda") -> nn.Module:
    """
    Loads the model from the given path

    :param path: Path to the model
    :type path: str
    :param config: Configuration object
    :type config: argparse.Namespace
    :param device: Device to load the model on.
    :type device: str
    :return: The model with weights loaded
    :rtype: nn.Module
    """
    model = utils.get_model(config)
    if config.dim=='3d':
        model=nn.DataParallel(model)
    model.load_state_dict(
        torch.load(f"{config.model_path}.pth", map_location=device))
    model.eval()
    return model.cuda()


def get_predict_dataset(path_souris,
                        contrast=True,
                        dim='2d'):
    """
    Creates an image array from a file path (tiff file).

    :param path_souris: Path to the mouse file.
    :type path_souris: str
    :param contrast: Flag to run contrast and reshape
    :type contrast: Bool
    :return: Images array containing the whole mouse
    :rtype: np.array
    """
    mouse = io.imread(path_souris, plugin="tifffile").astype(np.uint8)
    mouse = transform.resize(mouse, (len(mouse), 128, 128), anti_aliasing=True)
    mouse = np.array(mouse) / np.amax(mouse)
    if contrast:
        mouse = data.contrast_and_reshape(mouse)
    else:
        if dim=='2d':
            mouse = np.array(mouse).reshape(-1, 1, 128, 128)
        else:
            mouse = np.array(mouse).reshape(-1, 128, 128, 128)
    return mouse


def get_labels(path: str) -> List["np.ndarray"]:
    """
    Loads the labels from the given path

    :param path: Path to the labels
    :type path: str
    :return: List of labels
    :rtype: List[np.array]
    """
    file_list = utils.list_files_path(path)
    return [io.imread(file, plugin="tifffile").astype(np.uint8) for file in file_list]


def process_img(mouse: torch.Tensor,
                net: nn.Module) -> List:
    """runs inference on each 2D slice form a 3D stack. returns 
    a list where each item is an prediction mask for a 2d slice
    Args:
        mouse (torch.Tensor): 3D stack to be used as input
        net (nn.Module): net to be used for inference
    Return:
        list of 2d prediction masks, constituting a 3d prediction stack
    """
    output_list = []
    with torch.no_grad():
        for slice in mouse:
            slice = slice.reshape(1, 1, 128, 128)
            slice = torch.from_numpy(slice).float().cuda()
            output = net(slice)
            output = output.max(1).indices
            output_list.append(output.cpu().detach().numpy())
    return output_list

def process_img3d(mouse: torch.Tensor,
                  net: nn.Module) -> List:
    """runs inference on the whole 3D stack and returns 3d prediction
    mask
    Args:
        mouse (torch.Tensor): 3D stack to be used as input
        net (nn.Module): net to be used for inference
    Return:
        3D prediction mask
    """
    output_list = []
    slice = mouse
    slice = slice.reshape(1,1, 128, 128, 128)
    slice = torch.from_numpy(slice).float().cuda()
    with torch.no_grad():
        output = net(slice)
        output = output.max(1).indices.squeeze()
        output_list.append(output.cpu().detach().numpy())
    return output_list

def stats3d(args:argparse.Namespace,
            pt_stack:list,
            gt_stack:list,
            metric:str=None) -> list:
    """calculates the 3d score between prediction and ground truth 
    mask
    Args:
        args (argparse.Namespace) - args to be used
        pt_stack (list) - list of 3d stacks that are predicted
        gt_stack (list) - list of 3d stacks that are gt
        metric (str) - metric of choice, iou or f1. If none, 
                        metric from args is taken
    Return:
        list of scores
    """
    if metric is None:
        metric=args.metric
    if gt_stack is not None:
        res = []
        img_f = np.array(pt_stack).flatten()
        label_f = np.array(gt_stack).flatten()
        if args.classes==3:
            labels=[0,1,2]
        else:
            labels=[0,1]
        res.append(
            utils.get_metric(metric)(
                label_f, img_f, average=None, labels=labels, zero_division=1
            )[1:]
        )
        return res

def stats(args:argparse.Namespace,
          pt_stack:list,
          gt_stack:list,
          metric:str=None) -> list:
    """calculates the 2d score between prediction and ground truth 
    mask
    Args:
        args (argparse.Namespace) - args to be used
        pt_stack (list) - list of 2d slices that are predicted
        gt_stack (list) - list of 2d slices that are gt
        metric (str) - metric of choice, iou or f1. If none, 
                        metric from args is taken
    Return:
        list of scores
    """
    if metric is None:
        metric=args.metric
    if gt_stack is not None:
        res = []
        if len(pt_stack)>1:
            for i, output in enumerate(pt_stack):
                img_f = output.flatten()
                label_f = gt_stack[i].flatten()
                if args.classes==3:
                    labels=[0,1,2]
                else:
                    labels=[0,1]
                res.append(
                    utils.get_metric(metric)(
                        label_f, img_f, average=None, labels=labels, zero_division=1)[1:]
                )
        else:
            output=pt_stack[0]
            for i, output_i in enumerate(output):
                img_f = output_i.flatten()
                label_f = gt_stack[i].flatten()
                if args.classes==3:
                    labels=[0,1,2]
                else:
                    labels=[0,1]
                res.append(
                    utils.get_metric(metric)(
                        label_f, img_f, average=None, labels=labels, zero_division=1)[1:]
                )
        return res

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

def save_imgs(args:argparse.Namespace,
              save_name:str,
              mouse_list:list,
              output_list:list,
              mouse_labels:list):
    """save a figure of input, gt, and pt under specified directory
    Args:
        args(argparse.Namespace) -config object
        save_name (str): name under which file should be saved
        mouse_list (list): list of 2d/3d inputs to be saved
        output_list (list): list of 2d/3d pt to be saved
        mouse_labels (list): list of 2d/3d labels to be saved
    """
    os.makedirs(os.path.join(args.save_dir,save_name), exist_ok=True)
    if len(output_list)==1:
        output_list =output_list[0]
    if len(mouse_list)==1:
        mouse_list = mouse_list[0]
    for j, (slice2d, output, label) in enumerate(
        zip(mouse_list, output_list, mouse_labels)):
        utils.save_pred(
            slice2d.reshape(128, 128),
            output.reshape(128, 128),
            label.reshape(128, 128),
            args.classes,
            f"{args.save_dir}/{save_name}/{j}.png",
        )

def main():
    args = get_args()
    model = load_model(args).cuda()
    pprint.print_gre(f"Model {args.model_path} loaded")
    test_names = [
        ("8_m2PL_day25", True),
        ("28_2Pc_day50", True),
        ("56_2PLc_day106", True),
        ("m2Pc_c1_10Corr_1", False),
    ]
    iou_list_stacks, f1_list_stacks,iou_list_slices, f1_list_slices = [],[],[],[]
    for name, contrast in tqdm(test_names):
        mouse = get_predict_dataset(f"{args.data_dir}/{name}.tif", contrast=contrast, dim=args.dim)
        mouse_labels = get_labels(f"{args.data_dir}/{name}/3classes/")
        if args.dim=='2d':
            output_stack = process_img(mouse, model)
        else:
            output_stack = process_img3d(mouse, model)
        if args.postprocess:
            output_stack = pp.postprocess(mouse, np.array(output_stack))
            mouse_labels = pp.postprocess(mouse, np.array(mouse_labels))
        res_2d= stats(args, output_stack, mouse_labels, 'iou')
        res_3d = stats3d(args, output_stack, mouse_labels, 'iou')
        iou_list_stacks.extend(res_3d)
        iou_list_slices.extend(res_2d)
        res_2d= stats(args, output_stack, mouse_labels, 'f1')
        res_3d = stats3d(args, output_stack, mouse_labels, 'f1')
        f1_list_stacks.extend(res_3d)
        f1_list_slices.extend(res_2d)
        if args.save:
            save_imgs(args, name, mouse, output_stack, mouse_labels)
    print('2D')
    print_results(iou_list_slices,
                  f1_list_slices,
                  args.classes,
                  f"{args.save_dir}/{args.model}_results_2d.json")
    print('3D')
    print_results(iou_list_stacks,
                  f1_list_stacks,
                  args.classes,
                  f"{args.save_dir}/{args.model}_results_3d.json")
    print('results saved')
    print("\n\n\n")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
