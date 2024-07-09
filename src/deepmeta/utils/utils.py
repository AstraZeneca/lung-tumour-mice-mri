"""misc utils, taken from https://github.com/cbib/DeepMeta/tree/master/src/utils"""
import argparse
import os
import random
import re
import numpy as np
import torch

from typing import Any, List, Tuple
from sklearn import metrics
from torch.nn import init
from matplotlib import pyplot as plt
from src.deepmeta.models import unet
import src.deepmeta.loss as ls

def list_files_path(path: str) -> list:
    """
    List files from a path.

    :param path: Folder path
    :type path: str
    :return: A list containing all files in the folder
    :rtype: List
    """
    return sorted_alphanumeric(
        [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    )


def shuffle_lists(lista: List, listb: List, seed: int = 42) -> Tuple[List, List]:
    """
    Shuffle two list with the same seed.

    :param lista: List of elements
    :type lista: List
    :param listb: List of elements
    :type listb: List
    :param seed: Seed number
    :type seed: int
    :return: lista and listb shuffled
    :rtype: (List, List)
    """
    random.seed(seed)
    random.shuffle(lista)
    random.seed(seed)
    random.shuffle(listb)
    return lista, listb


def sorted_alphanumeric(data: List[str]) -> list:
    """
    Sort function.

    :param data: str list
    :type data: List
    :return: Sorted list
    :rtype: List
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa
    return sorted(data, key=alphanum_key)

def display_img(
    img: np.ndarray, pred: np.ndarray, label: np.ndarray, cmap: str = "gray"
):
    """
    Display an image.

    :param img: Image to display
    :type img: numpy.ndarray
    :param pred: Prediction
    :type pred: numpy.ndarray
    :param label: Label
    :type label: numpy.ndarray
    :param cmap: Color map
    :type cmap: str
    """
    if img.shape[0] == 1:
        img = np.moveaxis(img, 0, -1)
        label = np.moveaxis(label, 0, -1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(img, cmap=cmap)
    ax1.set_title("Input image")
    ax2.imshow(pred, cmap=cmap)
    ax2.set_title("Prediction")
    ax3.imshow(label, cmap=cmap)
    ax3.set_title("Label")
    plt.show()


def save_pred(img: np.ndarray, pred: np.ndarray, label: np.ndarray, classes:int, path: str):
    """
    Save an image.

    :param img: Image to save
    :type img: numpy.ndarray
    :param pred: Prediction
    :type pred: numpy.ndarray
    :param label: Label
    :type label: numpy.ndarray
    :param path: Path to save
    :type path: str
    """
    if classes==3:
        labels=[0,1,2]
    else:
        labels=[0,1]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(img, cmap="gray")
    ax1.set_title("Input image")
    ax2.imshow(pred, cmap="gray")
    ax2.set_title("Prediction")
    if label is not None:
        ax3.imshow(label, cmap="gray")
        ax3.set_title("Label")
    iou = metrics.jaccard_score(
                    label.flatten().round(),
        pred.flatten().round(),
        average=None,
        labels=labels,
        zero_division=1
                )
    dice= metrics.f1_score(
                    label.flatten().round(),
        pred.flatten().round(),
        average=None,
        labels=labels,
        zero_division=1
                )
    fig.text(.5, .05, f'IoU: {iou}; \n Dice: {dice}', ha='center')
    plt.savefig(path)
    plt.close()


def weights_init_kaiming(m: Any) -> None:  # noqa
    # He initialization
    classname = m.__class__.__name__
    if classname.find("SeparableConv") != -1:
        init.kaiming_normal_(m.depthwise.weight.data, a=0, mode="fan_in")
        init.kaiming_normal_(m.pointwise.weight.data, a=0, mode="fan_in")
    elif classname.find("DoubleConv") != -1:
        for elt in m.double_conv:
            if elt.__class__.__name__.find("SeparableConv") != -1:
                init.kaiming_normal_(elt.depthwise.weight.data, a=0, mode="fan_in")
                init.kaiming_normal_(elt.pointwise.weight.data, a=0, mode="fan_in")
            elif elt.__class__.__name__.find("Conv") != -1:
                init.kaiming_normal_(elt.weight.data, a=0, mode="fan_in")
    elif classname.find("OutConv") != -1:
        try:
            init.kaiming_normal_(m.conv.weight.data, a=0, mode="fan_in")
        except Exception:
            init.kaiming_normal_(m.conv.depthwise.weight.data, a=0, mode="fan_in")
            init.kaiming_normal_(m.conv.pointwise.weight.data, a=0, mode="fan_in")
    elif classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def plot_learning_curves(
    hist_train: List[float], hist_val: List[float], path: str = "data/plots/"
) -> None:
    """
    Plot training curves.

    :param hist_train: List of epoch loss value for training
    :type hist_train: list
    :param hist_val: List of epoch loss value for validation
    :type hist_val: list
    :param path: Saving path
    :type path: str

    """
    os.system(f"mkdir -p {path}")
    epochs = range(len(hist_train))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, hist_train, label="Train loss")
    plt.plot(epochs, hist_val, label="Validation loss")
    plt.title("Loss - Training vs. Validation.")
    plt.ylabel("Loss ")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"{path}training_curves_unet3p.png")


def get_model(args: argparse.Namespace) -> torch.nn.Module:
    """
    Load and init the model.

    :param args: Arguments
    :type args: argparse.Namespace
    :return: Model
    :rtype: torch.nn.Module
    """
    if args.model == "unet":
        model = unet.Unet(
            classes=args.classes, dim=args.dim
        )
    elif args.model == "unet3p":
        model = unet.Unet3plus(
            classes=args.classes, dim=args.dim
        )
    elif args.model == "deepmeta":
        model = unet.Unet3plus(
            classes=args.classes, dim=args.dim
        )
    else:
        raise ValueError(f"Unknown model {args.model}")
    model.apply(weights_init_kaiming)
    return model


def get_metric(arg:str) -> Any:
    """
    Get the metric.
    """
    if arg== "f1":
        metric = metrics.f1_score
    elif arg == "iou":
        metric = metrics.jaccard_score
    else:
        raise ValueError(f"Unknown metric {args.metric}")
    return metric

def get_custom_weights(nclass: int) -> list:
    return [1.0, 5.0, 15.0] if nclass == 3 else [1.0, 15.0]

def get_loss(args: argparse.Namespace,
             device: str = "cuda",
             nclass=3,
             dim='2d') -> Any:
    """
    Get the loss function.

    :param args: Arguments
    :type args: argparse.Namespace
    :param device: device
    :type device: str
    :return: Convolutional layer
    :rtype: torch.nn.Module
    """
    loss_dict = {
        'weighted_ce': {
            '2d': ls._2d.Weighted_Cross_Entropy_Loss,
            '3d': ls._3d.Weighted_Cross_Entropy_Loss
        },
        'fusion': {
            '2d': lambda: ls._2d.FusionLoss(device=device,
                                            custom_weights=get_custom_weights(nclass)),
            '3d': lambda: ls._3d.FusionLoss(device=device,
                                            custom_weights=get_custom_weights(nclass))
        },
        'unet3p_loss': {
            '2d': ls._2d.U3PLloss,
            '3d': ls._3d.U3PLloss
        }}

    if args.loss not in loss_dict:
        raise ValueError(f"Unknown loss function {args.loss}")
    if dim not in loss_dict[args.loss]:
        raise ValueError(f"Unknown dimension {dim}")
    criterion = loss_dict[args.loss][dim]().to(device)
    return criterion
