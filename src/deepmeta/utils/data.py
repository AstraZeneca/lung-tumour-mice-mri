"""data preparation script, taken from https://github.com/cbib/DeepMeta/tree/master/src/utils"""
import argparse
from typing import Any, Tuple, Dict

import numpy as np
import skimage.exposure as exposure  # type: ignore
import skimage.io as io  # type: ignore
import sklearn.model_selection as sk  # type: ignore
import torch  # type: ignore
import torch.utils.data as tud  # type: ignore
from torch.utils.data import DataLoader  # type: ignore

import src.deepmeta.utils.utils as utils


def get_datasets(
    path_imgs: str, path_labels: str, args: argparse.Namespace
) -> Dict[str, DataLoader[Any]]:
    """
    Get the datasets for the training and the validation set.

    :param path_imgs: Path to the images
    :type path_imgs: str
    :param path_labels: Path to the labels
    :type path_labels: str
    :param args: Arguments
    :type args: argparse.Namespace
    :return: Dictionary of the datasets
    :rtype: Dict[str, DataLoader[Any]]
    """
    img_path_list = utils.list_files_path(path_imgs)
    label_path_list = utils.list_files_path(path_labels)
    img_path_list, label_path_list = utils.shuffle_lists(img_path_list, label_path_list)
    # not good if we need to do metrics
    img_train, img_val = sk.train_test_split(
        img_path_list, test_size=0.2, random_state=42
    )
    label_train, label_val = sk.train_test_split(
        label_path_list, test_size=0.2, random_state=42
    )
    dataset_train = Deepmeta_dataset(args.batch_size, 128, img_train, label_train, dim=args.dim)
    dataset_val = Deepmeta_dataset(args.batch_size, 128, img_val, label_val, dim=args.dim)
    dataset_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.nworkers,
    )
    dataset_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers
    )
    return {"Train": dataset_train, "Val": dataset_val}

def contrast_and_reshape(img: "np.ndarray") -> "np.ndarray":
    """
    For some mice, we need to readjust the contrast.

    :param img: Slices of the mouse we want to segment
    :type img: np.array
    :return: Images list with readjusted contrast
    :rtype: np.array

    .. warning:
       If the contrast should not be readjusted,
        the network will fail prediction.
       Same if the image should be contrasted and you do not run it.
    """
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return np.array(img_adapteq)

class Deepmeta_dataset(tud.Dataset):
    """
    Dataset for the DeepMeta model.
    """
    def __init__(
        self,
        batch_size: int,
        img_size: int,
        input_img_paths: str,
        input_label_paths: str,
        dim: str = '2d'
    ) -> None:
        """
        Initialize the dataset.

        :param batch_size: Batch size
        :type batch_size: int
        :param img_size: Size of the images
        :type img_size: int
        :param input_img_paths: Path to the images
        :type input_img_paths: str
        :param input_label_paths: Path to the labels
        :type input_label_paths: str
        """
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.input_label_paths = input_label_paths
        self.dim = dim
        print(f"Nb of images : {len(input_img_paths)}")

    def __len__(self):
        return len(self.input_img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns tuple (input, target) correspond to batch #idx.
        """
        path = self.input_img_paths[idx]
        img = torch.Tensor(np.array(io.imread(path)) / 255)
        if self.dim=='2d':
            tensor_shape=(1,128,128)
        elif self.dim=='3d':
            tensor_shape=(1,128,128,128)
        img = img.expand(tensor_shape)
        path_lab = self.input_label_paths[idx]
        label = torch.Tensor(np.array(io.imread(path_lab)))
        label = label.expand(tensor_shape)
        return img, label
