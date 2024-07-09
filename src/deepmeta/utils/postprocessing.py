"""utils for postprocessing models from deepmeta paper, 
taken from https://github.com/cbib/DeepMeta/tree/master/src/utils"""
import cc3d
import cv2
import numpy as np
from skimage import measure
from scipy import ndimage

def remove_blobs(mask: "np.ndarray", min_size: int = 10) -> "np.ndarray":
    mask = mask.reshape(128, 128).astype(np.uint8)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    sizes = stats[1:, -1]
    nb_components = nb_components - 1  # remove background
    img2 = np.zeros(output.shape)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2


def dilate_and_erode(img: "np.ndarray", k1: int = 3, k2: int = 3) -> "np.ndarray":
    kernel1 = np.ones((k1, k1), np.uint8)
    kernel2 = np.ones((k2, k2), np.uint8)
    img_dilation = cv2.dilate(img, kernel1, iterations=1)
    return cv2.erode(img_dilation, kernel2, iterations=1)


def mean_vol_per_meta(mask: "np.ndarray", vol: float = 0.0047) -> float:
    _, num = measure.label(mask, return_num=True)
    nb_pix = mask.sum()
    return (nb_pix * vol) / num


def vol_mask(mask: "np.ndarray", vol: float = 0.0047) -> float:
    nb_pix = mask.sum()
    return nb_pix * vol


def process_meta_number(meta_masks: "np.ndarray") -> int:
    _, N = cc3d.connected_components(
        meta_masks, return_N=True, connectivity=18
    )
    return N


def laplace(img_stack: "np.ndarray", mask_list: "np.ndarray") -> "np.ndarray":
    """
    Remove false positives in lung segmentation. Apply a
    laplace of gaussian filter on slices, if the mean value of the
    result is <1 we remove the mask.

    .. note::
       We process only first and last slices (until we find a value >1).
       This ensure that we do not remove false
       negative slices.

    :param img_stack: Full image stack (dataset).
    :type img_stack: np.array
    :param mask_list: Full lung segmentation output
    :type mask_list: np.array
    :return: Updated mask list
    :rtype: np.array
    """
    img_stack2 = (img_stack * 255).astype(np.uint8)
    for i, img in enumerate(img_stack2):
        new_im = ndimage.gaussian_laplace(img, sigma=7)
        if np.mean(new_im) < 1:
            mask_list[i] = np.zeros((128, 128))
        else:
            break
    for i, img in enumerate(img_stack2[::-1]):
        new_im = ndimage.gaussian_laplace(img, sigma=7)
        if np.mean(new_im) < 1:
            mask_list[(len(mask_list) - 1) - i] = np.zeros((128, 128))
        else:
            break
    return mask_list


def sanity_check(mask_list: "np.ndarray") -> "np.ndarray":
    """
    Check if there is some false positive. If mask < 15px -> mask is null.
    If i-1 and i+1 do not contain mask, i does not contains a mask either.

    :param mask_list: Lungs segmentation output
    :type mask_list: np.array
    :return: Checked segmentation output
    :rtype: np.array
    """
    mask_list[0] = np.zeros((128, 128))
    mask_list[-1] = np.zeros((128, 128))
    for i in range(1, len(mask_list) - 1):
        if mask_list[i].sum() > 15:
            if mask_list[i - 1].sum() < 15 and mask_list[i + 1].sum() < 15:
                mask_list[i] = np.zeros((128, 128))
        else:
            mask_list[i] = np.zeros((128, 128))
    return mask_list


def postprocess(inputs: "np.ndarray", masks: "np.ndarray") -> "np.ndarray":
    masks = laplace(inputs, masks)
    lungs_masks = masks > 0.5
    metas_masks = masks > 1.5
    lungs_masks = sanity_check(lungs_masks)
    _lungs_masks = [remove_blobs(mask, 10) for mask in lungs_masks]
    __lungs_masks = (
        np.array([dilate_and_erode(mask, 3, 3) for mask in _lungs_masks]) / 255
    )
    _metas_masks = [remove_blobs(mask, 3) for mask in metas_masks]
    __metas_masks = (
        np.array([dilate_and_erode(mask, 3, 3) for mask in _metas_masks]) / 255
    )
    # return lungs_masks + metas_masks
    return np.where((__metas_masks == 1), 2, __lungs_masks)
