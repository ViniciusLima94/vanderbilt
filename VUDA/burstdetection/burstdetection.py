import numba as nb
import numpy as np
import xarray as xr
from functools import partial
from tqdm import tqdm
from skimage import measure as ski


@nb.jit(nopython=True)
def arrays_equal(a, b):
    """
    Check if two arrays are equal.

    Parameters:
    - a (numpy.ndarray): First array.
    - b (numpy.ndarray): Second array.

    Returns:
    - bool: True if the arrays are equal, False otherwise.

    This function checks whether two arrays are equal by comparing their shapes
    and element-wise values.
    """
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True


def return_labeled_image(img: np.ndarray = None, threshold: float = None):
    """
    Label regions in a binary image using a given threshold.

    Parameters:
    - img (numpy.ndarray): The binary image to label.
    - threshold (float): The threshold for labeling.

    Returns:
    - numpy.ndarray: A labeled image with connected regions.
    - numpy.ndarray: Array of unique labels.
    - int: The number of unique labels.

    This function labels connected regions in a binary image based on a given threshold.
    It uses the `ski.measure.label` function from the scikit-image library to perform
    the labeling. The resulting labeled image contains connected regions with unique
    labels, and the number of labels is also returned.
    """
    labeled_image = ski.label(img > threshold, background=0)
    labels = np.unique(labeled_image)[1:]
    nlabels = len(labels)
    return labeled_image, labels, nlabels


def detect_bursts(
    spectra: xr.DataArray,
    init_threshold: float = None,
    min_threshold: float = 0,
    gamma: float = 1,
    zscore_dims: tuple = None,
    verbose: bool = False,
):
    """
    Detect bursts in spectra using a dynamic thresholding approach.

    Parameters:
    - spectra (xarray.DataArray): The input spectra to analyze.
    - init_threshold (float): The initial threshold for labeling. If none uses the floor
                              of the maximum value in the zsored data.
    - min_threshold (float): The minimum threshold to stop the labeling process.
    - gamma (float): The step size for updating the threshold.
    - zscore_dims (tuple): Dimensions along which to compute z-scores.

    Returns:
    - numpy.ndarray: An image with labeled bursts.

    This function detects bursts in a given spectra by iteratively updating a labeling
    based on threshold values. It starts with an initial threshold, and in each iteration,
    it updates the labeling using a lower threshold. The process continues until the
    threshold reaches the minimum threshold value. The resulting labeled image contains
    burst regions.

    Note: The input spectra are first z-scored before applying labeling.
    """

    size = spectra.shape
    z = (spectra - spectra.mean(zscore_dims)) / spectra.std(zscore_dims)
    return_labeled_image_partial = partial(return_labeled_image, img=z)

    if init_threshold is None:
        init_threshold = int(z.max(zscore_dims).min())  # int(np.max(z))

    # Create first labeld image
    labeled_image, labels, nlabels = return_labeled_image(z, init_threshold)

    thresholds = np.arange(init_threshold, min_threshold, -gamma, dtype=float) - gamma

    __iter = tqdm(thresholds) if verbose else thresholds

    # thr = init_threshold - gamma

    for thr in __iter:
        if verbose:
            __iter.set_description(f"thrshold = {thr}")
        new_labeled_image, new_labels, new_nlabels = return_labeled_image_partial(
            threshold=thr
        )

        indexes = np.zeros(new_labeled_image.shape, dtype=np.bool_)

        for label in new_labels:
            mask = new_labeled_image == label
            unique_labels = np.unique(mask * labeled_image)
            if len(unique_labels) > 2:
                indexes = np.logical_or(indexes, mask)

        not_indexes = np.logical_not(indexes)
        new_labeled_image = new_labeled_image * not_indexes + labeled_image * indexes

        old_labeled_image = labeled_image.copy()

        labeled_image, labels, nlabels = return_labeled_image(new_labeled_image, 0)

        # Break if there is no evolution of patches
        if arrays_equal(labeled_image > 0, old_labeled_image > 0):
            break

        del old_labeled_image

    return labeled_image
