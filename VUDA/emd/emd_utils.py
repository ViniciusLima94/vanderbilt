from functools import partial
import numpy as np


def get_extreme_size(x, f_max_min=None):
    """
    Get the maximum or minimum size among the input list of arrays.

    Parameters:
    - x (list of np.ndarray): List of arrays for which to find the size.
    - f_max_min (function, optional): Function to compute the maximum or minimum.
                                     Defaults to np.min.

    Returns:
    - int: Maximum or minimum size among the input arrays.
    """
    return f_max_min([x_.shape[0] for x_ in x])


get_min_size = partial(get_extreme_size, f_max_min=np.min)
get_max_size = partial(get_extreme_size, f_max_min=np.max)


def get_data_blocks(n_samples, n_blocks, use_min_block_size):
    """
    Divide a range of samples into blocks.

    Parameters:
    - n_samples (int): Total number of samples to be divided.
    - n_blocks (int): Number of blocks to divide the samples into.
    - use_min_block_size (bool): Flag to indicate whether to use the minimum block size.

    Returns:
    - list of np.ndarray: List of arrays representing the blocks of samples.
    """
    # Split the range of samples into blocks
    blocks = np.array_split(np.arange(n_samples), n_blocks)
    # If using the minimum block size, truncate each block to the minimum size
    if use_min_block_size:
        min_block_size = get_min_size(blocks)
        blocks = np.array([block[:min_block_size] for block in blocks])
    return blocks
