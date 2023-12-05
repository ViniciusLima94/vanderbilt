import numpy as np
import xarray as xr
import logging
from functools import partial
from frites.utils import parallel_func
import PyEMD
import emd
from .emd_utils import get_data_blocks


def emd_pyEMD(
    data: np.ndarray,
    trials: int,
    max_imf: int = 10,
    block_size=None,
    n_jobs: int = 1,
    verbose: bool = False,
    seed: int = 0,
    use_min_block_size: bool = False,
    method="emd",
):
    """
    Perform Empirical Mode Decomposition (EMD) or Ensemble Empirical Mode Decomposition (EEMD) on the input data.

    Parameters:
    - data (np.ndarray): Input data for EMD.
    - trials (int): Number of trials for EEMD. Set to 1 for standard EMD.
    - max_imf (int): Maximum number of Intrinsic Mode Functions (IMFs) to extract.
    - block_size: Size of data blocks for parallel processing. If None, the entire data is processed as a single block.
    - n_jobs (int): Number of parallel jobs.
    - verbose (bool): If True, print progress information.
    - seed (int): Seed for reproducibility.
    - use_min_block_size (bool): If True, use the minimum block size for parallel processing.
    - method (str): EMD method, either "emd" for standard EMD or "eemd" for Ensemble EMD.

    Returns:
    - out: List of IMFs for each trial.
    """

    assert method in ["emd", "eemd"]

    if method == "emd":
        logging.warning("For method EMD, trials is set to 1.")
        trials = 1

    n_samples = data.shape[0]

    _emd = PyEMD.EEMD(trials=trials)
    _emd.noise_seed(seed)

    if method == "emd":
        f_emd = partial(_emd.emd, T=None, max_imf=max_imf)
    else:
        f_emd = partial(_emd.eemd, progress=False, max_imf=max_imf)

    if isinstance(block_size, int) and (block_size > 1):
        blocks = get_data_blocks(n_samples, block_size, use_min_block_size)
    else:
        blocks = [np.arange(n_samples)]

    nblocks = len(blocks)

    def _for_block(block):
        """Perform EMD on a data block."""
        return f_emd(data[block])

    # Define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_block, verbose=verbose, n_jobs=n_jobs, total=nblocks
    )

    out = parallel(p_fun(block) for block in blocks)

    return out


def emd_vec(
    x,
    times,
    method="emd",
    max_imfs=None,
    nensembles=None,
    block_size=None,
    verbose=False,
    use_min_block_size=False,
    imf_opts={},
    n_jobs=1,
):
    """
    Perform Empirical Mode Decomposition (EMD) or
    Ensemble Empirical Mode Decomposition (EEMD) on a vector.

    Base code by: @e.combrisson

    Parameters:
    - x: Input vector for EMD.
    - times: Time array of the time-series in the vector x.
    - method (str): EMD method, either "emd" for standard EMD or "eemd" for Ensemble EMD.
    - max_imfs (int): Maximum number of Intrinsic Mode Functions (IMFs) to extract.
    - nensembles (int): Number of ensembles for EEMD. Set to 1 for standard EMD.
    - block_size: Size of data blocks for parallel processing. If None, the entire data is processed as a single block.
    - verbose (bool): If True, print progress information.
    - use_min_block_size (bool): If True, use the minimum block size for parallel processing.
    - imf_opts (dict): Additional options for IMF extraction.
    - n_jobs (int): Number of parallel jobs.

    Returns:
    - out: List of IMFs for each vector block.
    """

    assert method in ["emd", "eemd"]

    f_emd = dict(emd=emd.sift.sift, eemd=emd.sift.ensemble_sift)
    f_emd_args = dict(max_imfs=max_imfs, imf_opts=imf_opts)

    if method == "emd":
        logging.warning("For method EMD, ensemble size is set to 1.")
        nensembles = 1

    n_samples = x.shape[0]

    if isinstance(block_size, int) and (block_size > 1):
        blocks = get_data_blocks(n_samples, block_size, use_min_block_size)
        block_times = np.vstack(
            [(times[block].min(), times[block].max()) for block in blocks]
        )
    else:
        blocks = [np.arange(n_samples)]

    nblocks = len(blocks)

    if method == "emd":
        f_emd = partial(f_emd[method], **f_emd_args)
    else:
        f_emd = partial(f_emd[method], nensembles=nensembles, **f_emd_args)

    def _for_block(block):
        """Perform EMD on a vector."""
        # extract single trial imf
        imf = f_emd(x[block]).T
        imf = xr.DataArray(imf, dims=("IMFs", "times"),)
        return imf

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(
        _for_block, verbose=verbose, n_jobs=n_jobs, total=nblocks
    )

    out = parallel(p_fun(block) for block in blocks)

    out = xr.concat( out, "blocks")

    out.attrs["block_times"] = block_times

    return out
