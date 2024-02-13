import os
import argparse
from functools import partial
import numpy as np
import xarray as xr
import scipy
import jax
import jax.numpy as jnp
from tqdm import tqdm
from frites.utils import parallel_func
from mne.time_frequency import tfr_array_morlet
from config import metadata, method, max_imfs
from skimage import measure as ski


##############################################################################
# Argument parsing
##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("SESSION_ID", help="id of the session to be used", type=int)
parser.add_argument(
    "CONDITION",
    help="to select sleep or task condition",
    type=str,
    choices=["task", "sleep"],
)
parser.add_argument(
    "STD_IMFS",
    help="whether to standardize number of IMFs per block or not",
    type=int,
    choices=[0, 1],
)

args = parser.parse_args()

monkey = args.MONKEY
sid = args.SESSION_ID
condition = args.CONDITION
std = bool(args.STD_IMFS)

session = metadata["monkey"][monkey]["dates"][sid]


##############################################################################
# Helper functions
##############################################################################


# def return_labeled_image(img: np.ndarray = None, threshold: float = None):
    # """
    # Label regions in a binary image using a given threshold.

    # Parameters:
    # - img (numpy.ndarray): The binary image to label.
    # - threshold (float): The threshold for labeling.

    # Returns:
    # - numpy.ndarray: A labeled image with connected regions.
    # - numpy.ndarray: Array of unique labels.
    # - int: The number of unique labels.

    # This function labels connected regions in a binary image based on a given threshold.
    # It uses the `ski.measure.label` function from the scikit-image library to perform
    # the labeling. The resulting labeled image contains connected regions with unique
    # labels, and the number of labels is also returned.
    # """
    # labeled_image = ski.label(img > threshold, background=0)
    # labels = np.unique(labeled_image)[1:]
    # nlabels = len(labels)
    # return labeled_image, labels, nlabels


# def detect_bursts(
    # spectra: xr.DataArray,
    # init_threshold: float,
    # min_threshold: float,
    # gamma: float,
    # relative: bool = True,
# ):
    # """
    # Detect bursts in spectra using a dynamic thresholding approach.

    # Parameters:
    # - spectra (numpy.ndarray): The input spectra to analyze.
    # - init_threshold (float): The initial threshold for labeling.
    # - min_threshold (float): The minimum threshold to stop the labeling process.
    # - gamma (float): The step size for updating the threshold.
    # - relative (bool, optional): If True, z-score is computed relative to the mean
                                 # along the "times" dimension; otherwise, it's computed
                                 # relative to the overall mean.

    # Returns:
    # - numpy.ndarray: An image with labeled bursts.

    # This function detects bursts in a given spectra by iteratively updating a labeling
    # based on threshold values. It starts with an initial threshold, and in each iteration,
    # it updates the labeling using a lower threshold. The process continues until the
    # threshold reaches the minimum threshold value. The resulting labeled image contains
    # burst regions.

    # Note: The input spectra are first z-scored before applying labeling.
    # """

    # size = spectra.shape
    # mean_dim = "times" if relative else None
    # z = (spectra - spectra.mean(mean_dim)) / spectra.std(mean_dim)
    # return_labeled_image_partial = partial(return_labeled_image, img=z)

    # labeled_image, _, _ = return_labeled_image(z, init_threshold)
    # thr = init_threshold - gamma

    # while thr >= min_threshold:
        # new_labeled_image, new_labels, _ = return_labeled_image_partial(threshold=thr)

        # indexes = np.zeros(new_labeled_image.shape, dtype=np.bool_)

        # for label in new_labels:
            # mask = new_labeled_image == label
            # unique_labels = np.unique(mask * labeled_image)
            # if len(unique_labels) > 2:
                # indexes = np.logical_or(indexes, mask)

        # not_indexes = np.logical_not(indexes)
        # new_labeled_image = new_labeled_image * not_indexes + labeled_image * indexes

        # labeled_image, _, _ = return_labeled_image(new_labeled_image, 0)
        # thr -= gamma

    # return labeled_image


from VUDA.burstdetection import detect_burts


##############################################################################
# Load data
##############################################################################


composites_path = os.path.expanduser(
    f"~/funcog/HoffmanData/{monkey}/{session}/composite_signals_task_method_eemd_max_imfs_None_std_{std}.nc"
)

ps_composites_path = os.path.expanduser(
    f"~/funcog/HoffmanData/{monkey}/{session}/ps_composite_signals_task_method_eemd_max_imfs_None_std_{std}.nc"
)


composites = xr.open_dataset(composites_path)
ps_composites = xr.open_dataset(ps_composites_path)

channels = list(composites.keys())
freqs = ps_composites.freqs.data

n_channels = len(channels)
n_blocks = composites.sizes["blocks"]

X = []

bands = {}

for channel in tqdm(channels):
    data = ps_composites[channel].load().dropna("IMFs")
    x = composites[channel].load().dropna("IMFs")

    freqs = data.freqs.data
    kernel = np.hanning(50)

    data_sm = xr.DataArray(
        scipy.signal.fftconvolve(data, kernel[None, None, :], mode="same", axes=2),
        dims=data.dims,
        coords=data.coords,
    )

    freqs = data.freqs.data

    n_blocks, n_IMFs, n_freqs = data.shape

    peaks = freqs[data_sm.argmax("freqs")]

    # Defining limits of slow and fast rythms
    min_peaks = peaks.min(0)
    # min_theta = min_peaks[np.argmin(np.abs(min_peaks - 3))]
    min_theta = peaks.min()

    max_peaks = peaks.max(0)
    max_theta = max(np.ceil(max_peaks[np.argmin(np.abs(max_peaks - 10))]), 10)

    min_gamma = min(peaks.min(0)[-1], 50)

    slow_idx = np.logical_and(
        peaks.flatten() >= min_theta, peaks.flatten() <= max_theta
    ).reshape(peaks.shape)

    fast_idx = np.logical_and(
        peaks.flatten() >= min_gamma, peaks.flatten() <= np.inf
    ).reshape(peaks.shape)

    slow = (x * slow_idx[..., None]).sum("IMFs")
    fast = (x * fast_idx[..., None]).sum("IMFs")
    bands[channel] = {}
    bands[channel]["theta"] = [min_theta, max_theta]
    bands[channel]["gamma"] = [min_gamma, 150]
    X += [xr.concat((slow, fast), "components")]

X = xr.concat(X, "channels")
X = X.assign_coords({"channels": channels})

##############################################################################
# Decompose the signal in time and frequency domain
##############################################################################


def _for_batch(W):
    init = int(((W.max("times") - W.mean("times")) / W.std("times")).max().data.item())
    init = np.floor(init)
    return detect_bursts(W, init, 0, 0.1,
                         zscore_dims=("times", "freqs"),
                         verbose=True)


parallel, p_fun = parallel_func(_for_batch, verbose=False, n_jobs=20, total=n_blocks)

BURSTS_SLOW = []
BURSTS_FAST = []

# Path in which to save the bursts
SAVE_TO = os.path.expanduser(f"~/funcog/HoffmanData/{monkey}/{session}/bursts")

# If path does not exist
if not os.path.exists(SAVE_TO):
    os.makedirs(SAVE_TO)

for channel in channels:

    fvec_theta = np.linspace(bands[channel]["theta"][0], bands[channel]["theta"][1], 30)
    fvec_gamma = np.linspace(bands[channel]["gamma"][0], bands[channel]["gamma"][1], 30)

    n_cycles_theta = np.floor(X.shape[-1] / 10000 * (2.0 * np.pi * fvec_theta[0]))
    W_theta = tfr_array_morlet(
        X.sel(components=0, channels=[channel]).transpose(
            "blocks", "channels", "times"
        ),
        1000,
        fvec_theta,
        n_cycles=n_cycles_theta,
        decim=10,
        n_jobs=20,
    ).squeeze()

    dims = ("batches", "freqs", "times")
    coords = dict(freqs=fvec_theta)
    W_theta = xr.DataArray((W_theta * W_theta.conj()).real, dims=dims, coords=coords)

    W_gamma = tfr_array_morlet(
        X.sel(components=1, channels=[channel]).transpose(
            "blocks", "channels", "times"
        ),
        1000,
        fvec_gamma,
        decim=10,
        n_cycles=fvec_gamma / 2,
        n_jobs=20,
    ).squeeze()

    dims = ("batches", "freqs", "times")
    coords = dict(freqs=fvec_gamma)
    W_gamma = xr.DataArray((W_gamma * W_gamma.conj()).real, dims=dims, coords=coords)

    # Detect burts
    # slow
    labeled_theta_bursts = np.stack(parallel(p_fun(W) for W in W_theta))

    # fast
    labeled_gamma_bursts = np.stack(parallel(p_fun(W) for W in W_gamma))

    # Attributes from composites
    attrs = composites[channel].attrs
    # File name in which to save bursts
    FILE_NAME_BURSTS_SLOW = (
    f"labeled_bursts_slow_{channel}_{condition}_method_{method}_max_imfs_{max_imfs}_std_{std}.nc"
    )
    FILE_NAME_BURSTS_FAST = (
    f"labeled_bursts_fast_{channel}_{condition}_method_{method}_max_imfs_{max_imfs}_std_{std}.nc"
    )

    labeled_theta_bursts = xr.DataArray(
        labeled_theta_bursts,
        dims=("blocks", "freqs", "times"),
        coords={"freqs": fvec_theta},
    )
    labeled_theta_bursts.attrs = attrs
    labeled_theta_bursts.attrs["band"] = bands[channel]["theta"]
    # BURSTS_SLOW += [labeled_theta_bursts.copy()]
    labeled_theta_bursts.to_netcdf(os.path.join(SAVE_TO, FILE_NAME_BURSTS_SLOW))
    del labeled_theta_bursts

    labeled_gamma_bursts = xr.DataArray(
        labeled_gamma_bursts,
        dims=("blocks", "freqs", "times"),
        coords={"freqs": fvec_gamma},
    )
    labeled_gamma_bursts.attrs = attrs
    labeled_gamma_bursts.attrs["band"] = bands[channel]["gamma"]
    # BURSTS_FAST += [labeled_gamma_bursts.copy()]
    labeled_gamma_bursts.to_netcdf(os.path.join(SAVE_TO, FILE_NAME_BURSTS_FAST))
    del labeled_gamma_bursts

# BURSTS_SLOW = xr.Dataset({f"channel{i + 1}": BURSTS_SLOW[i] for i in range(n_channels)})
# BURSTS_FAST = xr.Dataset({f"channel{i + 1}": BURSTS_FAST[i] for i in range(n_channels)})

##############################################################################
# Save labeled burts
##############################################################################


# FILE_NAME_BURSTS_SLOW = (
# f"labeled_bursts_slow_{condition}_method_{method}_max_imfs_{max_imfs}_std_{std}.nc"
# )
# FILE_NAME_BURSTS_FAST = (
# f"labeled_bursts_fast_{condition}_method_{method}_max_imfs_{max_imfs}_std_{std}.nc"
# )

# BURSTS_SLOW.to_netcdf(os.path.join(SAVE_TO, FILE_NAME_BURSTS_SLOW))
# BURSTS_FAST.to_netcdf(os.path.join(SAVE_TO, FILE_NAME_BURSTS_FAST))
