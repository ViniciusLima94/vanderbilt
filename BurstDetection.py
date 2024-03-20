import os
import argparse
from functools import partial
import numpy as np
import xarray as xr
# import scipy
import jax
import jax.numpy as jnp
from tqdm import tqdm
from frites.utils import parallel_func
from mne.time_frequency import tfr_array_morlet, tfr_array_multitaper
from config import metadata, method, max_imfs
from skimage import measure as ski
from VUDA.burstdetection import detect_bursts


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

print(session)

##############################################################################
# Load data
##############################################################################


composites_path = os.path.expanduser(
    f"~/funcog/HoffmanData/{monkey}/{session}/composite_signals_{condition}_method_eemd_max_imfs_None_std_{std}.nc"
)

ps_composites_path = os.path.expanduser(
    f"~/funcog/HoffmanData/{monkey}/{session}/ps_composite_signals_{condition}_method_eemd_max_imfs_None_std_{std}.nc"
)


composites = xr.open_dataset(composites_path)
ps_composites = xr.open_dataset(ps_composites_path)

channels = list(composites.keys())
freqs = ps_composites.freqs.data

n_channels = len(channels)
n_blocks = composites.sizes["blocks"]

##############################################################################
# Decompose the signal in time and frequency domain
##############################################################################
def get_init_threshold(W):
    z = (W - W.mean(("times", "freqs"))) / W.std(("times", "freqs"))
    return np.ceil(z.quantile(0.95).data.item()), max(
        0.1, np.round(z.quantile(0.75).data.item(), 2)
    )

def _for_batch(W):
    t0, tf = get_init_threshold(W)
    return detect_bursts(W, t0, tf, 0.1,
                         zscore_dims=("times", "freqs"),
                         verbose=False)


parallel, p_fun = parallel_func(_for_batch, verbose=False, n_jobs=20, total=n_blocks)

# Path in which to save the bursts
SAVE_TO = os.path.expanduser(f"~/funcog/HoffmanData/{monkey}/{session}/bursts")

# If path does not exist
if not os.path.exists(SAVE_TO):
    os.makedirs(SAVE_TO)

fvec = np.linspace(4, 150, 100)

for channel in channels:

    X = composites[channel].dropna("IMFs")

    W = tfr_array_multitaper(
        X.transpose("blocks", "IMFs", "times"),
        1000,
        fvec,
        n_cycles=6,#np.maximum(fvec / 4, 1),
        time_bandwidth=4,
        decim=10,
        output="power",
        n_jobs=20,
    ).squeeze()

    dims = ("batches", "IMFs", "freqs", "times")
    coords = dict(freqs=fvec)

    W = xr.DataArray(W, dims=dims, coords=coords)

    labeled_bursts = []
    for ii in range(W.shape[1]):
        labeled_bursts.append( np.stack(parallel(p_fun(_W) for _W in W[:, ii])) )
    labeled_bursts = np.stack(labeled_bursts, axis=1)

    # Attributes from composites
    attrs = composites[channel].attrs
    # File name in which to save bursts
    FILE_NAME_BURSTS = (
    f"labeled_bursts_{channel}_{condition}_method_{method}_max_imfs_{max_imfs}_std_{std}.nc"
    )
    FILE_NAME_SPEC = (
    f"spectogram_{channel}_{condition}_method_{method}_max_imfs_{max_imfs}_std_{std}.nc"
    )

    labeled_bursts = xr.DataArray(
        labeled_bursts,
        dims=("blocks", "IMFs", "freqs", "times"),
        coords={"freqs": fvec},
    )
    labeled_bursts.attrs = attrs

    print(f"saving files {channel}")

    labeled_bursts.to_netcdf(os.path.join(SAVE_TO, FILE_NAME_BURSTS))
    W.astype(int).to_netcdf(os.path.join(SAVE_TO, FILE_NAME_SPEC))

    del labeled_bursts
