import os
import argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
from VUDA.emd import emd_vec
from config import metadata, method, max_imfs, imf_opts, nensembles, block_size


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
    choices=[0, 1]
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
def standardize_imf_per_block(IMFs):
    """
    Standardizes the number of intrinsic mode functions (IMFs) per block in an xarray Dataset.
    It sums slower IMFs in case a given block has more IMFs, thant the block that has the least
    number of IMFs.

    Parameters:
    - IMFs (xarray.Dataset): Input Dataset containing IMFs with dimensions ('blocks', 'IMFs', ...).

    Returns:
    - xarray.Dataset: Output Dataset with standardized IMFs per block.

    The function standardizes the number of IMFs per block by either summing the first
    (10 - n_imfs_min + 1) IMFs or keeping the original IMFs if the number is already less
    than or equal to n_imfs_min.

    Note:
    - The function assumes that the input Dataset has dimensions ('blocks', 'IMFs', ...).
    - The result is a new Dataset with standardized IMFs per block.
    """

    assert isinstance(IMFs, xr.DataArray)
    np.testing.assert_array_equal(IMFs.dims, ("blocks", "IMFs", "times"))

    n_imfs_min = IMFs.n_imfs_per_block.min()
    IMFs_coords = np.arange(n_imfs_min, dtype=int)
    attrs = IMFs.attrs

    reduced = []

    for i in range(IMFs.sizes["blocks"]):

        temp = IMFs[i].dropna("IMFs").drop_vars("IMFs")
        n_imfs = temp.shape[0]

        if n_imfs > n_imfs_min:

            reduced += [
                xr.concat(
                    (
                        temp[0 : n_imfs - n_imfs_min + 1].sum("IMFs", keepdims=True),
                        temp[n_imfs - n_imfs_min + 1 :],
                    ),
                    "IMFs",
                )
            ]

        else:
            reduced += [temp]

    # Now all blocks have n_imfs_min IMFs
    attrs["n_imfs_per_block"] = np.ones_like( IMFs.n_imfs_per_block ) * n_imfs_min
    IMFs = xr.concat(reduced, "blocks").assign_coords({"IMFs": IMFs_coords})
    IMFs.attrs = attrs

    return IMFs


##############################################################################
# Load data
##############################################################################

# Define path to data
ROOT = os.path.expanduser(f"~/funcog/HoffmanData/{monkey}/{session}")
filepath = os.path.join(ROOT, "aHPC_B_cnct.nc")


# Load data
data = xr.load_dataarray(filepath)
# Get zero timestamp
t_init = data.times.data[0]
# Get final timestamp
t_end = data.times.data[0]
# End of treehouse
t_th_end = float(data.attrs["TH_end"].split(", ")[1])
# Beggining of sleep
# t_sleep_init = float(data.attrs["Sleep_start"].split(", ")[1])

if condition == "task":
    data = data.sel(times=slice(t_init, t_th_end))
else:
    data = data.sel(times=slice(t_sleep_init, t_end))

# Time array
times = data.times.values
# Channels array
channels = data.channels.values
# Nulber of samples and channels
n_times, n_channels = len(times), len(channels)
# Get data attrs
attrs = data.attrs


##############################################################################
# Extract IMFs
##############################################################################

IMFs = []

__iter = tqdm(channels)

for channel in __iter:

    __iter.set_description(
        f"Decomposing signal from channel {channel} into IMFs."
    )

    # Get IMFs
    imf = emd_vec(
        data.sel(channels=channel).values,
        times,
        method=method,
        max_imfs=max_imfs,
        block_size=block_size,
        nensembles=nensembles,
        use_min_block_size=True,
        remove_fastest_imf=True,
        n_jobs=30,
        imf_opts=imf_opts,
        verbose=False
    )
    # Set same number of IMFs for all blocks
    if std:
        imf = standardize_imf_per_block(imf)
    IMFs += [imf]

IMFs = xr.Dataset({f"channel{i + 1}": IMFs[i] for i in range(n_channels)})

for key in attrs.keys():
    IMFs.attrs[key] = attrs[key]
IMFs.attrs["nensembles"] = nensembles


##############################################################################
# Save annotated dataset
##############################################################################

SAVE_TO = os.path.expanduser(f"~/funcog/HoffmanData/{monkey}/{session}")
FILE_NAME = f"IMFs_{condition}_method_{method}_max_imfs_{max_imfs}_std_{std}.nc"


if not os.path.exists(SAVE_TO):
    os.makedirs(SAVE_TO)

IMFs.to_netcdf(os.path.join(SAVE_TO, FILE_NAME))
