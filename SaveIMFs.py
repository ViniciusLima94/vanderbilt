import os
import argparse
import logging
from functools import partial
import numpy as np
import xarray as xr
from mne.time_frequency import psd_array_multitaper
from sklearn.cluster import DBSCAN
from VUDA.emd import emd_vec
from VUDA.utils import sum_for_cluster, average_for_cluster
from config import metadata, method, max_imfs, imf_opts, nensembles, block_size


##############################################################################
# Argument parsing
##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("SESSION_ID", help="id of the session to be used", type=int)
parser.add_argument("CONDITION", help="to select sleep or task condition", type=str, choices=["task", "sleep"])

args = parser.parse_args()

monkey = args.MONKEY
sid = args.SESSION_ID
condition = args.CONDITION

session = metadata["monkey"]["FN"]["dates"][sid]

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
t_sleep_init = float(data.attrs["Sleep_start"].split(", ")[1])

if condition == "task":
    data = data.sel(times=slice(t_init, t_th_end))
else:
    data = data.sel(times=slice(t_sleep_init, t_end))

# Time array
times = data.times.values
# Channels array
channels = data.channels.values
# Get data attrs
attrs = data.attrs


##############################################################################
# Extract IMFs
##############################################################################

# For DBSCAN clustering
clustering = DBSCAN(eps=0.03, min_samples=20)


IMFs = []
COMPOSITE = []
SPECTRA = []

for channel in data.channels.data:

    # Get IMFs
    logging.info(f"Decomposing the signal in IMFs for channel {channel + 1}.")
    imf = emd_vec(
        data.sel(channels=channel).values,
        times,
        method=method,
        max_imfs=max_imfs,
        block_size=block_size,
        nensembles=nensembles,
        use_min_block_size=True,
        n_jobs=20,
        imf_opts=imf_opts,
    )

    # Remove ultra fast IMF
    imf = imf[:, 1:, :]

    IMFs += [imf]

IMFs = xr.concat(IMFs, "channels")
IMFs = IMFs.assign_coords({"channels": channels})

for key in attrs.keys():
    IMFs.attrs[key] = attrs[key]
IMFs.attrs["nensembles"] = nensembles


##############################################################################
# Save annotated dataset
##############################################################################

SAVE_TO = os.path.expanduser(f"~/funcog/HoffmanData/{monkey}/{session}")
FILE_NAME = f"IMFs_{condition}_method_{method}_max_imfs_{max_imfs}.nc"


if not os.path.exists(SAVE_TO):
    os.makedirs(SAVE_TO)

IMFs.to_netcdf(os.path.join(SAVE_TO, FILE_NAME))

"""
# Decompose IMFs
SXX, f = psd_array_multitaper(
    imf,
    fmin=0,
    fmax=300,
    sfreq=1000,
    verbose=False,
    bandwidth=4,
    n_jobs=20,
)

SXX = xr.DataArray(SXX, dims=("blocks", "IMF", "freqs"), coords={"freqs": f})

SPECTRA += [SXX]

SXX = SXX.stack(IMFs=("blocks", "IMF")).T.values
# Normalize power spectrum
SXX_norm = SXX / SXX.sum(1)[:, None]

# Fit spectra and get cluster labels
cluster_labels = clustering.fit_predict(SXX_norm)
unique_labels = np.unique(clustering.labels_)
n_clusters = len(unique_labels)
logging.info(f"Found {n_clusters} clusters based on IMF's power.")

# Get average spectra per cluster
partial_average_for_cluster = partial(
    average_for_cluster, data=SXX_norm, cluster_labels=cluster_labels
)
out = np.stack(
    [partial_average_for_cluster(label=label) for label in unique_labels]
)
# Cluster labels per block
cluster_labels = cluster_labels.reshape(-1, max_imfs - 1)

# Sort cluster based on the peak of the mean spectra
order = np.argsort(out.argmax(axis=1))

COMPOSITE_TEMP = []
for nb in range(block_size):
    COMPOSITE_TEMP += [
        np.stack(
            [sum_for_cluster(imf[nb], cluster_labels[nb], label) for label in order]
        )
    ]
COMPOSITE += [np.stack(COMPOSITE_TEMP)]
"""
