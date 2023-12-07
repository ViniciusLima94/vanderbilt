import os
import argparse
from functools import partial
import numpy as np
import xarray as xr
from mne.time_frequency import psd_array_multitaper
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import DBSCAN
from config import metadata, method, max_imfs


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

args = parser.parse_args()

monkey = args.MONKEY
sid = args.SESSION_ID
condition = args.CONDITION

session = metadata["monkey"]["FN"]["dates"][sid]

# DBSCAN Parameters
eps = 200
min_samples = 20


##############################################################################
# Helper functions
##############################################################################


def reassign_noise_points(data, labels):
    """
    Reassigns noise points in the clustering labels to the closest cluster.

    Parameters:
    - data (numpy.ndarray): Input data points.
    - labels (numpy.ndarray): Cluster labels assigned to each data point.

    Returns:
    - numpy.ndarray: Updated cluster labels with noise points reassigned to the closest cluster.

    If there are noise points represented by the label -1 in the input labels, this function
    identifies those points, finds the closest cluster for each noise point, and reassigns
    them to the cluster with the most occurrences among the closest clusters.
    """
    # Set noise points to the closest cluster
    if -1 in np.unique(labels):
        # Identify noise points
        noise_points = data[labels == -1]

        # Find the closest cluster for each noise point
        closest_cluster_indices = pairwise_distances_argmin_min(
            noise_points, data[labels != -1]
        )[0]
        closest_cluster_labels = labels[closest_cluster_indices]

        # Assign noise points to the closest cluster
        labels[labels == -1] = np.argmax(np.bincount(closest_cluster_labels))

        return labels
    return labels


def average_for_cluster(data=None, cluster_labels=None, label=None):
    """
    Calculates the average of data points belonging to a specific cluster.

    Parameters:
    - data (numpy.ndarray): Input data points.
    - cluster_labels (numpy.ndarray): Cluster labels assigned to each data point.
    - label: The cluster label for which the average is to be calculated.

    Returns:
    - numpy.ndarray: Average of data points belonging to the specified cluster.

    Given the input data and corresponding cluster labels, this function calculates
    the average of the data points that belong to the specified cluster label.
    """
    return data[cluster_labels == label].mean(0)


f_mt = partial(
    psd_array_multitaper,
    fmin=0,
    fmax=300,
    sfreq=1000,
    verbose=True,
    bandwidth=4,
    n_jobs=20,
)

##############################################################################
# Load data
##############################################################################

# Define path to data
ROOT = os.path.expanduser(f"~/funcog/HoffmanData/{monkey}/{session}")
filepath = os.path.join(
    ROOT, f"IMFs_{condition}_method_{method}_max_imfs_{max_imfs}.nc"
)

IMFs_dataset = xr.open_dataset(filepath)
channels = list(IMFs_dataset.keys())
n_channels = len(channels)

##############################################################################
# Get composites
##############################################################################
# IMFs power spectra
IMF_PS = []
# Composite signal spectra
CMP_PS = []
# COMPOSITE TIME-SERIES
CMP = []

for channel in channels:
    #################################################################
    # Get IMFs for channel
    #################################################################
    IMFs_single = IMFs_dataset[channel].load()
    IMFs = IMFs_single.stack(samples=("blocks", "IMFs")).T.dropna("samples")
    attrs = IMFs_single.attrs

    #################################################################
    # Compute power-spectrum for IMFs
    #################################################################
    SXX, f_imf = f_mt(IMFs)
    SXX_norm = SXX / SXX.mean(1)[:, None]

    # Cluster power-spectra
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(SXX_norm)
    clustering.labels_ = reassign_noise_points(SXX_norm, clustering.labels_)
    unique_labels = np.unique(clustering.labels_)
    n_cluster = len(unique_labels)

    #################################################################
    # Order labels by power peak
    #################################################################
    partial_average_for_cluster = partial(
        average_for_cluster, data=SXX_norm, cluster_labels=clustering.labels_
    )

    out = np.stack(
        [partial_average_for_cluster(label=label) for label in unique_labels]
    )

    order = np.argsort(out.argmax(axis=1))

    #################################################################
    # Generate composite signals
    #################################################################
    n_blocks = IMFs_single.shape[0]
    cluster_labels = clustering.labels_.reshape(-1, IMFs_single.shape[1])

    composite = []

    for i in range(n_blocks):
        labels = cluster_labels[i]
        composite += [
            IMFs_single[i]
            .dropna("IMFs")
            .assign_coords({"IMFs": labels})
            .groupby("IMFs")
            .sum("IMFs")
        ]

    composite = xr.concat(composite, "blocks")
    composite = composite.assign_coords({"IMFs": range(n_cluster + 1)})
    composite.attrs = attrs

    CMP += [composite]

CMP = xr.Dataset({f"channel{i + 1}": CMP[i] for i in range(n_channels)})
CMP.attrs["DBSCAN_eps"] = 200
CMP.attrs["DBSCAN_min_smamples"] = 20

##############################################################################
# Save composite signals
##############################################################################

SAVE_TO = os.path.expanduser(f"~/funcog/HoffmanData/{monkey}/{session}")
FILE_NAME = f"composite_signals_{condition}_method_{method}_max_imfs_{max_imfs}.nc"


if not os.path.exists(SAVE_TO):
    os.makedirs(SAVE_TO)

CMP.to_netcdf(os.path.join(SAVE_TO, FILE_NAME))
