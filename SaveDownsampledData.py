import os
import argparse
import h5py
import pandas as pd
import numpy as np
import xarray as xr
from frites.utils import parallel_func
from config import metadata, fsample, decim, _sel_attrs
from VUDA.io.loadbinary import LoadBinary

##############################################################################
# Argument parsing
##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("SESSION_ID", help="id of the session to be used", type=int)

args = parser.parse_args()

monkey = args.MONKEY
sid = args.SESSION_ID

session = metadata["monkey"]["FN"]["dates"][sid]

##############################################################################
# Load data
##############################################################################

# Root path for data
ROOT = os.path.expanduser(f"~/funcog/Neural Data/{monkey} - LaDy/{session}")
# Path for file
filepath = os.path.join(ROOT, "aHPC_B_cnct.dat")
# Path for timestamps
tspath = os.path.join(ROOT, "timestamps.mat")

# Load info sheet
info = pd.read_excel(metadata["rec_info"])

##############################################################################
# Select and filter metadata attributes
##############################################################################
attrs = info.loc[np.logical_and(info.Date == session, info.Animal_ID == monkey)]
attrs_dict = attrs.to_dict()
values = []
for _sel_attr in _sel_attrs:
    for key in attrs_dict[_sel_attr].keys():
        values += [attrs_dict[_sel_attr][key]]
attrs_dict = dict(zip(_sel_attrs, values))

# Total number of channels
n_channels = attrs["Num_chan"].values[0]
# Number of channels to load
n_channels_to_load = metadata["monkey"][monkey]["n_channels_to_load"] = 40
timestamps = np.asarray(h5py.File(tspath).get("timestamps")).squeeze()


def _load_channel(n):
    return LoadBinary(
        filepath,
        frequency=fsample,
        nSamplesPerChannel=np.inf,
        channels=[n],
        downsample=decim,
        bitVolts=0.195,
        nChannels=n_channels,
        precision=np.int16,
        timestamps=timestamps,
        attrs=attrs.to_dict(),
        verbose=False,
    )


# define the function to compute in parallel
parallel, p_fun = parallel_func(
    _load_channel, n_jobs=20, verbose=True, total=n_channels_to_load
)
# Compute the single trial coherence
data = parallel(p_fun(n) for n in range(n_channels_to_load))

data = xr.concat(data, "channels")

##############################################################################
# Save annotated dataset
##############################################################################

SAVE_TO = os.path.expanduser(f"~/funcog/HoffmanData/{monkey}/{session}/aHPC_B_cnct.nc")

if not os.path.exists(SAVE_TO):
    os.makedirs(SAVE_TO)

data.to_netcdf(SAVE_TO)
