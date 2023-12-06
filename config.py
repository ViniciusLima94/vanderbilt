##############################################################################
# Parameters to load data
##############################################################################
# Original sampling frequency
fsample = 30000
# Decimation factor
decim = 30

metadata = {}
metadata["rec_info"] = r"/home/vinicius/funcog/Neural Data/Data_Processed.xlsx"
metadata["monkey"] = {"FN": {}, "WI": {}}
metadata["monkey"]["FN"]["n_channels"] = 64
metadata["monkey"]["WI"]["n_channels"] = 128
metadata["monkey"]["FN"]["n_channels_to_load"] = 40
metadata["monkey"]["WI"]["n_channels"] = 128

metadata["monkey"]["FN"]["dates"] = [
    "10-13-2022",
    "10-16-2022",
    "10-20-2022",
    "10-23-2022",
    "10-27-2022",
    "10-30-2022",
    "10-14-2022",
    "10-18-2022",
    "10-21-2022",
    "10-24-2022",
    "10-28-2022",
    "10-31-2022",
    "10-15-2022",
    "10-19-2022",
    "10-22-2022",
    "10-25-2022",
    "10-29-2022",
    "11-01-2022",
]

metadata["monkey"]["WI"]["dates"] = [
    "2021-09-24",
    "2021-09-26",
    "2021-09-30",
    "2021-10-06",
    "2021-09-25",
    "2021-09-29",
    "2021-10-01",
    "2021-10-07",
]

# Attributes in the metadata spreadsheet to be kept
_sel_attrs = [
    "TH_end",
    "Sleep_start",
    "Sleep_errors",
    "Bad Channels",
    "Spindle_Chan",
    "Ripple_Chan_TH",
    "Ripple_Chan_Sleep",
    "Reversal_Chan",
    "Pos_Pol_Chan",
    "Noise_Chan",
    "Ripple_detected",
    "Spindle_detected",
    "SO_detected",
]

##############################################################################
# Parameters for the EMD
##############################################################################
method = "eemd"
max_imfs = None
nensembles = 5
imf_opts = {"stop_method": "fixed", "max_iters": 5}
block_size = 200
