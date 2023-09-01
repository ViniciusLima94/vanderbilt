import os
import json
import numpy as np
import xarray as xr
from .loadbinary import LoadBinary


class DataLoader:
    """
    Class to load data and preprocess it.
    """

    def __init__(self, filename: str, rec_info: str):
        # Data File
        self.filename = filename
        # Recording info file
        self.rec_info = rec_info

    def loadbinary(self, start: float = 0,
                   duration: float = None, offset: int = 0,
                   nSamplesPerChannel: int = None, channels: list = None,
                   downsample: int = None, verbose=False
                   ):

        # Load recording info
        with open(self.rec_info, 'r') as file:
            rec_params = json.load(file)["info"]
        # Evalueate precision
        rec_params["precision"] = eval(rec_params["precision"])

        # Load data from binaries
        self.data = LoadBinary(self.filename, start=start, duration=duration,
                          offset=offset, nSamplesPerChannel=nSamplesPerChannel,
                          channels=channels, downsample=downsample,
                          verbose=verbose, **rec_params)
        
        if isinstance(downsample, int):
            self.fsample = int(rec_params["frequency"] / downsample)


    def filter(self, l_freq: float, h_freq: float, kw_filter: dict = {}):
        from mne.filter import filter_data
        print(hasattr(self, 'data'))

        assert hasattr(self, 'data'), "Raw data not loaded (call loadbinary method)."

        dims, coords = self.data.dims, self.data.coords

        self.data = filter_data(self.data.data.T, self.fsample,
                                l_freq, h_freq).T

        self.data = xr.DataArray(self.data, dims=dims, coords=coords)
