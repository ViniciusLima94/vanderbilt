import os
import json
import numpy as np
import xarray as xr
from .loadbinary import LoadBinary
from dataclasses import dataclass, field


@dataclass(order=True)
class DataLoader:
    """
    Load and preprocess data.

    Parameters
    ----------
    filename : str
        The path to the data file.
    rec_info : str
        The path to the recording information file.
    data : xr.DataArray
        The loaded data (initialized as None).
    fsample : int
        The sample frequency (initialized as None).
    """

    filename: str = field(repr=False, compare=False)
    rec_info: str = field(repr=False, compare=False)
    data: xr.DataArray = field(init=False, repr=True)
    fsample: int = field(init=False, repr=False, compare=False)


    def loadbinary(self, start: float = 0,
                   duration: float = None, offset: int = 0,
                   nSamplesPerChannel: int = None, channels: list = None,
                   downsample: int = None, timestamps = None, verbose=False
                   ):
        """
        Load data from binary files.

        Parameters
        ----------
        start : float, optional
            The start time for loading data (default: 0).
        duration : float, optional
            The duration of data to load (default: None, loads all available).
        offset : int, optional
            Offset for data loading (default: 0).
        nSamplesPerChannel : int, optional
            Number of samples per channel to load (default: None, loads all).
        channels : list, optional
            List of channels to load (default: None, loads all available).
        downsample : int, optional
            Downsample factor (default: None, no downsampling).
        verbose : bool, optional
            Verbose mode (default: False).
        """

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
        """
        Apply a bandpass filter to the loaded data.

        Parameters
        ----------
        l_freq : float
            Lower frequency for the bandpass filter.
        h_freq : float
            Upper frequency for the bandpass filter.
        kw_filter : dict, optional
            Additional keyword arguments for the filter.
        """
        from mne.filter import filter_data

        assert hasattr(self, 'data'), "Raw data not loaded (call loadbinary)."

        dims, coords = self.data.dims, self.data.coords

        self.data = filter_data(self.data.data.T, self.fsample,
                                l_freq, h_freq).T

        self.data = xr.DataArray(self.data, dims=dims, coords=coords)


    def __str__(self):
        """
        Return a string representation of the loaded data if available.

        Returns
        -------
        str
            A string representation of the loaded data or an empty
            string if data is not loaded.
        """
        if hasattr(self, 'data'):
            return f'data = {self.data}'
        return ''
