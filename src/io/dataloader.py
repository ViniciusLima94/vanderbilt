import os
import json
import numpy as np
from .loadbinary import LoadBinary


class DataLoader:
    """
    Class to load data and preprocess it.
    """

    def __init__(self,):
        pass

    def loadbinary(self, filename: str, start: float = 0,
                   duration: float = None, offset: int = 0,
                   nSamplesPerChannel: int = None, channels: list = None,
                   downsample: int = None, verbose=False
                   ):
        dirname = os.path.dirname(__file__)
        json_path = os.path.join(dirname, "data", "recording_params.json")

        # Load recording info
        with open(json_path, 'r') as file:
            rec_params = json.load(file)["info"]
        # Evalueate precision
        rec_params["precision"] = eval(rec_params["precision"])

        # Load data from binaries
        data = LoadBinary(filename, start=start, duration=duration,
                          offset=offset, nSamplesPerChannel=nSamplesPerChannel,
                          channels=channels, downsample=downsample,
                          verbose=verbose, **rec_params)

        return data

    def filter(self,):
        pass
