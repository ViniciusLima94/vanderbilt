import logging
import numpy as np
import xarray as xr
from tqdm import tqdm
from .errors import error_msg
from .io_utils import fread


def _load_batches(
    f,
    nChannels: int,
    nSamples: int,
    nSamplesPerChannel: int,
    channels: list,
    precision: type,
    skip: int,
    maxSamplesPerBatch: int = 1000,
    verbose: bool = False,
):
    """
    Load and process data in batches from a file.

    Parameters
    ----------
    f : file
        The file object to read data from.
    nChannels : int
        Number of channels in the data.
    nSamples : int
        Total number of samples in the data.
    nSamplesPerChannel : int
        Samples per channel.
    channels : list
        Channel indices to read from.
    precision : type
        Data type precision for reading.
    skip : int
        Samples to skip after each read.
    maxSamplesPerBatch : int | 1000
        Max samples per batch.

    Returns
    -------
    data : ndarray
        A numpy array containing the processed data.
    """
    # Determine chunk duration and number of chunks
    nSamplesPerBatch = int(np.floor(maxSamplesPerBatch / nChannels)) * nChannels
    nBatchs = int(np.floor(nSamples / nSamplesPerBatch))
    # Allocate memory
    data = np.zeros((nSamplesPerChannel, len(channels)))
    # Read all chunks
    i = 0
    __iter = tqdm(range(nBatchs)) if verbose else range(nBatchs)
    for batch in __iter:
        if verbose: __iter.set_description(f"Loading batches: {batch}/{nBatchs}")
        d = fread(f, nChannels, channels, nSamplesPerBatch, precision, skip)
        m, n = d.shape
        if m == 0:
            break
        data[i : i + m, :] = d
        i = i + m
    # If the data size is not a multiple of the chunk size, read the remainder
    remainder = nSamples - nBatchs * nSamplesPerBatch
    if remainder != 0:
        d = fread(f, nChannels, channels, remainder, precision, skip)
        m, n = d.shape
        if m != 0:
            data[i : i + m, :] = d

    return data


def LoadBinary(
    filename: str,
    frequency: int = 30000,
    start: float = 0,
    duration: float = None,
    offset: int = 0,
    nSamplesPerChannel: int = None,
    nChannels: int = 1,
    channels: list = None,
    precision: type = np.int16,
    downsample: int = None,
    bitVolts: float = 0.195,
    verbose: bool = False,
) -> xr.DataArray:
    """
    Load data binaries and apply the appropriate parameters.

    Parameters
    ----------
    filename: str
        Name of the file to be read.
    frequency: int | 30 kHz
        Sampling rate in Hertz.
    start: float | 0
        Position to start reading in seconds.
    duration: float | None
        Duration to read in seconds. If None takes whole duration.
    offset: int | 0
        Position to start reading (in samples per channel.
    nSamplesPerChannel: int | None
        Number of samples (per channel) to read. If None read all.
    nChannels: int | 1
        Number of data channels in the file.
    channels: array_like | None
        Channels to be read. If None read all.
    precision: type | np.int16
        Sample precision.
    downsample: int | None
        Factor by which to downsample. If None, no downsample is applied.
    bitVolts: float | 0.195
        If provided LFP will be converted to double precision with this
        factor (the default value converts LFP to muVolts).


    Returns
    -------
    data: array_like
        A multidimensional array containing the data.
    """

    time = False
    samples = False

    ##########################################################################
    # Assertions
    ##########################################################################
    if isinstance(duration, (int, float)):
        assert duration > 0
        time = True

    if isinstance(nSamplesPerChannel, (int, float)):
        assert nSamplesPerChannel > 0
        samples = True

    if time & samples:
        raise ValueError(error_msg["TimeSamplesException"])

    # By default, load all channels
    if not isinstance(channels, (list, tuple, np.ndarray)):
        channels = np.arange(1, nChannels + 1, 1, dtype=int)
    else:
        assert isinstance(channels, (list, tuple, np.ndarray))
        channels = np.asarray(channels)

    # Check consistency between channel IDs and number of channels
    if any(channels > nChannels):
        raise ValueError(error_msg["ChannelIDException"])

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.info(
            f"Loading binaries from {filename} with:\n"
            f"fsample = {frequency}, start = {start}, "
            f"duration = {duration}, offset = {offset}, "
            f"nSamplesPerChannel = {nSamplesPerChannel}, "
            f"nChannels = {nChannels}, "
            f"precision = {precision}, downsample = {downsample}, "
            f"bitVolts = {bitVolts}."
        )

    # Size of one data point (in bytes)
    sampleSize = precision(0).nbytes

    ##########################################################################
    # Loading Files
    ##########################################################################
    # Open file
    f = open(filename, mode="rb")

    # Position and number of samples (per channel) of the data subset
    if time:
        dataOffset = int(np.floor(start * frequency)) * nChannels * sampleSize
        nSamplesPerChannel = np.round(duration * frequency)
    else:
        dataOffset = offset * nChannels * sampleSize

    # Position file index for reading
    f.seek(dataOffset, 0)

    # Determine total number of samples in file
    fileStart = f.tell()
    fileStop = f.seek(0, 2)

    # (floor in case all channels do not have the same number of samples)
    # Compute maximum number of samples per channel
    maxNSamplesPerChannel = np.floor(((fileStop - fileStart) / nChannels / sampleSize))
    # Reposition at start of file and then on offset
    f.seek(0, 0)
    f.seek(dataOffset, 0)

    if ((not time) and (not samples)) or (nSamplesPerChannel > maxNSamplesPerChannel):
        nSamplesPerChannel = int(maxNSamplesPerChannel)

    # Nsamples of final data array after downsample
    if isinstance(downsample, int):
        skip = int(nChannels * (downsample - 1) * sampleSize)
        nSamplesPerChannel = int(np.floor(nSamplesPerChannel / downsample))
    else:
        skip = None

    # For large amounts of data, read in batches
    nSamples = nSamplesPerChannel * nChannels

    maxSamplesPerBatch = 10000
    # Wheter it will be converter to volts
    if nSamples <= maxSamplesPerBatch:
        data = fread(f, nChannels, channels, nSamples, precision, skip)
    else:
        data = _load_batches(
            f,
            nChannels,
            nSamples,
            nSamplesPerChannel,
            channels,
            precision,
            skip,
            maxSamplesPerBatch,
            verbose,
        )

    # Convert to volts if necessary
    if bitVolts > 0:
        data = data * bitVolts
    # Close file
    f.close()

    data = xr.DataArray(data, dims=("times", "channels"), coords={"channels": channels})

    return data
