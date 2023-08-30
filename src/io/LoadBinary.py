import os
import numpy as np
import xarray as xr
from .errors import error_msg


def perpl_LoadBinary(
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
) -> xr.DataArray:
    """
    Load data binaries and apply the appropriate parameters.

    Parameters
    ----------
    filename: str
        Name of the file to be read.
    frequency: int | 20 kHz
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

    ##########################################################################
    # Loading Files
    ##########################################################################
    # Open file
    f = open(filename, mode="rb")

    # Size of one data point (in bytes)
    sampleSize = precision(0).nbytes

    # Position and number of samples (per channel) of the data subset
    if time:
        dataOffset = np.floor(start * frequency) * nChannels * sampleSize
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
        nSamplesPerChannel = maxNSamplesPerChannel

    if isinstance(downsample, int):
        nSamplesPerChannel = int(np.floor(nSamplesPerChannel / downsample))

    # For large amounts of data, read chunk by chunk
    maxSamplesPerChunk = 10000
    nSamples = nSamplesPerChannel * nChannels

    # Wheter it will be converter to volts
    if nSamples <= maxSamplesPerChunk:
        data = np.fromfile(f, dtype=precision, count=nSamples).reshape(
            int(nSamples / nChannels), nChannels
        )
    else:
        # Determine chunk duration and number of chunks
        nSamplesPerChunk = int(np.floor(maxSamplesPerChunk / nChannels)) * nChannels
        nChunks = int(np.floor(nSamples / nSamplesPerChunk))
        # Allocate memory
        data = np.zeros((nSamplesPerChannel, len(channels)))
        # Read all chunks
        i = 0
        for _ in range(nChunks):
            d = np.fromfile(f, dtype=precision, count=nSamplesPerChunk).reshape(
                int(nSamplesPerChunk / nChannels), nChannels
            )
            m, n = d.shape
            if m == 0:
                break
            data[i : i + m, :] = d
            i = i + m
        # If the data size is not a multiple of the chunk size, read the remainder
        remainder = nSamples - nChunks * nSamplesPerChunk
        if remainder != 0:
            d = np.fromfile(f, dtype=precision, count=remainder).reshape(
                int(remainder / nChannels), nChannels
            )
            m, n = d.shape
            if m != 0:
                data[i : i + m, :] = d

    # Convert to volts if necessary
    if bitVolts > 0:
        data = data * bitVolts
    # Close file
    f.close()

    return data