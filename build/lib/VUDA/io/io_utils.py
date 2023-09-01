import numpy as np


def fread(f, nChannels, channels, nSamples, precision, skip):
    """
    Reproduces function fread from MATLAB, allowing data reading from a file with optional bit skipping.

    Parameters
    ----------
    f : file object
        The file to read data from.
    nChannels : int
        Number of data channels.
    channels : int or list or tuple or ndarray
        Indices of the desired channels to extract (1-based).
        If an integer is provided, only that channel will be extracted.
        If a sequence of integers is provided, multiple channels will be extracted and stacked.
    nSamples : int
        Total number of samples to read.
    precision : str
        Data type of each sample, as a numpy dtype string (e.g., 'float32').
    skip : int
        Number of bits to skip after each read. Use 0 for no skipping.

    Returns
    -------
    numpy.ndarray
        Extracted data matrix, where rows correspond to samples and columns correspond to channels.
    """
    n = int(nSamples / nChannels)
    if not isinstance(skip, int):
        data = np.fromfile(f, dtype=precision, count=nSamples).reshape(n, nChannels)
    else:
        data = []
        for _ in range(n):
            data += [np.fromfile(f, dtype=precision, count=nChannels)]
            f.seek(f.tell() + skip)
        data = np.stack(data, axis=0).reshape(n, nChannels)

    if isinstance(channels, (list, tuple, np.ndarray)):
        data = data[:, channels - 1]

    return data
