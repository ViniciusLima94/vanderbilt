import numpy as np


def fread(f, nChannels, channels, nSamples, precision, skip):
    """
    Reproduces function fread from matlab which allows to skip some bits
    after each read.
    """
    n = int(nSamples / nChannels)
    if skip == 0:
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
