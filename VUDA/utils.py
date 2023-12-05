import numpy as np

def sum_for_cluster(
    data: np.ndarray = None, cluster_labels: np.ndarray = None, label: int = None
):
    """
    Calculate the average of the data points belonging to a specific cluster.

    Parameters:
    - data (numpy.ndarray or pandas.DataFrame): The input data, where each row represents a data point.
    - cluster_labels (numpy.ndarray or pandas.Series): Cluster labels assigned to each data point.
    - label (int): The specific cluster label for which the average is to be calculated.

    Returns:
    - numpy.ndarray or pandas.Series: The average of the data points in the specified cluster.

    Example:
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> cluster_labels = np.array([0, 1, 0])
    >>> label = 0
    >>> average_for_cluster(data, cluster_labels, label)
    array([4., 5., 6.])
    """
    return data[cluster_labels == label].sum(0)

def average_for_cluster(
    data: np.ndarray = None, cluster_labels: np.ndarray = None, label: int = None
):
    """
    Calculate the average of the data points belonging to a specific cluster.

    Parameters:
    - data (numpy.ndarray or pandas.DataFrame): The input data, where each row represents a data point.
    - cluster_labels (numpy.ndarray or pandas.Series): Cluster labels assigned to each data point.
    - label (int): The specific cluster label for which the average is to be calculated.

    Returns:
    - numpy.ndarray or pandas.Series: The average of the data points in the specified cluster.

    Example:
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> cluster_labels = np.array([0, 1, 0])
    >>> label = 0
    >>> average_for_cluster(data, cluster_labels, label)
    array([4., 5., 6.])
    """
    return data[cluster_labels == label].mean(0)
