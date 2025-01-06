import numpy as np
import pandas as pd

def print_optimal_index(data, metric: str, direction="max"):
    """
    Prints and returns index of optimal values from a dataframe.

    :param data: input, can be pd.DataFrame or pd.Series
    :param metric: metric/column name. Can also be a tuple or list if multiple columns should be checked.
    :param direction: string with "max" (default) or "min". Has to also be a tuple or a list if metric is.

    :return int or list of ints with optimal index/indices.
    """
    if isinstance(metric, (list, tuple)):
        # trouble check:
        if not isinstance(direction, (list, tuple)): raise ValueError("For metric tuples also direction tuples need to be provided!")
        if not isinstance(data, pd.DataFrame): raise ValueError("For metric tuples data needs to be a dataframe!")

        # iterate through submetrics:
        optimal_indices = list()
        for submetric, subdirection in zip(metric, direction):
            if subdirection == "min":
                optimal_index = np.argmin(data.loc[:, submetric])
            else:
                optimal_index = np.argmax(data.loc[:, submetric])
            print(f"Optimal ({subdirection}) value for {submetric} found at {optimal_index}.")
            optimal_indices += [optimal_index]
        return optimal_indices

    # if no sequence of metrics:
    elif not isinstance(metric, (list, tuple)):
        # select columns:
        if isinstance(data, pd.DataFrame): subset = data.loc[:, metric]
        elif isinstance(data, pd.Series): subset = data
        else: raise AttributeError("data needs to be either pd.DataFrame or pd.Series type!")
        # calculate optimal index:
        if direction == "min":
            optimal_index = np.argmin(subset)
        else:
            optimal_index = np.argmax(subset)
        # print optimal:
        print(f"Optimal ({direction}) value for {metric} found at {optimal_index}.")
        return optimal_index

def file_title(title: str, dtype_suffix=".svg", short=False):
    '''
    Creates a file title containing the current time and a data-type suffix.

    Parameters
    ----------
    title: string
            File title to be used
    dtype_suffix: (default is ".svg") string
            Suffix determining the file type.
    Returns
    -------
    file_title: string
            String to be used as the file title.
    '''
    if short:
        return datetime.now().strftime('%Y%m%d') + " " + title + dtype_suffix
    else:
        return datetime.now().strftime('%Y%m%d %H_%M_%S') + " " + title + dtype_suffix