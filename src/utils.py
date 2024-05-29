import argparse
import pickle
import re

import pandas as pd


def read_csv(file_path):
    """
    Reads a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): The file path of the CSV.

    Returns:
        pd.DataFrame: The Pandas DataFrame containing the CSV data.
    """
    return pd.read_csv(file_path)


def read_pickle(file_path):
    """
    Reads a Pickle file.

    Args:
        file_path (str): The file path of the Pickle file.

    Returns:
        dict: The dictionary loaded from the Pickle file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def write_pickle(data, file_path):
    """
    Writes data to a Pickle file.

    Args:
        data (dict): The data to pickle.
        file_path (str): The file path where the Pickle file will be saved.
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def remove_comments(text):
    """Remove C-style /*comments*/ from a string."""
    p = r'/\*[^*]*\*+([^/*][^*]*\*+)*/|("(\\.|[^"\\])*"|\'(\\.|[^\'\\])*\'|.[^/"\'\\]*)'
    return ''.join(m.group(2) for m in re.finditer(p, text, re.M | re.S) if m.group(2))
    

class AttrDict(dict):
    """
    A dictionary subclass that allows access to its keys through attribute notation.

    Example:
        d = AttrDict({'a': 1, 'b': 2})
        print(d.a)  # prints 1
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

