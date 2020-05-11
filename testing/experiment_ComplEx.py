#a lot of this is taken from sameh's code

import os
import itertools
import gzip
import numpy as np
from tqdm import tqdm

import argparse

from ..embeddings import ComplEx
from libkge import KgDataset


def main():
    seed = 1234
    nb_epochs_then_check = None
