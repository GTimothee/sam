import nibabel as nib
from math import ceil
from gzip import GzipFile
from io import BytesIO
import sys
import numpy as np
from time import time
import os
import logging
import gzip
import threading
from nibabel import fileslice

from naive import split, merge
from multiple import multiple_writes, multiple_reads
from clustered import clustered_writes, clustered_reads


class ImageUtils:
    """ Core utility class for performing operations on images."""

    def __init__(self, filepath, first_dim=None, second_dim=None,
                 third_dim=None, dtype=None):
        """
        Keyword arguments:
            filepath                                : filepath to image
            first_dim, second_dim, third_dim        : the shape of the image.
                                                      Only required if image
                                                      needs to be generated
            dtype                                   : the numpy dtype of the
                                                      image. Only required if
                                                      image needs
                                                      to be generated
        """

        # get file manager
        from utils import split_ext
        ext = split_ext(filepath)[1]
        if ext == "nii" or ext == "minc":
            from nibabel import Nibabel
            self.file_manager = Nibabel(filepath)  # manager contains self.proxy
        elif ext == "hdf5":
            from hdf5 import HDF5_manager
            self.file_manager = HDF5_manager(filepath)
        else:
            raise ValueError("File format not supported yet")


    def split_multiple(self):
        multiple_writes()


    def merge_multiple(self):
        multiple_reads()
    

    def split_clustered(self):
        clustered_writes(Y_splits, 
                         Z_splits, 
                         X_splits, 
                         out_dir,
                         mem, 
                         filename_prefix="bigbrain",
                         extension="nii", 
                         nThreads=1, 
                         benchmark=False)


    def merge_clustered(self):
        clustered_reads()


    def split_naive(self):
        split()
    

    def merge_naive(self):
        merge()
