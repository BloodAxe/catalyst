# flake8: noqa
# @TODO: code formatting issue for 20.07 release
import codecs
import gzip
import hashlib
import os
import tarfile
import zipfile

import numpy as np
import torch
from torch.utils.model_zoo import tqdm


def gen_bar_updater():
    
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath, chunk_size=1024 * 1024):  # noqa: WPS404
    
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)




def get_int(b):
    
    return int(codecs.encode(b, "hex"), 16)
