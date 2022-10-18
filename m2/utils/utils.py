import os
import random
import shutil

import numpy as np
import requests
import torch


def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)


def delete_folders(*folder_path):
    for folder in folder_path:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def create_folders(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
