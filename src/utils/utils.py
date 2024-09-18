import torch

import pathlib
import json

def get_asoca_dir():
    return pathlib.Path("data/datasets/asoca")

def get_asoca_json():
    return pathlib.Path(get_asoca_dir(), "datalist.json")

def get_imagecas_dir():
    return pathlib.Path("data/datasets/imagecas")

def get_imagecas_json():
    return pathlib.Path(get_imagecas_dir(), "datalist.json")

def json_reader(filename: str):
    with open(filename) as f:
        return json.load(f)

def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device


