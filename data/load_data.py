import os
import json
import numpy as np

import pyvips
from matplotlib import pyplot as plt

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset

import torchvision
from torchvision.utils import draw_bounding_boxes

from utils import encode, decode, bboxes_to_tiles_map

# def tile_id_to_coords(tile_id, tile_size=1024, offsets=None):
#     if offsets == None:
#         offsets = [0] * 2
#     return [(coord * tile_size) + offset for coord, offset in zip(decode(tile_id), offsets)] + ([tile_size] * 2)
#
# def vips_tile_crop(vips_img, tile_id, tile_size=1024):
#     # coords = [(coord * tile_size) + offset for coord, offset in zip(decode(tile_id), [int(vips_img.get(f'openslide.bounds-{coord}')) for coord in 'x y'.split()])] + ([tile_size] * 2)
#     offsets = [int(vips_img.get(f'openslide.bounds-{coord}')) for coord in 'x y'.split()]
#     coords = tile_id_to_coords(tile_id, offsets=offsets)
#     tile = vips_img[:3].crop(*coords)
#     return np.ndarray(buffer=tile.write_to_memory(), dtype=np.uint8, shape=(tile.height, tile.width, tile.bands))








class JsonDataset(Dataset):
    def __init__(self, json_name, stride=1024, size=1024, label_names='Core Diffuse Neuritic CAA'.split()):
        self.stride, self.size = stride, size
        self.labels, self.bboxes = [np.array(_) for _ in zip(*tuple(map(lambda json_obj: JsonDataset.json_to_instance(json_obj, label_names), json.loads(open(json_fname, 'r').read()).get('annotations'))))]
        self.tiles = bboxes_to_tiles_map(self.bboxes, self.stride, self.size)

    @staticmethod
    def json_to_instance(json_obj, label_names):
        return tuple(np.array(_, dtype=np.intc) for _ in (1 + label_names.index(json_obj.get('pathClasses')[0]), ([f(coords) for f in [np.amin, np.amax] for coords in [np.array(json_obj.get(c)).astype(np.intc) for c in 'x y'.split()]])))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = list(self.tiles.keys())[idx]
        tile_min = np.array(tile) * self.stride
        tile_max = tile_min + self.size

        idxs = self.tiles.get(tile)
        labels, bboxes = self.labels[idxs], self.bboxes[idxs]
        bboxes[:, :2] = np.maximum(tile_min, bboxes[:, :2]) - tile_min
        bboxes[:, 2:] = np.minimum(tile_max, bboxes[:, 2:]) - tile_min

        return dict(labels=torch.as_tensor(labels), boxes=torch.as_tensor(bboxes))

class VipsDataset(JsonDataset):
    def __init__(self, vips_img, json_name, **kwargs):
        super().__init__(json_name, **kwargs)
        self.vips_img = vips_img

    @staticmethod
    def vips_crop(vips_img, x, y, w, h):
        x, y = tuple(map(sum, zip((x, y), [int(vips_img.get(f'openslide.bounds-{coord}')) for coord in 'x y'.split()])))
        vips_crop = vips_img[:3].crop(x, y, w, h)
        return np.ndarray(buffer=vips_crop.write_to_memory(), dtype=np.uint8, shape=(vips_crop.height, vips_crop.width, vips_crop.bands))

    def __getitem__(self, idx):
        tile = [_ * self.stride for _ in list(self.tiles.keys())[idx]]
        tile_crop = VipsDataset.vips_crop(self.vips_img, *tile, *([self.size] * 2)).transpose(2, 0, 1)
        return torch.as_tensor(tile_crop), super().__getitem__(idx)



if __name__ == '__main__':
    vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg'
    vips_img_name = 'XE15-007_1_AmyB_1'

    vips_img_fname = os.path.join(vips_img_dir, f'{vips_img_name}.mrxs')
    vips_img_level = 0

    json_dir = '/home/gryan/QuPath/v0.3/json'
    json_fname = os.path.join(json_dir, f'{vips_img_name}.json')

    vips_img = pyvips.Image.new_from_file(vips_img_fname, level=vips_img_level)
    dataset = VipsDataset(vips_img, json_fname)

    to_pil = torchvision.transforms.ToPILImage()
    for image, target in dataset:
        if len(input('Show next tile: ')) > 0:
            break
        image = draw_bounding_boxes(image, target['boxes'], colors='black')
        to_pil(image).show()
