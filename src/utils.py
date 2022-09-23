from collections import OrderedDict
from math import sqrt

def encode(x, y):
    z = x + y
    z *= z + 1
    z //= 2
    return y + z

def decode(z):
    x = (int(sqrt((z * 8) + 1)) - 1) // 2
    y = z - encode(x, 0)
    return [x - y, y]


def intdiv(a, b): return (a // b) if a >= 0 else -((b - 1 - a) // b)

def coords_to_tiles(a, b, stride, size): return range(intdiv(a - size, stride) + 1, intdiv(b - 1, stride) + 1)

def bbox_to_tiles(x1, y1, x2, y2, stride, size): return [(xi, yi) for xi in coords_to_tiles(x1, x2, stride, size) for yi in coords_to_tiles(y1, y2, stride, size)]

def bboxes_to_tiles_map(bboxes, stride, size):
    tiles_map = OrderedDict()
    for idx, bbox in enumerate(bboxes):
        for tile in bbox_to_tiles(*bbox, stride, size):
            tiles_map.setdefault(tile, list()).append(idx)
    return tiles_map
