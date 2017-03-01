from os import path

import numpy as np

from skimage import io, util, filters, transform, morphology
from skimage.viewer import CollectionViewer, ImageViewer


SLICE_DIR = 'slices'
SLICE_PATTERN = '*.tif'


def load_uint8(f, **kwargs):
    return util.img_as_ubyte(io.imread(f), force_copy=True)


slice_collection = io.ImageCollection(path.join(SLICE_DIR, SLICE_PATTERN),
                                      load_func=load_uint8)

vol_img = slice_collection.concatenate()
otsu_threshold = filters.threshold_otsu(vol_img)

img_binary = vol_img >= otsu_threshold

viewer = CollectionViewer(img_binary)
viewer.show()

