from os import path

import numpy as np

from skimage import io, util, transform
from skimage.viewer import CollectionViewer, ImageViewer


SLICE_DIR = 'slices_processed'
SLICE_PATTERN = '*.TIF'
SLICE_BOTTOM = 222
SLICE_TOP = 936


def load_uint8(f, **kwargs):
    return util.img_as_ubyte(io.imread(f), force_copy=True)


slice_collection = io.ImageCollection(path.join(SLICE_DIR, SLICE_PATTERN),
                                      load_func=load_uint8)

vol_img = slice_collection.concatenate()[SLICE_BOTTOM:SLICE_TOP, :, :]

viewer = CollectionViewer(vol_img)
viewer.show()

