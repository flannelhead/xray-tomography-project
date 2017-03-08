from os import path, makedirs

import numpy as np
from scipy import ndimage as ndi

from skimage import color, io, util, filters, transform, morphology, measure, segmentation, draw, feature
from skimage.viewer import CollectionViewer, ImageViewer

SAMPLE_SLICES = [24, 71, 208, 290]
SAMPLE_DIR = 'samples'

SLICE_DIR = 'slices'
SLICE_PATTERN = '*.tif'

EDGE_THRESHOLD = 0.06
EROSION_SIZE = 5
EROSION_COUNT = 30


def sobel_edges(img):
    return filters.sobel(img)


def fill_interior(img):
    return ndi.binary_fill_holes(img)


def erode_to_infill(img):
    return ndi.binary_erosion(img, structure=morphology.square(EROSION_SIZE),
                              iterations=EROSION_COUNT)


def slice_by_slice(f, img, datatype=np.bool):
    nslices, _, _ = img.shape
    res = np.empty_like(img, dtype=datatype)
    for z in range(nslices):
        res[z, :, :] = f(img[z, :, :])
    return res


def otsu(img):
    return vol_img >= filters.threshold_otsu(img)


# img: a grayscale image
# mask: a binary image
# Sets the red channel to 255 for pixels where the mask is True
def red_overlay(img, mask):
    img_rgb = color.gray2rgb(img)
    img_rgb[:, :, :, 0] = np.maximum(img_rgb[:, :, :, 0],
                                     util.img_as_ubyte(mask))
    return img_rgb


def save_samples(prefix, img, color=False):
    if not path.isdir(SAMPLE_DIR):
        makedirs(SAMPLE_DIR)

    for slice in SAMPLE_SLICES:
        filename = path.join(SAMPLE_DIR, '{0}_{1}.png'.format(prefix, slice))
        ndim = len(img.shape)
        if ndim == 3:
            io.imsave(filename, util.img_as_ubyte(img[slice, :, :]))
        elif ndim == 4:
            io.imsave(filename, util.img_as_ubyte(img[slice, :, :, :]))


def load_uint8(f, **kwargs):
    return util.img_as_ubyte(io.imread(f), force_copy=True)


slice_collection = io.ImageCollection(path.join(SLICE_DIR, SLICE_PATTERN),
                                      load_func=load_uint8)

vol_img = slice_collection.concatenate()

img_otsu = otsu(vol_img)
save_samples('otsu_mask_initial', red_overlay(vol_img, img_otsu))
save_samples('otsu_initial', img_otsu)

# img_sobel = slice_by_slice(sobel_edges, vol_img, datatype=np.float)
# img_binary = img_sobel > EDGE_THRESHOLD
# save_samples('sobel', img_sobel)
# save_samples('sobel_binary', img_binary)
# save_samples('sobel_overlay', red_overlay(vol_img, img_binary))

img_binary = slice_by_slice(fill_interior, img_otsu)
save_samples('filled', red_overlay(vol_img, img_binary))

img_binary = slice_by_slice(erode_to_infill, img_binary)
save_samples('eroded', red_overlay(vol_img, img_binary))

img_binary[28:84, :, :] = 0
percentage = 100 * np.sum(img_otsu & img_binary) / np.sum(img_binary)
print('Infill percentage: {0} %'.format(percentage))

viewer = CollectionViewer(red_overlay(vol_img, img_binary))
viewer.show()

