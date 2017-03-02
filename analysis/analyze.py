from os import path

import numpy as np
from scipy import ndimage as ndi

from skimage import color, io, util, filters, transform, morphology, measure, segmentation, draw, feature
from skimage.viewer import CollectionViewer, ImageViewer


SLICE_DIR = 'slices'
SLICE_PATTERN = '*.tif'
CLOSING_RADIUS = 2
GAUSS_SIGMA = 3

CANNY_SIGMA = 1
CANNY_LOW = 0
CANNY_HIGH = 1

def load_uint8(f, **kwargs):
    return util.img_as_ubyte(io.imread(f), force_copy=True)


slice_collection = io.ImageCollection(path.join(SLICE_DIR, SLICE_PATTERN),
                                      load_func=load_uint8)

vol_img = slice_collection.concatenate()[:, :, :]
# img_blurred = util.img_as_ubyte(filters.gaussian(vol_img, sigma=GAUSS_SIGMA))

otsu_threshold = filters.threshold_otsu(vol_img)
img_binary = vol_img >= otsu_threshold
# img_binary = morphology.binary_closing(img_binary,
#                                        selem=morphology.ball(CLOSING_RADIUS))

nslices, _, _ = vol_img.shape
all_contours = np.empty_like(vol_img, dtype=np.bool)
for z in range(nslices):
    edges = feature.canny(
        img_binary[z, :, :], sigma=CANNY_SIGMA,
        low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH
    )
    # edges = morphology.binary_dilation(edges, selem=morphology.square(6))
    all_contours[z, :, :] = ndi.binary_fill_holes(edges)
    # contours = measure.find_contours(img_binary[z, :, :], level=0)
    # for contour in contours:
    #     rr, cc = draw.polygon_perimeter(contour[:, 0], contour[:, 1])
    #     all_contours[z, rr, cc] = 1
    # all_contours[z, :, :] = ndi.binary_fill_holes(all_contours[z, :, :])

# img_masked = vol_img & util.img_as_ubyte(img_binary)

# img_rgb = color.gray2rgb(vol_img)
# img_rgb[:, :, :, 0] = np.maximum(img_rgb[:, :, :, 0], util.img_as_ubyte(img_binary))


viewer = CollectionViewer(all_contours)
viewer.show()

