from os import path

from skimage import io
from skimage.viewer import ImageViewer

SLICE_DIR = 'slices'

io.use_plugin('pil')

img1 = io.imread(path.join(SLICE_DIR, 'testikuutio_rec0642.TIF'))

viewer = ImageViewer(img1)
viewer.show()

