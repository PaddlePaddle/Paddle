import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

from cv2 import resize

__all__ = [
    "load_image", "resize_short", "to_chw", "center_crop", "random_crop",
    "left_right_flip", "simple_transform", "load_and_transform"
]
"""
This file contains some common interfaces for image preprocess.
Many users are confused about the image layout. We introduce
the image layout as follows.

- CHW Layout
  - The abbreviations: C=channel, H=Height, W=Width
  - The default layout of image opened by cv2 or PIL is HWC.
    PaddlePaddle only supports the CHW layout. And CHW is simply
    a transpose of HWC. It must transpose the input image.

- Color format: RGB or BGR
  OpenCV use BGR color format. PIL use RGB color format. Both
  formats can be used for training. Noted that, the format should
  be keep consistent between the training and inference peroid.
"""


def load_image(file, is_color=True):
    """
    Load an color or gray image from the file path.

    Example usage:
    
    .. code-block:: python
        im = load_image('cat.jpg')

    :param file: the input image path.
    :type file: string
    :param is_color: If set is_color True, it will load and
                     return a color image. Otherwise, it will
                     load and return a gray image.
    """
    # cv2.IMAGE_COLOR for OpenCV3
    # cv2.CV_LOAD_IMAGE_COLOR for older OpenCV Version
    # cv2.IMAGE_GRAYSCALE for OpenCV3
    # cv2.CV_LOAD_IMAGE_GRAYSCALE for older OpenCV Version
    # Here, use constant 1 and 0
    # 1: COLOR, 0: GRAYSCALE
    flag = 1 if is_color else 0
    im = cv2.imread(file, flag)
    return im


def resize_short(im, size):
    """ 
    Resize an image so that the length of shorter edge is size.

    Example usage:
    
    .. code-block:: python
        im = load_image('cat.jpg')
        im = resize_short(im, 256)
    
    :param im: the input image with HWC layout.
    :type im: ndarray
    :param size: the shorter edge size of image after resizing.
    :type size: int
    """
    assert im.shape[-1] == 1 or im.shape[-1] == 3
    h, w = im.shape[:2]
    h_new, w_new = size, size
    if h > w:
        h_new = size * h / w
    else:
        w_new = size * w / h
    im = resize(im, (h_new, w_new), interpolation=cv2.INTER_CUBIC)
    return im


def to_chw(im, order=(2, 0, 1)):
    """
    Transpose the input image order. The image layout is HWC format
    opened by cv2 or PIL. Transpose the input image to CHW layout
    according the order (2,0,1).

    Example usage:
    
    .. code-block:: python
        im = load_image('cat.jpg')
        im = resize_short(im, 256)
        im = to_chw(im)
    
    :param im: the input image with HWC layout.
    :type im: ndarray
    :param order: the transposed order.
    :type order: tuple|list 
    """
    assert len(im.shape) == len(order)
    im = im.transpose(order)
    return im


def center_crop(im, size, is_color=True):
    """
    Crop the center of image with size.

    Example usage:
    
    .. code-block:: python
        im = center_crop(im, 224)
    
    :param im: the input image with HWC layout.
    :type im: ndarray
    :param size: the cropping size.
    :type size: int
    :param is_color: whether the image is color or not.
    :type is_color: bool
    """
    h, w = im.shape[:2]
    h_start = (h - size) / 2
    w_start = (w - size) / 2
    h_end, w_end = h_start + size, w_start + size
    if is_color:
        im = im[h_start:h_end, w_start:w_end, :]
    else:
        im = im[h_start:h_end, w_start:w_end]
    return im


def random_crop(im, size, is_color=True):
    """
    Randomly crop input image with size.

    Example usage:
    
    .. code-block:: python
        im = random_crop(im, 224)
    
    :param im: the input image with HWC layout.
    :type im: ndarray
    :param size: the cropping size.
    :type size: int
    :param is_color: whether the image is color or not.
    :type is_color: bool
    """
    h, w = im.shape[:2]
    h_start = np.random.randint(0, h - size + 1)
    w_start = np.random.randint(0, w - size + 1)
    h_end, w_end = h_start + size, w_start + size
    if is_color:
        im = im[h_start:h_end, w_start:w_end, :]
    else:
        im = im[h_start:h_end, w_start:w_end]
    return im


def left_right_flip(im):
    """
    Flip an image along the horizontal direction.
    Return the flipped image.

    Example usage:
    
    .. code-block:: python
        im = left_right_flip(im)
    
    :paam im: input image with HWC layout
    :type im: ndarray
    """
    if len(im.shape) == 3:
        return im[:, ::-1, :]
    else:
        return im[:, ::-1, :]


def simple_transform(im, resize_size, crop_size, is_train, is_color=True):
    """
    Simply data argumentation for training. These operations include
    resizing, croping and flipping.

    Example usage:
    
    .. code-block:: python
        im = simple_transform(im, 256, 224, True)

    :param im: The input image with HWC layout.
    :type im: ndarray
    :param resize_size: The shorter edge length of the resized image.
    :type resize_size: int
    :param crop_size: The cropping size.
    :type crop_size: int
    :param is_train: Whether it is training or not.
    :type is_train: bool
    """
    im = resize_short(im, resize_size)
    if is_train:
        im = random_crop(im, crop_size)
        if np.random.randint(2) == 0:
            im = left_right_flip(im)
    else:
        im = center_crop(im, crop_size)
    im = to_chw(im)

    return im


def load_and_transform(filename,
                       resize_size,
                       crop_size,
                       is_train,
                       is_color=True):
    """
    Load image from the input file `filename` and transform image for
    data argumentation. Please refer to the `simple_transform` interface
    for the transform operations.

    Example usage:
    
    .. code-block:: python
        im = load_and_transform('cat.jpg', 256, 224, True)

    :param filename: The file name of input image.
    :type filename: string
    :param resize_size: The shorter edge length of the resized image.
    :type resize_size: int
    :param crop_size: The cropping size.
    :type crop_size: int
    :param is_train: Whether it is training or not.
    :type is_train: bool
    """
    im = load_image(filename)
    im = simple_transform(im, resize_size, crop_size, is_train, is_color)
    return im
