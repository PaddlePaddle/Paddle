# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division

import math
import numbers

import paddle
from paddle.nn.functional import affine_grid, grid_sample


def _assert_paddle_image(img):
    if not isinstance(img, paddle.Tensor) or img.ndim != 3:
        raise RuntimeError('not support dim={} paddle image'.format(img.ndim))


def _assert_data_format(data_format):
    assert data_format.lower() in ('chw', 'hwc'
                                   ), "data_format should in ('chw', 'hwc')"


def _get_image_channels(img, data_format='CHW'):
    if data_format.lower() == 'chw':
        return img.shape[-3]
    elif data_format.lower() == 'hwc':
        return img.shape[-1]
    raise ValueError('data_format')


def _get_image_size(img, data_format='CHW'):
    if data_format.lower() == 'chw':
        return img.shape[-1], img.shape[-2]
    elif data_format.lower() == 'hwc':
        return img.shape[-2], img.shape[-3]
    raise ValueError('data_format')


def _get_image_h_axis(img, data_format='CHW'):
    if data_format.lower() == 'chw':
        return -2
    elif data_format.lower() == 'hwc':
        return -3
    raise ValueError('data_format')


def _get_image_w_axis(img, data_format='CHW'):
    if data_format.lower() == 'chw':
        return -1
    elif data_format.lower() == 'hwc':
        return -2
    raise ValueError('data_format')


def _get_image_c_axis(img, data_format='CHW'):
    if data_format.lower() == 'chw':
        return -3
    elif data_format.lower() == 'hwc':
        return -1


def normalize(img, mean, std, data_format='CHW'):
    """Normalizes a tensor image with mean and standard deviation.

    Args:
        img (paddle.Tensor): input data to be normalized.
        mean (list|tuple): Sequence of means for each channel.
        std (list|tuple): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        Tensor: Normalized mage.

    """
    assert data_format.lower() in ('chw', 'hwc'
                                   ), "data_format should in ('chw', 'hwc')"

    mean = paddle.to_tensor(mean, place=img.place)
    std = paddle.to_tensor(std, place=img.place)

    if data_format.lower == 'chw':
        mean = mean.reshape([-1, 1, 1])
        std = std.reshape([-1, 1, 1])

    return (img - mean) / std


def to_grayscale(img, num_output_channels=1, data_format='CHW'):
    """Converts image to grayscale version of image.

    Args:
        img (paddel.Tensor): Image to be converted to grayscale.
        num_output_channels (int, optionl[1, 3]):
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel 
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        PIL.Image: Grayscale version of the image.
    """
    assert data_format.lower() in ('chw', 'hwc'
                                   ), "data_format should in ('chw', 'hwc')"
    assert num_output_channels in (1, 3), ""

    rgb_weights = paddle.to_tensor(
        [0.2989, 0.5870, 0.1140], place=img.place).astype(img.dtype)

    if data_format.lower() == 'chw':
        _c_index = -3
        rgb_weights = rgb_weights.reshape((-1, 1, 1))
    else:
        _c_index = -1

    img = (img * rgb_weights).sum(axis=_c_index, keepdim=True)
    _shape = img.shape
    _shape[_c_index] = num_output_channels

    return img.expand(_shape)


def _affine_grid(theta, w, h, ow, oh):
    '''
    '''
    d = 0.5
    # tic = time.time()
    base_grid = paddle.ones((1, oh, ow, 3), dtype=theta.dtype)
    # print(time.time() - tic)

    x_grid = paddle.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, ow)
    base_grid[..., 0] = x_grid
    y_grid = paddle.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, oh).unsqueeze_(-1)
    base_grid[..., 1] = y_grid

    scaled_theta = theta.transpose(
        (0, 2, 1)) / paddle.to_tensor([0.5 * w, 0.5 * h])
    output_grid = base_grid.reshape((1, oh * ow, 3)).bmm(scaled_theta)

    return output_grid.reshape((1, oh, ow, 2))


def rotate(
        img,
        angle,
        interpolation='nearest',
        expand=False,
        center=None,
        fill=None,
        translate=None, ):
    '''
    https://github.com/python-pillow/Pillow/blob/11de3318867e4398057373ee9f12dcb33db7335c/src/PIL/Image.py#L2054
    '''

    angle = -angle % 360

    n, c, h, w = img.shape

    # image center is (0, 0) in matrix
    if translate is None:
        post_trans = [0, 0]
    else:
        post_trans = translate

    if center is None:
        rotn_center = [0, 0]
    else:
        rotn_center = [(p - s * 0.5) for p, s in zip(center, [w, h])]

    angle = -math.radians(angle)
    matrix = [
        math.cos(angle),
        math.sin(angle),
        0.0,
        -math.sin(angle),
        math.cos(angle),
        0.0,
    ]

    matrix[2] += matrix[0] * (-rotn_center[0] - post_trans[0]) + matrix[1] * (
        -rotn_center[1] - post_trans[1])
    matrix[5] += matrix[3] * (-rotn_center[0] - post_trans[0]) + matrix[4] * (
        -rotn_center[1] - post_trans[1])

    matrix[2] += rotn_center[0]
    matrix[5] += rotn_center[1]

    matrix = paddle.to_tensor(matrix, place=img.place)
    matrix = matrix.reshape((1, 2, 3))

    if expand:
        # calculate output size
        corners = paddle.to_tensor(
            [[-0.5 * w, -0.5 * h, 1.0], [-0.5 * w, 0.5 * h, 1.0],
             [0.5 * w, 0.5 * h, 1.0], [0.5 * w, -0.5 * h, 1.0]],
            place=matrix.place).astype(matrix.dtype)

        _pos = corners.reshape(
            (1, -1, 3)).bmm(matrix.transpose((0, 2, 1))).reshape((1, -1, 2))
        _min = _pos.min(axis=-2).floor()
        _max = _pos.max(axis=-2).ceil()

        npos = _max - _min
        nw = npos[0][0]
        nh = npos[0][1]

        ow, oh = int(nw.numpy()[0]), int(nh.numpy()[0])

    else:
        ow, oh = w, h

    grid = _affine_grid(matrix, w, h, ow, oh)

    out = grid_sample(img, grid, mode=interpolation)

    return out


def vflip(img, data_format='CHW'):
    """Vertically flips the given paddle tensor.

    Args:
        img (paddle.Tensor): Image to be flipped.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        paddle.Tensor:  Vertically flipped image.

    """
    assert data_format.lower() in ('chw', 'hwc'
                                   ), "data_format should in ('chw', 'hwc')"

    if data_format.lower() == 'chw':
        return img.flip(axis=[-2])
    else:
        return img.flip(axis=[-3])


def hflip(img, data_format='CHW'):
    """Horizontally flips the given paddle.Tensor Image.

    Args:
        img (paddle.Tensor): Image to be flipped.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        paddle.Tensor:  Horizontall flipped image.

    """
    assert data_format.lower() in ('chw', 'hwc'
                                   ), "data_format should in ('chw', 'hwc')"

    if data_format.lower() == 'chw':
        return img.flip(axis=[-1])
    else:
        return img.flip(axis=[-2])


def crop(img, top, left, height, width, data_format='CHW'):
    """Crops the given paddle.Tensor Image.

    Args:
        img (paddle.Tensor): Image to be cropped. (0,0) denotes the top left 
            corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
    Returns:
        paddle.Tensor: Cropped image.

    """
    assert data_format.lower() in ('chw', 'hwc'
                                   ), "data_format should in ('chw', 'hwc')"

    if data_format.lower() == 'chw':
        return img[:, top:top + height, left:left + width]
    else:
        return img[top:top + height, left:left + width, :]


def center_crop(img, output_size, data_format='CHW'):
    """Crops the given paddle.Tensor Image and resize it to desired size.

        Args:
            img (paddle.Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions   
            data_format (str, optional): Data format of img, should be 'HWC' or 
                'CHW'. Default: 'CHW'.     
        Returns:
            paddle.Tensor: Cropped image.

        """
    _assert_paddle_image(img)
    _assert_data_format(data_format)

    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))

    image_width, image_height = _get_image_size(img, data_format)
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(
        img,
        crop_top,
        crop_left,
        crop_height,
        crop_width,
        data_format=data_format)
