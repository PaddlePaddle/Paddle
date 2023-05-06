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

import math
import numbers

import numpy as np

import paddle
import paddle.nn.functional as F

from ...fluid.framework import Variable

__all__ = []


def _assert_image_tensor(img, data_format):
    if (
        not isinstance(img, (paddle.Tensor, Variable))
        or img.ndim < 3
        or img.ndim > 4
        or not data_format.lower() in ('chw', 'hwc')
    ):
        raise RuntimeError(
            'not support [type={}, ndim={}, data_format={}] paddle image'.format(
                type(img), img.ndim, data_format
            )
        )


def _get_image_h_axis(data_format):
    if data_format.lower() == 'chw':
        return -2
    elif data_format.lower() == 'hwc':
        return -3


def _get_image_w_axis(data_format):
    if data_format.lower() == 'chw':
        return -1
    elif data_format.lower() == 'hwc':
        return -2


def _get_image_c_axis(data_format):
    if data_format.lower() == 'chw':
        return -3
    elif data_format.lower() == 'hwc':
        return -1


def _get_image_n_axis(data_format):
    if len(data_format) == 3:
        return None
    elif len(data_format) == 4:
        return 0


def _is_channel_last(data_format):
    return _get_image_c_axis(data_format) == -1


def _is_channel_first(data_format):
    return _get_image_c_axis(data_format) == -3


def _get_image_num_batches(img, data_format):
    if _get_image_n_axis(data_format):
        return img.shape[_get_image_n_axis(data_format)]
    return None


def _get_image_num_channels(img, data_format):
    return img.shape[_get_image_c_axis(data_format)]


def _get_image_size(img, data_format):
    return (
        img.shape[_get_image_w_axis(data_format)],
        img.shape[_get_image_h_axis(data_format)],
    )


def _rgb_to_hsv(img):
    """Convert a image Tensor from RGB to HSV. This implementation is based on Pillow (
    https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Convert.c)
    """
    maxc = img.max(axis=-3)
    minc = img.min(axis=-3)

    is_equal = paddle.equal(maxc, minc)
    one_divisor = paddle.ones_like(maxc)
    c_delta = maxc - minc
    # s is 0 when maxc == minc, set the divisor to 1 to avoid zero divide.
    s = c_delta / paddle.where(is_equal, one_divisor, maxc)

    r, g, b = img.unbind(axis=-3)
    c_delta_divisor = paddle.where(is_equal, one_divisor, c_delta)
    # when maxc == minc, there is r == g == b, set the divisor to 1 to avoid zero divide.
    rc = (maxc - r) / c_delta_divisor
    gc = (maxc - g) / c_delta_divisor
    bc = (maxc - b) / c_delta_divisor

    hr = (maxc == r).astype(maxc.dtype) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)).astype(maxc.dtype) * (rc - bc + 2.0)
    hb = ((maxc != r) & (maxc != g)).astype(maxc.dtype) * (gc - rc + 4.0)
    h = (hr + hg + hb) / 6.0 + 1.0
    h = h - h.trunc()
    return paddle.stack([h, s, maxc], axis=-3)


def _hsv_to_rgb(img):
    """Convert a image Tensor from HSV to RGB."""
    h, s, v = img.unbind(axis=-3)
    f = h * 6.0
    i = paddle.floor(f)
    f = f - i
    i = i.astype(paddle.int32) % 6

    p = paddle.clip(v * (1.0 - s), 0.0, 1.0)
    q = paddle.clip(v * (1.0 - s * f), 0.0, 1.0)
    t = paddle.clip(v * (1.0 - s * (1.0 - f)), 0.0, 1.0)

    mask = paddle.equal(
        i.unsqueeze(axis=-3),
        paddle.arange(6, dtype=i.dtype).reshape((-1, 1, 1)),
    ).astype(img.dtype)
    matrix = paddle.stack(
        [
            paddle.stack([v, q, p, p, t, v], axis=-3),
            paddle.stack([t, v, v, q, p, p], axis=-3),
            paddle.stack([p, p, t, v, v, q], axis=-3),
        ],
        axis=-4,
    )
    return paddle.einsum("...ijk, ...xijk -> ...xjk", mask, matrix)


def _blend_images(img1, img2, ratio):
    max_value = 1.0 if paddle.is_floating_point(img1) else 255.0
    return (
        paddle.lerp(img2, img1, float(ratio))
        .clip(0, max_value)
        .astype(img1.dtype)
    )


def normalize(img, mean, std, data_format='CHW'):
    """Normalizes a tensor image given mean and standard deviation.

    Args:
        img (paddle.Tensor): input data to be normalized.
        mean (list|tuple): Sequence of means for each channel.
        std (list|tuple): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of img, should be 'HWC' or
            'CHW'. Default: 'CHW'.

    Returns:
        Tensor: Normalized mage.

    """
    _assert_image_tensor(img, data_format)

    mean = paddle.to_tensor(mean, place=img.place)
    std = paddle.to_tensor(std, place=img.place)

    if _is_channel_first(data_format):
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
        paddle.Tensor: Grayscale version of the image.
    """
    _assert_image_tensor(img, data_format)

    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')

    rgb_weights = paddle.to_tensor(
        [0.2989, 0.5870, 0.1140], place=img.place
    ).astype(img.dtype)

    if _is_channel_first(data_format):
        rgb_weights = rgb_weights.reshape((-1, 1, 1))

    _c_index = _get_image_c_axis(data_format)

    img = (img * rgb_weights).sum(axis=_c_index, keepdim=True)
    _shape = img.shape
    _shape[_c_index] = num_output_channels

    return img.expand(_shape)


def _affine_grid(theta, w, h, ow, oh):
    d = 0.5
    base_grid = paddle.ones((1, oh, ow, 3), dtype=theta.dtype)

    x_grid = paddle.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, ow)
    base_grid[..., 0] = x_grid

    if paddle.in_dynamic_mode():
        y_grid = paddle.linspace(
            -oh * 0.5 + d, oh * 0.5 + d - 1, oh
        ).unsqueeze_(-1)
        base_grid[..., 1] = y_grid
        tmp = paddle.to_tensor([0.5 * w, 0.5 * h])
    else:
        # To eliminate the warning:
        # In static mode, unsqueeze_() is the same as unsqueeze() and does not perform inplace operation.
        y_grid = paddle.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, oh).unsqueeze(
            -1
        )
        base_grid[..., 1] = y_grid
        tmp = paddle.assign(np.array([0.5 * w, 0.5 * h], dtype="float32"))

    scaled_theta = theta.transpose((0, 2, 1)) / tmp
    output_grid = base_grid.reshape((1, oh * ow, 3)).bmm(scaled_theta)

    return output_grid.reshape((1, oh, ow, 2))


def _grid_transform(img, grid, mode, fill):
    if img.shape[0] > 1:
        grid = grid.expand(
            shape=[img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]]
        )

    if fill is not None:
        dummy = paddle.ones(
            (img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype
        )
        img = paddle.concat((img, dummy), axis=1)

    img = F.grid_sample(
        img, grid, mode=mode, padding_mode="zeros", align_corners=False
    )

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # n 1 h w
        img = img[:, :-1, :, :]  # n c h w
        mask = mask.tile([1, img.shape[1], 1, 1])
        len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1

        if paddle.in_dynamic_mode():
            fill_img = (
                paddle.to_tensor(fill)
                .reshape((1, len_fill, 1, 1))
                .astype(img.dtype)
                .expand_as(img)
            )
        else:
            fill = np.array(fill).reshape(len_fill).astype("float32")
            fill_img = paddle.ones_like(img) * paddle.assign(fill).reshape(
                [1, len_fill, 1, 1]
            )

        if mode == 'nearest':
            mask = paddle.cast(mask < 0.5, img.dtype)
            img = img * (1.0 - mask) + mask * fill_img
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    return img


def affine(img, matrix, interpolation="nearest", fill=None, data_format='CHW'):
    """Affine to the image by matrix.

    Args:
        img (paddle.Tensor): Image to be rotated.
        matrix (float or int): Affine matrix.
        interpolation (str, optional): Interpolation method. If omitted, or if the
            image has only one channel, it is set NEAREST . when use pil backend,
            support method are as following:
            - "nearest"
            - "bilinear"
            - "bicubic"
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.
        data_format (str, optional): Data format of img, should be 'HWC' or
            'CHW'. Default: 'CHW'.

    Returns:
        paddle.Tensor: Affined image.

    """
    ndim = len(img.shape)
    if ndim == 3:
        img = img.unsqueeze(0)

    img = img if data_format.lower() == 'chw' else img.transpose((0, 3, 1, 2))

    matrix = paddle.to_tensor(matrix, place=img.place)
    matrix = matrix.reshape((1, 2, 3))
    shape = img.shape

    grid = _affine_grid(
        matrix, w=shape[-1], h=shape[-2], ow=shape[-1], oh=shape[-2]
    )

    if isinstance(fill, int):
        fill = tuple([fill] * 3)

    out = _grid_transform(img, grid, mode=interpolation, fill=fill)

    out = out if data_format.lower() == 'chw' else out.transpose((0, 2, 3, 1))
    out = out.squeeze(0) if ndim == 3 else out

    return out


def rotate(
    img,
    angle,
    interpolation='nearest',
    expand=False,
    center=None,
    fill=None,
    data_format='CHW',
):
    """Rotates the image by angle.

    Args:
        img (paddle.Tensor): Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        interpolation (str, optional): Interpolation method. If omitted, or if the
            image has only one channel, it is set NEAREST . when use pil backend,
            support method are as following:
            - "nearest"
            - "bilinear"
            - "bicubic"
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.

    Returns:
        paddle.Tensor: Rotated image.

    """

    angle = -angle % 360
    img = img.unsqueeze(0)

    # n, c, h, w = img.shape
    w, h = _get_image_size(img, data_format=data_format)

    img = img if data_format.lower() == 'chw' else img.transpose((0, 3, 1, 2))

    post_trans = [0, 0]

    if center is None:
        rotn_center = [0, 0]
    else:
        rotn_center = [(p - s * 0.5) for p, s in zip(center, [w, h])]

    if paddle.in_dynamic_mode():
        angle = math.radians(angle)
        matrix = [
            math.cos(angle),
            math.sin(angle),
            0.0,
            -math.sin(angle),
            math.cos(angle),
            0.0,
        ]
        matrix = paddle.to_tensor(matrix, place=img.place)
    else:
        angle = angle / 180 * math.pi
        matrix = paddle.concat(
            [
                paddle.cos(angle),
                paddle.sin(angle),
                paddle.zeros([1]),
                -paddle.sin(angle),
                paddle.cos(angle),
                paddle.zeros([1]),
            ]
        )

    matrix[2] += matrix[0] * (-rotn_center[0] - post_trans[0]) + matrix[1] * (
        -rotn_center[1] - post_trans[1]
    )
    matrix[5] += matrix[3] * (-rotn_center[0] - post_trans[0]) + matrix[4] * (
        -rotn_center[1] - post_trans[1]
    )

    matrix[2] += rotn_center[0]
    matrix[5] += rotn_center[1]

    matrix = matrix.reshape((1, 2, 3))

    if expand:
        # calculate output size
        if paddle.in_dynamic_mode():
            corners = paddle.to_tensor(
                [
                    [-0.5 * w, -0.5 * h, 1.0],
                    [-0.5 * w, 0.5 * h, 1.0],
                    [0.5 * w, 0.5 * h, 1.0],
                    [0.5 * w, -0.5 * h, 1.0],
                ],
                place=matrix.place,
            ).astype(matrix.dtype)
        else:
            corners = paddle.assign(
                [
                    [-0.5 * w, -0.5 * h, 1.0],
                    [-0.5 * w, 0.5 * h, 1.0],
                    [0.5 * w, 0.5 * h, 1.0],
                    [0.5 * w, -0.5 * h, 1.0],
                ],
            ).astype(matrix.dtype)

        _pos = (
            corners.reshape((1, -1, 3))
            .bmm(matrix.transpose((0, 2, 1)))
            .reshape((1, -1, 2))
        )
        _min = _pos.min(axis=-2).floor()
        _max = _pos.max(axis=-2).ceil()

        npos = _max - _min
        nw = npos[0][0]
        nh = npos[0][1]

        if paddle.in_dynamic_mode():
            ow, oh = int(nw), int(nh)
        else:
            ow, oh = nw.astype("int32"), nh.astype("int32")

    else:
        ow, oh = w, h

    grid = _affine_grid(matrix, w, h, ow, oh)

    out = _grid_transform(img, grid, mode=interpolation, fill=fill)

    out = out if data_format.lower() == 'chw' else out.transpose((0, 2, 3, 1))

    return out.squeeze(0)


def _perspective_grid(img, coeffs, ow, oh, dtype):
    theta1 = coeffs[:6].reshape([1, 2, 3])
    tmp = paddle.tile(coeffs[6:].reshape([1, 2]), repeat_times=[2, 1])
    dummy = paddle.ones((2, 1), dtype=dtype)
    theta2 = paddle.concat((tmp, dummy), axis=1).unsqueeze(0)

    d = 0.5
    base_grid = paddle.ones((1, oh, ow, 3), dtype=dtype)

    x_grid = paddle.linspace(d, ow * 1.0 + d - 1.0, ow)
    base_grid[..., 0] = x_grid
    y_grid = paddle.linspace(d, oh * 1.0 + d - 1.0, oh).unsqueeze_(-1)
    base_grid[..., 1] = y_grid

    scaled_theta1 = theta1.transpose((0, 2, 1)) / paddle.to_tensor(
        [0.5 * ow, 0.5 * oh]
    )
    output_grid1 = base_grid.reshape((1, oh * ow, 3)).bmm(scaled_theta1)
    output_grid2 = base_grid.reshape((1, oh * ow, 3)).bmm(
        theta2.transpose((0, 2, 1))
    )

    output_grid = output_grid1 / output_grid2 - 1.0
    return output_grid.reshape((1, oh, ow, 2))


def perspective(
    img, coeffs, interpolation="nearest", fill=None, data_format='CHW'
):
    """Perspective the image.

    Args:
        img (paddle.Tensor): Image to be rotated.
        coeffs (list[float]): coefficients (a, b, c, d, e, f, g, h) of the perspective transforms.
        interpolation (str, optional): Interpolation method. If omitted, or if the
            image has only one channel, it is set NEAREST. When use pil backend,
            support method are as following:
            - "nearest"
            - "bilinear"
            - "bicubic"
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.

    Returns:
        paddle.Tensor: Perspectived image.

    """

    ndim = len(img.shape)
    if ndim == 3:
        img = img.unsqueeze(0)

    img = img if data_format.lower() == 'chw' else img.transpose((0, 3, 1, 2))
    ow, oh = img.shape[-1], img.shape[-2]
    dtype = img.dtype if paddle.is_floating_point(img) else paddle.float32

    coeffs = paddle.to_tensor(coeffs, place=img.place)
    grid = _perspective_grid(img, coeffs, ow=ow, oh=oh, dtype=dtype)
    out = _grid_transform(img, grid, mode=interpolation, fill=fill)

    out = out if data_format.lower() == 'chw' else out.transpose((0, 2, 3, 1))
    out = out.squeeze(0) if ndim == 3 else out

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
    _assert_image_tensor(img, data_format)

    h_axis = _get_image_h_axis(data_format)

    return img.flip(axis=[h_axis])


def hflip(img, data_format='CHW'):
    """Horizontally flips the given paddle.Tensor Image.

    Args:
        img (paddle.Tensor): Image to be flipped.
        data_format (str, optional): Data format of img, should be 'HWC' or
            'CHW'. Default: 'CHW'.

    Returns:
        paddle.Tensor:  Horizontall flipped image.

    """
    _assert_image_tensor(img, data_format)

    w_axis = _get_image_w_axis(data_format)

    return img.flip(axis=[w_axis])


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
    _assert_image_tensor(img, data_format)

    if _is_channel_first(data_format):
        return img[:, top : top + height, left : left + width]
    else:
        return img[top : top + height, left : left + width, :]


def erase(img, i, j, h, w, v, inplace=False):
    """Erase the pixels of selected area in input Tensor image with given value.

    Args:
         img (paddle.Tensor): input Tensor image.
         i (int): y coordinate of the top-left point of erased region.
         j (int): x coordinate of the top-left point of erased region.
         h (int): Height of the erased region.
         w (int): Width of the erased region.
         v (paddle.Tensor): value used to replace the pixels in erased region.
         inplace (bool, optional): Whether this transform is inplace. Default: False.

     Returns:
         paddle.Tensor: Erased image.

    """
    _assert_image_tensor(img, 'CHW')
    if not inplace:
        img = img.clone()

    img[..., i : i + h, j : j + w] = v
    return img


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
    _assert_image_tensor(img, data_format)

    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))

    image_width, image_height = _get_image_size(img, data_format)
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return crop(
        img,
        crop_top,
        crop_left,
        crop_height,
        crop_width,
        data_format=data_format,
    )


def pad(img, padding, fill=0, padding_mode='constant', data_format='CHW'):
    """
    Pads the given paddle.Tensor on all sides with specified padding mode and fill value.

    Args:
        img (paddle.Tensor): Image to be padded.
        padding (int|list|tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (float, optional): Pixel fill value for constant fill. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant. Default: 0.
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default: 'constant'.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value on the edge of the image

            - reflect: pads with reflection of image (without repeating the last value on the edge)

                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image (repeating the last value on the edge)

                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        paddle.Tensor: Padded image.

    """
    _assert_image_tensor(img, data_format)

    if not isinstance(padding, (numbers.Number, list, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, list, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, (list, tuple)) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a "
            + f"{len(padding)} element tuple"
        )

    assert padding_mode in [
        'constant',
        'edge',
        'reflect',
        'symmetric',
    ], 'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    padding = [pad_left, pad_right, pad_top, pad_bottom]

    if padding_mode == 'edge':
        padding_mode = 'replicate'
    elif padding_mode == 'symmetric':
        raise ValueError('Do not support symmetric mode')

    img = img.unsqueeze(0)
    #  'constant', 'reflect', 'replicate', 'circular'
    img = F.pad(
        img,
        pad=padding,
        mode=padding_mode,
        value=float(fill),
        data_format='N' + data_format,
    )

    return img.squeeze(0)


def resize(img, size, interpolation='bilinear', data_format='CHW'):
    """
    Resizes the image to given size

    Args:
        input (paddle.Tensor): Image to be resized.
        size (int|list|tuple): Target size of input data, with (height, width) shape.
        interpolation (int|str, optional): Interpolation method. when use paddle backend,
            support method are as following:
            - "nearest"
            - "bilinear"
            - "bicubic"
            - "trilinear"
            - "area"
            - "linear"
        data_format (str, optional): paddle.Tensor format
            - 'CHW'
            - 'HWC'
    Returns:
        paddle.Tensor: Resized image.

    """
    _assert_image_tensor(img, data_format)

    if not (
        isinstance(size, int)
        or (isinstance(size, (tuple, list)) and len(size) == 2)
    ):
        raise TypeError(f'Got inappropriate size arg: {size}')

    if isinstance(size, int):
        w, h = _get_image_size(img, data_format)
        # TODO(Aurelius84): In static graph mode, w and h will be -1 for dynamic shape.
        # We should consider to support this case in future.
        if w <= 0 or h <= 0:
            raise NotImplementedError(
                "Not support while w<=0 or h<=0, but received w={}, h={}".format(
                    w, h
                )
            )
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
    else:
        oh, ow = size

    img = img.unsqueeze(0)
    img = F.interpolate(
        img,
        size=(oh, ow),
        mode=interpolation.lower(),
        data_format='N' + data_format.upper(),
    )

    return img.squeeze(0)


def adjust_brightness(img, brightness_factor):
    """Adjusts brightness of an Image.

    Args:
        img (paddle.Tensor): Image to be adjusted.
        brightness_factor (float): How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        paddle.Tensor: Brightness adjusted image.

    """
    _assert_image_tensor(img, 'CHW')
    assert brightness_factor >= 0, "brightness_factor should be non-negative."
    assert _get_image_num_channels(img, 'CHW') in [
        1,
        3,
    ], "channels of input should be either 1 or 3."

    extreme_target = paddle.zeros_like(img, img.dtype)
    return _blend_images(img, extreme_target, brightness_factor)


def adjust_contrast(img, contrast_factor):
    """Adjusts contrast of an image.

    Args:
        img (paddle.Tensor): Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        paddle.Tensor: Contrast adjusted image.

    """
    _assert_image_tensor(img, 'chw')
    assert contrast_factor >= 0, "contrast_factor should be non-negative."

    channels = _get_image_num_channels(img, 'CHW')
    dtype = img.dtype if paddle.is_floating_point(img) else paddle.float32
    if channels == 1:
        extreme_target = paddle.mean(
            img.astype(dtype), axis=(-3, -2, -1), keepdim=True
        )
    elif channels == 3:
        extreme_target = paddle.mean(
            to_grayscale(img).astype(dtype), axis=(-3, -2, -1), keepdim=True
        )
    else:
        raise ValueError("channels of input should be either 1 or 3.")

    return _blend_images(img, extreme_target, contrast_factor)


def adjust_saturation(img, saturation_factor):
    """Adjusts color saturation of an image.

    Args:
        img (paddle.Tensor): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        paddle.Tensor: Saturation adjusted image.

    """
    _assert_image_tensor(img, 'CHW')
    assert saturation_factor >= 0, "saturation_factor should be non-negative."
    channels = _get_image_num_channels(img, 'CHW')
    if channels == 1:
        return img
    elif channels == 3:
        extreme_target = to_grayscale(img)
    else:
        raise ValueError("channels of input should be either 1 or 3.")

    return _blend_images(img, extreme_target, saturation_factor)


def adjust_hue(img, hue_factor):
    """Adjusts hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    Args:
        img (paddle.Tensor): Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        paddle.Tensor: Hue adjusted image.

    """
    _assert_image_tensor(img, 'CHW')
    assert (
        hue_factor >= -0.5 and hue_factor <= 0.5
    ), "hue_factor should be in range [-0.5, 0.5]"
    channels = _get_image_num_channels(img, 'CHW')
    if channels == 1:
        return img
    elif channels == 3:
        dtype = img.dtype
        if dtype == paddle.uint8:
            img = img.astype(paddle.float32) / 255.0

        img_hsv = _rgb_to_hsv(img)
        h, s, v = img_hsv.unbind(axis=-3)
        h = h + hue_factor
        h = h - h.floor()
        img_adjusted = _hsv_to_rgb(paddle.stack([h, s, v], axis=-3))

        if dtype == paddle.uint8:
            img_adjusted = (img_adjusted * 255.0).astype(dtype)
    else:
        raise ValueError("channels of input should be either 1 or 3.")

    return img_adjusted
