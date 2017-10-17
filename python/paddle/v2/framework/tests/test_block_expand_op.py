import unittest
import numpy as np
from op_test import OpTest


def get_output_shape(attrs, X):
    img_height = X.shape[2]
    img_width = X.shpe[3]
    padding_height = attrs['padding_height']
    padding_width = attrs['padding_width']
    block_height = attrs['block_height']
    block_width = attrs['block_width']
    stride_height = attrs['stride_height']
    stride_width = attrs['stride_width']
    output_height = \
      1 +  \
      (img_height + 2 * padding_height - block_height + stride_height - 1) / \
          stride_height

    output_width = \
      1 + \
      (img_width + 2 * padding_width - block_width + stride_width - 1) / \
          stride_width

    return output_height, output_width


"""
img: {CHW}
col:
    {output_height, output_width, inputChannels, filterHeight, filterWidth}
"""


def img2col(attrs, im, col):
    input_channels = im.shape.dims[0]
    input_height = im.shape.dims[1]
    input_width = im.shape.dims[2]
    filter_height = col.shape.dims[3]
    filter_width = col.shape.dims[4]
    output_height = col.shape.dims[0]
    output_width = col.shape.dims[1]

    for col_row_idx in range(0, output_height):
        for col_col_idx in range(0, output_width):
            for channel in range(0, input_channels):
                for filter_row_idx in range(0, filter_height):
                    for filter_col_idx in range(0, filter_width):
                        im_row_offset = col_row_idx * stride_height \
                            + filter_row_idx - padding_height
                        im_col_offset = col_col_idx * stride_width \
                            + filter_col_idx - padding_width
                        if (im_row_offset < 0 or
                                im_row_offset >= input_height or
                                im_col_offset < 0 or
                                im_col_offset >= input_width):
                            col[col_row_idx][col_col_idx][channel][
                                filter_row_idx][filter_col_idx] = 0.0
                        else:
                            im_offset = (channel * input_height + im_row_offset
                                         ) * input_width + im_col_offset
                            col[col_row_idx][col_col_idx][channel][
                                filter_row_idx][filter_col_idx] = im[channel][
                                    im_row_offset][im_col_offset]


"""
img: {CHW}
col:
    {output_height, output_width, inputChannels, filterHeight, filterWidth}
"""


def col2img(attrs, col, img):
    input_channels = im.shape.dims[0]
    input_height = im.shape.dims[1]
    input_width = im.shape.dims[2]
    filter_height = col.shape.dims[3]
    filter_width = col.shape.dims[4]
    output_height = col.shape.dims[0]
    output_width = col.shape.dims[1]

    for col_row_idx in range(0, output_height):
        for col_col_idx in range(0, output_width):
            for channel in range(0, input_channels):
                for filter_row_idx in range(0, filter_height):
                    for filter_col_idx in range(0, filter_width):
                        im_row_offset = \
                            col_row_idx * stride_height + filter_row_idx - padding_height
                        im_col_offset = \
                            col_col_idx * stride_width + filter_col_idx - padding_width
                        if (im_row_offset >= 0 and
                                im_row_offset < input_height and
                                im_col_offset >= 0 and
                                im_col_offset < input_width):
                            im[channel][im_row_offset][im_col_offset] = \
                                col[col_row_idx][col_col_idx][channel][filter_row_idx][filter_col_idx]


class TestBlockExpandMulOp(OpTest):
    def setUp(self):
        self.op_type = "block_expand"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 9, 9]).astype("float64"),
        }
        self.attrs = {
            'block_height': 3,
            'block_width': 3,
            'stride_height': 2,
            'stride_width': 2,
            'padding_height': 3,
            'padding_width': 3,
        }

        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')
