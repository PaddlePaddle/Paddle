import unittest
import numpy as np
from op_test import OpTest


def get_output_shape(attrs, x):
    img_height = x.shape[1]
    img_width = x.shape[2]

    padding_height = attrs['paddingHeight']
    padding_width = attrs['paddingWidth']
    block_height = attrs['blockHeight']
    block_width = attrs['blockWidth']
    stride_height = attrs['strideHeight']
    stride_width = attrs['strideWidth']

    output_height = \
      1 +  \
      (img_height + 2 * padding_height - block_height + stride_height - 1) / \
          strideHeight

    output_width = \
      1 + \
      (img_width + 2 * padding_width - block_width + stride_width - 1) / \
          stride_width

    return output_height, output_width


def im2col(attrs, im, col):
    """
    im: {CHW}
    col:
        {outputHeight, outputWidth, inputChannels, filterHeight, filterWidth}
    """
    input_channels = im.shape[0]
    input_height = im.shape[1]
    input_width = im.shape[2]

    output_height = col.shape[0]
    output_width = col.shape[1]
    filter_height = col.shape[3]
    filter_width = col.shape[4]

    stride_height = attrs['strideHeight']
    stride_width = attrs['strideWidth']
    padding_height = attrs['paddingHeight']
    padding_width = attrs['paddingWidth']

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
                            col[col_row_idx][col_col_idx][channel][\
                                filter_row_idx][filter_col_idx] = 0.0
                        else:
                            im_offset = (channel * input_height + im_row_offset \
                                         ) * input_width + im_col_offset

                            col[col_row_idx][col_col_idx][channel][\
                                filter_row_idx][filter_col_idx] = im[channel][ \
                                    im_row_offset][im_col_offset]


def col2img(attrs, col, img):
    """
    img: {CHW}
    col:
        {output_height, outputWidth, inputChannels, filterHeight, filterWidth}
    """
    input_channels = im.shape[0]
    input_height = im.shape[1]
    input_width = im.shape[2]

    output_height = col.shape[0]
    output_width = col.shape[1]
    filter_height = col.shape[3]
    filter_width = col.shape[4]

    stride_height = attrs['strideHeight']
    stride_width = attrs['strideWidth']
    padding_height = attrs['paddingHeight']
    padding_width = attrs['paddingWidth']

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


def get_input_data(C, H, W):
    x = np.random.uniform(0.1, 1, [C, H, W]).astype("float32")
    for c in range(0, C):
        for h in range(0, H):
            for w in range(0, W):
                #x[c][h][w] = c * H * W + h *W + w
                x[c][h][w] = 0.2 + 0.01 * (c * H * W + h * W + w)
        return x


class TestBlockExpandOp(OpTest):
    def setUp(self):
        C = 3
        H = 4
        W = 4
        x = get_input_data(C, H, W)

        attrs = {
            'blockHeight': 2,
            'blockWidth': 2,
            'strideHeight': 1,
            'strideWidth': 1,
            'paddingHeight': 1,
            'paddingWidth': 1,
        }

        output_height, output_width = get_output_shape(attrs, x)
        out = np.random.uniform(0.1, 1,\
                    [output_height, output_width, x.shape[0], \
                     attrs['blockHeight'], attrs['blockWidth']]).astype("float32")

        self.op_type = "block_expand"
        self.inputs = {'X': x.reshape(1, C, H, W)}
        self.attrs = attrs

        im2col(attrs, x, out)
        self.outputs = {
            'Out':out.reshape(1, output_height, output_width, x.shape[0], \
                     attrs['blockHeight'], attrs['blockWidth'])
            }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestBlockExpandOp2(OpTest):
    def setUp(self):
        C = 3
        H = 4
        W = 5
        x = get_input_data(C, H, W)

        attrs = {
            'blockHeight': 2,
            'blockWidth': 1,
            'strideHeight': 2,
            'strideWidth': 1,
            'paddingHeight': 2,
            'paddingWidth': 1,
        }

        output_height, output_width = get_output_shape(attrs, x)
        out = np.random.uniform(0.1, 1,\
                    [output_height, output_width, x.shape[0], \
                     attrs['blockHeight'], attrs['blockWidth']]).astype("float32")

        self.op_type = "block_expand"
        self.inputs = {'X': x.reshape(1, C, H, W)}
        self.attrs = attrs

        im2col(attrs, x, out)
        self.outputs = {
            'Out':out.reshape(1, output_height, output_width, x.shape[0], \
                     attrs['blockHeight'], attrs['blockWidth'])
            }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
