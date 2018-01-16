import unittest
import numpy as np
from op_test import OpTest


def get_output_shape(attrs, x):
    img_height = x.shape[2]
    img_width = x.shape[3]

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

    stride_height = attrs['stride_height']
    stride_width = attrs['stride_width']
    padding_height = attrs['padding_height']
    padding_width = attrs['padding_width']

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


def block_expand(inputs, attrs):
    output_height, output_width = get_output_shape(attrs, inputs)
    img_channels = inputs.shape[1]
    batch_size = inputs.shape[0]
    out = np.zeros([
        batch_size, output_height, output_width, img_channels,
        attrs['block_height'], attrs['block_width']
    ]).astype("float32")

    for i in range(len(inputs)):
        im2col(attrs, inputs[i], out[i])

    out = out.reshape([
        batch_size * output_height * output_width,
        img_channels * attrs['block_height'] * attrs['block_width']
    ])
    return out


class TestBlockExpandOp(OpTest):
    def config(self):
        self.batch_size = 1
        self.img_channels = 3
        self.img_height = 4
        self.img_width = 4
        self.attrs = {
            'block_height': 2,
            'block_width': 2,
            'stride_height': 1,
            'stride_width': 1,
            'padding_height': 1,
            'padding_width': 1,
        }

    def setUp(self):
        self.config()
        self.op_type = "block_expand"
        #x = np.random.uniform(0.1, 1,
        x = np.random.randint(0, 10, [
            self.batch_size, self.img_channels, self.img_height, self.img_width
        ]).astype("float32")

        out = block_expand(x, self.attrs)
        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestBlockExpandOpCase2(TestBlockExpandOp):
    def config(self):
        self.batch_size = 2
        self.img_channels = 3
        self.img_height = 4
        self.img_width = 5
        self.attrs = {
            'block_height': 2,
            'block_width': 1,
            'stride_height': 2,
            'stride_width': 1,
            'padding_height': 2,
            'padding_width': 1,
        }


class TestBlockExpandOpCase3(TestBlockExpandOp):
    def config(self):
        self.batch_size = 3
        self.img_channels = 1
        self.img_height = 4
        self.img_width = 5
        self.attrs = {
            'block_height': 2,
            'block_width': 1,
            'stride_height': 2,
            'stride_width': 1,
            'padding_height': 2,
            'padding_width': 0,
        }


class TestBlockExpandOpCase4(TestBlockExpandOp):
    def config(self):
        self.batch_size = 2
        self.img_channels = 2
        self.img_height = 3
        self.img_width = 3
        self.attrs = {
            'block_height': 2,
            'block_width': 2,
            'stride_height': 1,
            'stride_width': 1,
            'padding_height': 0,
            'padding_width': 0,
        }


if __name__ == '__main__':
    unittest.main()
