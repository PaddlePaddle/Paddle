import unittest
import numpy as np
from op_test import OpTest


def get_output_shape(attrs, x):
    imgHeight = x.shape[1]
    imgWidth = x.shape[2]

    paddingHeight = attrs['paddingHeight']
    paddingWidth = attrs['paddingWidth']
    blockHeight = attrs['blockHeight']
    blockWidth = attrs['blockWidth']
    strideHeight = attrs['strideHeight']
    strideWidth = attrs['strideWidth']

    outputHeight = \
      1 +  \
      (imgHeight + 2 * paddingHeight - blockHeight + strideHeight - 1) / \
          strideHeight

    outputWidth = \
      1 + \
      (imgWidth + 2 * paddingWidth - blockWidth + strideWidth - 1) / \
          strideWidth

    return outputHeight, outputWidth


"""
im: {CHW}
col:
    {outputHeight, outputWidth, inputChannels, filterHeight, filterWidth}
"""


def im2col(attrs, im, col):
    input_channels = im.shape[0]
    inputHeight = im.shape[1]
    inputWidth = im.shape[2]

    outputHeight = col.shape[0]
    outputWidth = col.shape[1]
    filterHeight = col.shape[3]
    filterWidth = col.shape[4]

    strideHeight = attrs['strideHeight']
    strideWidth = attrs['strideWidth']
    paddingHeight = attrs['paddingHeight']
    paddingWidth = attrs['paddingWidth']

    for col_row_idx in range(0, outputHeight):
        for col_col_idx in range(0, outputWidth):
            for channel in range(0, input_channels):
                for filter_row_idx in range(0, filterHeight):
                    for filter_col_idx in range(0, filterWidth):
                        im_row_offset = col_row_idx * strideHeight \
                            + filter_row_idx - paddingHeight

                        im_col_offset = col_col_idx * strideWidth \
                            + filter_col_idx - paddingWidth

                        if (im_row_offset < 0 or im_row_offset >= inputHeight or
                                im_col_offset < 0 or
                                im_col_offset >= inputWidth):
                            col[col_row_idx][col_col_idx][channel][\
                                filter_row_idx][filter_col_idx] = 0.0
                        else:
                            im_offset = (channel * inputHeight + im_row_offset \
                                         ) * inputWidth + im_col_offset

                            col[col_row_idx][col_col_idx][channel][\
                                filter_row_idx][filter_col_idx] = im[channel][ \
                                    im_row_offset][im_col_offset]


"""
img: {CHW}
col:
    {outputHeight, outputWidth, inputChannels, filterHeight, filterWidth}
"""


def col2img(attrs, col, img):
    input_channels = im.shape[0]
    inputHeight = im.shape[1]
    inputWidth = im.shape[2]

    outputHeight = col.shape[0]
    outputWidth = col.shape[1]
    filterHeight = col.shape[3]
    filterWidth = col.shape[4]

    strideHeight = attrs['strideHeight']
    strideWidth = attrs['strideWidth']
    paddingHeight = attrs['paddingHeight']
    paddingWidth = attrs['paddingWidth']

    for col_row_idx in range(0, outputHeight):
        for col_col_idx in range(0, outputWidth):
            for channel in range(0, input_channels):
                for filter_row_idx in range(0, filterHeight):
                    for filter_col_idx in range(0, filterWidth):
                        im_row_offset = \
                            col_row_idx * strideHeight + filter_row_idx - paddingHeight
                        im_col_offset = \
                            col_col_idx * strideWidth + filter_col_idx - paddingWidth
                        if (im_row_offset >= 0 and
                                im_row_offset < inputHeight and
                                im_col_offset >= 0 and
                                im_col_offset < inputWidth):
                            im[channel][im_row_offset][im_col_offset] = \
                                col[col_row_idx][col_col_idx][channel][filter_row_idx][filter_col_idx]


class TestBlockExpandOp(OpTest):
    def get_input_data(self, C, H, W):
        x = np.random.uniform(0.1, 1, [C, H, W]).astype("float32")
        for c in range(0, C):
            for h in range(0, H):
                for w in range(0, W):
                    #x[c][h][w] = c * H * W + h *W + w
                    x[c][h][w] = 0.2 + 0.01 * (c * H * W + h * W + w)
        return x

    def setUp(self):
        C = 3
        H = 4
        W = 4
        x = self.get_input_data(C, H, W)
        #print x

        attrs = {
            'blockHeight': 2,
            'blockWidth': 2,
            'strideHeight': 1,
            'strideWidth': 1,
            'paddingHeight': 1,
            'paddingWidth': 1,
        }

        outputHeight, outputWidth = get_output_shape(attrs, x)
        out = np.random.uniform(0.1, 1,\
                    [outputHeight, outputWidth, x.shape[0], \
                     attrs['blockHeight'], attrs['blockWidth']]).astype("float32")

        self.op_type = "block_expand"
        self.inputs = {'X': x.reshape(1, C, H, W)}
        self.attrs = attrs

        im2col(attrs, x, out)
        self.outputs = {
            'Out':out.reshape(1, outputHeight, outputWidth, x.shape[0], \
                     attrs['blockHeight'], attrs['blockWidth'])
            }

    """
    def test_check_output(self):
        self.check_output()
    """

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.01)


if __name__ == '__main__':
    unittest.main()
