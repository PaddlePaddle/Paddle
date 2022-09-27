#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci


def get_output_shape(attrs, in_shape, img_real_size):
    batchsize = in_shape[0]
    img_height = in_shape[2]
    img_width = in_shape[3]
    paddings = np.array(attrs['paddings']).astype("int32")
    kernels = np.array(attrs['kernels']).astype("int32")
    strides = np.array(attrs['strides']).astype("int32")
    output_height = np.zeros((1, batchsize)).astype("int32")
    output_width = np.zeros((1, batchsize)).astype("int32")
    if len(img_real_size):
        out_stride = np.array(attrs['out_stride']).astype("int32")
        imgreal_h = 0
        imgreal_w = 0
        for index in range(batchsize):
            if img_real_size[index, 0] % out_stride[0] == 0:
                imgreal_h = img_real_size[index, 0] / out_stride[0]
            else:
                imgreal_h = img_real_size[index, 0] / out_stride[0] + 1
            if img_real_size[index, 0] % out_stride[1] == 0:
                imgreal_w = img_real_size[index, 1] / out_stride[1]
            else:
                imgreal_w = img_real_size[index, 0] / out_stride[1] + 1
            output_height[0,index] = \
              1 +  \
              (imgreal_h + paddings[0] + paddings[2] - kernels[0] + strides[0] - 1) / \
                  strides[0]

            output_width[0,index] = \
              1 + \
              (imgreal_w + paddings[1] + paddings[3] - kernels[1] + strides[1] - 1) / \
                  strides[1]
    else:
        for index in range(batchsize):
            output_height[0,index] = \
              1 +  \
              (img_height + paddings[0] + paddings[2] - kernels[0] + strides[0] - 1) / \
                  strides[0]

            output_width[0,index] = \
              1 + \
              (img_width + paddings[1] + paddings[3] - kernels[1] + strides[1] - 1) / \
                  strides[1]

    return output_height, output_width


def im2col(attrs, im, col):
    """
    im: {CHW}
    col:
        {outputHeight, outputWidth, inputChannels, filterHeight, filterWidth}
    """
    input_channels, input_height, input_width = im.shape
    output_height, output_width, _, filter_height, filter_width = col.shape

    stride_height, stride_width = attrs['strides']
    padding_height, padding_width = attrs['paddings'][0:2]

    for col_row_idx in range(0, output_height):
        for col_col_idx in range(0, output_width):
            for channel in range(0, input_channels):
                for filter_row_idx in range(0, filter_height):
                    for filter_col_idx in range(0, filter_width):
                        im_row_offset = col_row_idx * stride_height \
                            + filter_row_idx - padding_height

                        im_col_offset = col_col_idx * stride_width \
                            + filter_col_idx - padding_width

                        if (im_row_offset < 0 or im_row_offset >= input_height
                                or im_col_offset < 0
                                or im_col_offset >= input_width):
                            col[col_row_idx][col_col_idx][channel][\
                                filter_row_idx][filter_col_idx] = 0.0
                        else:
                            im_offset = (channel * input_height + im_row_offset \
                                         ) * input_width + im_col_offset

                            col[col_row_idx][col_col_idx][channel][\
                                filter_row_idx][filter_col_idx] = im[channel][ \
                                    im_row_offset][im_col_offset]


def Im2Sequence(inputs, img_real_size, attrs):
    output_height, output_width = get_output_shape(attrs, inputs.shape,
                                                   img_real_size)
    img_channels = inputs.shape[1]
    batch_size = inputs.shape[0]
    out = []
    for index in range(batch_size):
        tmp = np.zeros([
            output_height[0, index], output_width[0, index], img_channels,
            attrs['kernels'][0], attrs['kernels'][1]
        ]).astype("float32")
        out.append(tmp)
    for index in range(len(inputs)):
        im2col(attrs, inputs[index], out[index])
        out[index] = out[index].reshape([
            output_height[0, index] * output_width[0, index],
            img_channels * attrs['kernels'][0] * attrs['kernels'][1]
        ])
    out = np.concatenate(out, axis=0)
    return out


class TestBlockExpandOp(OpTest):

    def config(self):
        self.batch_size = 1
        self.img_channels = 3
        self.img_height = 4
        self.img_width = 10
        self.attrs = {
            'kernels': [2, 2],
            'strides': [1, 1],
            'paddings': [1, 1, 1, 1],
        }

    def setUp(self):
        self.config()
        self.op_type = "im2sequence"
        x = np.random.uniform(0.1, 1, [
            self.batch_size, self.img_channels, self.img_height, self.img_width
        ]).astype("float32")
        real_size = np.array([]).astype("float32")
        out = Im2Sequence(x, real_size, self.attrs)
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
            'kernels': [2, 1],
            'strides': [2, 1],
            'paddings': [2, 1, 2, 1],
        }


class TestBlockExpandOpCase3(TestBlockExpandOp):

    def config(self):
        self.batch_size = 6
        self.img_channels = 1
        self.img_height = 4
        self.img_width = 5
        self.attrs = {
            'kernels': [2, 1],
            'strides': [2, 1],
            'paddings': [2, 0, 2, 0],
        }


class TestBlockExpandOpCase4(TestBlockExpandOp):

    def config(self):
        self.batch_size = 6
        self.img_channels = 2
        self.img_height = 3
        self.img_width = 3
        self.attrs = {
            'kernels': [2, 2],
            'strides': [1, 1],
            'paddings': [0, 0, 0, 0],
        }


@skip_check_grad_ci(
    reason=
    "Since 'real_size' is used just in forward computation, we don't test the gradient here."
)
class TestBlockExpandOpCase5(OpTest):

    def config(self):
        self.batch_size = 1
        self.img_channels = 3
        self.img_height = 4
        self.img_width = 5
        self.attrs = {
            'kernels': [2, 1],
            'strides': [2, 1],
            'paddings': [2, 1, 2, 1],
            'out_stride': [2, 2],
        }
        self.real_size = np.array([[8, 10], [5, 8]]).astype("float32")

    def setUp(self):
        self.config()
        self.op_type = "im2sequence"
        x = np.random.uniform(0.1, 1, [
            self.batch_size, self.img_channels, self.img_height, self.img_width
        ]).astype("float32")
        out = np.array(Im2Sequence(x, self.real_size, self.attrs))
        self.inputs = {'X': x, 'Y': self.real_size}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()


class TestBlockExpandOpCase6(TestBlockExpandOpCase5):

    def config(self):
        self.batch_size = 3
        self.img_channels = 1
        self.img_height = 4
        self.img_width = 5
        self.attrs = {
            'kernels': [2, 1],
            'strides': [1, 1],
            'paddings': [0, 0, 0, 0],
            'out_stride': [1, 1],
        }
        self.real_size = np.array([[8, 10], [5, 8], [5, 8]]).astype("float32")


class TestBlockExpandOpCase7(TestBlockExpandOpCase6):

    def config(self):
        self.batch_size = 2
        self.img_channels = 2
        self.img_height = 3
        self.img_width = 3
        self.attrs = {
            'kernels': [2, 2],
            'strides': [1, 1],
            'paddings': [1, 0, 1, 0],
            'out_stride': [2, 2],
        }
        self.real_size = np.array([[6, 6], [4, 4]]).astype("float32")


if __name__ == '__main__':
    unittest.main()
