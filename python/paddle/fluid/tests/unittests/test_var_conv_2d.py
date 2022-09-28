#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci


class TestVarConv2DOp(OpTest):

    def setUp(self):
        self.init_op_type()
        self.set_data()
        self.compute()

    def init_op_type(self):
        self.op_type = "var_conv_2d"

    def set_data(self):
        input_channel = 8
        output_channel = 2
        filter_size = [2, 3]
        stride = [1, 1]
        row = [2, 4]
        col = [3, 2]
        self.init_data(input_channel, output_channel, filter_size, stride, row,
                       col)

    def init_data(self, input_channel, output_channel, filter_size, stride, row,
                  col):

        feature = [row[i] * col[i] for i in range(len(row))]
        numel = sum(feature) * input_channel
        x_data = np.random.random((numel, 1)).astype('float32')
        x_lod = [[x * input_channel for x in feature]]
        row_data = np.random.random((sum(row), 10)).astype('float32')
        col_data = np.random.random((sum(col), 10)).astype('float32')
        w_shape = (output_channel,
                   input_channel * filter_size[0] * filter_size[1])
        w_data = np.random.random(w_shape).astype('float32')
        self.inputs = {
            'X': (x_data, x_lod),
            'ROW': (row_data, [row]),
            'COLUMN': (col_data, [col]),
            'W': w_data
        }
        self.attrs = {
            'InputChannel': input_channel,
            'OutputChannel': output_channel,
            'StrideH': stride[0],
            'StrideW': stride[1],
            'KernelH': filter_size[0],
            'KernelW': filter_size[1],
        }

    def compute(self):
        in_ch = self.attrs['InputChannel']
        out_ch = self.attrs['OutputChannel']
        kernel_h = self.attrs['KernelH']
        kernel_w = self.attrs['KernelW']
        stride_h = self.attrs['StrideH']
        stride_w = self.attrs['StrideW']
        row_data, row_lod = self.inputs['ROW']
        col_data, col_lod = self.inputs['COLUMN']
        x_data, x_lod = self.inputs['X']
        w_data = self.inputs['W']
        out_data = np.zeros((0, 1)).astype('float32')

        col_res_data, col_res_lod = self.Im2Col()
        out_lod = [[]]
        col_data_offset = 0
        batch_size = len(x_lod[0])
        for idx in range(batch_size):
            width = col_lod[0][idx]
            height = row_lod[0][idx]
            top_im_x = 0
            if width != 0:
                top_im_x = (width - 1) // stride_w + 1
            top_im_y = 0
            if height != 0:
                top_im_y = (height - 1) // stride_h + 1
            top_im_size = top_im_x * top_im_y
            out_lod[0].append(out_ch * top_im_size)
            if top_im_size == 0:
                out_tmp = np.zeros((out_ch * top_im_size, 1)).astype('float32')
            else:
                col_batch_data = col_res_data[col_data_offset:col_data_offset +
                                              col_res_lod[0][idx]]
                gemm_shape = (in_ch * kernel_h * kernel_w, top_im_size)
                col_batch_data = col_batch_data.reshape(gemm_shape)
                out_tmp = np.dot(w_data, col_batch_data).reshape(-1, 1)
            out_data = np.vstack((out_data, out_tmp))

            col_data_offset += col_res_lod[0][idx]

        self.outputs = {
            'Out': (out_data.astype('float32'), out_lod),
            'Col': (col_res_data, col_res_lod)
        }

    def Im2Col(self):
        in_ch = self.attrs['InputChannel']
        kernel_h = self.attrs['KernelH']
        kernel_w = self.attrs['KernelW']
        stride_h = self.attrs['StrideH']
        stride_w = self.attrs['StrideW']
        row_data, row_lod = self.inputs['ROW']
        col_data, col_lod = self.inputs['COLUMN']
        x_data, x_lod = self.inputs['X']
        col_res_lod = [[]]
        top_size = 0
        batch_size = len(x_lod[0])
        for idx in range(batch_size):
            width = col_lod[0][idx]
            height = row_lod[0][idx]
            top_im_x = 0
            if width != 0:
                top_im_x = (width - 1) // stride_w + 1
            top_im_y = 0
            if height != 0:
                top_im_y = (height - 1) // stride_h + 1
            top_x = top_im_x * top_im_y
            top_y = in_ch * kernel_h * kernel_w
            col_res_lod[0].append(top_x * top_y)
            top_size += top_x * top_y

        col_res = np.zeros((top_size, 1)).astype('float32')

        kernel_win_size = kernel_h * kernel_w
        half_kernel_h = kernel_h // 2
        half_kernel_w = kernel_w // 2
        t_offset, b_offset = 0, 0
        for idx in range(batch_size):
            width = col_lod[0][idx]
            height = row_lod[0][idx]
            if width == 0 or height == 0:
                continue
            top_im_x = (width - 1) // stride_w + 1
            top_im_y = (height - 1) // stride_h + 1
            top_x = top_im_x * top_im_y
            for z in range(in_ch):
                row_offset = kernel_win_size * z
                im_offset = z * width * height
                for y in range(0, height, stride_h):
                    for x in range(0, width, stride_w):
                        col_offset = x // stride_w + y // stride_h * top_im_x
                        for ky in range(kernel_h):
                            for kx in range(kernel_w):
                                im_y = y + ky - half_kernel_h
                                im_x = x + kx - half_kernel_w
                                if im_x >= 0 and im_x < width and im_y >= 0 and im_y < height:
                                    col_res[t_offset +
                                        (row_offset + ky * kernel_w + kx) * top_x +
                                        col_offset] = \
                                    x_data[b_offset + im_offset + im_y * width + im_x]

            t_offset += col_res_lod[0][idx]
            b_offset += x_lod[0][idx]

        return col_res, col_res_lod

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.check_grad(['X'],
                        'Out',
                        max_relative_error=0.005,
                        check_dygraph=False)


class TestVarConv2DOpCase1(TestVarConv2DOp):

    def set_data(self):
        # set in_ch 1
        input_channel = 1
        output_channel = 2
        filter_size = [2, 3]
        stride = [1, 1]
        row = [1, 10]
        col = [40, 6]
        self.init_data(input_channel, output_channel, filter_size, stride, row,
                       col)


class TestVarConv2DOpCase2(TestVarConv2DOp):

    def set_data(self):
        # set out_ch 1
        input_channel = 2
        output_channel = 1
        filter_size = [3, 3]
        stride = [2, 2]
        row = [6, 7]
        col = [8, 2]
        self.init_data(input_channel, output_channel, filter_size, stride, row,
                       col)


class TestVarConv2DOpCase3(TestVarConv2DOp):

    def set_data(self):
        # set batch 1
        input_channel = 2
        output_channel = 1
        filter_size = [3, 3]
        stride = [2, 2]
        row = [14]
        col = [4]
        self.init_data(input_channel, output_channel, filter_size, stride, row,
                       col)


class TestVarConv2DOpCase4(TestVarConv2DOp):

    def set_data(self):
        # set filter size very large
        input_channel = 3
        output_channel = 4
        filter_size = [6, 6]
        stride = [2, 2]
        row = [4, 7]
        col = [5, 2]
        self.init_data(input_channel, output_channel, filter_size, stride, row,
                       col)


class TestVarConv2DOpCase5(TestVarConv2DOp):

    def set_data(self):
        # set input very small
        input_channel = 50
        output_channel = 3
        filter_size = [3, 3]
        stride = [1, 1]
        row = [1, 1]
        col = [1, 1]
        self.init_data(input_channel, output_channel, filter_size, stride, row,
                       col)


@skip_check_grad_ci(
    reason=
    "[skip shape check] Use shape of input_channel, row and col all is 1 to test special LoDTensor."
)
class TestVarConv2DOpCase6(TestVarConv2DOp):

    def set_data(self):
        input_channel = 1
        output_channel = 3
        filter_size = [3, 3]
        stride = [1, 1]
        row = [1, 1]
        col = [1, 1]
        self.init_data(input_channel, output_channel, filter_size, stride, row,
                       col)


class TestVarConv2DOpCase7(TestVarConv2DOp):

    def set_data(self):
        input_channel = 2
        output_channel = 3
        filter_size = [3, 3]
        stride = [1, 1]
        row = [5, 4]
        col = [6, 7]
        self.init_data(input_channel, output_channel, filter_size, stride, row,
                       col)


class TestVarConv2DApi(unittest.TestCase):

    def test_api(self):
        import paddle.fluid as fluid

        x = fluid.layers.data(name='x', shape=[1], lod_level=1)
        row = fluid.layers.data(name='row', shape=[6], lod_level=1)
        col = fluid.layers.data(name='col', shape=[6], lod_level=1)
        out = fluid.contrib.var_conv_2d(input=x,
                                        row=row,
                                        col=col,
                                        input_channel=3,
                                        output_channel=5,
                                        filter_size=[3, 3],
                                        stride=1)

        place = fluid.CPUPlace()
        x_tensor = fluid.create_lod_tensor(
            np.random.rand(116, 1).astype('float32'), [[60, 56]], place)
        row_tensor = fluid.create_lod_tensor(
            np.random.rand(9, 6).astype('float32'), [[5, 4]], place)
        col_tensor = fluid.create_lod_tensor(
            np.random.rand(13, 6).astype('float32'), [[6, 7]], place)

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        ret = exe.run(feed={
            'x': x_tensor,
            'row': row_tensor,
            'col': col_tensor
        },
                      fetch_list=[out],
                      return_numpy=False)


if __name__ == '__main__':
    unittest.main()
