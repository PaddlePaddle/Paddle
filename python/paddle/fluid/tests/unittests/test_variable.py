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

from __future__ import print_function

import unittest
from functools import reduce

import paddle
from paddle.fluid.framework import default_main_program, Program, convert_np_dtype_to_dtype_, in_dygraph_mode
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import numpy as np

paddle.enable_static()


class TestVariable(unittest.TestCase):
    def test_np_dtype_convert(self):
        DT = core.VarDesc.VarType
        convert = convert_np_dtype_to_dtype_
        self.assertEqual(DT.FP32, convert(np.float32))
        self.assertEqual(DT.FP16, convert("float16"))
        self.assertEqual(DT.FP64, convert("float64"))
        self.assertEqual(DT.INT32, convert("int32"))
        self.assertEqual(DT.INT16, convert("int16"))
        self.assertEqual(DT.INT64, convert("int64"))
        self.assertEqual(DT.BOOL, convert("bool"))
        self.assertEqual(DT.INT8, convert("int8"))
        self.assertEqual(DT.UINT8, convert("uint8"))

    def test_var(self):
        b = default_main_program().current_block()
        w = b.create_var(
            dtype="float64", shape=[784, 100], lod_level=0, name="fc.w")
        self.assertNotEqual(str(w), "")
        self.assertEqual(core.VarDesc.VarType.FP64, w.dtype)
        self.assertEqual((784, 100), w.shape)
        self.assertEqual("fc.w", w.name)
        self.assertEqual("fc.w@GRAD", w.grad_name)
        self.assertEqual(0, w.lod_level)

        w = b.create_var(name='fc.w')
        self.assertEqual(core.VarDesc.VarType.FP64, w.dtype)
        self.assertEqual((784, 100), w.shape)
        self.assertEqual("fc.w", w.name)
        self.assertEqual("fc.w@GRAD", w.grad_name)
        self.assertEqual(0, w.lod_level)

        self.assertRaises(ValueError,
                          lambda: b.create_var(name="fc.w", shape=(24, 100)))

        w = b.create_var(
            dtype=paddle.fluid.core.VarDesc.VarType.STRINGS,
            shape=[1],
            name="str_var")
        self.assertEqual(None, w.lod_level)

    def test_element_size(self):
        with fluid.program_guard(Program(), Program()):
            x = paddle.static.data(name='x1', shape=[2], dtype='bool')
            self.assertEqual(x.element_size(), 1)

            x = paddle.static.data(name='x2', shape=[2], dtype='float16')
            self.assertEqual(x.element_size(), 2)

            x = paddle.static.data(name='x3', shape=[2], dtype='float32')
            self.assertEqual(x.element_size(), 4)

            x = paddle.static.data(name='x4', shape=[2], dtype='float64')
            self.assertEqual(x.element_size(), 8)

            x = paddle.static.data(name='x5', shape=[2], dtype='int8')
            self.assertEqual(x.element_size(), 1)

            x = paddle.static.data(name='x6', shape=[2], dtype='int16')
            self.assertEqual(x.element_size(), 2)

            x = paddle.static.data(name='x7', shape=[2], dtype='int32')
            self.assertEqual(x.element_size(), 4)

            x = paddle.static.data(name='x8', shape=[2], dtype='int64')
            self.assertEqual(x.element_size(), 8)

            x = paddle.static.data(name='x9', shape=[2], dtype='uint8')
            self.assertEqual(x.element_size(), 1)

    def test_step_scopes(self):
        prog = Program()
        b = prog.current_block()
        var = b.create_var(
            name='step_scopes', type=core.VarDesc.VarType.STEP_SCOPES)
        self.assertEqual(core.VarDesc.VarType.STEP_SCOPES, var.type)

    def _test_slice(self, place):
        b = default_main_program().current_block()
        w = b.create_var(dtype="float64", shape=[784, 100, 100], lod_level=0)

        for i in range(3):
            nw = w[i]
            self.assertEqual((100, 100), nw.shape)

        nw = w[:]
        self.assertEqual((784, 100, 100), nw.shape)

        nw = w[:, :]
        self.assertEqual((784, 100, 100), nw.shape)

        nw = w[:, :, -1]
        self.assertEqual((784, 100), nw.shape)

        nw = w[1, 1, 1]

        self.assertEqual(len(nw.shape), 1)
        self.assertEqual(nw.shape[0], 1)

        nw = w[:, :, :-1]
        self.assertEqual((784, 100, 99), nw.shape)

        self.assertEqual(0, nw.lod_level)

        main = fluid.Program()
        with fluid.program_guard(main):
            exe = fluid.Executor(place)
            tensor_array = np.array(
                [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                 [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                 [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]).astype('float32')
            var = fluid.layers.assign(tensor_array)
            var1 = var[0, 1, 1]
            var2 = var[1:]
            var3 = var[0:1]
            var4 = var[::-1]
            var5 = var[1, 1:, 1:]
            var_reshape = fluid.layers.reshape(var, [3, -1, 3])
            var6 = var_reshape[:, :, -1]
            var7 = var[:, :, :-1]
            var8 = var[:1, :1, :1]
            var9 = var[:-1, :-1, :-1]
            var10 = var[::-1, :1, :-1]
            var11 = var[:-1, ::-1, -1:]
            var12 = var[1:2, 2:, ::-1]
            var13 = var[2:10, 2:, -2:-1]
            var14 = var[1:-1, 0:2, ::-1]
            var15 = var[::-1, ::-1, ::-1]

            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            y = fluid.layers.fc(input=x, size=1, act=None)
            y_1 = y[:, 0]
            feeder = fluid.DataFeeder(place=place, feed_list=[x])
            data = []
            data.append((np.random.randint(10, size=[13]).astype('float32')))
            exe.run(fluid.default_startup_program())

            local_out = exe.run(main,
                                feed=feeder.feed([data]),
                                fetch_list=[
                                    var, var1, var2, var3, var4, var5, var6,
                                    var7, var8, var9, var10, var11, var12,
                                    var13, var14, var15
                                ])

            self.assertTrue(
                np.array_equal(local_out[1], tensor_array[0, 1, 1:2]))
            self.assertTrue(np.array_equal(local_out[2], tensor_array[1:]))
            self.assertTrue(np.array_equal(local_out[3], tensor_array[0:1]))
            self.assertTrue(np.array_equal(local_out[4], tensor_array[::-1]))
            self.assertTrue(
                np.array_equal(local_out[5], tensor_array[1, 1:, 1:]))
            self.assertTrue(
                np.array_equal(local_out[6],
                               tensor_array.reshape((3, -1, 3))[:, :, -1]))
            self.assertTrue(
                np.array_equal(local_out[7], tensor_array[:, :, :-1]))
            self.assertTrue(
                np.array_equal(local_out[8], tensor_array[:1, :1, :1]))
            self.assertTrue(
                np.array_equal(local_out[9], tensor_array[:-1, :-1, :-1]))
            self.assertTrue(
                np.array_equal(local_out[10], tensor_array[::-1, :1, :-1]))
            self.assertTrue(
                np.array_equal(local_out[11], tensor_array[:-1, ::-1, -1:]))
            self.assertTrue(
                np.array_equal(local_out[12], tensor_array[1:2, 2:, ::-1]))
            self.assertTrue(
                np.array_equal(local_out[13], tensor_array[2:10, 2:, -2:-1]))
            self.assertTrue(
                np.array_equal(local_out[14], tensor_array[1:-1, 0:2, ::-1]))
            self.assertTrue(
                np.array_equal(local_out[15], tensor_array[::-1, ::-1, ::-1]))

    def _test_slice_index_tensor(self, place):
        data = np.random.rand(2, 3).astype("float32")
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(data)
            idx0 = [1, 0]
            idx1 = [0, 1]
            idx2 = [0, 0]
            idx3 = [1, 1]

            out0 = x[paddle.assign(np.array(idx0))]
            out1 = x[paddle.assign(np.array(idx1))]
            out2 = x[paddle.assign(np.array(idx2))]
            out3 = x[paddle.assign(np.array(idx3))]

        exe = paddle.static.Executor(place)
        result = exe.run(prog, fetch_list=[out0, out1, out2, out3])

        expected = [data[idx0], data[idx1], data[idx2], data[idx3]]

        self.assertTrue((result[0] == expected[0]).all())
        self.assertTrue((result[1] == expected[1]).all())
        self.assertTrue((result[2] == expected[2]).all())
        self.assertTrue((result[3] == expected[3]).all())

        with self.assertRaises(IndexError):
            one = paddle.ones(shape=[1])
            res = x[one, [0, 0]]

    def _test_slice_index_list(self, place):
        data = np.random.rand(2, 3).astype("float32")
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(data)
            idx0 = [1, 0]
            idx1 = [0, 1]
            idx2 = [0, 0]
            idx3 = [1, 1]

            out0 = x[idx0]
            out1 = x[idx1]
            out2 = x[idx2]
            out3 = x[idx3]

        exe = paddle.static.Executor(place)
        result = exe.run(prog, fetch_list=[out0, out1, out2, out3])

        expected = [data[idx0], data[idx1], data[idx2], data[idx3]]

        self.assertTrue((result[0] == expected[0]).all())
        self.assertTrue((result[1] == expected[1]).all())
        self.assertTrue((result[2] == expected[2]).all())
        self.assertTrue((result[3] == expected[3]).all())

    def _test_slice_index_ellipsis(self, place):
        data = np.random.rand(2, 3, 4).astype("float32")
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(data)
            y = paddle.assign([1, 2, 3, 4])
            out1 = x[0:, ..., 1:]
            out2 = x[0:, ...]
            out3 = x[..., 1:]
            out4 = x[...]
            out5 = x[[1, 0], [0, 0]]
            out6 = x[([1, 0], [0, 0])]
            out7 = y[..., 0]

        exe = paddle.static.Executor(place)
        result = exe.run(prog,
                         fetch_list=[out1, out2, out3, out4, out5, out6, out7])

        expected = [
            data[0:, ..., 1:], data[0:, ...], data[..., 1:], data[...],
            data[[1, 0], [0, 0]], data[([1, 0], [0, 0])], np.array([1])
        ]

        self.assertTrue((result[0] == expected[0]).all())
        self.assertTrue((result[1] == expected[1]).all())
        self.assertTrue((result[2] == expected[2]).all())
        self.assertTrue((result[3] == expected[3]).all())
        self.assertTrue((result[4] == expected[4]).all())
        self.assertTrue((result[5] == expected[5]).all())
        self.assertTrue((result[6] == expected[6]).all())

        with self.assertRaises(IndexError):
            res = x[[1.2, 0]]

    def _test_slice_index_list_bool(self, place):
        data = np.random.rand(2, 3, 4).astype("float32")
        np_idx = np.array([[True, False, False], [True, False, True]])
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(data)
            idx0 = [True, False]
            idx1 = [False, True]
            idx2 = [True, True]
            idx3 = [False, False, 1]
            idx4 = [True, False, 0]
            idx5 = paddle.assign(np_idx)

            out0 = x[idx0]
            out1 = x[idx1]
            out2 = x[idx2]
            out3 = x[idx3]
            out4 = x[idx4]
            out5 = x[idx5]
            out6 = x[x < 0.36]
            out7 = x[x > 0.6]

        exe = paddle.static.Executor(place)
        result = exe.run(
            prog, fetch_list=[out0, out1, out2, out3, out4, out5, out6, out7])

        expected = [
            data[idx0], data[idx1], data[idx2], data[idx3], data[idx4],
            data[np_idx], data[data < 0.36], data[data > 0.6]
        ]

        self.assertTrue((result[0] == expected[0]).all())
        self.assertTrue((result[1] == expected[1]).all())
        self.assertTrue((result[2] == expected[2]).all())
        self.assertTrue((result[3] == expected[3]).all())
        self.assertTrue((result[4] == expected[4]).all())
        self.assertTrue((result[5] == expected[5]).all())
        self.assertTrue((result[6] == expected[6]).all())
        self.assertTrue((result[7] == expected[7]).all())

        with self.assertRaises(IndexError):
            res = x[[True, False, False]]
        with self.assertRaises(ValueError):
            res = x[[False, False]]

    def test_slice(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            self._test_slice(place)
            self._test_slice_index_tensor(place)
            self._test_slice_index_list(place)
            self._test_slice_index_ellipsis(place)
            self._test_slice_index_list_bool(place)

    def _tostring(self):
        b = default_main_program().current_block()
        w = b.create_var(dtype="float64", lod_level=0)
        self.assertTrue(isinstance(str(w), str))

        if core.is_compiled_with_cuda():
            wc = b.create_var(dtype="int", lod_level=0)
            self.assertTrue(isinstance(str(wc), str))

    def test_tostring(self):
        with fluid.dygraph.guard():
            self._tostring()

        with fluid.program_guard(default_main_program()):
            self._tostring()

    def test_fake_interface_only_api(self):
        b = default_main_program().current_block()
        var = b.create_var(dtype="float64", lod_level=0)
        with fluid.dygraph.guard():
            self.assertRaises(AssertionError, var.numpy)
            self.assertRaises(AssertionError, var.backward)
            self.assertRaises(AssertionError, var.gradient)
            self.assertRaises(AssertionError, var.clear_gradient)

    def test_variable_in_dygraph_mode(self):
        b = default_main_program().current_block()
        var = b.create_var(dtype="float64", shape=[1, 1])
        with fluid.dygraph.guard():
            self.assertTrue(var.to_string(True).startswith('name:'))

            self.assertFalse(var.persistable)
            var.persistable = True
            self.assertTrue(var.persistable)

            self.assertFalse(var.stop_gradient)
            var.stop_gradient = True
            self.assertTrue(var.stop_gradient)

            self.assertTrue(var.name.startswith('_generated_var_'))
            self.assertEqual(var.shape, (1, 1))
            self.assertEqual(var.dtype, fluid.core.VarDesc.VarType.FP64)
            self.assertEqual(var.type, fluid.core.VarDesc.VarType.LOD_TENSOR)

    def test_create_selected_rows(self):
        b = default_main_program().current_block()

        var = b.create_var(
            name="var",
            shape=[1, 1],
            dtype="float32",
            type=fluid.core.VarDesc.VarType.SELECTED_ROWS,
            persistable=True)

        def _test():
            var.lod_level()

        self.assertRaises(Exception, _test)

    def test_size(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(np.random.rand(2, 3, 4).astype("float32"))
            exe = paddle.static.Executor(fluid.CPUPlace())
            exe.run(paddle.static.default_startup_program())

            output = exe.run(prog, fetch_list=[x.size()])
            self.assertEqual(output[0], [24])

    def test_detach(self):
        b = default_main_program().current_block()
        x = b.create_var(shape=[2, 3, 5], dtype="float64", lod_level=0)
        detach_x = x.detach()
        self.assertEqual(x.persistable, detach_x.persistable)
        self.assertEqual(x.shape, detach_x.shape)
        self.assertEqual(x.dtype, detach_x.dtype)
        self.assertEqual(x.type, detach_x.type)
        self.assertTrue(detach_x.stop_gradient)

        xx = b.create_var(name='xx', type=core.VarDesc.VarType.STEP_SCOPES)
        self.assertRaises(AssertionError, xx.detach)

        startup = paddle.static.Program()
        main = paddle.static.Program()
        scope = fluid.core.Scope()
        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data(
                    name='x', shape=[3, 2, 1], dtype='float32')
                x.persistable = True
                feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32)
                detach_x = x.detach()
                exe = paddle.static.Executor(paddle.CPUPlace())
                exe.run(startup)
                result = exe.run(main,
                                 feed={'x': feed_data},
                                 fetch_list=[x, detach_x])
                self.assertTrue((result[1] == feed_data).all())
                self.assertTrue((result[0] == result[1]).all())

                modified_value = np.zeros(shape=[3, 2, 1], dtype=np.float32)
                detach_x.set_value(modified_value, scope)
                result = exe.run(main, fetch_list=[x, detach_x])
                self.assertTrue((result[1] == modified_value).all())
                self.assertTrue((result[0] == result[1]).all())

                modified_value = np.random.uniform(
                    -1, 1, size=[3, 2, 1]).astype('float32')
                x.set_value(modified_value, scope)
                result = exe.run(main, fetch_list=[x, detach_x])
                self.assertTrue((result[1] == modified_value).all())
                self.assertTrue((result[0] == result[1]).all())


class TestVariableSlice(unittest.TestCase):
    def _test_item_none(self, place):
        data = np.random.rand(2, 3, 4).astype("float32")
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(data)
            out0 = x[0:, None, 1:]
            out1 = x[0:, None]
            out2 = x[None, 1:]
            out3 = x[None]
            out4 = x[..., None, :, None]

        outs = [out0, out1, out2, out3, out4]
        exe = paddle.static.Executor(place)
        result = exe.run(prog, fetch_list=outs)

        expected = [
            data[0:, None, 1:], data[0:, None], data[None, 1:], data[None],
            data[..., None, :, None]
        ]
        for i in range(len(outs)):
            self.assertEqual(outs[i].shape, expected[i].shape)
            self.assertTrue((result[i] == expected[i]).all())

    def _test_item_none_and_decrease(self, place):
        data = np.random.rand(2, 3, 4).astype("float32")
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(data)
            out0 = x[0, 1:, None]
            out1 = x[0, None]
            out2 = x[None, 1]
            out3 = x[None]
            out4 = x[0, 0, 0, None]
            out5 = x[None, 0, 0, 0, None]

        outs = [out0, out1, out2, out3, out4, out5]
        exe = paddle.static.Executor(place)
        result = exe.run(prog, fetch_list=outs)
        expected = [
            data[0, 1:, None], data[0, None], data[None, 1], data[None],
            data[0, 0, 0, None], data[None, 0, 0, 0, None]
        ]

        for i in range(len(outs)):
            self.assertEqual(outs[i].shape, expected[i].shape)
            self.assertTrue((result[i] == expected[i]).all())

    def test_slice(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            self._test_item_none(place)
            self._test_item_none_and_decrease(place)


class TestListIndex(unittest.TestCase):
    def numel(self, shape):
        return reduce(lambda x, y: x * y, shape)

    def test_static_graph_list_index(self):
        paddle.enable_static()

        inps_shape = [3, 4, 5, 2]
        array = np.arange(
            self.numel(inps_shape), dtype='float32').reshape(inps_shape)

        index_shape = [3, 3, 2, 1]
        index = np.arange(self.numel(index_shape)).reshape(index_shape)

        for _ in range(3):
            program = paddle.static.Program()

            index_mod = (index % (array.shape[0])).tolist()

            with paddle.static.program_guard(program):
                x = paddle.static.data(
                    name='x', shape=array.shape, dtype='float32')

                y = x[index_mod]

                place = paddle.fluid.CPUPlace(
                ) if not paddle.fluid.core.is_compiled_with_cuda(
                ) else paddle.fluid.CUDAPlace(0)

                prog = paddle.static.default_main_program()
                exe = paddle.static.Executor(place)

                exe.run(paddle.static.default_startup_program())
                fetch_list = [y.name]

                getitem_np = array[index_mod]
                getitem_pp = exe.run(prog,
                                     feed={x.name: array},
                                     fetch_list=fetch_list)
                self.assertTrue(np.array_equal(getitem_np, getitem_pp[0]))

            array = array[0]
            index = index[0]

    def test_dygraph_list_index(self):
        paddle.disable_static()

        inps_shape = [3, 4, 5, 3]
        array = np.arange(self.numel(inps_shape)).reshape(inps_shape)

        index_shape = [2, 3, 4, 5, 6]
        index = np.arange(self.numel(index_shape)).reshape(index_shape)
        for _ in range(len(inps_shape) - 1):

            pt = paddle.to_tensor(array)
            index_mod = (index % (array.shape[-1])).tolist()
            try:
                getitem_np = array[index_mod]

            except:
                with self.assertRaises(ValueError):
                    getitem_pp = pt[index_mod]
                array = array[0]
                index = index[0]
                continue
            getitem_pp = pt[index_mod]
            self.assertTrue(np.array_equal(getitem_np, getitem_pp.numpy()))

            array = array[0]
            index = index[0]

    def test_static_graph_list_index_muti_dim(self):
        paddle.enable_static()
        inps_shape = [3, 4, 5]
        array = np.arange(
            self.numel(inps_shape), dtype='float32').reshape(inps_shape)

        index_shape = [2, 2]
        index1 = np.arange(self.numel(index_shape)).reshape(index_shape)
        index2 = np.arange(self.numel(index_shape)).reshape(index_shape) + 2

        value_shape = [3, 2, 2, 3]
        value_np = np.arange(
            self.numel(value_shape), dtype='float32').reshape(value_shape) + 100

        index_mod1 = (index1 % (min(array.shape))).tolist()
        index_mod2 = (index2 % (min(array.shape))).tolist()

        program = paddle.static.Program()
        with paddle.static.program_guard(program):

            x = paddle.static.data(name='x', shape=array.shape, dtype='float32')

            value = paddle.static.data(
                name='value', shape=value_np.shape, dtype='float32')
            index1 = paddle.static.data(
                name='index1', shape=index1.shape, dtype='int32')
            index2 = paddle.static.data(
                name='index2', shape=index2.shape, dtype='int32')

            y = x[index1, index2]

            place = paddle.fluid.CPUPlace(
            ) if not paddle.fluid.core.is_compiled_with_cuda(
            ) else paddle.fluid.CUDAPlace(0)

            prog = paddle.static.default_main_program()
            exe = paddle.static.Executor(place)

            exe.run(paddle.static.default_startup_program())
            fetch_list = [y.name]
            array2 = array.copy()

            y2 = array2[index_mod1, index_mod2]

            getitem_pp = exe.run(prog,
                                 feed={
                                     x.name: array,
                                     index1.name: index_mod1,
                                     index2.name: index_mod2
                                 },
                                 fetch_list=fetch_list)

            self.assertTrue(
                np.array_equal(y2, getitem_pp[0]),
                msg='\n numpy:{},\n paddle:{}'.format(y2, getitem_pp[0]))

    def test_dygraph_list_index_muti_dim(self):
        paddle.disable_static()
        inps_shape = [3, 4, 5]
        array = np.arange(
            self.numel(inps_shape), dtype='float32').reshape(inps_shape)

        index_shape = [2, 2]
        index1 = np.arange(self.numel(index_shape)).reshape(index_shape)
        index2 = np.arange(self.numel(index_shape)).reshape(index_shape) + 2

        value_shape = [3, 2, 2, 3]
        value_np = np.arange(
            self.numel(value_shape), dtype='float32').reshape(value_shape) + 100

        index_mod1 = (index1 % (min(array.shape))).tolist()
        index_mod2 = (index2 % (min(array.shape))).tolist()

        x = paddle.to_tensor(array)
        index_t1 = paddle.to_tensor(index_mod1)
        index_t2 = paddle.to_tensor(index_mod2)

        y_np = array[index_t1, index_t2]
        y = x[index_t1, index_t2]
        self.assertTrue(np.array_equal(y.numpy(), y_np))

    def run_setitem_list_index(self, array, index, value_np):
        x = paddle.static.data(name='x', shape=array.shape, dtype='float32')

        value = paddle.static.data(
            name='value', shape=value_np.shape, dtype='float32')

        x[index] = value
        y = x
        place = paddle.fluid.CPUPlace()

        prog = paddle.static.default_main_program()
        exe = paddle.static.Executor(place)

        exe.run(paddle.static.default_startup_program())
        fetch_list = [y.name]
        array2 = array.copy()

        try:
            array2[index] = value_np
        except:
            with self.assertRaises(ValueError):
                setitem_pp = exe.run(
                    prog,
                    feed={x.name: array,
                          value.name: value_np},
                    fetch_list=fetch_list)
            return
        setitem_pp = exe.run(prog,
                             feed={x.name: array,
                                   value.name: value_np},
                             fetch_list=fetch_list)

        self.assertTrue(
            np.array_equal(array2, setitem_pp[0]),
            msg='\n numpy:{},\n paddle:{}'.format(array2, setitem_pp[0]))

    def test_static_graph_setitem_list_index(self):
        paddle.enable_static()
        # case 1:
        inps_shape = [3, 4, 5, 2, 3]
        array = np.arange(
            self.numel(inps_shape), dtype='float32').reshape(inps_shape)

        index_shape = [3, 3, 1, 2]
        index = np.arange(self.numel(index_shape)).reshape(index_shape)

        value_shape = inps_shape[3:]
        value_np = np.arange(
            self.numel(value_shape), dtype='float32').reshape(value_shape) + 100

        for _ in range(3):
            program = paddle.static.Program()

            index_mod = (index % (min(array.shape))).tolist()

            with paddle.static.program_guard(program):
                self.run_setitem_list_index(array, index_mod, value_np)

            array = array[0]
            index = index[0]

        # case 2:
        inps_shape = [3, 4, 5, 4, 3]
        array = np.arange(
            self.numel(inps_shape), dtype='float32').reshape(inps_shape)

        index_shape = [4, 3, 2, 2]
        index = np.arange(self.numel(index_shape)).reshape(index_shape)

        value_shape = [3]
        value_np = np.arange(
            self.numel(value_shape), dtype='float32').reshape(value_shape) + 100

        for _ in range(4):
            program = paddle.static.Program()
            index_mod = (index % (min(array.shape))).tolist()

            with paddle.static.program_guard(program):
                self.run_setitem_list_index(array, index_mod, value_np)

            array = array[0]
            index = index[0]

        # case 3:
        inps_shape = [3, 4, 5, 3, 3]
        array = np.arange(
            self.numel(inps_shape), dtype='float32').reshape(inps_shape)

        index_shape = [4, 3, 2, 2]
        index = np.arange(self.numel(index_shape)).reshape(index_shape)

        value_shape = [3, 2, 2, 3]
        value_np = np.arange(
            self.numel(value_shape), dtype='float32').reshape(value_shape) + 100
        index_mod = (index % (min(array.shape))).tolist()
        self.run_setitem_list_index(array, index_mod, value_np)

    def test_static_graph_tensor_index_setitem_muti_dim(self):
        paddle.enable_static()
        inps_shape = [3, 4, 5, 4]
        array = np.arange(
            self.numel(inps_shape), dtype='float32').reshape(inps_shape)

        index_shape = [2, 3, 4]
        index1 = np.arange(
            self.numel(index_shape), dtype='int32').reshape(index_shape)
        index2 = np.arange(
            self.numel(index_shape), dtype='int32').reshape(index_shape) + 2

        value_shape = [4]
        value_np = np.arange(
            self.numel(value_shape), dtype='float32').reshape(value_shape) + 100
        for _ in range(3):

            index_mod1 = index1 % (min(array.shape))
            index_mod2 = index2 % (min(array.shape))

            array2 = array.copy()
            array2[index_mod1, index_mod2] = value_np
            array3 = array.copy()
            array3[index_mod1] = value_np

            program = paddle.static.Program()
            with paddle.static.program_guard(program):

                x1 = paddle.static.data(
                    name='x1', shape=array.shape, dtype='float32')
                x2 = paddle.static.data(
                    name='x2', shape=array.shape, dtype='float32')

                value = paddle.static.data(
                    name='value', shape=value_np.shape, dtype='float32')
                index_1 = paddle.static.data(
                    name='index_1', shape=index1.shape, dtype='int32')
                index_2 = paddle.static.data(
                    name='index_2', shape=index2.shape, dtype='int32')

                x1[index_1, index_2] = value
                x2[index_1] = value

                place = paddle.fluid.CPUPlace(
                ) if not paddle.fluid.core.is_compiled_with_cuda(
                ) else paddle.fluid.CUDAPlace(0)

                prog = paddle.static.default_main_program()
                exe = paddle.static.Executor(place)

                exe.run(paddle.static.default_startup_program())
                fetch_list = [x1.name, x2.name]

                setitem_pp = exe.run(prog,
                                     feed={
                                         x1.name: array,
                                         x2.name: array,
                                         value.name: value_np,
                                         index_1.name: index_mod1,
                                         index_2.name: index_mod2
                                     },
                                     fetch_list=fetch_list)
                self.assertTrue(
                    np.array_equal(array2, setitem_pp[0]),
                    msg='\n numpy:{},\n paddle:{}'.format(array2,
                                                          setitem_pp[0]))
                self.assertTrue(
                    np.array_equal(array3, setitem_pp[1]),
                    msg='\n numpy:{},\n paddle:{}'.format(array3,
                                                          setitem_pp[1]))
            array = array[0]
            index1 = index1[0]
            index2 = index2[0]

    def test_static_graph_array_index_muti_dim(self):
        paddle.enable_static()
        inps_shape = [3, 4, 5, 4]
        array = np.arange(
            self.numel(inps_shape), dtype='float32').reshape(inps_shape)

        index_shape = [2, 3, 4]
        index1 = np.arange(
            self.numel(index_shape), dtype='int32').reshape(index_shape)
        index2 = np.arange(
            self.numel(index_shape), dtype='int32').reshape(index_shape) + 2

        for _ in range(3):
            index_mod1 = index1 % (min(array.shape))
            index_mod2 = index2 % (min(array.shape))

            array2 = array.copy()
            array2[index_mod1, index_mod2] = 1
            y_np1 = array2[index_mod2, index_mod1]
            array3 = array.copy()
            array3[index_mod1] = 2.5
            y_np2 = array3[index_mod2]

            program = paddle.static.Program()
            with paddle.static.program_guard(program):

                x1 = paddle.static.data(
                    name='x1', shape=array.shape, dtype='float32')
                x2 = paddle.static.data(
                    name='x2', shape=array.shape, dtype='float32')

                x1[index_mod1, index_mod2] = 1
                x2[index_mod1] = 2.5
                y1 = x1[index_mod2, index_mod1]
                y2 = x2[index_mod2]
                place = paddle.fluid.CPUPlace(
                ) if not paddle.fluid.core.is_compiled_with_cuda(
                ) else paddle.fluid.CUDAPlace(0)

                prog = paddle.static.default_main_program()
                exe = paddle.static.Executor(place)
                exe.run(paddle.static.default_startup_program())
                fetch_list = [x1.name, x2.name, y1.name, y2.name]

                setitem_pp = exe.run(prog,
                                     feed={x1.name: array,
                                           x2.name: array},
                                     fetch_list=fetch_list)
                self.assertTrue(
                    np.array_equal(array2, setitem_pp[0]),
                    msg='\n numpy:{},\n paddle:{}'.format(array2,
                                                          setitem_pp[0]))
                self.assertTrue(
                    np.array_equal(array3, setitem_pp[1]),
                    msg='\n numpy:{},\n paddle:{}'.format(array3,
                                                          setitem_pp[1]))

                self.assertTrue(
                    np.array_equal(y_np1, setitem_pp[2]),
                    msg='\n numpy:{},\n paddle:{}'.format(y_np1, setitem_pp[2]))
                self.assertTrue(
                    np.array_equal(y_np2, setitem_pp[3]),
                    msg='\n numpy:{},\n paddle:{}'.format(y_np2, setitem_pp[3]))
            array = array[0]
            index1 = index1[0]
            index2 = index2[0]

    def test_dygraph_array_index_muti_dim(self):
        paddle.disable_static()
        inps_shape = [3, 4, 5, 4]
        array = np.arange(
            self.numel(inps_shape), dtype='float32').reshape(inps_shape)
        index_shape = [2, 3, 4]
        index1 = np.arange(
            self.numel(index_shape), dtype='int32').reshape(index_shape)
        index2 = np.arange(
            self.numel(index_shape), dtype='int32').reshape(index_shape) + 2

        for _ in range(3):

            index_mod1 = index1 % (min(array.shape))
            index_mod2 = index2 % (min(array.shape))
            index_mod_t1 = paddle.to_tensor(index_mod1)
            index_mod_t2 = paddle.to_tensor(index_mod2)
            # 2 dim getitem
            array1 = array.copy()
            y_np1 = array1[index_mod2, index_mod1]
            tensor1 = paddle.to_tensor(array)

            y_t1 = tensor1[index_mod_t2, index_mod_t1]

            self.assertTrue(
                np.array_equal(y_t1.numpy(), y_np1),
                msg='\n numpy:{},\n paddle:{}'.format(y_np1, y_t1.numpy()))
            # 1 dim getitem
            array2 = array.copy()
            y_np2 = array2[index_mod2]
            tensor2 = paddle.to_tensor(array)

            y_t2 = tensor2[index_mod_t2]
            self.assertTrue(
                np.array_equal(y_t2.numpy(), y_np2),
                msg='\n numpy:{},\n paddle:{}'.format(y_np2, y_t2.numpy()))

            # 2 dim setitem
            array1 = array.copy()
            array1[index_mod1, index_mod2] = 1
            tensor1[index_mod_t1, index_mod_t2] = 1
            self.assertTrue(
                np.array_equal(tensor1.numpy(), array1),
                msg='\n numpy:{},\n paddle:{}'.format(array1, tensor1.numpy()))
            # 1 dim setitem
            array2 = array.copy()

            array2[index_mod1] = 2.5

            tensor2[index_mod_t1] = 2.5

            self.assertTrue(
                np.array_equal(tensor2.numpy(), array2),
                msg='\n numpy:{},\n paddle:{}'.format(array2, tensor2.numpy()))

            array = array[0]
            index1 = index1[0]
            index2 = index2[0]


if __name__ == '__main__':
    unittest.main()
