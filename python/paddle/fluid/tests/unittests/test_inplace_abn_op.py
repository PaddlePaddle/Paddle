#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import os
import six
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid import compiler
import paddle.fluid.unique_name as unique_name


class TestInplaceANBOpTraining(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float64
        self.N = 32
        self.C = 16
        self.H = 64
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]

    # def append_activation(self, input_var):
    #     act = self.kwargs.get('act', None)
    #     if act is None:
    #         return input_var
    #     if isinstance(act, six.string_types):
    #         act = {'type': act}
    #     else:
    #         raise TypeError(str(act) + " should be unicode or str")
    #
    #     act_type = act.pop('type')
    #
    #     tmp = self.create_variable_for_type_inference(dtype=input_var.dtype)
    #     self.append_op(
    #         type=act_type,
    #         inputs={"X": [input_var]},
    #         outputs={"Out": [tmp]},
    #         attrs=act)
    #     return tmp
    #
    # def batch_norm(self, block, input,
    #                act=None,
    #                is_test=False,
    #                momentum=0.9,
    #                epsilon=1e-05,
    #                param_attr=None,
    #                bias_attr=None,
    #                data_layout='NCHW',
    #                in_place=False,
    #                name=None,
    #                moving_mean_name=None,
    #                moving_variance_name=None,
    #                do_model_average_for_mean_and_var=False,
    #                fuse_with_relu=False,
    #                use_global_stats=False):
    #     dtype = input.dtype
    #
    #     # use fp32 for bn parameter
    #     if dtype == core.VarDesc.VarType.FP16:
    #         dtype = core.VarDesc.VarType.FP32
    #
    #     input_shape = input.shape
    #     if data_layout == 'NCHW':
    #         channel_num = input_shape[1]
    #     else:
    #         if data_layout == 'NHWC':
    #             channel_num = input_shape[-1]
    #         else:
    #             raise ValueError("unsupported data layout:" + data_layout)
    #
    #     param_shape = [channel_num]
    #
    #
    #     # create parameter
    #     block.create_var(
    #         name=unique_name.generate('scale'),
    #         persistable=True,
    #         type=core.VarDesc.VarType.RAW,
    #         shape=param_shape,
    #         dtype=dtype,
    #         default_initializer=fluid.Constant(1.0))
    #     block.create_var(
    #         name=unique_name.generate('scale'),
    #         persistable=True,
    #         type=core.VarDesc.VarType.RAW,
    #         shape=param_shape,
    #         dtype=dtype,
    #         default_initializer=fluid.Constant(1.0))
    #
    #     bias = helper.create_parameter(
    #         attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)
    #
    #     mean = helper.create_parameter(
    #         attr=fluid.ParamAttr(
    #             name=moving_mean_name,
    #             initializer=fluid.Constant(0.0),
    #             trainable=False,
    #             do_model_average=do_model_average_for_mean_and_var),
    #         shape=param_shape,
    #         dtype=dtype)
    #     mean.stop_gradient = True
    #
    #     variance = helper.create_parameter(
    #         attr=fluid.ParamAttr(
    #             name=moving_variance_name,
    #             initializer=fluid.Constant(1.0),
    #             trainable=False,
    #             do_model_average=do_model_average_for_mean_and_var),
    #         shape=param_shape,
    #         dtype=dtype)
    #     variance.stop_gradient = True
    #
    #     # create output
    #     # mean and mean_out share the same memory
    #     mean_out = mean
    #     # variance and variance out share the same memory
    #     variance_out = variance
    #     saved_mean = helper.create_variable_for_type_inference(
    #         dtype=dtype, stop_gradient=True)
    #     saved_variance = helper.create_variable_for_type_inference(
    #         dtype=dtype, stop_gradient=True)
    #
    #     use_mkldnn = False
    #     op_type = "batch_norm"
    #
    #     batch_norm_out = input if in_place else helper.create_variable_for_type_inference(
    #         dtype)
    #
    #     attrs = {
    #         "momentum": momentum,
    #         "epsilon": epsilon,
    #         "is_test": is_test,
    #         "data_layout": data_layout,
    #         "use_mkldnn": use_mkldnn,
    #         "fuse_with_relu": fuse_with_relu,
    #         "use_global_stats": use_global_stats
    #     }
    #
    #     helper.append_op(
    #         type=op_type,
    #         inputs={
    #             "X": input,
    #             "Scale": scale,
    #             "Bias": bias,
    #             "Mean": mean,
    #             "Variance": variance
    #         },
    #         outputs={
    #             "Y": batch_norm_out,
    #             "MeanOut": mean_out,
    #             "VarianceOut": variance_out,
    #             "SavedMean": saved_mean,
    #             "SavedVariance": saved_variance
    #         },
    #         attrs=attrs)
    #
    #     return self.append_activation(batch_norm_out)

    def build_program(self,
                      place,
                      layout,
                      seed,
                      inplace_abn=False,
                      only_forward=False,
                      activation="identity"):
        main = fluid.Program()
        startup = fluid.Program()
        main.random_seed = seed
        startup.random_seed = seed
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                data = fluid.layers.data(
                    name='input',
                    shape=self.dshape,
                    dtype=self.dtype,
                    append_batch_size=False)
                conv = fluid.layers.conv2d(
                    input=data,
                    num_filters=32,
                    filter_size=1,
                    param_attr=fluid.ParamAttr(name='conv2d_weight'),
                    bias_attr=False,
                    use_cudnn=False)
                bn = fluid.layers.batch_norm(
                    conv,
                    in_place=inplace_abn,
                    act=activation,
                    param_attr=fluid.ParamAttr(name='bn_scale'),
                    bias_attr=fluid.ParamAttr(name='bn_bias'),
                    moving_mean_name='bn_moving_mean',
                    moving_variance_name='bn_moving_variance',
                    data_layout=layout,
                    is_test=only_forward)

                sigmoid = fluid.layers.sigmoid(bn)
                out = fluid.layers.reduce_sum(sigmoid)
                # if not inplace_abn:
                #     out = out / core.get_cuda_device_count()
                if not only_forward:
                    sgd_opt = fluid.optimizer.SGD(learning_rate=0.0)
                    sgd_opt.backward(out)
        return main, startup, [out, conv, bn]

    def compare(self, place, layout, only_forward, activation):
        seed = 10
        os.environ['FLAGS_cudnn_deterministic'] = "1"
        data = np.random.random(size=self.dshape).astype(self.dtype) * 4. - 2
        # Single-GPU, N = 32 per GPU
        main, startup, outs = self.build_program(place, layout, seed, False,
                                                 only_forward, activation)
        exe = fluid.Executor(place)
        exe.run(startup)
        fetch_names = [v.name for v in outs] + [
            'bn_moving_mean', 'bn_moving_variance', 'bn_scale', 'bn_bias'
        ]
        if not only_forward:
            others = [
                'batch_norm_0.tmp_0', 'batch_norm_0.tmp_1', 'bn_scale@GRAD',
                'bn_bias@GRAD', 'batch_norm_0.tmp_2@GRAD', 'conv2d_0.tmp_0@GRAD'
            ]
            fetch_names += others
        bn_fetches = exe.run(program=main,
                             feed={'input': data},
                             fetch_list=fetch_names)

        #####################################################################
        # Multi-GPUs, self.N / core.get_cuda_device_count() per GPU
        # assert core.get_cuda_device_count() > 1
        main, startup, outs = self.build_program(place, layout, seed, True,
                                                 only_forward, activation)
        exe = fluid.Executor(place)
        exe.run(startup)
        fetch_names = [v.name for v in outs] + [
            'bn_moving_mean', 'bn_moving_variance', 'bn_scale', 'bn_bias'
        ]
        if not only_forward:
            others = [
                'batch_norm_0.tmp_0', 'batch_norm_0.tmp_1', 'bn_scale@GRAD',
                'bn_bias@GRAD', 'batch_norm_0.tmp_2@GRAD', 'conv2d_0.tmp_0@GRAD'
            ]
            fetch_names += others
        for nm in fetch_names:
            fv = fluid.framework._get_var(str(nm), program=main)
            fv.persistable = True
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = False
        build_strategy.memory_optimize = False
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 1
        comp_prog = compiler.CompiledProgram(main).with_data_parallel(
            outs[0].name if not only_forward else None,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
        inplace_abn_fetches = exe.run(program=comp_prog,
                                      feed={'input': data},
                                      fetch_list=fetch_names)

        for i in six.moves.xrange(1, len(inplace_abn_fetches)):
            bn_val = bn_fetches[i]
            inplace_abn_val = inplace_abn_fetches[i]
            if inplace_abn_val.shape != bn_val.shape:
                inplace_abn_val = inplace_abn_val[:bn_val.shape[0]]
            self.assertTrue(
                np.allclose(
                    bn_val, inplace_abn_val, atol=1e-3),
                "Output (" + fetch_names[i] + ") has diff. \n" + "\nBN     " +
                str(bn_val) + "\n" + "Inplace ABN " + str(inplace_abn_val))

    def test_train(self):
        # if not core.is_compiled_with_cuda():
        #     return

        # places = [core.CUDAPlace(0)]
        places = [core.CPUPlace()]
        for place in places:
            for layout in ["NCHW", "NHWC"]:
                for activation in ['elu']:
                    self.compare(place, layout, False, activation)

    def test_infer(self):
        # if not core.is_compiled_with_cuda():
        #     return

        # places = [core.CUDAPlace(0)]
        places = [core.CPUPlace()]
        for place in places:
            for layout in ["NCHW", "NHWC"]:
                for activation in ['elu']:
                    self.compare(place, layout, True, activation)


if __name__ == '__main__':
    unittest.main()
