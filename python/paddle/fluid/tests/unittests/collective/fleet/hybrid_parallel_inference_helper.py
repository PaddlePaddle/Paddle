# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import paddle
import numpy as np
import paddle.fluid.layers as layers
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.utils.hybrid_parallel_inference import HybridParallelInferenceHelper

paddle.enable_static()


def numpy_while(x, w1=1.0, w2=2.0, max_len=2):
    data = [x]
    weight1 = np.empty([2, 5], dtype='float32')
    weight1.fill(w1)
    weight2 = np.empty([5, 2], dtype='float32')
    weight2.fill(w2)
    for i in range(max_len):
        input = data[i]
        hidden1 = np.matmul(input, weight1)
        hidden2 = np.matmul(hidden1, weight2)
        data.append(hidden2)

    return data


class TestHybridParallelInferenceHelperClass(unittest.TestCase):

    def setUp(self):
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)
        np.random.seed(2333)

    def test_hybrid_parallel_inference_helper_mp1pp2(self):

        nranks = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
        rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
        dev_id = int(os.getenv("FLAGS_selected_gpus", 0))

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        device = "gpu"

        with paddle.static.program_guard(main_program, startup_program):
            with paddle.fluid.device_guard(f'{device}:0'):
                X = paddle.static.data(name='X',
                                       shape=[None, 2],
                                       dtype='float32')

            with paddle.fluid.device_guard(f'{device}:all'):
                max_len = layers.fill_constant(shape=[1],
                                               dtype="int64",
                                               value=2,
                                               force_cpu=False,
                                               name="n")
                step_idx = layers.fill_constant(shape=[1],
                                                dtype="int64",
                                                value=0,
                                                force_cpu=False,
                                                name="i")

                data = layers.array_write(X, step_idx)

                cond_int = layers.fill_constant(shape=[1],
                                                dtype="int64",
                                                value=0,
                                                force_cpu=False,
                                                name="cond_int")
                cond = layers.less_than(x=step_idx, y=max_len)
                while_op = layers.While(cond, is_test=True)

            with while_op.block():
                with paddle.fluid.device_guard(f'{device}:all'):
                    input = layers.array_read(array=data, i=step_idx)
                    layers.increment(x=step_idx, value=1.0, in_place=True)
                    layers.array_write(input, i=step_idx, array=data)

                with paddle.fluid.device_guard(f'{device}:0'):
                    param_attr = paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(1.0))
                    weight1 = paddle.static.create_parameter(shape=[2, 5],
                                                             dtype='float32',
                                                             attr=param_attr,
                                                             is_bias=False)
                    hidden1 = paddle.matmul(input, weight1)

                with paddle.fluid.device_guard(f'{device}:1'):
                    param_attr = paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(2.0))
                    weight2 = paddle.static.create_parameter(shape=[5, 2],
                                                             dtype='float32',
                                                             attr=param_attr,
                                                             is_bias=False)
                    hidden2 = paddle.matmul(hidden1, weight2)

                    layers.array_write(hidden2, i=step_idx, array=data)

                    # update cond and assign to cond_int, we will sync cond_int
                    layers.less_than(x=step_idx, y=max_len, cond=cond)
                    layers.assign(layers.cast(cond, dtype="int32"), cond_int)

                with paddle.fluid.device_guard(f'{device}:all'):
                    # the code below must at end of while block and exists in device:all
                    layers.assign(layers.cast(cond_int, dtype='bool'), cond)

            with paddle.fluid.device_guard(f'{device}:all'):
                out = layers.create_array(data.dtype)
                layers.assign(data, out)

            with paddle.fluid.device_guard(f'{device}:all'):
                # use a empty lod_tensor_array to clear lod_tensor_array
                layers.assign(layers.create_array(data.dtype), data)

        helper = HybridParallelInferenceHelper(
            startup_program,
            main_program,
            micro_batch_size=2,
            num_mp=1,
            num_pp=2,
            init_comm=nranks > 1,
        )
        helper.gen_infer_program(['array_write_0.out'], ['cond_int.tmp_0'],
                                 debug=True)

        exe = paddle.static.Executor(paddle.CUDAPlace(dev_id))
        exe.run(startup_program)

        for step in range(2):
            init_data = np.random.uniform(low=0.0, high=1.0,
                                          size=[2, 2]).astype('float32')
            [res] = exe.run(main_program,
                            feed={"X": init_data},
                            fetch_list=[out])
            res_np = numpy_while(init_data)

            assert len(res) == len(res_np)
            for d1, d2 in zip(res, res_np):
                np.testing.assert_allclose(d1, d2)


if __name__ == '__main__':
    unittest.main()
