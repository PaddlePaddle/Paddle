# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^ShuffleNet^ShuffleNetV2_x2_0
# api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape
import unittest
import os                                                                                                                                                                                                           
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'                                                                                                                                                                  
os.environ['FLAGS_group_schedule_tiling_first'] = '1'                                                                                                                                                                  
os.environ['FLAGS_prim_all'] = 'true'                                                                                                                                                                               
os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'                                                                                                                                                                            
os.environ['FLAGS_use_cinn'] = '1'                                                                                                                                                                                  
os.environ['FLAGS_cinn_bucket_compile'] = '1'                                                                                                                                                                                  
#os.environ['GLOG_vmodule'] = 'op_lowering_impl=4'
import paddle                                                                                                                                                                                                       
import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [10, 122, 28, 28], dtype: paddle.float32, stop_gradient: True)
        var_1,  # (shape: [10, 122, 28, 28], dtype: paddle.float32, stop_gradient: True)
    ):
        var_2 = paddle.tensor.manipulation.concat([var_1, var_0], axis=1)
        var_3 = paddle.tensor.manipulation.reshape(
            x=var_2, shape=[10, 2, 122, 28, 28]
        )
        var_4 = paddle.tensor.linalg.transpose(x=var_3, perm=[0, 2, 1, 3, 4])
        var_5 = paddle.tensor.manipulation.reshape(
            x=var_4, shape=[10, 244, 28, 28]
        )
        return var_5


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[10, 122, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[10, 122, 28, 28], dtype=paddle.float32),
        )
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()