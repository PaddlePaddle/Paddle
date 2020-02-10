# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.dygraph.nn import *
import numpy as np


class TestDygraphLoadStatic(unittest.TestCase):
    def testLoadStaticModel(self):
        # static mode
        a = fluid.data(name="a", shape=[10, 10])
        conv_in = fluid.data(name="conv_in", shape=[None, 10, 10, 10])

        fc_out1 = fluid.layers.fc(a, 10)
        fc_out2 = fluid.layers.fc(a, 20)

        conv_out_1 = fluid.layers.conv2d(
            conv_in, num_filters=10, filter_size=5, act="relu")
        conv_out_2 = fluid.layers.conv2d(
            conv_in, num_filters=10, filter_size=5, act="relu")

        conv3d_in = fluid.data(
            name='conv3d_in', shape=[None, 3, 12, 32, 32], dtype='float32')
        conv3d_out_1 = fluid.layers.conv3d(
            input=conv3d_in, num_filters=2, filter_size=3, act="relu")
        conv3d_out_2 = fluid.layers.conv3d(
            input=conv3d_in, num_filters=2, filter_size=3, act="relu")

        batchnorm_in = fluid.data(
            name="batchnorm_in", shape=[None, 10], dtype='float32')
        batchnorm_out_1 = fluid.layers.batch_norm(batchnorm_in)
        batchnorm_out_2 = fluid.layers.batch_norm(batchnorm_in)

        emb_in = fluid.data(name='emb_in', shape=[None, 10], dtype='int64')
        emb_out_1 = fluid.embedding(emb_in, [1000, 100])
        emb_out_2 = fluid.embedding(emb_in, [2000, 200])

        layernorm = fluid.data(name="ln", shape=[None, 10], dtype='float32')
        layernorm_1 = fluid.layers.layer_norm(layernorm)
        layernorm_2 = fluid.layers.layer_norm(layernorm)

        nce_in = fluid.data(name="nce_in", shape=[None, 100], dtype='float32')
        nce_label = fluid.data(
            name="nce_label", shape=[None, 10], dtype='int64')
        nce_out_1 = fluid.layers.nce(nce_in, nce_label, 10000)
        nce_out_2 = fluid.layers.nce(nce_in, nce_label, 10000)

        prelu_in = fluid.data(
            name="prelu_in", shape=[None, 5, 10, 10], dtype='float32')
        prelu_out_1 = fluid.layers.prelu(prelu_in, "channel")
        prelu_out_2 = fluid.layers.prelu(prelu_in, "channel")

        bilinear_tensor_pro_x = fluid.data(
            "t1", shape=[None, 5], dtype="float32")
        bilinear_tensor_pro_y = fluid.data(
            "t2", shape=[None, 4], dtype="float32")

        bilinear_tensor_pro_out_1 = fluid.layers.bilinear_tensor_product(
            x=bilinear_tensor_pro_x, y=bilinear_tensor_pro_y, size=1000)
        bilinear_tensor_pro_out_2 = fluid.layers.bilinear_tensor_product(
            x=bilinear_tensor_pro_x, y=bilinear_tensor_pro_y, size=1000)

        conv2d_trans_in = fluid.data(
            name="conv2d_trans_in", shape=[None, 10, 10, 10])

        conv2d_trans_out_1 = fluid.layers.conv2d_transpose(
            conv2d_trans_in, num_filters=10, filter_size=5, act="relu")
        conv2d_trans_out_2 = fluid.layers.conv2d_transpose(
            conv2d_trans_in, num_filters=10, filter_size=5, act="relu")

        conv3d_trans_in = fluid.data(
            name='conv3d_trans_in',
            shape=[None, 3, 12, 32, 32],
            dtype='float32')
        conv3d_trans_out_1 = fluid.layers.conv3d_transpose(
            input=conv3d_trans_in, num_filters=2, filter_size=3, act="relu")
        conv3d_trans_out_2 = fluid.layers.conv3d_transpose(
            input=conv3d_trans_in, num_filters=2, filter_size=3, act="relu")

        groupnorm_in = fluid.data(
            name='groupnorm_in', shape=[None, 8, 32, 32], dtype='float32')
        groupnorm_out1 = fluid.layers.group_norm(input=groupnorm_in, groups=4)
        groupnorm_out2 = fluid.layers.group_norm(input=groupnorm_in, groups=4)
        '''
        spec_norm = fluid.data(name='spec_norm', shape=[2, 8, 32, 32], dtype='float32')
        spe_norm_out_1 = fluid.layers.spectral_norm(weight=spec_norm, dim=1, power_iters=2)
        spe_norm_out_2 = fluid.layers.spectral_norm(weight=spec_norm, dim=1, power_iters=2)
        '''

        nodes_vector = fluid.data(
            name='vectors', shape=[None, 10, 5], dtype='float32')
        edge_set = fluid.data(
            name='edge_set', shape=[None, 10, 2], dtype='float32')
        tree_conv_out1 = fluid.contrib.layers.tree_conv(nodes_vector, edge_set,
                                                        6, 1, 2)
        tree_conv_out2 = fluid.contrib.layers.tree_conv(nodes_vector, edge_set,
                                                        6, 1, 2)

        para1 = fluid.layers.create_parameter(
            [100, 100], 'float32', name="weight_test_1")
        para2 = fluid.layers.create_parameter(
            [20, 200], 'float32', name="weight_test_2")

        para_list = fluid.default_main_program().list_vars()

        exe = fluid.Executor(fluid.CPUPlace(
        ) if not fluid.is_compiled_with_cuda() else fluid.CUDAPlace(0))
        out = exe.run(framework.default_startup_program())

        fluid.save(framework.default_main_program(), "./test_1")

        para_dict = fluid.load_program_state("./test_1")

        new_dict = {}
        for k, v in para_dict.items():
            #print( k, v.shape )
            if k.startswith("fc"):
                name = k.replace("fc", "linear", 1)
                new_dict[name] = v
            else:
                new_dict[k] = v

        with fluid.dygraph.guard():

            class MyTest(fluid.dygraph.Layer):
                def __init__(self):
                    super(MyTest, self).__init__()

                    self.linear1 = Linear(10, 10)
                    self.lienar2 = Linear(10, 20)

                    self.conv2d_1 = Conv2D(
                        num_channels=10,
                        num_filters=10,
                        filter_size=5,
                        act="relu")
                    self.conv2d_2 = Conv2D(
                        num_channels=10,
                        num_filters=10,
                        filter_size=5,
                        act="relu")

                    self.conv3d_1 = Conv3D(
                        num_channels=3,
                        num_filters=2,
                        filter_size=3,
                        act="relu")
                    self.conv3d_2 = Conv3D(
                        num_channels=3,
                        num_filters=2,
                        filter_size=3,
                        act="relu")

                    self.batch_norm_1 = BatchNorm(10)
                    self.batch_norm_2 = BatchNorm(10)

                    self.emb1 = Embedding([1000, 100])
                    self.emb2 = Embedding([2000, 200])

                    self.layer_norm_1 = LayerNorm([10])
                    self.layer_norm_2 = LayerNorm(10)

                    self.nce1 = NCE(10000, 100)
                    self.nce2 = NCE(10000, 100)

                    self.prelu1 = PRelu("channel", channel=5)
                    self.prelu2 = PRelu("channel", channel=5)

                    self.group_norm1 = GroupNorm(8, 4)
                    self.gourp_norm2 = GroupNorm(8, 4)

                    self.w_1 = self.create_parameter(
                        [100, 100], dtype='float32', attr="weight_test_1")
                    self.w_2 = self.create_parameter(
                        [20, 200], dtype='float32', attr="weight_test_2")

            my_test = MyTest()
            my_test.set_dict(new_dict, use_structured_name=False)
            for k, v in my_test.state_dict().items():
                self.assertTrue(np.array_equal(v.numpy(), new_dict[v.name]))


if __name__ == '__main__':
    unittest.main()
