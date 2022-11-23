# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
from paddle.fluid.core import PassVersionChecker


class FcGruFusePassTest(InferencePassTest):

    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            dict_dim, emb_dim = 128, 64
            data = fluid.data(name='step_data',
                              shape=[None],
                              dtype='int64',
                              lod_level=1)
            emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim = 512
            x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
            hidden = fluid.layers.dynamic_gru(input=x,
                                              size=hidden_dim,
                                              bias_attr=True,
                                              origin_mode=False,
                                              is_reverse=True)

        batch = 16
        lod_tensor = fluid.LoDTensor()
        lod_tensor.set(
            np.random.randint(0, dict_dim, size=[batch]).astype("int64"),
            fluid.CPUPlace())
        lod_tensor.set_lod([[0, batch]])
        self.feeds = {"step_data": lod_tensor}
        self.fetch_list = [hidden]

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)
        self.assertTrue(PassVersionChecker.IsCompatible('fc_gru_fuse_pass'))


class MulGruFusePassTest(InferencePassTest):

    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            dict_dim, emb_dim = 128, 64
            data = fluid.data(name='step_data',
                              shape=[None],
                              dtype='int64',
                              lod_level=1)
            emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim = 512
            x = fluid.layers.fc(input=emb, size=hidden_dim * 3, bias_attr=False)
            hidden = fluid.layers.dynamic_gru(input=x,
                                              size=hidden_dim,
                                              bias_attr=True,
                                              origin_mode=False,
                                              is_reverse=True)

        batch = 16
        lod_tensor = fluid.LoDTensor()
        lod_tensor.set(
            np.random.randint(0, dict_dim, size=[batch]).astype("int64"),
            fluid.CPUPlace())
        lod_tensor.set_lod([[0, batch]])
        self.feeds = {"step_data": lod_tensor}
        self.fetch_list = [hidden]

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)
        self.assertTrue(PassVersionChecker.IsCompatible('mul_gru_fuse_pass'))


if __name__ == "__main__":
    unittest.main()
