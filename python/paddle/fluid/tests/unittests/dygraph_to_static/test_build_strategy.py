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

import os
import paddle
import unittest
import numpy as np
from paddle.jit import ProgramTranslator

from test_resnet import ResNet, ResNetHelper

program_translator = ProgramTranslator()


class TestResnetWithPass(unittest.TestCase):

    def setUp(self):
        self.build_strategy = paddle.static.BuildStrategy()
        self.build_strategy.fuse_elewise_add_act_ops = True
        self.build_strategy.fuse_bn_act_ops = True
        self.build_strategy.fuse_bn_add_act_ops = True
        self.build_strategy.enable_addto = True
        self.resnet_helper = ResNetHelper()
        # NOTE: for enable_addto
        paddle.fluid.set_flags({"FLAGS_max_inplace_grad_add": 8})

    def train(self, to_static):
        program_translator.enable(to_static)
        return self.resnet_helper.train(to_static, self.build_strategy)

    def verify_predict(self):
        image = np.random.random([1, 3, 224, 224]).astype('float32')
        dy_pre = self.resnet_helper.predict_dygraph(image)
        st_pre = self.resnet_helper.predict_static(image)
        dy_jit_pre = self.resnet_helper.predict_dygraph_jit(image)
        predictor_pre = self.resnet_helper.predict_analysis_inference(image)
        np.testing.assert_allclose(
            dy_pre,
            st_pre,
            rtol=1e-05,
            err_msg='dy_pre:\n {}\n, st_pre: \n{}.'.format(dy_pre, st_pre))
        np.testing.assert_allclose(
            dy_jit_pre,
            st_pre,
            rtol=1e-05,
            err_msg='dy_jit_pre:\n {}\n, st_pre: \n{}.'.format(
                dy_jit_pre, st_pre))
        np.testing.assert_allclose(
            predictor_pre,
            st_pre,
            rtol=1e-05,
            err_msg='predictor_pre:\n {}\n, st_pre: \n{}.'.format(
                predictor_pre, st_pre))

    def test_resnet(self):
        static_loss = self.train(to_static=True)
        dygraph_loss = self.train(to_static=False)
        np.testing.assert_allclose(
            static_loss,
            dygraph_loss,
            rtol=1e-05,
            err_msg='static_loss: {} \n dygraph_loss: {}'.format(
                static_loss, dygraph_loss))
        self.verify_predict()

    def test_in_static_mode_mkldnn(self):
        paddle.fluid.set_flags({'FLAGS_use_mkldnn': True})
        try:
            if paddle.fluid.core.is_compiled_with_mkldnn():
                self.resnet_helper.train(True, self.build_strategy)
        finally:
            paddle.fluid.set_flags({'FLAGS_use_mkldnn': False})


class TestError(unittest.TestCase):

    def test_type_error(self):

        def foo(x):
            out = x + 1
            return out

        with self.assertRaises(TypeError):
            static_foo = paddle.jit.to_static(foo, build_strategy="x")


if __name__ == '__main__':
    unittest.main()
