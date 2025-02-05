# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import numpy as np

import paddle
import paddle.inference as paddle_infer
from paddle.framework import in_pir_mode

paddle.enable_static()


class UnittestBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.init_info()

    def tearDwon(self):
        self.temp_dir.cleanup()

    def init_info(self):
        self.shapes = None
        self.save_path = None

    def path_prefix(self):
        return type(self).__name__

    def infer_prog(self):
        if in_pir_mode():
            config = paddle_infer.Config(
                self.save_path + '.json', self.save_path + '.pdiparams'
            )
            config.enable_new_ir()
            config.enable_new_executor()
        else:
            config = paddle_infer.Config(
                self.save_path + '.pdmodel', self.save_path + '.pdiparams'
            )
        config.disable_mkldnn()
        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        for i, shape in enumerate(self.shapes):
            input_handle = predictor.get_input_handle(input_names[i])
            self.fake_input = np.random.randn(*shape).astype("float32")
            input_handle.reshape(shape)
            input_handle.copy_from_cpu(self.fake_input)
        predictor.run()
        output_names = predictor.get_output_names()
        res = []
        for out_name in output_names:
            output_handle = predictor.get_output_handle(out_name)
            output_data = output_handle.copy_to_cpu()
            res.append(output_data)

        if len(output_names) == 1:
            res = res[0]

        return res


class TestDropout(UnittestBase):
    def init_info(self):
        self.shapes = [[10, 10]]
        self.save_path = os.path.join(self.temp_dir.name, 'dropout')

    def test_static(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            fc = paddle.nn.Linear(10, 10)
            x = paddle.randn(self.shapes[0])
            x.stop_gradient = False
            feat = fc(x)
            # p is a Variable
            p = paddle.randn([1])
            out = paddle.nn.functional.dropout(feat, p=p)
            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))

            exe = paddle.static.Executor()
            exe.run(startup_prog)
            res = exe.run(fetch_list=[x, out])
            # export model
            paddle.static.save_inference_model(self.save_path, [x], [out], exe)

            # Test for Inference Predictor
            infer_out = self.infer_prog()
            self.assertEqual(infer_out.shape, (10, 10))

            if not in_pir_mode():
                self.assertTrue("Var[" in str(main_prog))
                self.assertEqual(
                    main_prog.block(0).ops[4].all_attrs()['dropout_prob'].name,
                    p.name,
                )


if __name__ == '__main__':
    unittest.main()
