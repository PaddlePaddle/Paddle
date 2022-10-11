# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from simple_nets import simple_fc_net, init_data


class TestMNIST(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.save_dirname = "./"
        cls.model_filename = "test_parallel_executor_run_load_infer_program_model"
        cls.params_filename = "test_parallel_executor_run_load_infer_program_parameter"
        cls.place = fluid.CPUPlace()
        cls.exe = fluid.Executor(cls.place)
        img, label = init_data()
        cls.batch_data = []
        for img, label in zip(img, label):
            cls.batch_data.append([img, label])

    def test_simple_fc(self):
        exe_loss = self.run_with_executor()

        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(self.save_dirname,
                                                        self.exe,
                                                        self.model_filename,
                                                        self.params_filename)

        train_exe = fluid.ParallelExecutor(use_cuda=False,
                                           main_program=inference_program)
        feed_vars = [
            inference_program.global_block().var(var_name)
            for var_name in ["image", "label"]
        ]
        feeder = fluid.DataFeeder(place=self.place, feed_list=feed_vars)

        pe_loss = train_exe.run(feed=feeder.feed(self.batch_data),
                                fetch_list=[fetch_targets[0].name])
        assert exe_loss == pe_loss

    def run_with_executor(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = simple_fc_net()

        feed_vars = [
            main.global_block().var(var_name)
            for var_name in ["image", "label"]
        ]
        feeder = fluid.DataFeeder(place=self.place, feed_list=feed_vars)

        self.exe.run(startup)

        loss_data = self.exe.run(main,
                                 feed=feeder.feed(self.batch_data),
                                 fetch_list=[loss.name])

        fluid.io.save_inference_model(self.save_dirname, ["image", "label"],
                                      [loss],
                                      self.exe,
                                      model_filename=self.model_filename,
                                      params_filename=self.params_filename,
                                      main_program=main)

        return loss_data


if __name__ == '__main__':
    unittest.main()
