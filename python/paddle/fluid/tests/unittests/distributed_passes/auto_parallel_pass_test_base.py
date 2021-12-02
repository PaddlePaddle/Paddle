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

import paddle
import os
import random
import sys
import pickle
import shlex
import shutil
import inspect
import numpy as np
from collections import OrderedDict
from dist_pass_test_base import DistPassTestBase


class AutoPallelPassTestBase(DistPassTestBase):
    def setUp(self):
        paddle.enable_static()
        seed = int(os.environ.get('SEED', -1))
        if seed <= 0:
            seed = np.random.randint(low=1, high=1000000, size=[1])[0]
            os.environ['SEED'] = str(seed)
        self.seed = seed
        paddle.seed(self.seed)

        self.rtol = 1e-5
        self.atol = 1e-8
        self.equal_nan = False

        self.init()

    def init(self):
        pass

    def get_model(self, place, **kwargs):
        raise NotImplementedError()

    def apply_passes(self):
        raise NotImplementedError()

    def apply_no_passes(self):
        raise NotImplementedError()

    def check_main(self, gpus=None, **kwargs):
        no_pass_rets = self._distributed_launch(
            apply_pass=False, gpus=gpus, **kwargs)
        pass_rets = self._distributed_launch(
            apply_pass=True, gpus=gpus, **kwargs)
        self.check_results(no_pass_rets, pass_rets)

    def _run_gpu_main(self, apply_pass, dump_file, **kwargs):
        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        place = paddle.CUDAPlace(gpu_id)
        scope = paddle.static.Scope()
        if apply_pass:
            self.apply_passes()
        else:
            self.apply_no_passes()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            with paddle.static.scope_guard(scope):
                with paddle.fluid.unique_name.guard():
                    main_prog, startup_prog, inputs, outputs, reader = self.get_model(
                        place, **kwargs)
                    inputs = self._to_var_names(main_prog, inputs)
                    outputs = self._to_var_names(main_prog, outputs)

        all_fetch_values = []
        exe = paddle.static.Executor(place)
        with paddle.static.scope_guard(scope):
            exe.run(startup_prog)
            for batch_id, input_data in enumerate(reader()):
                assert len(input_data) == len(inputs), "{} vs {}".format(
                    len(input_data), len(inputs))
                feed = dict(zip(inputs, input_data))
                fetch_values = exe.run(main_prog, feed=feed, fetch_list=outputs)
                if paddle.distributed.get_rank() == 0:
                    output_dict = OrderedDict(zip(outputs, fetch_values))
                    print('batch {}, outputs {}'.format(batch_id, output_dict))
                all_fetch_values.append(fetch_values)
        with open(dump_file, "wb") as f:
            pickle.dump(all_fetch_values, f)
