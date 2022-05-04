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

import unittest

import numpy as np
import paddle
from pass_test import PassTest
import paddle.fluid as fluid
import paddle.fluid.core as core

def get_input(shape=[4,4]):
    ret = np.ones(shape=shape).astype("float32")
    return ret

class FCFusePassTest(PassTest):
    def make_2_4_weights(self):
        for param in self.main_program.list_vars():
            if "sparse_2_4" not in param.name or "w_0" not in param.name: continue
            print(param.name)
            v_param = np.ones(shape=[4,4], dtype="float32")
            for i in range(4):
                v_param[i, :2] = 0
                v_param[i, 2:] = 2
            param.set_value(v_param, fluid.global_scope())
            print(param.get_value())
    def setUp(self):
        shape=[4,4]
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=shape, dtype="float32", lod_level=0)
            tmp_0 = fluid.layers.fc(input=data,
                                    size=4,
                                    num_flatten_dims=1,
                                    act=None,
                                    name="weight_sparse_2_4")

        place = paddle.CUDAPlace(0)
        self.exe = fluid.Executor(place)
        self.exe.run(self.startup_program)

        self.make_2_4_weights()
        ret = get_input(shape)
        self.feeds = {"data": ret}
        self.fetch_list = [tmp_0]
        self.pass_names = "replace_dense_with_sparse_pass"
        self.fused_op_type = "fc"
        self.num_fused_ops = 0

    def test_check_output(self):
        use_gpu_set = [False]
        if core.is_compiled_with_cuda():
            use_gpu_set.append(True)
        # fluid.io.save_inference_model("./test", feeded_var_names=["data"], target_vars=self.fetch_list, 
        #                               executor=self.exe, main_program=self.main_program)
        for use_gpu in use_gpu_set:
            self.pass_attrs = {"replace_dense_with_sparse_pass": {"use_gpu": use_gpu}}
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            opt_prog = self.check_output_with_place(place, startup_on_cpu=True)
        fluid.io.save_inference_model("./test", feeded_var_names=["data"], target_vars=self.fetch_list, 
                                      executor=self.exe, main_program=opt_prog)
  
if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
