#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle

from test_collective_api_base import TestDistBase
import os


class TestCollectiveSelectScatterAPI(TestDistBase):
    def _setup_config(self):
        pass

    def test_global_scatter_nccl(self):
        paddle.enable_static()
        self.check_with_place("collective_global_scatter.py", "global_scatter",
                              "nccl")

    def test_global_scatter_nccl_dygraph(self):
        self.check_with_place(
            "collective_global_scatter_dygraph.py",
            "global_scatter",
            "nccl",
            static_mode="0")

    def check_with_place(self,
                         model_file,
                         col_type,
                         backend="nccl",
                         path_id="0",
                         static_mode="1",
                         check_error_log=False,
                         need_envs={}):
        if backend == "nccl" or backend == "bkcl":
            with_gloo = '0'
        else:
            with_gloo = '1'
        required_envs = {
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_eager_delete_tensor_gb": "0.0",
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
            "FLAGS_call_stack_level": "2",
            "GLOG_v": "3",
            "NCCL_P2P_DISABLE": "1",
            "STATIC_MODE": static_mode,
            "PADDLE_WITH_GLOO": with_gloo,
            "BACKEND": backend,
            "PATH_ID": path_id
        }
        required_envs.update(need_envs)
        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"
            required_envs["GLOO_LOG_LEVEL"] = "TRACE"
        tr0_out, tr1_out, pid0, pid1 = self._run_cluster(model_file,
                                                         required_envs)

        if col_type == "global_scatter":
            np.random.seed(pid0)
            local_expert_count1 = np.random.randint(1, 4, size=4).astype("int")
            fwd_expert_count = sum(local_expert_count1)
            local_input_buf1 = np.random.rand(fwd_expert_count,
                                              2).astype("float32")
            expert_ptr1 = np.ones(4, dtype=np.int32)
            expert_ptr1[0] = 0
            for i in range(1, 4):
                expert_ptr1[i] = expert_ptr1[i - 1] + local_expert_count1[i - 1]
            np.random.seed(pid1)
            local_expert_count2 = np.random.randint(1, 4, size=4).astype("int")
            fwd_expert_count = sum(local_expert_count2)
            local_input_buf2 = np.random.rand(fwd_expert_count,
                                              2).astype("float32")
            expert_ptr2 = np.ones(4, dtype=np.int32)
            expert_ptr2[0] = 0
            for i in range(1, 4):
                expert_ptr2[i] = expert_ptr2[i - 1] + local_expert_count2[i - 1]

            output1 = []
            output2 = []
            for i in range(2):
                for j in range(2):
                    idx = j * 2 + i
                    if j == 0:
                        # send data to 0 card
                        output1.append(local_input_buf1[expert_ptr1[idx]: \
                            expert_ptr1[idx]+local_expert_count1[idx]])
                        output1.append(local_input_buf2[expert_ptr2[idx]:\
                            expert_ptr2[idx]+local_expert_count2[idx]])
                    else:
                        output2.append(local_input_buf1[expert_ptr1[idx]: \
                            expert_ptr1[idx]+local_expert_count1[idx]])
                        output2.append(local_input_buf2[expert_ptr2[idx]:\
                            expert_ptr2[idx]+local_expert_count2[idx]])
            if output1 == []:
                output1 = np.array([])
            else:
                output1 = np.concatenate(output1)
            if output2 == []:
                output2 = np.array([])
            else:
                output2 = np.concatenate(output2)

            if tr0_out[0] is None or tr0_out[0].shape[0] == 0:
                tr0_out[0] = np.array([])

            if tr1_out[0] is None or tr1_out[0].shape[0] == 0:
                tr1_out[0] = np.array([])

            self.assertTrue(
                np.allclose(
                    tr0_out[0], output1, rtol=1e-05, atol=1e-05))
            self.assertTrue(
                np.allclose(
                    tr1_out[0], output2, rtol=1e-05, atol=1e-05))
            if static_mode == 0:
                self.assertTrue(
                    np.allclose(
                        tr0_out[1],
                        2 * local_input_buf1,
                        rtol=1e-05,
                        atol=1e-05))
                self.assertTrue(
                    np.allclose(
                        tr1_out[1],
                        2 * local_input_buf2,
                        rtol=1e-05,
                        atol=1e-05))


if __name__ == '__main__':
    unittest.main()
