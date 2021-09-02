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


class TestCollectiveGlobalGatherAPI(TestDistBase):
    def _setup_config(self):
        pass

    def test_global_gather_nccl(self):
        paddle.enable_static()
        self.check_with_place("collective_global_gather.py", "global_gather",
                              "nccl")

    def test_global_gather_nccl_dygraph(self):
        self.check_with_place(
            "collective_global_gather_dygraph.py",
            "global_gather",
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

        if col_type == "global_gather":
            in_feat = 2
            n_expert = 2
            world_size = 2
            tot_expert = n_expert * world_size

            np.random.seed(pid0)
            local_expert_count1 = np.random.randint(
                0, 4, size=tot_expert).astype("int")
            expert_ptr1 = np.ones(tot_expert, dtype=np.int32)
            expert_ptr1[0] = 0
            for i in range(1, tot_expert):
                expert_ptr1[i] = expert_ptr1[i - 1] + local_expert_count1[i - 1]

            np.random.seed(pid1)
            local_expert_count2 = np.random.randint(
                0, 4, size=tot_expert).astype("int")
            expert_ptr2 = np.ones(tot_expert, dtype=np.int32)
            expert_ptr2[0] = 0
            for i in range(1, tot_expert):
                expert_ptr2[i] = expert_ptr2[i - 1] + local_expert_count2[i - 1]

            global_expert_count1 = np.zeros(tot_expert).astype("int")
            global_expert_count2 = np.zeros(tot_expert).astype("int")
            global_expert_count1[0:n_expert] = local_expert_count1[0:n_expert]
            global_expert_count1[n_expert:] = local_expert_count2[0:n_expert]
            global_expert_count2[0:n_expert] = local_expert_count1[n_expert:]
            global_expert_count2[n_expert:] = local_expert_count2[n_expert:]

            np.random.seed(pid0)
            fwd_expert_count = sum(global_expert_count1).astype("int")
            local_input_buf1 = np.random.rand(fwd_expert_count,
                                              in_feat).astype("float32")
            np.random.seed(pid1)
            fwd_expert_count = sum(global_expert_count2).astype("int")
            local_input_buf2 = np.random.rand(fwd_expert_count,
                                              in_feat).astype("float32")
            output1 = [[], [], [], []]
            output2 = [[], [], [], []]
            send_ptr1 = 0
            send_ptr2 = 0

            for i in range(n_expert):
                for j in range(world_size):
                    idx = j * n_expert + i
                    if j == 0:
                        output1_part1 = local_input_buf1[send_ptr1: \
                            send_ptr1 + global_expert_count1[idx], :]
                        output1_part2 = local_input_buf2[send_ptr2: \
                            send_ptr2 + global_expert_count2[idx], :]
                        output1[i].extend(output1_part1)
                        output1[i + n_expert].extend(output1_part2)
                    else:
                        output2_part1 = local_input_buf1[send_ptr1: \
                            send_ptr1 + global_expert_count1[idx]]
                        output2_part2 = local_input_buf2[send_ptr2: \
                            send_ptr2 + global_expert_count2[idx]]
                        output2[i].extend(output2_part1)
                        output2[i + n_expert].extend(output2_part2)
                    send_ptr1 = send_ptr1 + global_expert_count1[idx]
                    send_ptr2 = send_ptr2 + global_expert_count2[idx]
            result1 = []
            result2 = []
            for i in range(tot_expert):
                for arr in output1[i]:
                    result1.append(arr)
            for i in range(tot_expert):
                for arr in output2[i]:
                    result2.append(arr)
            output1 = np.concatenate(
                result1, axis=0).reshape(sum(local_expert_count1), in_feat)
            output2 = np.concatenate(
                result2, axis=0).reshape(sum(local_expert_count2), in_feat)

            self.assertTrue(
                np.allclose(
                    tr0_out[0], output1, rtol=1e-05, atol=1e-05))
            self.assertTrue(
                np.allclose(
                    tr1_out[0], output2, rtol=1e-05, atol=1e-05))

        else:
            pass


if __name__ == '__main__':
    unittest.main()
