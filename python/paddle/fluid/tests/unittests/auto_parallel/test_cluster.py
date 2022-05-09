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

import unittest
import os
import json

import paddle
from paddle.distributed.auto_parallel.cluster import Cluster

cluster_json = """
{ 
    "alpha_latency": {"inter": {"ring": "NET", "tree": "NET"},
                    "intra": {"ring": "NVL", "tree": "PHB"},
                    "base": {"ring": 8.4, "tree": 0},
                    "switch": 10.0},
    "machines": [
        {
            "hostname": "yq01-sys-hic-v100-box-a225-0266",
            "addr": "10.127.9.147",
            "port": "60009",
            "devices": [
                {
                    "global_id": 0,
                    "local_id": 0,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 1,
                    "local_id": 1,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 2,
                    "local_id": 2,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 3,
                    "local_id": 3,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 4,
                    "local_id": 4,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 5,
                    "local_id": 5,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 6,
                    "local_id": 6,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 7,
                    "local_id": 7,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 8,
                    "local_id": 0,
                    "type": "CPU",
                    "arch": "x86_64",
                    "vendor": "GenuineIntel",
                    "model": "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GH",
                    "memory": "502",
                    "sp_gflops": "150",
                    "dp_gflops": "75"
                },
                {
                    "global_id": 9,
                    "local_id": 0,
                    "type": "NIC",
                    "width": 12.5,
                    "ip": "10.127.9.147"
                }
            ],
            "links": [
                {
                    "source_global_id": 0,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 5,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 6,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 7,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 4,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 6,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 7,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 4,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 5,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 7,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 4,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 5,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 6,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 1,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 2,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 3,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 0,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 2,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 3,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 0,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 1,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 3,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 0,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 1,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 2,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 0,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 1,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 2,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 3,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 4,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 5,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 6,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 7,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 0,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 1,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 2,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 3,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 4,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 5,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 6,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 7,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                }
            ]
        }
    ]
}
"""

multi_cluster_json = """{
    "machines": [
        {
            "hostname": "yq01-sys-hic-v100-box-a225-0266",
            "addr": "10.127.9.147",
            "port": "60009",
            "devices": [
                {
                    "global_id": 0,
                    "local_id": 0,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 1,
                    "local_id": 1,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 2,
                    "local_id": 2,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 3,
                    "local_id": 3,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 4,
                    "local_id": 4,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 5,
                    "local_id": 5,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 6,
                    "local_id": 6,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 7,
                    "local_id": 7,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 8,
                    "local_id": 0,
                    "type": "CPU",
                    "arch": "x86_64",
                    "vendor": "GenuineIntel",
                    "model": "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GH",
                    "memory": "502",
                    "sp_gflops": "150",
                    "dp_gflops": "75"
                },
                {
                    "global_id": 9,
                    "local_id": 0,
                    "type": "NIC",
                    "width": 12.5,
                    "ip": "10.127.9.147"
                }
            ],
            "links": [
                {
                    "source_global_id": 0,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 5,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 6,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 7,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 4,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 6,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 7,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 4,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 5,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 7,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 4,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 5,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 6,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 1,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 2,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 3,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 0,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 2,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 3,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 0,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 1,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 3,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 0,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 1,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 2,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 0,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 1,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 2,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 3,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 4,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 5,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 6,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 7,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 0,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 1,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 2,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 3,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 4,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 5,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 6,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 7,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 19,
                    "type": "NET",
                    "bandwidth": 24.0
                }
            ]
        },
        {
            "hostname": "yq01-sys-hic-k8s-v100-box-a225-0751",
            "addr": "10.127.43.24",
            "port": "60009",
            "devices": [
                {
                    "global_id": 10,
                    "local_id": 0,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 11,
                    "local_id": 1,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 12,
                    "local_id": 2,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 13,
                    "local_id": 3,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 14,
                    "local_id": 4,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 15,
                    "local_id": 5,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 16,
                    "local_id": 6,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 17,
                    "local_id": 7,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 18,
                    "local_id": 0,
                    "type": "CPU",
                    "arch": "x86_64",
                    "vendor": "GenuineIntel",
                    "model": "Intel(R) Xeon(R) Gold 6271C CPU @ 2.60G",
                    "memory": "503",
                    "sp_gflops": "150",
                    "dp_gflops": "75"
                },
                {
                    "global_id": 19,
                    "local_id": 0,
                    "type": "NIC",
                    "width": 12.5,
                    "ip": "10.127.43.24"
                }
            ],
            "links": [
                {
                    "source_global_id": 10,
                    "target_global_id": 11,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 10,
                    "target_global_id": 12,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 10,
                    "target_global_id": 13,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 10,
                    "target_global_id": 14,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 10,
                    "target_global_id": 15,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 10,
                    "target_global_id": 16,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 10,
                    "target_global_id": 17,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 10,
                    "target_global_id": 18,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 10,
                    "target_global_id": 19,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 11,
                    "target_global_id": 10,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 11,
                    "target_global_id": 12,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 11,
                    "target_global_id": 13,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 11,
                    "target_global_id": 14,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 11,
                    "target_global_id": 15,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 11,
                    "target_global_id": 16,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 11,
                    "target_global_id": 17,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 11,
                    "target_global_id": 18,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 11,
                    "target_global_id": 19,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 12,
                    "target_global_id": 10,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 12,
                    "target_global_id": 11,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 12,
                    "target_global_id": 13,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 12,
                    "target_global_id": 14,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 12,
                    "target_global_id": 15,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 12,
                    "target_global_id": 16,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 12,
                    "target_global_id": 17,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 12,
                    "target_global_id": 18,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 12,
                    "target_global_id": 19,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 13,
                    "target_global_id": 10,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 13,
                    "target_global_id": 11,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 13,
                    "target_global_id": 12,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 13,
                    "target_global_id": 14,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 13,
                    "target_global_id": 15,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 13,
                    "target_global_id": 16,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 13,
                    "target_global_id": 17,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 13,
                    "target_global_id": 18,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 13,
                    "target_global_id": 19,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 14,
                    "target_global_id": 10,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 14,
                    "target_global_id": 11,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 14,
                    "target_global_id": 12,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 14,
                    "target_global_id": 13,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 14,
                    "target_global_id": 15,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 14,
                    "target_global_id": 16,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 14,
                    "target_global_id": 17,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 14,
                    "target_global_id": 18,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 14,
                    "target_global_id": 19,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 15,
                    "target_global_id": 10,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 15,
                    "target_global_id": 11,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 15,
                    "target_global_id": 12,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 15,
                    "target_global_id": 13,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 15,
                    "target_global_id": 14,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 15,
                    "target_global_id": 16,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 15,
                    "target_global_id": 17,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 15,
                    "target_global_id": 18,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 15,
                    "target_global_id": 19,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 16,
                    "target_global_id": 10,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 16,
                    "target_global_id": 11,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 16,
                    "target_global_id": 12,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 16,
                    "target_global_id": 13,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 16,
                    "target_global_id": 14,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 16,
                    "target_global_id": 15,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 16,
                    "target_global_id": 17,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 16,
                    "target_global_id": 18,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 16,
                    "target_global_id": 19,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 17,
                    "target_global_id": 10,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 17,
                    "target_global_id": 11,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 17,
                    "target_global_id": 12,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 17,
                    "target_global_id": 13,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 17,
                    "target_global_id": 14,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 17,
                    "target_global_id": 15,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 17,
                    "target_global_id": 16,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 17,
                    "target_global_id": 18,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 17,
                    "target_global_id": 19,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 18,
                    "target_global_id": 10,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 18,
                    "target_global_id": 11,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 18,
                    "target_global_id": 12,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 18,
                    "target_global_id": 13,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 18,
                    "target_global_id": 14,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 18,
                    "target_global_id": 15,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 18,
                    "target_global_id": 16,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 18,
                    "target_global_id": 17,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 18,
                    "target_global_id": 19,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 19,
                    "target_global_id": 10,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 19,
                    "target_global_id": 11,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 19,
                    "target_global_id": 12,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 19,
                    "target_global_id": 13,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 19,
                    "target_global_id": 14,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 19,
                    "target_global_id": 15,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 19,
                    "target_global_id": 16,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 19,
                    "target_global_id": 17,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 19,
                    "target_global_id": 18,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 19,
                    "target_global_id": 9,
                    "type": "NET",
                    "bandwidth": 24.0
                }
            ]
        }
    ]
}
"""


class TestCluster(unittest.TestCase):
    def test_single_machine(self):
        # Build cluster
        file_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_json_path = os.path.join(file_dir, "auto_parallel_cluster.json")
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)

        beta = cluster.get_beta(0, 1)
        hop = cluster.get_hop(0, 1)
        cross_machine = cluster.cross_machine([0, 1])
        devices = cluster.convert_rank_to_device_id([0, 1, 2, 3])
        involved_machine_count = cluster.get_involved_machine_count(devices)
        self.assertTrue(beta > 0)
        self.assertTrue(hop == 0)
        self.assertTrue(not cross_machine)
        self.assertTrue(devices == [0, 1, 2, 3])
        self.assertTrue(involved_machine_count == 1)

        # Remove unnecessary files
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)

    def test_multi_machine(self):
        # Build cluster
        file_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_json_path = os.path.join(file_dir, "auto_parallel_cluster.json")
        cluster_json_object = json.loads(multi_cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)

        beta = cluster.get_beta(0, 11)
        hop = cluster.get_hop(0, 11)
        cross_machine = cluster.cross_machine([0, 11])
        devices = cluster.convert_rank_to_device_id([5, 6, 7, 8])
        involved_machine_count = cluster.get_involved_machine_count(devices)
        self.assertTrue(beta > 0)
        self.assertTrue(hop >= 0)
        self.assertTrue(cross_machine)
        self.assertTrue(devices == [5, 6, 7, 10])
        self.assertTrue(involved_machine_count == 2)

        # Remove unnecessary files
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)


if __name__ == "__main__":
    unittest.main()
