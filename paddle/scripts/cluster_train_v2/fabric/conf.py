# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

HOSTS = [
    "root@10.1.9.7",
    "root@10.1.18.7",
    "root@10.1.32.9",
]
'''
workspace configuration
'''
#root dir for workspace, can be set as any director with real user account
ROOT_DIR = "/root"
'''
network configuration
'''
#pserver nics
PADDLE_NIC = "eth0"
#pserver port
PADDLE_PORT = 7164
#pserver ports num
PADDLE_PORTS_NUM = 1
#pserver sparse ports num
PADDLE_PORTS_NUM_FOR_SPARSE = 1
#trainer whether use gpu
PADDLE_USE_GPU = "False"
#environments setting for all processes in cluster job
LD_LIBRARY_PATH = "/usr/local/cuda/lib64:/usr/lib64"
