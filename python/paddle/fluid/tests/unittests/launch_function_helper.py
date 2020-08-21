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
from multiprocessing import Pool, Process
import os
import socket
from contextlib import closing


def launch_func(func, env_dict):
    for key in env_dict:
        os.environ[key] = env_dict[key]
    proc = Process(target=func)
    return proc


def _find_free_port(port_set):
    def __free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    while True:
        port = __free_port()
        if port not in port_set:
            port_set.add(port)
            return port
