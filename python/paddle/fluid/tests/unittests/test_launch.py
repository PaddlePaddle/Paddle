# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import unittest


class TestLaunch(unittest.TestCase):
    def test_launch_multi_process(self):
        node_ips = "127.0.0.1"
        node_id = "0"
        current_node_ip = "127.0.0.1"

        distributed_args = "--node_ips {node_ips} --node_id {node_id} --current_node_ip {current_node_ip}".format(
            node_ips=node_ips, node_id=node_id, current_node_ip=current_node_ip)

        default_env = os.environ.copy()
        cmd = "python -m paddle.distributed.launch {distributed_args} multi_process.py".format(
            distributed_args=distributed_args)
        print("cmd", cmd)

        self.assertEqual(os.system(cmd), 0)


if __name__ == '__main__':
    unittest.main()
