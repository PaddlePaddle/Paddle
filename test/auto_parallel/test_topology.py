# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distributed.launch.utils.topology import SingleNodeTopology


def check_empty_json_object(json_object):
    return json_object is not None


class TestSingleNodeTopology(unittest.TestCase):
    def test_empty_topology_json_object(self):
        topo = SingleNodeTopology()
        topo.detect()

        self.assertTrue(check_empty_json_object(topo.json_object))


if __name__ == "__main__":
    unittest.main()
