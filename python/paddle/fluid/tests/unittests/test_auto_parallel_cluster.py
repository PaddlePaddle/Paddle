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

import tempfile
import unittest
import os
import json
from paddle.distributed.auto_parallel.cluster import Cluster
from paddle.distributed.auto_parallel.cluster import DeviceType
from paddle.distributed.auto_parallel.cluster import LinkType

cluster_json = """
{
  "machines": [
    {
      "hostname": "machine0",
      "addr": "0.0.0.1",
      "port": "768",
      "devices": [
        {
          "global_id": 0,
          "local_id": 0,
          "type": "GPU",
          "model": "A100-SXM4-40GB",
          "sp_gflops": 19500,
          "dp_gflops": 9700,
          "memory": 40
        },
        {
          "global_id": 1,
          "local_id": 1,
          "type": "GPU",
          "model": "A100-SXM4-40GB",
          "sp_gflops": 19500,
          "dp_gflops": 9700,
          "memory": 40
        },
        {
          "global_id": 2,
          "local_id": 0,
          "type": "CPU",
          "model": "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GH",
          "arch": "x86_64",
          "vendor": "GenuineIntel",
          "sp_gflops": 150,
          "dp_gflops": 75,
          "memory": 1510
        },
        {
          "global_id": 3,
          "local_id": 0,
          "type": "NIC"
        }
      ],
      "links": [
        {
          "source_global_id": 0,
          "target_global_id": 1,
          "type": "NVL",
          "bandwidth": 252
        },
        {
          "source_global_id": 0,
          "target_global_id": 2,
          "type": "PHB",
          "bandwidth": 12
        },
        {
          "source_global_id": 1,
          "target_global_id": 2,
          "type": "PHB",
          "bandwidth": 12
        },
        {
          "source_global_id": 0,
          "target_global_id": 3,
          "type": "NET",
          "bandwidth": 1
        },
        {
          "source_global_id": 1,
          "target_global_id": 3,
          "type": "NET",
          "bandwidth": 1
        },
        {
          "source_global_id": 2,
          "target_global_id": 3,
          "type": "NET",
          "bandwidth": 1
        },
        {
          "source_global_id": 3,
          "target_global_id": 7,
          "type": "NET",
          "bandwidth": 1
        }
      ]
    },
    {
      "hostname": "machine1",
      "addr": "0.0.0.2",
      "port": "768",
      "devices": [
        {
          "global_id": 4,
          "local_id": 0,
          "type": "GPU",
          "model": "Tesla V100-SXM2-32GB",
          "sp_gflops": 15700,
          "dp_gflops": 7800,
          "memory": 32
        },
        {
          "global_id": 5,
          "local_id": 1,
          "type": "GPU",
          "model": "Tesla V100-SXM2-32GB",
          "sp_gflops": 15700,
          "dp_gflops": 7800,
          "memory": 32
        },
        {
          "global_id": 6,
          "local_id": 0,
          "type": "CPU",
          "model": "Intel(R) Xeon(R) Gold 6271C CPU @ 2.60G",
          "arch": "x86_64",
          "vendor": "GenuineIntel",
          "sp_gflops": 150,
          "dp_gflops": 75,
          "memory": "503"
        },
        {
          "global_id": 7,
          "local_id": 0,
          "type": "NIC"
        }
      ],
      "links": [
        {
          "source_global_id": 4,
          "target_global_id": 5,
          "type": "NVL",
          "bandwidth": 42
        },
        {
          "source_global_id": 4,
          "target_global_id": 6,
          "type": "PHB",
          "bandwidth": 12
        },
        {
          "source_global_id": 5,
          "target_global_id": 6,
          "type": "PHB",
          "bandwidth": 12
        },
        {
          "source_global_id": 4,
          "target_global_id": 7,
          "type": "NET",
          "bandwidth": 1
        },
        {
          "source_global_id": 5,
          "target_global_id": 7,
          "type": "NET",
          "bandwidth": 1
        },
        {
          "source_global_id": 6,
          "target_global_id": 7,
          "type": "NET",
          "bandwidth": 1
        },
        {
          "source_global_id": 7,
          "target_global_id": 3,
          "type": "NET",
          "bandwidth": 1
        }
      ]
    }
  ]
}
"""


class TestAutoParallelCluster(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cluster(self):
        cluster_json_path = os.path.join(self.temp_dir.name,
                                         "auto_parallel_cluster.json")
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)

        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)

        self.assertEqual(len(cluster.get_all_devices("GPU")), 4)
        self.assertEqual(len(cluster.get_all_devices("CPU")), 2)
        self.assertEqual(len(cluster.get_all_devices("NIC")), 2)
        self.assertEqual(len(cluster.machines), 2)

        # machine0
        machine0 = cluster.machines[0]
        self.assertEqual(machine0.id, 0)
        self.assertEqual(machine0.hostname, "machine0")
        self.assertEqual(machine0.addr, "0.0.0.1")
        self.assertEqual(machine0.port, "768")
        self.assertEqual(len(machine0.devices), 4)
        self.assertEqual(len(machine0.links), 7)

        # device0
        device0_machine0 = machine0.devices[0]
        self.assertEqual(device0_machine0.global_id, 0)
        self.assertEqual(device0_machine0.local_id, 0)
        self.assertEqual(device0_machine0.type, DeviceType.GPU)
        self.assertEqual(device0_machine0.model, "A100-SXM4-40GB")
        self.assertAlmostEqual(device0_machine0.sp_gflops, 19500)
        self.assertAlmostEqual(device0_machine0.dp_gflops, 9700)
        self.assertAlmostEqual(device0_machine0.memory, 40)

        # device0, link0
        link0_machine0 = machine0.links[(0, 1)]
        self.assertEqual(link0_machine0.source.global_id, 0)
        self.assertEqual(link0_machine0.target.global_id, 1)
        self.assertEqual(link0_machine0.type, LinkType.NVL)
        self.assertAlmostEqual(link0_machine0.bandwidth, 252)
        self.assertAlmostEqual(link0_machine0.latency, 0)

        # device 0, link 1
        link1_machine0 = machine0.links[(0, 2)]
        self.assertEqual(link1_machine0.source.global_id, 0)
        self.assertEqual(link1_machine0.target.global_id, 2)
        self.assertEqual(link1_machine0.type, LinkType.PHB)
        self.assertAlmostEqual(link1_machine0.bandwidth, 12)
        self.assertAlmostEqual(link1_machine0.latency, 0)

        # device0, link2
        link2_machine0 = machine0.links[(0, 3)]
        self.assertEqual(link2_machine0.source.global_id, 0)
        self.assertEqual(link2_machine0.target.global_id, 3)
        self.assertEqual(link2_machine0.type, LinkType.NET)
        self.assertAlmostEqual(link2_machine0.bandwidth, 1)
        self.assertAlmostEqual(link2_machine0.latency, 0)

        # device1
        device1_machine0 = machine0.devices[1]
        self.assertEqual(device1_machine0.global_id, 1)
        self.assertEqual(device1_machine0.local_id, 1)
        self.assertEqual(device1_machine0.type, DeviceType.GPU)
        self.assertEqual(device1_machine0.model, "A100-SXM4-40GB")
        self.assertAlmostEqual(device1_machine0.sp_gflops, 19500)
        self.assertAlmostEqual(device1_machine0.dp_gflops, 9700)
        self.assertAlmostEqual(device1_machine0.memory, 40)

        # device1, link0
        link0_machine0 = machine0.links[(1, 2)]
        self.assertEqual(link0_machine0.source.global_id, 1)
        self.assertEqual(link0_machine0.target.global_id, 2)
        self.assertEqual(link0_machine0.type, LinkType.PHB)
        self.assertAlmostEqual(link0_machine0.bandwidth, 12)
        self.assertAlmostEqual(link0_machine0.latency, 0)

        # device1, link1
        link1_machine0 = machine0.links[(1, 3)]
        self.assertEqual(link1_machine0.source.global_id, 1)
        self.assertEqual(link1_machine0.target.global_id, 3)
        self.assertEqual(link1_machine0.type, LinkType.NET)
        self.assertAlmostEqual(link1_machine0.bandwidth, 1)
        self.assertAlmostEqual(link1_machine0.latency, 0)

        # device2
        device2_machine0 = machine0.devices[2]
        self.assertEqual(device2_machine0.global_id, 2)
        self.assertEqual(device2_machine0.local_id, 0)
        self.assertEqual(device2_machine0.type, DeviceType.CPU)
        self.assertEqual(device2_machine0.model,
                         "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GH")
        self.assertAlmostEqual(device2_machine0.sp_gflops, 150)
        self.assertAlmostEqual(device2_machine0.dp_gflops, 75)
        self.assertAlmostEqual(device2_machine0.memory, 1510)

        # device2, link0
        link0_machine0 = machine0.links[(2, 3)]
        self.assertEqual(link0_machine0.source.global_id, 2)
        self.assertEqual(link0_machine0.target.global_id, 3)
        self.assertEqual(link0_machine0.type, LinkType.NET)
        self.assertAlmostEqual(link0_machine0.bandwidth, 1)
        self.assertAlmostEqual(link0_machine0.latency, 0)

        # device3
        device3_machine0 = machine0.devices[3]
        self.assertEqual(device3_machine0.global_id, 3)
        self.assertEqual(device3_machine0.local_id, 0)
        self.assertEqual(device3_machine0.type, DeviceType.NIC)
        self.assertAlmostEqual(device3_machine0.model, None)
        self.assertAlmostEqual(device3_machine0.sp_gflops, 0)
        self.assertAlmostEqual(device3_machine0.dp_gflops, 0)
        self.assertAlmostEqual(device3_machine0.memory, 0)

        link0_machine0 = machine0.links[(3, 7)]
        # device3, link0
        self.assertEqual(link0_machine0.source.global_id, 3)
        self.assertEqual(link0_machine0.target.global_id, 7)
        self.assertEqual(link0_machine0.type, LinkType.NET)
        self.assertAlmostEqual(link0_machine0.bandwidth, 1)
        self.assertAlmostEqual(link0_machine0.latency, 0)

        # machine1
        machine1 = cluster.machines[1]
        self.assertEqual(machine1.id, 1)
        self.assertEqual(machine1.hostname, "machine1")
        self.assertEqual(machine1.addr, "0.0.0.2")
        self.assertEqual(machine1.port, "768")
        self.assertEqual(len(machine1.devices), 4)
        self.assertEqual(len(machine1.links), 7)

        # device4
        device4_machine1 = machine1.devices[4]
        self.assertEqual(device4_machine1.global_id, 4)
        self.assertEqual(device4_machine1.local_id, 0)
        self.assertEqual(device4_machine1.type, DeviceType.GPU)
        self.assertEqual(device4_machine1.model, "Tesla V100-SXM2-32GB")
        self.assertAlmostEqual(device4_machine1.sp_gflops, 15700)
        self.assertAlmostEqual(device4_machine1.dp_gflops, 7800)
        self.assertAlmostEqual(device4_machine1.memory, 32)

        # device4, link0
        link0_machine1 = machine1.links[(4, 5)]
        self.assertEqual(link0_machine1.source.global_id, 4)
        self.assertEqual(link0_machine1.target.global_id, 5)
        self.assertEqual(link0_machine1.type, LinkType.NVL)
        self.assertAlmostEqual(link0_machine1.bandwidth, 42)
        self.assertAlmostEqual(link0_machine1.latency, 0)

        # device 4, link 1
        link1_machine1 = machine1.links[(4, 6)]
        self.assertEqual(link1_machine1.source.global_id, 4)
        self.assertEqual(link1_machine1.target.global_id, 6)
        self.assertEqual(link1_machine1.type, LinkType.PHB)
        self.assertAlmostEqual(link1_machine1.bandwidth, 12)
        self.assertAlmostEqual(link1_machine1.latency, 0)

        # device4, link2
        link2_machine1 = machine1.links[(4, 7)]
        self.assertEqual(link2_machine1.source.global_id, 4)
        self.assertEqual(link2_machine1.target.global_id, 7)
        self.assertEqual(link2_machine1.type, LinkType.NET)
        self.assertAlmostEqual(link2_machine1.bandwidth, 1)
        self.assertAlmostEqual(link2_machine1.latency, 0)

        # device5
        device5_machine1 = machine1.devices[5]
        self.assertEqual(device5_machine1.global_id, 5)
        self.assertEqual(device5_machine1.local_id, 1)
        self.assertEqual(device5_machine1.type, DeviceType.GPU)
        self.assertEqual(device4_machine1.model, "Tesla V100-SXM2-32GB")
        self.assertAlmostEqual(device4_machine1.sp_gflops, 15700)
        self.assertAlmostEqual(device4_machine1.dp_gflops, 7800)
        self.assertAlmostEqual(device4_machine1.memory, 32)

        # device5, link0
        link0_machine1 = machine1.links[(5, 6)]
        self.assertEqual(link0_machine1.source.global_id, 5)
        self.assertEqual(link0_machine1.target.global_id, 6)
        self.assertEqual(link0_machine1.type, LinkType.PHB)
        self.assertAlmostEqual(link0_machine1.bandwidth, 12)
        self.assertAlmostEqual(link0_machine1.latency, 0)

        # device5, link1
        link1_machine1 = machine1.links[(5, 7)]
        self.assertEqual(link1_machine1.source.global_id, 5)
        self.assertEqual(link1_machine1.target.global_id, 7)
        self.assertEqual(link1_machine1.type, LinkType.NET)
        self.assertAlmostEqual(link1_machine1.bandwidth, 1)
        self.assertAlmostEqual(link1_machine1.latency, 0)

        # device6
        device6_machine1 = machine1.devices[6]
        self.assertEqual(device6_machine1.global_id, 6)
        self.assertEqual(device6_machine1.local_id, 0)
        self.assertEqual(device6_machine1.type, DeviceType.CPU)
        self.assertEqual(device6_machine1.model,
                         "Intel(R) Xeon(R) Gold 6271C CPU @ 2.60G")
        self.assertAlmostEqual(device6_machine1.sp_gflops, 150)
        self.assertAlmostEqual(device6_machine1.dp_gflops, 75)
        self.assertAlmostEqual(device6_machine1.memory, 503)

        # device6, link0
        link0_machine1 = machine1.links[(6, 7)]
        self.assertEqual(link0_machine1.source.global_id, 6)
        self.assertEqual(link0_machine1.target.global_id, 7)
        self.assertEqual(link0_machine1.type, LinkType.NET)
        self.assertAlmostEqual(link0_machine1.bandwidth, 1)
        self.assertAlmostEqual(link0_machine1.latency, 0)

        # device7
        device7_machine1 = machine1.devices[7]
        self.assertEqual(device7_machine1.global_id, 7)
        self.assertEqual(device7_machine1.local_id, 0)
        self.assertEqual(device7_machine1.type, DeviceType.NIC)
        self.assertAlmostEqual(device7_machine1.model, None)
        self.assertAlmostEqual(device7_machine1.sp_gflops, 0)
        self.assertAlmostEqual(device7_machine1.dp_gflops, 0)
        self.assertAlmostEqual(device7_machine1.memory, 0)

        # device3, link0
        link0_machine1 = machine1.links[(7, 3)]
        self.assertEqual(link0_machine1.source.global_id, 7)
        self.assertEqual(link0_machine1.target.global_id, 3)
        self.assertEqual(link0_machine1.type, LinkType.NET)
        self.assertAlmostEqual(link0_machine1.bandwidth, 1)
        self.assertAlmostEqual(link0_machine1.latency, 0)

        str = "cluster: {}".format(cluster)


if __name__ == '__main__':
    unittest.main()
