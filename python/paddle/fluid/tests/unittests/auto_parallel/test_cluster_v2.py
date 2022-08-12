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
# limitations under the License

import unittest
from paddle.distributed.auto_parallel.cluster_v2 import Device
from paddle.distributed.auto_parallel.cluster_v2 import Link
from paddle.distributed.auto_parallel.cluster_v2 import DeviceMesh


class TestDeviceMesh(unittest.TestCase):

    def test_device_mesh(self):
        name = "my_device_mesh"
        mesh = [[0, 1, 2], [3, 4, 5]]
        device_mesh = DeviceMesh(name, mesh, dim_names=["x", "y"])
        device_mesh1 = DeviceMesh("another_mesh", [0, 1, 2, 3])
        self.assertEqual(device_mesh.name, "my_device_mesh")
        self.assertEqual(device_mesh.shape, [2, 3])
        self.assertEqual(device_mesh.device_ids, [0, 1, 2, 3, 4, 5])
        self.assertEqual(device_mesh.dim_names, ["x", "y"])
        self.assertEqual(device_mesh.device_type, "UNKNOWN")
        self.assertEqual(device_mesh.size, 6)
        self.assertEqual(device_mesh.ndim, 2)
        self.assertEqual(device_mesh.dim_size(0), 2)
        self.assertEqual(device_mesh.dim_size(-1), 3)
        self.assertEqual(device_mesh.dim_size("x"), 2)
        self.assertEqual(device_mesh.dim_size("y"), 3)
        self.assertEqual(device_mesh.empty(), False)
        self.assertEqual(device_mesh.contains(0), True)
        self.assertEqual(device_mesh.contains(6), False)

        dev0 = Device(global_id=0, local_id=0, machine_id=0, type="GPU")
        dev1 = Device(global_id=1, local_id=1, machine_id=0, type="GPU")
        dev2 = Device(global_id=2, local_id=2, machine_id=0, type="GPU")
        dev3 = Device(global_id=3, local_id=0, machine_id=1, type="GPU")
        dev4 = Device(global_id=4, local_id=1, machine_id=1, type="GPU")
        dev5 = Device(global_id=5, local_id=2, machine_id=1, type="GPU")
        device_mesh.add_device(dev0)
        device_mesh.add_device(dev1)
        device_mesh.add_device(dev2)
        device_mesh.add_device(dev3)
        device_mesh.add_device(dev4)
        device_mesh.add_device(dev5)
        self.assertEqual(device_mesh.device(0), dev0)
        self.assertEqual(device_mesh.device(1), dev1)
        self.assertEqual(device_mesh.device(2), dev2)
        self.assertEqual(device_mesh.device(3), dev3)
        self.assertEqual(device_mesh.device(4), dev4)
        self.assertEqual(device_mesh.device(5), dev5)

        link0 = Link(source_id=0, target_id=1, type="NVL")
        link1 = Link(source_id=1, target_id=0, type="NVL")
        link2 = Link(source_id=3, target_id=4, type="NVL")
        link3 = Link(source_id=4, target_id=5, type="NVL")
        device_mesh.add_link(link0)
        device_mesh.add_link(link1)
        device_mesh.add_link(link2)
        device_mesh.add_link(link3)
        self.assertEqual(device_mesh.link(0, 1), link0)
        self.assertEqual(device_mesh.link(1, 0), link1)
        self.assertEqual(device_mesh.link(3, 4), link2)
        self.assertEqual(device_mesh.link(4, 5), link3)

        self.assertEqual(device_mesh.machine(0).id, 0)
        self.assertEqual(device_mesh.machine(0).contains(3), False)
        self.assertEqual(device_mesh.machine(0).device(2), dev2)
        self.assertEqual(device_mesh.machine(1).link(3, 4), link2)
        self.assertEqual(
            device_mesh.machine(0).devices,
            device_mesh.machine(0).devices)
        self.assertEqual(
            device_mesh.machine(0).links,
            device_mesh.machine(0).links)

        self.assertEqual(device_mesh.device_type, "GPU")
        self.assertEqual(device_mesh.devices, device_mesh.devices)
        self.assertEqual(device_mesh.links, device_mesh.links)
        self.assertEqual(device_mesh.machines, device_mesh.machines)
        self.assertEqual(device_mesh, device_mesh)
        self.assertNotEqual(device_mesh, device_mesh1)
        self.assertEqual(str(device_mesh), str(device_mesh))

    def test_device(self):
        device = Device(global_id=0, local_id=1, machine_id=2, type="GPU")
        device.capability.sflops = 100
        device.capability.dflops = 200
        device.capability.memory = 32
        device.capability.rate = 2
        self.assertEqual(device.global_id, 0)
        self.assertEqual(device.local_id, 1)
        self.assertEqual(device.machine_id, 2)
        self.assertEqual(device.type, "GPU")
        self.assertAlmostEqual(device.capability.sflops, 100)
        self.assertAlmostEqual(device.capability.dflops, 200)
        self.assertAlmostEqual(device.capability.memory, 32)
        self.assertAlmostEqual(device.capability.rate, 2)
        self.assertEqual(device, device)
        self.assertEqual(str(device), str(device))

    def test_link(self):
        link = Link(source_id=0, target_id=1, type="NVL")
        link.capability.bandwidth = 100
        link.capability.latency = 1
        self.assertEqual(link.source_id, 0)
        self.assertEqual(link.target_id, 1)
        self.assertEqual(link.type, "NVL")
        self.assertAlmostEqual(link.capability.bandwidth, 100)
        self.assertAlmostEqual(link.capability.latency, 1)
        self.assertEqual(link, link)
        self.assertEqual(str(link), str(link))


if __name__ == "__main__":
    unittest.main()
