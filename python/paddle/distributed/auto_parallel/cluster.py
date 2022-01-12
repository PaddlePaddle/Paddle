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

import os
import json
from enum import IntEnum
from enum import unique


@unique
class DeviceType(IntEnum):
    UNKNOWN = 0
    CPU = 1
    GPU = 2
    XPU = 3
    NPU = 4
    DCU = 5
    NIC = 6


@unique
class LinkType(IntEnum):
    UNKNOWN = 0
    LOC = 1
    SYS = 2
    PHB = 3
    PIX = 4
    PIB = 5
    NVL = 6
    NVB = 7
    NET = 8


class Device:
    def __init__(self, global_id, local_id, machine):
        self._global_id = global_id
        self._local_id = local_id
        self._machine = machine
        self._type = None
        # Different device have different models, such as 
        # "Tesla V100-SXM2-32GB" and "A100-SXM4-40GB" etc.
        self._model = None
        # Double precision GFLOPS
        self._dp_gflops = None
        # Single precision GFLOPS
        self._sp_gflops = None
        # Memory is stored by GB 
        self._memory = None

    @property
    def global_id(self):
        return self._global_id

    @global_id.setter
    def global_id(self, value):
        self._global_id = value

    @property
    def local_id(self):
        return self._local_id

    @local_id.setter
    def local_id(self, value):
        self._local_id = value

    @property
    def machine(self):
        return self._machine

    @machine.setter
    def machine(self, value):
        self._machine = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def dp_gflops(self):
        return self._dp_gflops

    @dp_gflops.setter
    def dp_gflops(self, value):
        self._dp_gflops = value

    @property
    def sp_gflops(self):
        return self._sp_gflops

    @sp_gflops.setter
    def sp_gflops(self, value):
        self._sp_gflops = value

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, value):
        self._memory = value

    def __str__(self):
        str = ""
        str += "global_id: {}, local_id: {}, machine_id: {}, type: {}, model: {}, dp_flops: {}, sp_flops: {}, memory: {}".format(
            self.global_id, self.local_id, self.machine.id, self.type.name,
            self.model, self.dp_gflops, self.sp_gflops, self.memory)
        return str

    def __repr__(self):
        return self.__str__()


class Link:
    def __init__(self, source, target):
        self._src = source
        self._tgt = target
        self._type = None
        # bandwidth is stored by GB/s 
        self._bandwidth = None
        # latency is stored by millisecond 
        self._latency = None

    @property
    def source(self):
        return self._src

    @source.setter
    def source(self, value):
        self._source = value

    @property
    def target(self):
        return self._tgt

    @target.setter
    def target(self, value):
        self._target = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        self._bandwidth = value

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, value):
        self._latency = value

    def __str__(self):
        str = ""
        str += "source_global_id: {}, target_global_id: {}, type: {}, bandwidth: {}, latency: {}".format(
            self.source.global_id, self.target.global_id, self.type,
            self.bandwidth, self.latency)
        return str

    def __repr__(self):
        return self.__str__()


class Machine:
    def __init__(self, id):
        self._id = id
        self._hostname = None
        self._addr = None
        self._port = None
        self._devices = {}
        self._links = {}

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def hostname(self):
        return self._hostname

    @hostname.setter
    def hostname(self, value):
        self._hostname = value

    @property
    def addr(self):
        return self._addr

    @addr.setter
    def addr(self, value):
        self._addr = value

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, value):
        self._port = value

    @property
    def devices(self):
        return self._devices

    @property
    def links(self):
        return self._links

    def add_device(self, device):
        # Use the device global_id as the key
        self._devices[device.global_id] = device

    def add_link(self, link):
        # Use the source device global_id and target device global_id as the key
        self._links[(link.source.global_id, link.target.global_id)] = link

    def __str__(self):
        str = ""
        for device in self.devices.values():
            str += ", device: {}".format(device)
        for link in self.links.values():
            str += ", link: {}".format(link)
        return str

    def __repr__(self):
        return self.__str__()


class Cluster:
    """
    The cluster is an abstract of the hardware resource for training, which contains the cluster topology and
    related hardware information. It will serve the task mapping, cost model and auto searching.
    """

    def __init__(self):
        # Used to compute machine id
        self._num_machines = 0
        # Store all machines within the cluster
        self._machines = {}
        # Cluster graph topology
        self._topology = None

    @property
    def machines(self):
        return self._machines

    def add_machine(self, machine):
        assert isinstance(machine, Machine)
        self._machines[machine.id] = machine

    def add_device(self, device):
        assert isinstance(device, Device)
        device.machine.add_device(device)

    def add_link(self, link):
        assert isinstance(link, Link)
        # Only add the link to the source machine
        link.source.machine.add_link(link)

    def get_device(self, device_global_id):
        device = None
        for machine in self.machines.values():
            if device_global_id in machine.devices.keys():
                device = machine.devices[device_global_id]
        return device

    def build_from_file(self, json_file_path):
        with open(json_file_path) as json_file:
            cluster_info = json.load(json_file)
        machines_info = cluster_info["machines"]
        for machine_info in machines_info:
            machine_id = self._generate_machine_id()
            machine = Machine(machine_id)
            machine.hostname = machine_info.get("hostname")
            machine.addr = machine_info.get("addr")
            machine.port = machine_info.get("port")
            devices_info = machine_info.get("devices", [])
            for device_info in devices_info:
                device_global_id = device_info.get("global_id")
                device_local_id = device_info.get("local_id")
                device = Device(device_global_id, device_local_id, machine)
                device_type = device_info.get("type", None)
                if device_type is not None:
                    device_type = DeviceType[device_type]
                else:
                    device_type = DeviceType.UNKNOWN
                device.type = device_type
                device.model = device_info.get("model", None)
                device.dp_gflops = float(device_info.get("dp_gflops", 0))
                device.sp_gflops = float(device_info.get("sp_gflops", 0))
                device.memory = float(device_info.get("memory", 0))
                self.add_device(device)
            self.add_machine(machine)
        for machine_info in machines_info:
            links_info = machine_info.get("links", [])
            for link_info in links_info:
                source_global_id = link_info.get("source_global_id")
                target_global_id = link_info.get("target_global_id")
                source = self.get_device(source_global_id)
                target = self.get_device(target_global_id)
                link = Link(source, target)
                link_type = link_info.get("type", None)
                if link_type is not None:
                    link_type = LinkType[link_type]
                else:
                    link_type = LinkType.UNKNOWN
                link.type = link_type
                link.bandwidth = float(link_info.get("bandwidth", 0))
                link.latency = float(link_info.get("latency", 0))
                self.add_link(link)

    def _generate_machine_id(self):
        cur_machine_id = self._num_machines
        self._num_machines += 1
        return cur_machine_id

    def get_all_devices(self, device_type):
        devices = []
        for machine in self.machines.values():
            for device in machine.devices.values():
                if device.type == DeviceType[device_type]:
                    devices.append(device)
        return devices

    def __str__(self):
        str = ""
        for machine in self.machines.values():
            str += "machine: {}\n".format(machine)
        return str

    def __repr__(self):
        return self.__str__()
