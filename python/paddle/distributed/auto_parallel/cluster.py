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
    NON_ACCELERATOR_TYPE = [DeviceType.CPU, DeviceType.NIC, DeviceType.UNKNOWN]

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

    default_hop = 1
    default_nic_bandwith = 24

    def __init__(self, source, target):
        self._src = source
        self._tgt = target
        self._type = None
        # bandwidth is stored by GB/s 
        self._bandwidth = None
        # latency is stored by millisecond 
        self._latency = None
        self._hop = None

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

    @property
    def hop(self):
        return self._hop

    @hop.setter
    def hop(self, value):
        self._hop = value

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
        self._accelerators = {}
        self._non_accelerator_cumulative_count = 0

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

    @property
    def accelerators(self):
        return self._accelerators

    def add_device(self, device):
        # Use the device global_id as the key
        self._devices[device.global_id] = device
        if device.type not in Device.NON_ACCELERATOR_TYPE:
            self._accelerators[device.global_id] = device

    def add_link(self, link):
        # Use the source device global_id and target device global_id as the key
        self._links[(link.source.global_id, link.target.global_id)] = link

    def get_link(self, source_global_id, target_global_id):
        return self._links.get((source_global_id, target_global_id), None)

    def __str__(self):
        str = ""
        for device in self.devices.values():
            str += ", device: {}".format(device)
        for link in self.links.values():
            str += ", link: {}".format(link)
        return str

    def __repr__(self):
        return self.__str__()


class AlphaLatency:
    def __init__(self, alpha_latency):
        assert isinstance(alpha_latency, dict)
        self._base = alpha_latency.get("base", None)
        self._inter = alpha_latency.get("inter", None)
        self._intra = alpha_latency.get("intra", None)
        self._switch = alpha_latency.get("switch", None)
        if self._switch is not None:
            try:
                self._switch = float(self._switch)
            except:
                raise TypeError("The switch latency must be float")
        self._base_ring = self._base.get(
            "ring", None) if self._base is not None else None
        self._base_tree = self._base.get(
            "tree", None) if self._base is not None else None
        self._base_inter = self._base.get(
            "inter", None) if self._base is not None else None
        if self._base_ring is not None:
            try:
                self._base_ring = float(self._base_ring)
            except:
                raise TypeError("The base ring latency must be float.")
        if self._base_tree is not None:
            try:
                self._base_tree = float(self._base_tree)
            except:
                raise TypeError("The base ring latency must be float.")

        self._inter_ring = self._inter.get("ring", None)
        self._inter_tree = self._inter.get("tree", None)
        self._intra_ring = self._intra.get("ring", None)
        self._intra_tree = self._intra.get("tree", None)

        if self._inter_ring is not None:
            if isinstance(self._inter_ring, str):
                assert self._inter_ring in ["NET"]
                self._inter_ring = LinkType[self._inter_ring]
            else:
                try:
                    self._inter_ring = float(self._inter_ring)
                except:
                    raise TypeError("The inter ring latency must be float.")

        if self._inter_tree is not None:
            if isinstance(self._inter_tree, str):
                assert self._inter_tree in ["NET"]
                self._inter_tree = LinkType[self._inter_tree]
            else:
                try:
                    self._inter_tree = float(self._inter_tree)
                except:
                    raise TypeError("The inter tree latency must be float.")

        if self._intra_ring is not None:
            if isinstance(self._intra_ring, str):
                assert self._intra_ring in ["NVL", "PHB"]
                self._intra_ring = LinkType[self._intra_ring]
            else:
                try:
                    self._intra_ring = float(self._intra_ring)
                except:
                    raise TypeError("The intra ring latency must be float.")

        if self._intra_tree is not None:
            if isinstance(self._intra_tree, str):
                assert self._intra_tree in ["NVL", "PHB"]
                self._intra_tree = LinkType[self._intra_tree]
            else:
                try:
                    self._intra_tree = float(self._intra_tree)
                except:
                    raise TypeError("The intra tree latency must be float.")

    @property
    def base_ring(self):
        return self._base_ring

    @property
    def base_tree(self):
        return self._base_tree

    @property
    def switch(self):
        return self._switch

    @property
    def inter_ring(self):
        return self._inter_ring

    @property
    def inter_tree(self):
        return self._inter_tree

    @property
    def intra_ring(self):
        return self._intra_ring

    @property
    def intra_tree(self):
        return self._intra_tree


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
        # Latency for communication cost model
        self._alpha_latency = None
        self._rank_to_device_id = {}
        self._device_id_to_rank = {}

    @property
    def rank_to_device_id(self):
        return self._rank_to_device_id

    @property
    def device_id_to_rank(self):
        return self._device_id_to_rank

    @property
    def machines(self):
        return self._machines

    def add_machine(self, machine):
        assert isinstance(machine, Machine)
        self._machines[machine.id] = machine

        # map rank to device id and map device id to rank
        if machine.id != 0:
            prev_machine = self._machines[machine.id - 1]
            offset = prev_machine._non_accelerator_cumulative_count
            for global_id in machine.devices:
                if machine.devices[
                        global_id].type not in Device.NON_ACCELERATOR_TYPE:
                    rank_id = global_id - offset
                    self._rank_to_device_id[rank_id] = global_id
                    self._device_id_to_rank[global_id] = rank_id
            machine._non_accelerator_cumulative_count = len(
                machine.devices) - len(
                    machine.accelerators
                ) + prev_machine._non_accelerator_cumulative_count
        else:
            for global_id in machine.devices:
                if machine.devices[
                        global_id].type not in Device.NON_ACCELERATOR_TYPE:
                    rank_id = global_id
                    self._rank_to_device_id[rank_id] = global_id
                    self._device_id_to_rank[global_id] = rank_id
                    machine.accelerators[global_id] = machine.devices[global_id]
            machine._non_accelerator_cumulative_count = len(
                machine.devices) - len(machine.accelerators)

    @property
    def alpha_latency(self):
        return self._alpha_latency

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
                link.hop = link_info.get("hop", None)
                if link.hop is None:
                    # Set the default of hop: If in the same machine, hop is 0. And if in the different machine, hop is 1.
                    source_machine = source.machine
                    target_machine = target.machine
                    if source_machine.id == target_machine.id:
                        link.hop = 0
                    else:
                        link.hop = Link.default_hop
                self.add_link(link)

        if "alpha_latency" in cluster_info:
            self._alpha_latency = AlphaLatency(
                cluster_info.get("alpha_latency"))
        else:
            self._alpha_latecy = None

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

    def get_beta(self, source_device_id, target_device_id):
        # beta means the time transferring a byte, us/B
        beta = None
        convert_base = 1000
        device = self.get_device(source_device_id)
        machine = device.machine
        link = machine.get_link(source_device_id, target_device_id)
        bandwidth = None
        # None means the source and target are not connected directly, set NIC in default
        if link is None:
            bandwidth = Link.default_nic_bandwith
        else:
            bandwidth = link.bandwidth

        if bandwidth == 0.:
            beta = 0
        else:
            beta = 1 / (bandwidth * (convert_base**3 / 10**6))

        return beta

    def get_hop(self, source_device_id, target_device_id):
        beta = None
        hop = None
        device = self.get_device(source_device_id)
        machine = device.machine
        link = machine.get_link(source_device_id, target_device_id)
        if link is not None:
            hop = link.hop
        else:
            hop = Link.default_hop
        return hop

    def cross_machine(self, device_ids):
        machine_ids = set()
        for device_id in device_ids:
            device = self.get_device(device_id)
            machine_id = device.machine.id
            machine_ids.add(machine_id)
        if len(machine_ids) == 1:
            return False
        else:
            return True

    def convert_rank_to_device_id(self, group_ranks):
        # group_ranks is global id of the rank in paddle
        # task will use all of machine in this cluster with accelerators in default
        device_ids = []
        for rank in group_ranks:
            device_ids.append(self.rank_to_device_id[rank])
        return device_ids

    def get_involved_machine_count(self, device_ids):
        machine_ids = set()
        for device_id in device_ids:
            device = self.get_device(device_id)
            machine_id = device.machine.id
            machine_ids.add(machine_id)
        count = len(machine_ids)
        assert count > 0
        return count

    def __str__(self):
        str = ""
        for machine in self.machines.values():
            str += "machine: {}\n".format(machine)
        return str

    def __repr__(self):
        return self.__str__()
