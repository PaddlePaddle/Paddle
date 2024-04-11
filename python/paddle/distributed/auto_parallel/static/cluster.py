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

import json
import logging
import os
import re
from enum import IntEnum, unique

import paddle

from ...utils.log_utils import get_logger


@unique
class DeviceType(IntEnum):
    UNKNOWN = 0
    CPU = 1
    GPU = 2
    XPU = 3
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
        # Half precision GFLOPS
        self._hp_gflops = None
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
    def hp_gflops(self):
        return self._hp_gflops

    @hp_gflops.setter
    def hp_gflops(self, value):
        self._hp_gflops = value

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, value):
        self._memory = value

    def __str__(self):
        str = ""
        str += f"global_id: {self.global_id}, local_id: {self.local_id}, machine_id: {self.machine.id}, type: {self.type.name}, model: {self.model}, dp_flops: {self.dp_gflops}, sp_flops: {self.sp_gflops}, hp_flops: {self.hp_gflops}, memory: {self.memory}"
        return str

    def __repr__(self):
        return self.__str__()


class Link:
    default_hop = 1
    default_nic_bandwidth = 24

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
        str += f"source_global_id: {self.source.global_id}, target_global_id: {self.target.global_id}, type: {self.type}, bandwidth: {self.bandwidth}, latency: {self.latency}"
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
            str += f", device: {device}"
        for link in self.links.values():
            str += f", link: {link}"
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
        self._base_ring = (
            self._base.get("ring", None) if self._base is not None else None
        )
        self._base_tree = (
            self._base.get("tree", None) if self._base is not None else None
        )
        self._base_inter = (
            self._base.get("inter", None) if self._base is not None else None
        )
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
        # This property only be valid when the cluster consists of machines,
        # which have the same number accelerators.
        self._num_devices_per_machine = None
        self._gpu_model = None

    def gen_default_config_cluster(
        self,
        gpu_model="V100",
        cpu_model="6271C",
        node_count=1,
        device_count=1,
        gpu_memory=32,
        cpu_memory=503,
        inter_bandwidth=24,
        intra_bandwidth=235,
        gpu_dp_gflops=7800,
        gpu_sp_gflops=15700,
        gpu_hp_gflops=31400,
        cpu_dp_gflops=75,
        cpu_sp_gflops=150,
    ):
        """Generate cluster by default config."""
        gpu_models = ["V100", "A100", "H100", "A2", "A10", "A16", "A30", "A40"]
        xpu_models = ["XPU"]
        dcu_models = ["DCU"]
        all_gpu_models = gpu_models + xpu_models + dcu_models
        self._num_devices_per_machine = device_count
        self._gpu_model = gpu_model

        def _convert_to_type(gpu_model):
            type = None
            if gpu_model in gpu_models:
                type = "GPU"
            elif gpu_model in xpu_models:
                type = "XPU"
            elif gpu_model in dcu_models:
                type = "DCU"
            else:
                type = "GPU"
            assert type is not None

            return type

        def _convert_to_model(gpu_model, gpu_memory):
            model = None
            if gpu_model == "V100":
                model = "Tesla V100-SXM2-" + str(gpu_memory) + "GB"
            elif gpu_model == "A100":
                model = "Tesla A100-SXM-" + str(gpu_memory) + "GB"
            elif gpu_model == "A30":
                model = "Tesla A30-SXM-" + str(gpu_memory) + "GB"
            else:
                model = gpu_model + str(gpu_memory) + "GB"
            assert model is not None

            return model

        def _convert_to_cpu_info(cpu_model):
            arch, vendor, model = None, None, None
            if cpu_model == "6271C":
                arch = "x86_64"
                vendor = "GenuineIntel"
                model = "Intel(R) Xeon(R) Gold 6271C CPU @ 2.60G"
            elif cpu_model == "6148":
                arch = "x86_64"
                vendor = "GenuineIntel"
                model = "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40G"
            assert arch is not None
            assert vendor is not None
            assert model is not None

            return arch, vendor, model

        cluster_info = {}
        cluster_info["machines"] = []
        global_id = 0
        global_id_to_device_type = {}
        global_id_to_node = {}
        # NOTE: It will support NPU, XPU, DCU models in the future, it is just a fake value now
        for i in range(node_count):
            machine = {}
            # NOTE: The hostname is host_0, host_1, ...
            machine["hostname"] = "host_" + str(i)
            # NOTE: The addr is localhost, if need actual addr, it should be reset manually
            machine["addr"] = "127.0.0.1"
            # NOTE: The port is a default value
            machine["port"] = 60009
            machine["links"] = []

            devices = []
            local_id = 0

            for j in range(device_count):
                device = {}
                global_id = global_id if i == 0 and j == 0 else global_id + 1

                local_id += 1
                type = _convert_to_type(gpu_model)
                model = _convert_to_model(gpu_model, gpu_memory)
                memory = gpu_memory

                device["global_id"] = global_id
                device["local_id"] = local_id
                device["type"] = type
                device["model"] = model
                device["memory"] = memory
                device["sp_gflops"] = gpu_sp_gflops
                device["dp_gflops"] = gpu_dp_gflops
                device["hp_gflops"] = gpu_hp_gflops
                # hard code
                device["type"] = "GPU"
                global_id_to_device_type[global_id] = type
                global_id_to_node[global_id] = i
                devices.append(device)

            # add cpu device and nic device, just one cpu
            cpu_device = {}
            arch, vendor, model = _convert_to_cpu_info(cpu_model)
            sp_gflops = cpu_sp_gflops
            dp_gflops = cpu_dp_gflops
            global_id += 1
            local_id = 0
            memory = cpu_memory
            type = "CPU"
            cpu_device["arch"] = arch
            cpu_device["vendor"] = vendor
            cpu_device["model"] = model
            cpu_device["sp_gflops"] = sp_gflops
            cpu_device["dp_gflops"] = dp_gflops
            cpu_device["global_id"] = global_id
            cpu_device["local_id"] = local_id
            cpu_device["memory"] = memory
            cpu_device["type"] = type
            global_id_to_node[global_id] = i
            global_id_to_device_type[global_id] = type
            devices.append(cpu_device)

            nic_device = {}
            global_id += 1

            # add NIC
            type = "NIC"
            width = 12.5
            ip = "127.0.0.1"
            local_id = 0
            nic_device["type"] = type
            nic_device["local_id"] = type
            nic_device["global_id"] = global_id
            global_id_to_device_type[global_id] = type
            global_id_to_node[global_id] = i
            devices.append(nic_device)
            machine["devices"] = devices
            cluster_info["machines"].append(machine)

        # build link
        for i in range(0, global_id + 1):
            for j in range(0, global_id + 1):
                if i == j:
                    continue
                node_id_i = global_id_to_node[i]
                node_id_j = global_id_to_node[j]
                device_type_i = global_id_to_device_type[i]
                device_type_j = global_id_to_device_type[j]
                link = {}
                source_global_id = i
                target_global_id = j
                link["source_global_id"] = source_global_id
                link["target_global_id"] = target_global_id
                # the same node and device_type, set intra_bandwidth, NVL
                if node_id_i == node_id_j and device_type_i == device_type_j:
                    link["type"] = "NVL"
                    link["bandwidth"] = intra_bandwidth
                else:
                    link["type"] = "PHB"
                    link["bandwidth"] = inter_bandwidth
                cluster_info["machines"][node_id_i]["links"].append(link)

        self._build_from_dict(cluster_info)

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
                if (
                    machine.devices[global_id].type
                    not in Device.NON_ACCELERATOR_TYPE
                ):
                    rank_id = global_id - offset
                    self._rank_to_device_id[rank_id] = global_id
                    self._device_id_to_rank[global_id] = rank_id
            machine._non_accelerator_cumulative_count = (
                len(machine.devices)
                - len(machine.accelerators)
                + prev_machine._non_accelerator_cumulative_count
            )
        else:
            for global_id in machine.devices:
                if (
                    machine.devices[global_id].type
                    not in Device.NON_ACCELERATOR_TYPE
                ):
                    rank_id = global_id
                    self._rank_to_device_id[rank_id] = global_id
                    self._device_id_to_rank[global_id] = rank_id
                    machine.accelerators[global_id] = machine.devices[global_id]
            machine._non_accelerator_cumulative_count = len(
                machine.devices
            ) - len(machine.accelerators)

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

    def _build_from_dict(self, cluster_info):
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
                device.hp_gflops = float(device_info.get("hp_gflops", 0))
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
                cluster_info.get("alpha_latency")
            )
        else:
            self._alpha_latency = None

    def build_from_file(self, json_file_path):
        with open(json_file_path) as json_file:
            cluster_info = json.load(json_file)
        self._build_from_dict(cluster_info)

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
            bandwidth = Link.default_nic_bandwidth
        else:
            bandwidth = link.bandwidth

        if bandwidth == 0.0:
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

    def get_num_machines(self):
        return len(self._machines)

    def get_num_devices_per_machine(self):
        # Only return the number of accelerators of each machine.
        # All machines must has the same number of devices and same type of devices.
        assert self._num_devices_per_machine
        return self._num_devices_per_machine

    def __str__(self):
        str = ""
        for machine in self.machines.values():
            str += f"machine: {machine}\n"
        return str

    def __repr__(self):
        return self.__str__()


logger = get_logger(logging.INFO)


def get_default_cluster(json_config=None):
    def is_by_json_config(json_config):
        if not json_config:
            return False
        if "cluster" not in json_config:
            return False
        else:
            if "path" not in json_config["cluster"]:
                if "num_nodes" not in json_config["cluster"]:
                    return False
                if "num_gpus" not in json_config["cluster"]:
                    return False
                if "gpu_model" not in json_config["cluster"]:
                    return False
                if "gpu_memory" not in json_config["cluster"]:
                    return False
                return True
            else:
                return True

    cluster = Cluster()
    if json_config and is_by_json_config(json_config):
        # Get GPU info by json config
        if "path" in json_config["cluster"]:
            cluster.build_from_file(json_config["cluster"]["path"])
            return cluster
        else:
            node_count = json_config["cluster"]["num_nodes"]
            local_device_count = json_config["cluster"]["num_gpus"]
            gpu_model = json_config["cluster"]["gpu_model"]
            memory = json_config["cluster"]["gpu_memory"]
    else:
        # Get GPU info by get_device_properties
        local_device_count = os.getenv("PADDLE_LOCAL_SIZE")
        if local_device_count is None:
            local_device_count = 1
        else:
            local_device_count = int(local_device_count)

        global_device_count = os.getenv("PADDLE_GLOBAL_SIZE")
        if global_device_count is None:
            node_count = 1
        else:
            global_device_count = int(global_device_count)
            assert global_device_count % local_device_count == 0
            node_count = int(global_device_count) // local_device_count

        if os.getenv("PADDLE_DISTRI_BACKEND", None) == "xccl":
            gpu_name = os.getenv("PADDLE_XCCL_BACKEND", None)
            gpu_model = gpu_name
            memory = int(
                paddle.base.core.libpaddle._get_device_total_memory(gpu_name)
            ) // (1000**3)
        else:
            gpu_info = paddle.device.cuda.get_device_properties()
            assert gpu_info, "Auto parallel just runs on gpu now."

            gpu_name = gpu_info.name
            try:
                re_result = re.split(r'[ , -]', gpu_name)
                gpu_model = re_result[1]
                memory = int(re_result[-1][:-2])
            except:
                memory = int(gpu_info.total_memory) // (1000**3)
                gpu_model = gpu_name

    logger.info(
        "Node Count: {}, Local Device Size: {}, GPU Model: {}, GPU Memory: {}GB, World size: {}, EndPoint: {}.".format(
            node_count,
            local_device_count,
            gpu_model,
            memory,
            paddle.distributed.get_world_size(),
            os.getenv("PADDLE_CURRENT_ENDPOINT", None),
        )
    )

    gflops_info = {
        "V100": {"dp": 7800, "sp": 15700, "hp": 125000},
        "A100": {"dp": 9700, "sp": 19500, "hp": 624000},
    }
    default_gflops = (
        gflops_info["A100"] if gpu_model == "A100" else gflops_info["V100"]
    )

    cluster.gen_default_config_cluster(
        node_count=node_count,
        device_count=local_device_count,
        gpu_model=gpu_model,
        gpu_memory=memory,
        gpu_dp_gflops=default_gflops["dp"],
        gpu_sp_gflops=default_gflops["sp"],
        gpu_hp_gflops=default_gflops["hp"],
    )
    return cluster
