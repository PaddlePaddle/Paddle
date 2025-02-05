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

import json
import os
import subprocess
import warnings

import paddle


def call_cmd(cmd, err_msg, default_value):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
    )
    stdout, stderr = process.communicate()
    if stderr:
        warnings.warn(err_msg)
        stdout = default_value

    return stdout


class SingleNodeTopology:
    def __init__(self):
        self.pcie_latency = 0.0
        self.pcie_bandwidth = float('inf')
        self.nvlink_bandwidth = -1.0
        self.nb_devices = 8

        self.machine = {}
        self.devices = []
        self.links = []
        self.json_object = None

    def calculate_cpu_flops(self):
        # Get number sockets
        cmd = "lscpu | grep 'Socket(s)' | awk '{print $NF}'"
        err_msg = "Failed to get number of sockets"
        default_value = 4
        nb_sockets = call_cmd(cmd, err_msg, default_value)

        # Get number of cores per socket
        cmd = "lscpu | grep 'Core(s) per socket' | awk '{print $NF}'"
        err_msg = "Failed to get number of cores per socket"
        default_value = 20
        nb_cores_per_socket = call_cmd(cmd, err_msg, default_value)

        # Get clock speed
        cmd = "lscpu | grep GHz | awk -F '@' '{print $NF}' | awk -F 'G' '{print $1}'"
        err_msg = "Failed to get cpu clock rate"
        default_value = 2.4
        clock_rate = call_cmd(cmd, err_msg, default_value)

        # Get number of FMA units
        # TODO(changtao02): find a way to detect this value
        nb_fmas = 2

        # Get SIMD width
        simd_width_sp = 0
        simd_width_dp = 0

        cmd = "lscpu | grep sse"
        err_msg = "Failed to get cpu vector size"
        default_value = "sse"
        vector_size = call_cmd(cmd, err_msg, default_value)

        if vector_size:
            simd_width_sp = 4  # 128 / 32
            simd_width_dp = 2  # 128 / 64

        cmd = "lscpu | grep avx2"
        err_msg = "Failed to get cpu vector size"
        default_value = "avx2"
        vector_size = call_cmd(cmd, err_msg, default_value)

        if vector_size:
            simd_width_sp = 8  # 256 / 32
            simd_width_dp = 4  # 256 / 64

        cmd = "lscpu | grep avx512"
        err_msg = "Failed to get cpu vector size"
        default_value = "avx512"
        vector_size = call_cmd(cmd, err_msg, default_value)

        if vector_size:
            simd_width_sp = 16  # 512 / 32
            simd_width_dp = 8  # 512 / 64

        gflops_per_element = (
            int(nb_sockets)
            * int(nb_cores_per_socket)
            * float(clock_rate)
            * nb_fmas
        )
        sp_gflops = gflops_per_element * simd_width_sp
        dp_gflops = gflops_per_element * simd_width_dp

        self.machine['sp_gflops'] = sp_gflops
        self.machine['dp_gflops'] = dp_gflops

    def pcie_gen2bandwidth(self, pcie_generation):
        if pcie_generation == 1:
            return 0.25
        elif pcie_generation == 2:
            return 0.5
        elif pcie_generation == 3:
            return 1.0
        elif pcie_generation == 4:
            return 2.0
        elif pcie_generation == 5:
            return 4.0
        elif pcie_generation == 6:
            return 8.0

    def model2gflops(self, model):
        if "H100" in model and "SXM5" in model:
            return 60000, 30000
        elif "H100" in model and "PCIe" in model:
            return 48000, 24000
        elif "A100" in model:
            return 19500, 9700
        elif "A800" in model:
            return 19500, 9700
        elif "V100" in model:
            return 15700, 7800
        elif "P100" in model:
            return 10600, 5300

    def get_link_bandwidth(self, source_id, target_id):
        # Get link type
        row_id = 2 + source_id
        column_id = 2 + target_id

        cmd = (
            "cat /tmp/matrix.txt | awk 'FNR=="
            + str(row_id)
            + " {print $"
            + str(column_id)
            + "}'"
        )
        err_msg = "Failed to get topo matrix"
        default_value = "NVL"
        link_type = call_cmd(cmd, err_msg, default_value)

        link_bandwidth = self.pcie_bandwidth

        if "NV" in link_type:
            if self.nvlink_bandwidth == -1.0:
                cmd = "nvidia-smi nvlink -s -i 0 | tail -n 1 | awk '{print $3}'"
                err_msg = "Failed to get nvlink bandwidth"
                default_value = "25"
                self.nvlink_bandwidth = float(
                    call_cmd(cmd, err_msg, default_value)
                )

            link_bandwidth = int(link_type[2:]) * self.nvlink_bandwidth
            link_type = "NVL"

        return link_type, link_bandwidth

    def get_host_info(self):
        # Get hostname
        cmd = "hostname -s"
        err_msg = "Failed to get hostname"
        default_value = "localhost"
        hostname = call_cmd(cmd, err_msg, default_value).strip()

        # Get ip address
        cmd = "hostname -i"
        err_msg = "Failed to get host ip address"
        default_value = "127.0.0.1"
        ip_addr = call_cmd(cmd, err_msg, default_value).strip()

        # Get CPU memory (GB)
        cmd = "cat /proc/meminfo | grep 'MemAvailable' | awk -F ':' '{print $NF}' | awk '{print $1}'"
        err_msg = "Failed to get cpu memory"
        default_value = "41366484"
        cpu_memory = int(call_cmd(cmd, err_msg, default_value)) // 1e6

        # Get single-point flops and double-point flops (GFLOPs)
        self.calculate_cpu_flops()

        self.machine['hostname'] = hostname
        self.machine['addr'] = ip_addr
        self.machine['memory'] = cpu_memory

    def get_device_info(self):
        # Get device count
        cmd = "nvidia-smi -L | wc -l"
        err_msg = "Failed to get device count"
        default_value = "8"
        self.nb_devices = int(call_cmd(cmd, err_msg, default_value))

        local_size = int(os.getenv("PADDLE_LOCAL_SIZE"))
        if local_size < self.nb_devices:
            self.nb_devices = local_size
        # Get PCIe latency and bandwidth (ms, GB/s)
        for i in range(self.nb_devices):
            cmd = (
                "nvidia-smi --id="
                + str(i)
                + " --query-gpu=pcie.link.gen.max --format=csv,noheader"
            )
            err_msg = "Failed to get max pcie link generation"
            default_value = "4"
            pcie_generation = int(call_cmd(cmd, err_msg, default_value))

            cmd = (
                "nvidia-smi --id="
                + str(i)
                + " --query-gpu=pcie.link.width.max --format=csv,noheader"
            )
            err_msg = "Failed to get max pcie link width"
            default_value = "16"
            pcie_width = int(call_cmd(cmd, err_msg, default_value))

            self.pcie_bandwidth = min(
                self.pcie_bandwidth,
                self.pcie_gen2bandwidth(pcie_generation) * pcie_width,
            )

        dev_global_ids = []
        dev_local_ids = []
        dev_types = []
        dev_models = []
        dev_memories = []  # GiB
        dev_sp_gflops = []  # GB/s
        dev_dp_gflops = []  # GB/s

        # Get device info
        rank_first = paddle.paddle.distributed.get_rank()
        for i in range(self.nb_devices):
            dev_global_ids.append(i + rank_first)
            dev_local_ids.append(i)
            dev_types.append("GPU")

            cmd = (
                "nvidia-smi --id="
                + str(i)
                + " --query-gpu=name --format=csv,noheader"
            )
            err_msg = "Failed to get device name"
            default_value = "NVIDIA A100-SXM4-40GB"
            dev_models.append(call_cmd(cmd, err_msg, default_value).strip())

            cmd = (
                "nvidia-smi --id="
                + str(i)
                + " --query-gpu=memory.free --format=csv,noheader | awk '{print $1}'"
            )
            err_msg = "Failed to get device available memory"
            default_value = "40536"
            dev_memories.append(
                int(call_cmd(cmd, err_msg, default_value)) // 1e3
            )

            sp_gflops, dp_gflops = self.model2gflops(dev_models[i])
            dev_sp_gflops.append(sp_gflops)
            dev_dp_gflops.append(dp_gflops)

        for i in range(len(dev_global_ids)):
            device = {}
            device['global_id'] = dev_global_ids[i]
            device['local_id'] = dev_local_ids[i]
            device['type'] = dev_types[i]
            device['model'] = dev_models[i]
            device['memory'] = dev_memories[i]
            device['sp_gflops'] = dev_sp_gflops[i]
            device['dp_gflops'] = dev_dp_gflops[i]
            self.devices.append(device)

        self.machine['latency'] = self.pcie_latency
        self.machine['bandwidth'] = self.pcie_bandwidth
        self.machine['device_type'] = dev_types[0]
        self.machine['device_type_full'] = f"{dev_types[0]}-{dev_models[0]}"
        self.machine['devices'] = self.devices

    def get_link_info(self):
        link_source_global_ids = []
        link_target_global_ids = []
        link_types = []
        link_latencies = []  # ms
        link_bandwidths = []  # GB/s

        cmd = "nvidia-smi topo -m > /tmp/matrix.txt"
        err_msg = "Failed to get topo matrix"
        default_value = ""
        call_cmd(cmd, err_msg, default_value)

        rank_first = paddle.paddle.distributed.get_rank()
        # Get link info between devices
        for i in range(self.nb_devices):
            for j in range(self.nb_devices):
                if i == j:
                    link_types.append("X")
                    link_bandwidths.append(-1.0)
                else:
                    link_source_global_ids.append(i + rank_first)
                    link_target_global_ids.append(j + rank_first)
                    link_latencies.append(0.0)
                    if i > j:
                        index = j * self.nb_devices + i
                        link_types.append(link_types[index])
                        link_bandwidths.append(link_bandwidths[index])
                    elif i < j:
                        link_type, link_bandwidth = self.get_link_bandwidth(
                            i, j
                        )
                        link_types.append(link_type)
                        link_bandwidths.append(link_bandwidth)

        for i in reversed(range(self.nb_devices)):
            link_types.pop(i * self.nb_devices + i)
            link_bandwidths.pop(i * self.nb_devices + i)

        cmd = "rm /tmp/matrix.txt"
        err_msg = "Failed to delete matrix.txt"
        default_value = ""
        # call_cmd(cmd, err_msg, default_value)

        for i in range(len(link_types)):
            link = {}
            link['source_global_id'] = link_source_global_ids[i]
            link['target_global_id'] = link_target_global_ids[i]
            link['type'] = link_types[i]
            link['latency'] = link_latencies[i]
            link['bandwidth'] = link_bandwidths[i]
            self.links.append(link)

        self.machine['links'] = self.links

    def detect(self):
        # Get host info
        self.get_host_info()

        # Get device info
        self.get_device_info()

        # Get link info between devices
        self.get_link_info()

        self.json_object = json.dumps(self.machine, indent=4)

    def dump(self, output_path):
        with open(output_path, "w") as outfile:
            json.dump(self.machine, outfile, indent=4)
