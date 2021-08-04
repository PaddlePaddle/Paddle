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

from enum import IntEnum


class ProcessorKind(IntEnum):
    UNKNOWN = 0
    CPU = 1
    GPU = 2


class MemoryKind(IntEnum):
    UNKNOWN = 0
    HOST_MEM = 1
    DEVICE_MEM = 2


# The following info classes should act like c-struct.
# These classes may be re-implemented by nametuple (immutable) or dataclass (mutable)
class ProcessInfo:
    def __init__(self):
        self.kind = ProcessorKind.UNKNOWN


class MemoryInfo:
    def __init__(self):
        self.kind = ProcessorKind.UNKNOWN
        self.capacity = -1


class MachineInfo:
    def __init__(self):
        self.addr = None
        self.port = None


class ProcessorMemoryAffinity:
    def __init__(self, processor, memory, bandwidth, latency):
        self._proc = processor
        self._mem = memory
        self._bandwidth = bandwidth
        self._latency = latency

    @property
    def processor(self):
        return self._proc

    @property
    def memory(self):
        return self._mem

    @property
    def bandwidth(self):
        return self._bandwidth

    # @bandwidth.setter
    # def bandwidth(self, value):
    #     self._bandwidth = value

    @property
    def latency(self):
        return self._latency

    # @latency.setter
    # def latency(self, value):
    #     self._latency = value


class MemoryMemoryAffinity:
    def __init__(self, src_mem, tgt_mem, bandwidth, latency):
        self._src_mem = src_mem
        self._tgt_mem = tgt_mem
        self._bandwidth = bandwidth
        self._latency = latency

    @property
    def source_memory(self):
        return self._src_mem

    @property
    def target_memory(self):
        return self._tgt_mem

    @property
    def bandwidth(self):
        return self._bandwidth

    # @bandwidth.setter
    # def bandwidth(self, value):
    #     self._bandwidth = value

    @property
    def latency(self):
        return self._latency

    # @latency.setter
    # def latency(self, value):
    #     self._latency = value


class Processor:
    def __init__(self, machine_id, processor_id, processor_info):
        self._machine_id = machine_id
        self._proc_id = processor_id
        self._proc_info = processor_info
        self._proc_mem_affinities_all = {}
        self._proc_mem_affinities_local = {}

    @property
    def id(self):
        return self._proc_id

    @property
    def owner_machine_id(self):
        return self._machine_id

    def add_proc_mem_affinity(self, proc_mem_affinity):
        memory = proc_mem_affinity.memory
        self._proc_mem_affinities_all[memory.id] = proc_mem_affinity
        if self.owner_machine_id == memory.owner_machine_id:
            self._proc_mem_affinities_local[memory.id] = proc_mem_affinity


class Memory:
    def __init__(self, machine_id, memory_id, memory_info):
        self._machine_id = machine_id
        self._mem_id = memory_id
        self._mem_info = memory_info
        self._proc_mem_affinities_all = {}
        self._proc_mem_affinities_local = {}
        self._mem_mem_affinities_in_all = {}
        self._mem_mem_affinities_in_local = {}
        self._mem_mem_affinities_out_all = {}
        self._mem_mem_affinities_out_local = {}

    @property
    def id(self):
        return self._mem_id

    @property
    def owner_machine_id(self):
        return self._machine_id

    @property
    def info(self):
        return self._mem_info

    def add_proc_mem_affinity(self, proc_mem_affinity):
        processor = proc_mem_affinity.processor
        self._proc_mem_affinities_all[processor.id] = proc_mem_affinity
        if self.owner_machine_id == processor.owner_machine_id:
            self._proc_mem_affinities_local[processor.id] = proc_mem_affinity

    def add_mem_mem_affinity(self, mem_mem_affinity):
        source_memory = mem_mem_affinity.source_memory
        target_memory = mem_mem_affinity.target_memory
        if source_memory.id == self.id:
            self._mem_mem_affinities_out_all[
                target_memory.id] = mem_mem_affinity
            if self.owner_machine_id == target_memory.owner_machine_id:
                self._mem_mem_affinities_in_local[
                    target_memory.id] = mem_mem_affinity
        elif target_memory.id == self.id:
            self._mem_mem_affinities_in_all[source_memory.id] = mem_mem_affinity
            if self.owner_machine_id == source_memory.owner_machine_id:
                self._mem_mem_affinities_in_local[
                    source_memory.id] = mem_mem_affinity


class Machine:
    def __init__(self, machine_id, machine_info):
        self._machine_id = machine_id
        self._machine_info = machine_info
        self._num_procs = 0
        self._num_mems = 0
        self._procs = {}
        self._mems = {}

    @property
    def id(self):
        return self._machine_id

    @property
    def info(self):
        return self._machine_info

    # private
    @property
    def _next_processor_id(self):
        self._num_procs = self._num_procs + 1
        return self._num_procs

    # private
    @property
    def _next_memory_id(self):
        self._num_procs = self._num_procs + 1
        return self._num_procs

    def add_processor(self, processor_info):
        processor = Processor(self.id, self._next_processor_id, processor_info)
        self._procs[processor.id] = processor

    def add_memory(self, memory_info):
        memory = Memory(self.id, self._next_memory_id, memory_info)
        self._mems[memory.id] = memory


class Cluster:
    def __init__(self):
        self._num_machines = 0
        self._machines = {}
        self._proc_mem_affinities = []
        self._mem_mem_affinities = []

    # private
    @property
    def _next_machine_id(self):
        self._num_machines = self._num_machines + 1
        return self._num_machines

    def build_from_config_file(self, config_file):
        """
        # Expect the configuration file has the following format, for example
        # machine info
        machine1_id, machine1_info
        machine2_id, machine1_info
        machine3_id, machine1_info
        ...
        # processor info
        (machine1_id, processor1_id), processor1_info
        (machine1_id, processor2_id), processor2_info
        (machine2_id, processor1_id), processor1_info
        (machine3_id, processor1_id), processor1_info
        ...
        # memory info 
        (machine1_id, memory1_id), memory1_info
        (machine1_id, memory2_id), memory2_info
        (machine2_id, memory1_id), memory1_info
        (machine3_id, memory1_id), memory1_info
        ...
        # affinity info 
        (machine1_id, processor1_id), (machine1_id, memory1_id), bandwidth, latency
        (machine1_id, processor2_id), (machine1_id, memory2_id), bandwidth, latency
        (machine1_id, memory1_id), (machine1_id, memory2_id), bandwidth, latency
        (machine1_id, memory2_id), (machine1_id, memory1_id), bandwidth, latency
        (machine2_id, processor1_id), (machine2_id, memory1_id), bandwidth, latency
        (machine3_id, processor1_id), (machine3_id, memory1_id), bandwidth, latency
        (machine1_id, memory1_id), (machine2_id, memory1_id), bandwidth, latency
        (machine2_id, memory1_id), (machine3_id, memory1_id), bandwidth, latency
        ...
        """
        pass

    def has_proc_mem_affinity(self, processor, memory):
        pass

    def has_mem_mem_affinity(self, memory1, memory2):
        pass

    def get_proc_mem_affinity(self, processor, memory, local_only=True):
        pass

    def get_mem_mem_affinity(self, memory1, memory2, local_only=True):
        pass

    def add_machine(self, machine_info):
        machine = Machine(self._next_machine_id, machine_info)
        self._machines[machine.id] = machine

    def add_processor(self, machine, procssor_info):
        pass

    def add_memory(self, machine, memory_info):
        pass

    def add_proc_mem_affinity(self, proc_mem_affinity):
        pass

    def add_mem_mem_affinity(self, mem_mem_affinity):
        pass
