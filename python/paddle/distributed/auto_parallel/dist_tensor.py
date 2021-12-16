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
# limitations under the License

import copy
from paddle.fluid import core
from .dist_attribute import TensorDistributedAttribute
from .dist_attribute import get_tensor_dist_attr_field_keys
from utils import compute_partition_shape
from utils import compute_process_index


class DistributedTensor:
    @staticmethod
    def compute_partition_index(process, complete_shape, dims_mapping, topology,
                                processes):
        """Compute the partition index in complete tensor."""
        partition_shape = compute_partition_size(complete_shape, dims_mapping,
                                                 topology)
        process_index = compute_process_index(process, processes, topology)
        partition_index = []

        for i in range(len(complete_shape)):
            if dims_mapping[i] == -1:
                partition_index.append([0, partition_shape[i]])
            else:
                partition_index.append([
                    process_index[dims_mapping[i]] * partition_shape[i],
                    (process_index[dims_mapping[i]] + 1) * partition_shape[i]
                ])
        return partition_index

    def __init__(self, serial_tensor, dist_attr=None):
        self._serial_tensor = serial_tensor
        self._dist_attr = None
        self._batch_dim = 0
        # Reuse the dist_attr setter to initialize _dist_attr
        self.dist_attr = dist_attr
        sefl._dist_partition_info_map = {}

    @property
    def serial_tensor(self):
        return self._serial_tensor

    @property
    def dist_attr(self):
        return self._dist_attr

    @dist_attr.setter
    def dist_attr(self, dist_attr):
        if self._dist_attr is None:
            self._dist_attr = TensorDistributedAttribute()
        self._dist_attr.init(dist_attr)
        self._init_default_dist_attr()

    @property
    def dist_partition_info_map(self):
        return self._dist_partition_info_map

    def _init_default_dist_attr(self):
        if self._dist_attr.dims_mapping is None:
            if self.serial_tensor.type == core.VarDesc.VarType.READER:
                tensor_shape = []
            else:
                tensor_shape = self._serial_tensor.shape
            tensor_dims_mapping = [-1 for _ in range(len(tensor_shape))]
            self._dist_attr.dims_mapping = tensor_dims_mapping

    def validate_dist_attr(self):
        if self.serial_tensor.type == core.VarDesc.VarType.READER:
            return True
        tensor_shape = self.serial_tensor.shape
        if len(tensor_shape) != len(self.dist_attr.dims_mapping):
            return False
        for i in range(len(self.dist_attr.dims_mapping)):
            if self.dist_attr.dims_mapping[
                    i] < -1 or self.dist_attr.dims_mapping[i] >= len(
                        self.dist_attr.process_mesh.topology):
                return False
        for i in range(len(self.dist_attr.process_mesh.topology)):
            if self.dist_attr.dims_mapping.count(i) > 1:
                return False
        return True

    def get_local_partition_info(self, rank_id):
        if rank_id in self.dist_partition_info_map.keys():
            return self.dist_partition_info_map[rank_id]

        dims_mapping = self.dist_attr.dims_mapping
        process_mesh = self.dist_attr.process_mesh
        topology = process_mesh.topology
        processes = process_mesh.processes
        complete_shape = self.get_global_shape()
        partition_index = DistributedTensor.compute_partition_index(
            rank_id, complete_shape, dims_mapping, topology, processes)
        offset = []
        size = []
        for item in partition_index:
            offset.append(item[0])
            size.append(item[1] - item[0])
        partition_info = PartitionInfo(offset, size)
        self.dist_partition_info_map[rank_id] = partition_info

        return partition_info

    def get_global_shape(self):
        return self.serial_tensor.shape

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_serial_tensor":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __str__(self):
        str = "{{tensor name: {}, tensor id: {}".format(
            self.serial_tensor.desc.name(), self.serial_tensor.desc.id())

        # str += ", {}".format(self.dist_attr)
        # return str

        if self.dist_attr.is_annotated("process_mesh"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", process_mesh ({}): {}".format(annotated_str,
                                                self.dist_attr.process_mesh)

        str += ", is_parameter: {}".format(self.serial_tensor.is_parameter)

        if self.dist_attr.is_annotated("dims_mapping"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", dims_mapping ({}): {}".format(annotated_str,
                                                self.dist_attr.dims_mapping)

        if self.dist_attr.is_annotated("shard_mask"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", shard_mask ({}): {}".format(annotated_str, None)

        if self.dist_attr.is_annotated("offload_device"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", offload_device ({}): {} }}".format(annotated_str, None)
        return str


class PartitionInfo:
    def __init__(self, offset, size):
        self._offset = offset
        self._size = size

    @property
    def offset(self):
        return self._offset

    @property
    def size(self):
        return self._size
