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


class DistributedTensor:
    def __init__(self, serial_tensor, dist_attr=None):
        self._serial_tensor = serial_tensor
        self._dist_attr = None
        self._batch_dim = 0
        # Reuse the dist_attr setter to initialize _dist_attr
        self.dist_attr = dist_attr

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
