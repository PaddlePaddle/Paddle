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

import numpy as np


class ProcessMesh:
    def __init__(self, shape, process_group):
        """
        A class to describe the logical topology of all processes.

        shape (list): a list to describe process topology
        process_group (list): a list of processes belonging to this group
        Examples:
            dp_degree=pp_degree=mp_degree=2
            shape = ProcessMesh([dp_degree, pp_degree, mp_degree])
        """
        process_num = np.prod(shape)
        assert len(process_group) == process_num, \
            "ProcessMesh must have same processes as the process group argument. "
        self._shape = shape
        self._process_group = process_group

    def get_ndim(self):
        return len(self._shape)

    def get_shape(self):
        return self._shape

    def get_process_group(self):
        return self._process_group

    def __str__(self):
        str = "shape {} and process group {}".format(self._shape,
                                                     self._process_group)
        return str

    def __eq__(self, other):
        """Overrides the default __eq__ implementation"""
        if isinstance(other, ProcessMesh):
            if len(self._shape) != len(other._shape):
                return False
            if len(self._process_group) != len(other._process_group):
                return False
            for i in range(len(self._shape)):
                if self._shape[i] != other._shape[i]:
                    return False
            for i in range(len(self._process_group)):
                if self._process_group[i] != other._process_group[i]:
                    return False
            return True
        return NotImplemented

    def __ne__(self, other):
        """Overrides the default __ne__ implementation (unnecessary in Python 3)"""
        result = self.__eq__(other)
        if result is not NotImplemented:
            return not result
        return NotImplemented
