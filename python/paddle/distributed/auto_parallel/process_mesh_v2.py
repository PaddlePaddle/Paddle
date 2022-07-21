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


class ProcessMesh(object):

    def __init__(self, mesh, dim_names=None, device_type="GPU"):
        if not isinstance(mesh, list) and \
           not isinstance(mesh, np.ndarray):
            raise ValueError('mesh must be an instance of list or np.ndarray.')
        if isinstance(mesh, list):
            mesh = np.array(mesh)

        self._shape = list(mesh.shape)

        self._process_ids = mesh.flatten().tolist()
        assert all(isinstance(p, int) for p in self._process_ids), \
            ("All elements of mesh must be integer")
        assert min(
            self._process_ids) >= 0, ('All elements of mesh must be >= 0.')
        unique_process_ids = set(self._process_ids)
        assert len(unique_process_ids) == len(
            self._process_ids), ('All elements of mesh must be unique.')

        if dim_names is not None:
            assert len(dim_names) == len(self._shape), \
                ("The length of dims_names must be same as the shape of mesh.")
            self._dims_names = dim_names
        else:
            self._dim_names = ["d" + str(i) for i in range(len(self._shape))]

        self._device_type = device_type

    @property
    def shape(self):
        r"""
        Get the shape belonging to this ProcessMesh.
        """
        return self._shape

    @property
    def process_ids(self):
        r"""
        Get a list of all process_ids belonging to this ProcessMesh.
        """
        return self._process_ids

    @property
    def dim_names(self):
        r"""
        Get the names of all dimensions of ProcessMesh.
        """
        return self._dim_names

    @property
    def device_type(self):
        r"""
        Get the device type of ProcessMesh.
        """
        return self._device_type

    @device_type.setter
    def device_type(self, device_type):
        r"""
        Set the device type of ProcessMesh.
        """
        self._device_type = device_type

    @property
    def ndim(self):
        r"""
        Get the number of dimension of ProcessMesh.
        """
        return len(self._shape)

    def __eq__(self, other):
        if not isinstance(other, ProcessMesh):
            return False
        if self.shape != other.shape \
                or self.process_ids != other.process_ids\
                or self.device_type != other.device_type:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        str = "{{shape:{}, process_ids:{}, dim_names:{}, device_type:{}}}".format(
            self.shape, self.process_ids, self.dim_names, self.device_type)
        return str
