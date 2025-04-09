#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import copy

from paddle.static import InputSpec

from ..placement_type import get_shard_spec
from .utils import convert_to_dims_mapping


class DistributedInputSpec(InputSpec):
    def __init__(
        self,
        shape,
        dtype='float32',
        name=None,
        stop_gradient=False,
        mesh=None,
        placements=None,
        local_shape=None,
    ):
        super().__init__(shape, dtype, name, stop_gradient)
        self.mesh = copy.deepcopy(mesh)
        sharding_specs = get_shard_spec(mesh, placements, len(self.shape))
        self.dims_mapping = convert_to_dims_mapping(sharding_specs, mesh)
        self.local_shape = local_shape

    @classmethod
    def from_dtensor(cls, dtensor, name=None, shape=None):
        """
        Generates a DistributedInputSpec based on dist tensor.

        Args:
            dtensor: the dist tensor.

        Returns:
            A DistributedInputSpec instance generated from dtensor.
        """
        return cls(
            shape=dtensor.shape if shape is None else shape,
            dtype=dtensor.dtype,
            name=name,
            stop_gradient=dtensor.stop_gradient,
            mesh=dtensor.process_mesh,
            placements=dtensor.placements,
            local_shape=dtensor._local_value().shape,
        )

    def __repr__(self):
        return f"{super().__repr__()}, mesh:{self.mesh}, placements:{self.dims_mapping}"
