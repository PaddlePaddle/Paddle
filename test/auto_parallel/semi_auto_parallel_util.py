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

import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.placement_type import to_placements


class SemiAutoParallelTestBase:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def check_tensor_eq(self, a, b):
        if a is None:
            assert b is None
            return
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05, verbose=True)

    def flatten(self, inputs, terminal_cond):
        """
        inputs may be single tensor、tuple
        """

        if terminal_cond(inputs):
            return [inputs], "i"

        assert isinstance(inputs, (tuple, list))
        flattened = []
        structure = []
        for i in range(len(inputs)):
            tmp, tmp_structure = self.flatten(inputs[i], terminal_cond)
            flattened.extend(tmp)
            structure.append(tmp_structure)

        if isinstance(inputs, tuple):
            structure = tuple(structure)
        return flattened, structure

    def unflatten(self, inputs, structure, offset=0):
        """
        inputs may be single tensor
        """
        assert isinstance(inputs, list)
        assert offset < len(inputs)
        if structure == "i":
            offset = offset + 1
            # return a list
            return inputs[offset - 1], offset
        assert isinstance(structure, (tuple, list))
        unflattened = []
        for i in range(len(structure)):
            tmp, offset = self.unflatten(inputs, structure[i], offset)
            unflattened.append(tmp)
        if isinstance(structure, tuple):
            unflattened = tuple(unflattened)
        return unflattened, offset

    def runfunc_and_check(
        self, inputs_shape, inputs_specs, op_func, with_backward, **kwargs
    ):
        paddle.seed(self._seed)
        np.random.seed(self._seed)

        flat_inputs = []
        flat_dist_inputs = []

        def terminal_cond(x):
            return isinstance(x, list) and all(
                not isinstance(e, (list, tuple)) for e in x
            )

        flat_inputs_specs, inputs_structure = self.flatten(
            inputs_specs, terminal_cond
        )
        flat_inputs_shape, _ = self.flatten(inputs_shape, terminal_cond)
        assert len(flat_inputs_specs) == len(flat_inputs_shape)

        for shape, spec in zip(flat_inputs_shape, flat_inputs_specs):
            input_np = np.random.random(size=shape).astype(self._dtype)
            input = paddle.to_tensor(input_np)
            input.stop_gradient = not with_backward
            # retain dist_attr here.
            input_dist_attr = dist.DistAttr(
                mesh=self._mesh, sharding_specs=spec
            )
            # for dygraph auto_parallel, get placements by using to_placements
            placements = to_placements(input_dist_attr.dims_mapping, self._mesh)
            dist_input = dist.shard_tensor(input, self._mesh, placements)
            dist_input.stop_gradient = not with_backward
            flat_inputs.append(input)
            flat_dist_inputs.append(dist_input)

        inputs, _ = self.unflatten(flat_inputs, inputs_structure)
        dist_inputs, _ = self.unflatten(flat_dist_inputs, inputs_structure)

        def wrap_tuple(e):
            return e if isinstance(e, tuple) else (e,)

        op_inputs = wrap_tuple(inputs)
        op_dist_input = wrap_tuple(dist_inputs)

        out = op_func(*op_inputs, **kwargs)
        dist_out = op_func(*op_dist_input, **kwargs)

        if with_backward:

            def terminal_cond2(x):
                return not isinstance(x, (list, tuple))

            flat_out, _ = self.flatten(out, terminal_cond2)
            flat_dist_out, _ = self.flatten(dist_out, terminal_cond2)
            assert len(flat_out) == len(flat_dist_out)
            for output, dist_output in zip(flat_out, flat_dist_out):
                self.check_tensor_eq(output, dist_output)
                if output is not None:
                    output.backward()
                    dist_output.backward()

            for x, dist_x in zip(flat_inputs, flat_dist_inputs):
                self.check_tensor_eq(x.grad, dist_x.grad)

        return dist_inputs, dist_out
