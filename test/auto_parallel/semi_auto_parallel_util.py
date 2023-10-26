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


class SemiAutoParallelTestBase:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def check_tensor_eq(self, a, b):
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
            tmp, tmp_structure = self.flatten(inputs[i])
            flattened.extend(tmp)
            structure.append(tmp_structure)

        if isinstance(inputs, list):
            structure = tuple(structure)
        return flattened, structure

    def unflatten(self, inputs, structure, offset=0):
        """
        inputs may be single tensor
        """
        assert isinstance(inputs, list)
        assert offset < len(inputs)
        if structure == "i":
            assert len(inputs) == 1
            offset = offset + 1
            # return a list
            return inputs, offset
        assert isinstance(structure, (tuple, list))
        unflattened = []
        for i in range(len(structure)):
            tmp, offset = self.unflatten(inputs, structure[i], offset)
            unflattened.append(tmp)
        if isinstance(inputs, tuple):
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
            input.stop_gradient = False
            input_dist_attr = dist.DistAttr(
                mesh=self._mesh, sharding_specs=spec
            )
            dist_input = dist.shard_tensor(input, dist_attr=input_dist_attr)
            dist_input.stop_gradient = False
            flat_inputs.append(input)
            flat_dist_inputs.append(dist_input)

        inputs = self.unflatten(flat_inputs, inputs_structure)
        dist_inputs = self.unflatten(flat_dist_inputs, inputs_structure)
        out = op_func(**inputs, **kwargs)
        dist_out = op_func(**dist_inputs, **kwargs)

        if with_backward:

            def terminal_cond2(x):
                return not isinstance(x, (list, tuple))

            flat_out = self.flatten(out, terminal_cond2)
            flat_dist_out = self.flatten(dist_out, terminal_cond2)
            assert len(flat_out) == len(flat_dist_out)
            for output, dist_output in zip(flat_out, flat_dist_out):
                self.check_tensor_eq(out, dist_out)
                output.backward()
                dist_output.backward()

            for x, dist_x in zip(flat_inputs, flat_dist_inputs):
                self.check_tensor_eq(x.grad, dist_x.grad)

        if isinstance(dist_inputs, tuple) and len(dist_inputs) == 1:
            (dist_inputs,) = dist_inputs

        return dist_inputs, dist_out
