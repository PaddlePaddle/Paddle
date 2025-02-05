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

import unittest

import numpy as np

import paddle
from paddle.base import core, framework


def get_default_generator():
    """Get default generator for different devices."""
    place = framework._current_expected_place()
    if isinstance(place, core.CPUPlace):
        return core.default_cpu_generator()
    elif isinstance(place, core.CUDAPlace):
        return core.default_cuda_generator(0)
    elif isinstance(place, core.XPUPlace):
        return core.default_xpu_generator(0)
    elif isinstance(place, core.CustomPlace):
        return core.default_custom_device_generator(
            core.CustomPlace(place.get_device_type(), 0)
        )


def convert_state_to_seed_offset(state):
    """Get seed and offset from state."""
    device, seed, offset = (int(i) for i in str(state).split(' ')[:3])
    return np.array([device, seed, offset])


def generate_random_number_and_states(gen):
    """Concatenate random number and state for compare."""
    ret = []
    for i in range(3):
        x = paddle.uniform([10], dtype="float32", min=0.0, max=1.0).numpy()
        state = convert_state_to_seed_offset(gen.get_state())
        ret.append(np.concatenate([x, state]))
    return np.array(ret)


class TestRandomGeneratorSetGetState(unittest.TestCase):
    def test_random_generator_set_get_state(self):
        """Test Generator Get/Set state with Index."""
        paddle.seed(102)
        gen = get_default_generator()
        orig_state = gen.get_state()

        x = generate_random_number_and_states(gen)

        assert_array_equal = lambda x, y: np.testing.assert_array_equal(x, y)

        paddle.seed(102)

        assert_array_equal(x, generate_random_number_and_states(gen))

        gen.set_state(orig_state)

        assert_array_equal(x, generate_random_number_and_states(gen))

        gen.set_state(orig_state)
        orig_index = gen.get_state_index()
        new_index = gen.register_state_index(orig_state)

        assert_array_equal(x, generate_random_number_and_states(gen))

        gen.set_state_index(orig_index)

        assert_array_equal(x, generate_random_number_and_states(gen))


if __name__ == "__main__":
    unittest.main()
