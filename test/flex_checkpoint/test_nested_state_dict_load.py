# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import tempfile
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.flex_checkpoint.dcp.load_state_dict import (
    _metadata_manager,
)
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    make_replicated_sharded_weight,
)


class TestNestedStateDictLoad(unittest.TestCase):
    """Loading into a nested state_dict must rebuild the caller's layout.

    ``load_state_dict_impl`` flattens the target state_dict, loads into the
    flat view, then walks ``mapping`` to write the tensors back into the
    caller's nested dict. That walk needs the state_dict it was handed, so no
    loop in between may reuse the name.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_path = self._tmp.name
        _metadata_manager.clear()

    def tearDown(self):
        self._tmp.cleanup()

    def _save(self, state_dict):
        dist.save_state_dict(state_dict, self.ckpt_path)

    def test_loads_into_nested_state_dict(self):
        expected = np.arange(32, dtype=np.float32).reshape(4, 8)
        self._save(
            {
                "layer": {
                    "weight": make_replicated_sharded_weight(
                        "layer.weight", paddle.to_tensor(expected)
                    )
                }
            }
        )
        target = make_replicated_sharded_weight(
            "layer.weight", paddle.zeros([4, 8], dtype="float32")
        )
        state_dict = {"layer": {"weight": target}}

        # A non-empty aoa_config skips the local-resume fast path, so the load
        # goes through the flatten/restore round trip.
        dist.load_state_dict(
            state_dict,
            self.ckpt_path,
            aoa_config={"aoa_statements": ["layer.weight -> layer.weight"]},
        )

        np.testing.assert_array_equal(
            state_dict["layer"]["weight"].local_tensor.numpy(), expected
        )
        self.assertIs(state_dict["layer"]["weight"], target)

    def test_loads_into_deeply_nested_state_dict(self):
        expected = np.arange(6, dtype=np.float32).reshape(2, 3)
        self._save(
            {
                "opt": {
                    "master_weights": {
                        "w": make_replicated_sharded_weight(
                            "opt.master_weights.w", paddle.to_tensor(expected)
                        )
                    }
                }
            }
        )
        target = make_replicated_sharded_weight(
            "opt.master_weights.w", paddle.zeros([2, 3], dtype="float32")
        )
        state_dict = {"opt": {"master_weights": {"w": target}}}

        dist.load_state_dict(
            state_dict,
            self.ckpt_path,
            aoa_config={
                "aoa_statements": [
                    "opt.master_weights.w -> opt.master_weights.w"
                ]
            },
        )

        np.testing.assert_array_equal(
            state_dict["opt"]["master_weights"]["w"].local_tensor.numpy(),
            expected,
        )

    def test_loads_into_flat_state_dict(self):
        expected = np.arange(32, dtype=np.float32).reshape(4, 8)
        self._save(
            {
                "layer.weight": make_replicated_sharded_weight(
                    "layer.weight", paddle.to_tensor(expected)
                )
            }
        )
        target = make_replicated_sharded_weight(
            "layer.weight", paddle.zeros([4, 8], dtype="float32")
        )
        state_dict = {"layer.weight": target}

        dist.load_state_dict(
            state_dict,
            self.ckpt_path,
            aoa_config={"aoa_statements": ["layer.weight -> layer.weight"]},
        )

        np.testing.assert_array_equal(
            state_dict["layer.weight"].local_tensor.numpy(), expected
        )


if __name__ == "__main__":
    unittest.main()
