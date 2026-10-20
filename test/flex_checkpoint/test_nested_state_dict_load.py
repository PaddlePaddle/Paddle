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

import os
import shutil
import tempfile
import unittest
from dataclasses import replace

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

    def _rebuild_with_replica(self, replicated_key, relocated_key):
        """Rewrite the checkpoint so one shard is recorded twice.

        ``save_replicas=True`` keeps every copy of a shard, tagging the primary
        with ``replica_id=0`` and the others with ``replica_id>=1``. Producing
        that needs several ranks holding the same shard, so build the layout by
        hand instead: duplicate the data file, record ``replicated_key`` in
        both, and move ``relocated_key`` into the duplicate only. The rank then
        has to read both files, which is what makes the load drop the redundant
        copy of ``replicated_key`` instead of ignoring it.
        """
        primary, replica = "0_0.distcp", "1_0.distcp"
        shutil.copyfile(
            os.path.join(self.ckpt_path, primary),
            os.path.join(self.ckpt_path, replica),
        )

        metadata_file = os.path.join(self.ckpt_path, "0.metadata")
        metadata = paddle.load(metadata_file)
        storage_metadata = {}
        for index in metadata.storage_metadata:
            if index.tensor_key == replicated_key:
                storage_metadata[replace(index, replica_id=0)] = primary
                storage_metadata[replace(index, replica_id=1)] = replica
            elif index.tensor_key == relocated_key:
                storage_metadata[replace(index, replica_id=0)] = replica
            else:
                storage_metadata[replace(index, replica_id=0)] = primary
        metadata.storage_metadata = storage_metadata
        paddle.save(metadata, metadata_file)
        _metadata_manager.clear()

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

    def test_loads_checkpoint_with_replica_entries(self):
        """A replica recorded in an already loaded file must be dropped.

        With replicas in the storage metadata the load first evicts every
        ``replica_id != 0`` shard from the files it read, so only the primary
        copy reaches the reshard step. That eviction walks the per-file tensor
        tables, which is the other place that used to shadow ``state_dict``.
        """
        expected_w1 = np.arange(8, dtype=np.float32).reshape(2, 4)
        expected_w2 = np.arange(6, dtype=np.float32).reshape(2, 3)
        self._save(
            {
                "layer": {
                    "w1": make_replicated_sharded_weight(
                        "layer.w1", paddle.to_tensor(expected_w1)
                    ),
                    "w2": make_replicated_sharded_weight(
                        "layer.w2", paddle.to_tensor(expected_w2)
                    ),
                }
            }
        )
        self._rebuild_with_replica("layer.w1", "layer.w2")

        target_w1 = make_replicated_sharded_weight(
            "layer.w1", paddle.zeros([2, 4], dtype="float32")
        )
        target_w2 = make_replicated_sharded_weight(
            "layer.w2", paddle.zeros([2, 3], dtype="float32")
        )
        state_dict = {"layer": {"w1": target_w1, "w2": target_w2}}

        dist.load_state_dict(
            state_dict,
            self.ckpt_path,
            aoa_config={
                "aoa_statements": [
                    "layer.w1 -> layer.w1",
                    "layer.w2 -> layer.w2",
                ]
            },
        )

        np.testing.assert_array_equal(
            state_dict["layer"]["w1"].local_tensor.numpy(), expected_w1
        )
        np.testing.assert_array_equal(
            state_dict["layer"]["w2"].local_tensor.numpy(), expected_w2
        )
        self.assertIs(state_dict["layer"]["w1"], target_w1)
        self.assertIs(state_dict["layer"]["w2"], target_w2)


if __name__ == "__main__":
    unittest.main()
