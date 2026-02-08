# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distributed.flex_checkpoint.aoa.aoa_engine import AOAEngine
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    ShardedWeightDesc,
)


def create_shard_info(keys, shape=(4, 4), dtype="float32"):
    info = {}
    for k in keys:
        desc = ShardedWeightDesc(
            key=k,
            local_shape=shape,
            global_shape=shape,
            global_offset=(0, 0),
            dtype=dtype,
        )
        info[k] = [desc]
    return info


class TestMacroLayerOffsetError(unittest.TestCase):
    def setUp(self):
        self.source_keys = [f"model.layers.{i}.weight" for i in range(10)]
        self.dest_keys = [f"model.layers.{i}.weight_out" for i in range(10)] + [
            f"model.layers.{i}.weight_out2" for i in range(10)
        ]

        self.src_info = create_shard_info(self.source_keys)
        self.dst_info = create_shard_info(self.dest_keys)

    def test_macro_error_chain(self):
        """
        The statement contains fused_qkv_old and is missing a comma, expecting to trigger the assertion and print the chain.
        """
        aoa_config = {
            "aoa_statements": [
                "model.layers.$LAYER_ID.weight^T -> model.layers.$LAYER_ID.weight_out, axis=0 fused_qkv_old, num_heads=20,num_key_value_groups=4",
            ],
            "enable_traceback": True,
        }

        with self.assertRaises(AssertionError):
            AOAEngine(
                aoa_config=aoa_config,
                source_state_shard_info=self.src_info,
                destination_state_shard_info=self.dst_info,
            )

    def test_no_error_should_be_raised(self):
        # No error should be raised
        source_keys = ["model.layers.0.weight"]
        dest_keys = ["model.layers.0.weight_out"]
        src_info = create_shard_info(source_keys)
        dst_info = create_shard_info(dest_keys)
        aoa_config = {
            "aoa_statements": [
                "model.layers.0.weight^T -> model.layers.0.weight_out",
            ],
            "enable_traceback": True,
        }
        AOAEngine(
            aoa_config=aoa_config,
            source_state_shard_info=src_info,
            destination_state_shard_info=dst_info,
        )

    def test_shape_propagation_error_chain(self):
        """
        when split/concat, only support one attr named `axis`, but got multiple attrs.
        """
        aoa_config = {
            "aoa_statements": [
                "model.layers.0.weight -> model.layers.0.weight_out,model.layers.0.weight_out2,axis=0,axis=1",
            ],
            "enable_traceback": True,
        }

        with self.assertRaises(ValueError):
            AOAEngine(
                aoa_config=aoa_config,
                source_state_shard_info=self.src_info,
                destination_state_shard_info=self.dst_info,
            )


if __name__ == "__main__":
    unittest.main()
