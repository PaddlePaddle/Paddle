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

import re
import unittest

import paddle.distributed as dist
from paddle.distributed.flex_checkpoint.aoa.aoa_engine import AOAEngine
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    ShardedWeightDesc,
)


def build_src_state_shard_info_1():
    fc1_1 = ShardedWeightDesc(
        key="fc1.weight",
        local_shape=(1, 2),
        global_shape=(2, 2),
        global_offset=(0, 0),
    )
    fc1_2 = ShardedWeightDesc(
        key="fc1.weight",
        local_shape=(1, 2),
        global_shape=(2, 2),
        global_offset=(1, 0),
    )
    fc2_1 = ShardedWeightDesc(
        key="fc2.weight",
        local_shape=(2, 1),
        global_shape=(2, 2),
        global_offset=(0, 0),
    )
    fc2_2 = ShardedWeightDesc(
        key="fc2.weight",
        local_shape=(2, 1),
        global_shape=(2, 2),
        global_offset=(0, 1),
    )
    return {
        "fc1.weight": [fc1_1, fc1_2],
        "fc2.weight": [fc2_1, fc2_2],
    }


def build_dst_state_shard_info_1():
    fc3_1 = ShardedWeightDesc(
        key="fc3.weight",
        local_shape=(1, 4),
        global_shape=(2, 4),
        global_offset=(0, 0),
    )
    fc3_2 = ShardedWeightDesc(
        key="fc3.weight",
        local_shape=(1, 4),
        global_shape=(2, 4),
        global_offset=(1, 0),
    )
    return {
        "fc3.weight": [fc3_1, fc3_2],
    }


def build_src_state_shard_info_2():
    fc4_1 = ShardedWeightDesc(
        key="fc4.weight",
        local_shape=(512, 1024),
        global_shape=(512, 1024),
        global_offset=(0, 0),
    )
    return {
        "fc4.weight": [fc4_1],
    }


def build_dst_state_shard_info_2():
    fc5_1 = ShardedWeightDesc(
        key="fc5.weight",
        local_shape=(1024, 512),
        global_shape=(1024, 512),
        global_offset=(0, 0),
    )
    return {
        "fc5.weight": [fc5_1],
    }


def parse_slice_tuple(text: str):
    # text 形如: "(slice(0, 1, 1), slice(0, 8, 1))"
    nums = re.findall(
        r"slice\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text
    )
    return tuple(slice(int(a), int(b), int(c)) for a, b, c in nums)


class TestSaveLoadParamShapeMismatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.init_parallel_env()

    def test_merge_tensor_shape_mismatch(self):
        aoa_statements = [
            "fc1.weight,fc2.weight -> fc3.weight,axis = 1 \n",
        ]
        aoa_config = {"aoa_statements": aoa_statements}
        src_state_shard_info = build_src_state_shard_info_1()
        dst_state_shard_info = build_dst_state_shard_info_1()
        tgt_desc_1 = ShardedWeightDesc(
            key="fc3.weight",
            local_shape=(1, 8),
            global_shape=(2, 8),
            global_offset=(0, 0),
        )
        tgt_desc_2 = ShardedWeightDesc(
            key="fc3.weight",
            local_shape=(1, 8),
            global_shape=(2, 8),
            global_offset=(1, 0),
        )

        aoa_engine = AOAEngine(
            aoa_config, src_state_shard_info, dst_state_shard_info
        )
        with self.assertRaises(AssertionError) as cm:
            aoa_engine.find_shard_sources(tgt_desc_1)

        msg = str(cm.exception)
        m = re.search(
            r"current model (?:state dict )?param slice range:\s*(\(.+?\))\s*,\s*preloaded weights param slice range:\s*(\(.+?\))\s*,\s*Please check",
            msg,
        )
        self.assertIsNotNone(m)
        current_slices = parse_slice_tuple(m.group(1))
        preloaded_slices = parse_slice_tuple(m.group(2))
        self.assertEqual(current_slices[1].stop, 8)
        self.assertEqual(preloaded_slices[1].stop, 4)

        with self.assertRaises(AssertionError) as cm:
            aoa_engine.find_shard_sources(tgt_desc_2)

        msg = str(cm.exception)
        m = re.search(
            r"current model (?:state dict )?param slice range:\s*(\(.+?\))\s*,\s*preloaded weights param slice range:\s*(\(.+?\))\s*,\s*Please check",
            msg,
        )
        self.assertIsNotNone(m)
        current_slices = parse_slice_tuple(m.group(1))
        preloaded_slices = parse_slice_tuple(m.group(2))
        self.assertEqual(current_slices[1].stop, 8)
        self.assertEqual(preloaded_slices[1].stop, 4)

    def test_forget_transpose_before_merge(self):
        src_state_shard_info = build_src_state_shard_info_2()
        dst_state_shard_info = build_dst_state_shard_info_2()
        dst_desc_5 = ShardedWeightDesc(
            key="fc5.weight",
            local_shape=(1024, 512),
            global_shape=(1024, 512),
            global_offset=(0, 0),
        )
        aoa_statements = [
            "fc4.weight -> fc5.weight  \n",
        ]
        aoa_config = {"aoa_statements": aoa_statements}
        aoa_engine = AOAEngine(
            aoa_config, src_state_shard_info, dst_state_shard_info
        )
        with self.assertRaises(AssertionError) as cm:
            aoa_engine.find_shard_sources(dst_desc_5)

        msg = str(cm.exception)
        m = re.search(
            r"current model (?:state dict )?param slice range:\s*(\(.+?\))\s*,\s*preloaded weights param slice range:\s*(\(.+?\))\s*,\s*Please check",
            msg,
        )
        self.assertIsNotNone(m)
        current_slices = parse_slice_tuple(m.group(1))
        preloaded_slices = parse_slice_tuple(m.group(2))
        self.assertEqual(current_slices[0].stop, 1024)
        self.assertEqual(preloaded_slices[0].stop, 512)


if __name__ == "__main__":
    unittest.main()
