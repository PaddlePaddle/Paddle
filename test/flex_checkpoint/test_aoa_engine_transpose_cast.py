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

from paddle.distributed.flex_checkpoint.aoa.aoa_engine import (
    AOAEngine,
    ShardedWeightDesc,
    ShardMappingEntry,
)


class TestAOAEngineTransposeCast(unittest.TestCase):
    def setUp(self):
        self.setup_statements()
        self.aoa_engine = AOAEngine(
            aoa_config={"aoa_statements": self.aoa_statements},
            source_state_shard_info=self.source_state_shard_info,
            destination_state_shard_info=self.destination_state_shard_info,
        )
        self.generate_query_answer()

    def setup_statements(self):
        s0 = ShardedWeightDesc(
            key="s0",
            local_shape=(2, 2),
            global_shape=(2, 2),
            global_offset=(0, 0),
        )
        s1 = ShardedWeightDesc(
            key="s1",
            local_shape=(2, 2),
            global_shape=(2, 2),
            global_offset=(0, 0),
        )

        d0 = ShardedWeightDesc(
            key="d0",
            local_shape=(4, 1),
            global_shape=(4, 1),
            global_offset=(0, 0),
        )
        d1 = ShardedWeightDesc(
            key="d1",
            local_shape=(4, 1),
            global_shape=(4, 1),
            global_offset=(0, 0),
        )

        self.source_state_shard_info = {
            "s0": [s0],
            "s1": [s1],
        }
        self.destination_state_shard_info = {
            "d0": [d0],
            "d1": [d1],
        }

        self.aoa_statements = [
            "s0, s1 -> s, axis = 1 \n",
            "s -> s, dtype = 'float32'\n",
            "s^T -> d\n",
            "d -> d0, d1, axis = 1",
        ]

    def generate_query_answer(self):
        self.queries = []
        self.answers = []

        # ======================================================
        # Query 1:
        query = ShardedWeightDesc(
            key="d0",
            local_shape=(2, 1),
            global_shape=(4, 1),
            global_offset=(2, 0),
        )
        src_sharded_weight_desc = ShardedWeightDesc(
            key="s1",
            local_shape=(1, 2),
            global_shape=(2, 2),
            global_offset=(0, 0),
        )
        shard_mapping_entry = ShardMappingEntry(
            target_slice=query,
            source_slice=src_sharded_weight_desc,
            postprocess_list=["float32", "[1, 0]"],
        )
        answer = [shard_mapping_entry]
        self.queries.append(query)
        self.answers.append(answer)

        # ======================================================
        # Query 2:
        query = ShardedWeightDesc(
            key="d1",
            local_shape=(2, 1),
            global_shape=(4, 1),
            global_offset=(0, 0),
        )
        src_sharded_weight_desc = ShardedWeightDesc(
            key="s0",
            local_shape=(1, 2),
            global_shape=(2, 2),
            global_offset=(1, 0),
        )
        shard_mapping_entry = ShardMappingEntry(
            target_slice=query,
            source_slice=src_sharded_weight_desc,
            postprocess_list=["float32", "[1, 0]"],
        )
        answer = [shard_mapping_entry]
        self.queries.append(query)
        self.answers.append(answer)

        # ======================================================
        # Query 3:
        query = ShardedWeightDesc(
            key="d1",
            local_shape=(4, 1),
            global_shape=(4, 1),
            global_offset=(0, 0),
        )

        # d1[0:2, :] <--- s0[1, :]^T
        src_sharded_weight_desc0 = ShardedWeightDesc(
            key="s0",
            local_shape=(1, 2),
            global_shape=(2, 2),
            global_offset=(1, 0),
        )
        dst_sharded_weight_desc0 = ShardedWeightDesc(
            key="d1",
            local_shape=(2, 1),
            global_shape=(4, 1),
            global_offset=(0, 0),
        )

        # d1[2:4, :] <--- s1[1, :]^T
        src_sharded_weight_desc1 = ShardedWeightDesc(
            key="s1",
            local_shape=(1, 2),
            global_shape=(2, 2),
            global_offset=(1, 0),
        )
        dst_sharded_weight_desc1 = ShardedWeightDesc(
            key="d1",
            local_shape=(2, 1),
            global_shape=(4, 1),
            global_offset=(2, 0),
        )

        shard_mapping_entry0 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc0,
            source_slice=src_sharded_weight_desc0,
            postprocess_list=["float32", "[1, 0]"],
        )
        shard_mapping_entry1 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc1,
            source_slice=src_sharded_weight_desc1,
            postprocess_list=["float32", "[1, 0]"],
        )
        answer = [shard_mapping_entry0, shard_mapping_entry1]
        self.queries.append(query)
        self.answers.append(answer)

    def test_transpose(self):
        for idx in range(len(self.queries)):
            query = self.queries[idx]
            answer = self.answers[idx]
            result = self.aoa_engine.find_shard_sources(query)
            self.assertEqual(result, answer)


class TestAOAEngineTransposeCast2(TestAOAEngineTransposeCast):
    def setup_statements(self):
        s0 = ShardedWeightDesc(
            key="s0",
            local_shape=(4, 1),
            global_shape=(4, 1),
            global_offset=(0, 0),
        )
        s1 = ShardedWeightDesc(
            key="s1",
            local_shape=(4, 1),
            global_shape=(4, 1),
            global_offset=(0, 0),
        )

        d0 = ShardedWeightDesc(
            key="d0",
            local_shape=(2, 2),
            global_shape=(2, 2),
            global_offset=(0, 0),
        )
        d1 = ShardedWeightDesc(
            key="d1",
            local_shape=(2, 2),
            global_shape=(2, 2),
            global_offset=(0, 0),
        )

        self.source_state_shard_info = {
            "s0": [s0],
            "s1": [s1],
        }
        self.destination_state_shard_info = {
            "d0": [d0],
            "d1": [d1],
        }

        self.aoa_statements = [
            "s0^T -> s0\n",
            "s1^T -> s1\n",
            "s0, s1 -> s, axis = 0\n",
            "s -> s, dtype = 'float16'\n",
            "s -> d0, d1, axis = 1",
        ]

    def generate_query_answer(self):
        self.queries = []
        self.answers = []

        # ======================================================
        # Query 1:
        query = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 2),
            global_shape=(2, 2),
            global_offset=(1, 0),
        )
        src_sharded_weight_desc = ShardedWeightDesc(
            key="s1",
            local_shape=(2, 1),
            global_shape=(4, 1),
            global_offset=(0, 0),
        )
        shard_mapping_entry = ShardMappingEntry(
            target_slice=query,
            source_slice=src_sharded_weight_desc,
            postprocess_list=["[1, 0]", "float16"],
        )
        answer = [shard_mapping_entry]
        self.queries.append(query)
        self.answers.append(answer)

        # ======================================================
        # Query 2:
        query = ShardedWeightDesc(
            key="d1",
            local_shape=(1, 2),
            global_shape=(2, 2),
            global_offset=(0, 0),
        )
        src_sharded_weight_desc = ShardedWeightDesc(
            key="s0",
            local_shape=(2, 1),
            global_shape=(4, 1),
            global_offset=(2, 0),
        )
        shard_mapping_entry = ShardMappingEntry(
            target_slice=query,
            source_slice=src_sharded_weight_desc,
            postprocess_list=["[1, 0]", "float16"],
        )
        answer = [shard_mapping_entry]
        self.queries.append(query)
        self.answers.append(answer)

        # ======================================================
        # Query 3:
        query = ShardedWeightDesc(
            key="d1",
            local_shape=(2, 2),
            global_shape=(2, 2),
            global_offset=(0, 0),
        )

        # d1[0:1, :] <--- s0[2:4, :]^T
        src_sharded_weight_desc0 = ShardedWeightDesc(
            key="s0",
            local_shape=(2, 1),
            global_shape=(4, 1),
            global_offset=(2, 0),
        )
        dst_sharded_weight_desc0 = ShardedWeightDesc(
            key="d1",
            local_shape=(1, 2),
            global_shape=(2, 2),
            global_offset=(0, 0),
        )

        # d1[1:2, :] <--- s1[2:4, :]^T
        src_sharded_weight_desc1 = ShardedWeightDesc(
            key="s1",
            local_shape=(2, 1),
            global_shape=(4, 1),
            global_offset=(2, 0),
        )
        dst_sharded_weight_desc1 = ShardedWeightDesc(
            key="d1",
            local_shape=(1, 2),
            global_shape=(2, 2),
            global_offset=(1, 0),
        )

        shard_mapping_entry0 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc0,
            source_slice=src_sharded_weight_desc0,
            postprocess_list=["[1, 0]", "float16"],
        )
        shard_mapping_entry1 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc1,
            source_slice=src_sharded_weight_desc1,
            postprocess_list=["[1, 0]", "float16"],
        )
        answer = [shard_mapping_entry0, shard_mapping_entry1]
        self.queries.append(query)
        self.answers.append(answer)


class TestAOAEngineTransposeCast3(TestAOAEngineTransposeCast):
    def setup_statements(self):
        s0 = ShardedWeightDesc(
            key="s0",
            local_shape=(3, 4),
            global_shape=(3, 4),
            global_offset=(0, 0),
        )

        d0 = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 6),
            global_shape=(1, 6),
            global_offset=(0, 0),
        )
        d1 = ShardedWeightDesc(
            key="d1",
            local_shape=(6, 1),
            global_shape=(6, 1),
            global_offset=(0, 0),
        )

        self.source_state_shard_info = {
            "s0": [s0],
        }
        self.destination_state_shard_info = {
            "d0": [d0],
            "d1": [d1],
        }

        self.aoa_statements = [
            "s0 -> a1, a2, a3, a4, axis = 1\n",
            "a2^T -> b2\n",
            "a3^T -> b3\n",
            "b2, b3 -> d0, axis = 1\n",
            "a3, a4 -> d1, axis = 0\n",
        ]

    def generate_query_answer(self):
        self.queries = []
        self.answers = []

        # ======================================================
        # Query 1:
        query = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 6),
            global_shape=(1, 6),
            global_offset=(0, 0),
        )
        # d0[:, 0:3] <--- s0[:, 1:2]^T
        src_sharded_weight_desc0 = ShardedWeightDesc(
            key="s0",
            local_shape=(3, 1),
            global_shape=(3, 4),
            global_offset=(0, 1),
        )
        dst_sharded_weight_desc0 = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 3),
            global_shape=(1, 6),
            global_offset=(0, 0),
        )

        # d0[:, 3:6] <--- s0[:, 2:3]^T
        src_sharded_weight_desc1 = ShardedWeightDesc(
            key="s0",
            local_shape=(3, 1),
            global_shape=(3, 4),
            global_offset=(0, 2),
        )
        dst_sharded_weight_desc1 = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 3),
            global_shape=(1, 6),
            global_offset=(0, 3),
        )

        shard_mapping_entry0 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc0,
            source_slice=src_sharded_weight_desc0,
            postprocess_list=["[1, 0]"],
        )
        shard_mapping_entry1 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc1,
            source_slice=src_sharded_weight_desc1,
            postprocess_list=["[1, 0]"],
        )
        answer = [shard_mapping_entry0, shard_mapping_entry1]
        self.queries.append(query)
        self.answers.append(answer)

        # ======================================================
        # Query 2:
        query = ShardedWeightDesc(
            key="d1",
            local_shape=(6, 1),
            global_shape=(6, 1),
            global_offset=(0, 0),
        )
        # d1[0:3, :] <--- s0[:, 2:3]
        src_sharded_weight_desc0 = ShardedWeightDesc(
            key="s0",
            local_shape=(3, 1),
            global_shape=(3, 4),
            global_offset=(0, 2),
        )
        dst_sharded_weight_desc0 = ShardedWeightDesc(
            key="d1",
            local_shape=(3, 1),
            global_shape=(6, 1),
            global_offset=(0, 0),
        )

        # d1[3:6, :] <--- s0[:, 3:4]
        src_sharded_weight_desc1 = ShardedWeightDesc(
            key="s0",
            local_shape=(3, 1),
            global_shape=(3, 4),
            global_offset=(0, 3),
        )
        dst_sharded_weight_desc1 = ShardedWeightDesc(
            key="d1",
            local_shape=(3, 1),
            global_shape=(6, 1),
            global_offset=(3, 0),
        )

        shard_mapping_entry0 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc0,
            source_slice=src_sharded_weight_desc0,
            postprocess_list=None,
        )
        shard_mapping_entry1 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc1,
            source_slice=src_sharded_weight_desc1,
            postprocess_list=None,
        )
        answer = [shard_mapping_entry0, shard_mapping_entry1]
        self.queries.append(query)
        self.answers.append(answer)


class TestAOAEngineTransposeCast4(TestAOAEngineTransposeCast):
    def setup_statements(self):
        s0 = ShardedWeightDesc(
            key="s0",
            local_shape=(4, 1, 3),
            global_shape=(4, 1, 3),
            global_offset=(0, 0, 0),
        )
        s1 = ShardedWeightDesc(
            key="s1",
            local_shape=(4, 1, 3),
            global_shape=(4, 1, 3),
            global_offset=(0, 0, 0),
        )

        d0 = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 4, 4),
            global_shape=(1, 4, 4),
            global_offset=(0, 0, 0),
        )
        d1 = ShardedWeightDesc(
            key="d1",
            local_shape=(1, 4, 2),
            global_shape=(1, 4, 2),
            global_offset=(0, 0, 0),
        )

        self.source_state_shard_info = {
            "s0": [s0],
            "s1": [s1],
        }
        self.destination_state_shard_info = {
            "d0": [d0],
            "d1": [d1],
        }

        self.aoa_statements = [
            "s0, s1 -> s, axis = 1\n",
            "s -> s, dtype = 'bfloat16'\n",
            "s -> a, permute = '[2, 0, 1]'\n",
            "a -> b1, b2, b3, axis = 0\n",
            "b1 -> b1, permute = '[0, 2, 1]'\n",
            "b2 -> b2, permute = '[0, 2, 1]'\n",
            "b1, b2 -> d0, axis = 1\n",
            "b3 -> d1\n",
            "d1 -> d1, dtype = 'float32'",
        ]

    def generate_query_answer(self):
        self.queries = []
        self.answers = []

        # ======================================================
        # Query 1:
        query = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 4, 4),
            global_shape=(1, 4, 4),
            global_offset=(0, 0, 0),
        )
        # d0[:, 0:1, :] <--- s0[:, :, 0:1].transpose([2, 0, 1]).transpose([0, 2, 1])
        src_sharded_weight_desc0 = ShardedWeightDesc(
            key="s0",
            local_shape=(4, 1, 1),
            global_shape=(4, 1, 3),
            global_offset=(0, 0, 0),
        )
        dst_sharded_weight_desc0 = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 1, 4),
            global_shape=(1, 4, 4),
            global_offset=(0, 0, 0),
        )

        # d0[:, 1:2, :] <--- s1[:, :, 0:1].transpose([2, 0, 1]).transpose([0, 2, 1])
        src_sharded_weight_desc1 = ShardedWeightDesc(
            key="s1",
            local_shape=(4, 1, 1),
            global_shape=(4, 1, 3),
            global_offset=(0, 0, 0),
        )
        dst_sharded_weight_desc1 = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 1, 4),
            global_shape=(1, 4, 4),
            global_offset=(0, 1, 0),
        )

        # d0[:, 2:3, :] <--- s0[:, :, 1:2].transpose([2, 0, 1]).transpose([0, 2, 1])
        src_sharded_weight_desc2 = ShardedWeightDesc(
            key="s0",
            local_shape=(4, 1, 1),
            global_shape=(4, 1, 3),
            global_offset=(0, 0, 1),
        )
        dst_sharded_weight_desc2 = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 1, 4),
            global_shape=(1, 4, 4),
            global_offset=(0, 2, 0),
        )

        # d0[:, 3:4, :] <--- s1[:, :, 1:2].transpose([2, 0, 1]).transpose([0, 2, 1])
        src_sharded_weight_desc3 = ShardedWeightDesc(
            key="s1",
            local_shape=(4, 1, 1),
            global_shape=(4, 1, 3),
            global_offset=(0, 0, 1),
        )
        dst_sharded_weight_desc3 = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 1, 4),
            global_shape=(1, 4, 4),
            global_offset=(0, 3, 0),
        )

        shard_mapping_entry0 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc0,
            source_slice=src_sharded_weight_desc0,
            postprocess_list=["bfloat16", "[2, 0, 1]", "[0, 2, 1]"],
        )
        shard_mapping_entry1 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc1,
            source_slice=src_sharded_weight_desc1,
            postprocess_list=["bfloat16", "[2, 0, 1]", "[0, 2, 1]"],
        )
        shard_mapping_entry2 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc2,
            source_slice=src_sharded_weight_desc2,
            postprocess_list=["bfloat16", "[2, 0, 1]", "[0, 2, 1]"],
        )
        shard_mapping_entry3 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc3,
            source_slice=src_sharded_weight_desc3,
            postprocess_list=["bfloat16", "[2, 0, 1]", "[0, 2, 1]"],
        )
        answer = [
            shard_mapping_entry0,
            shard_mapping_entry1,
            shard_mapping_entry2,
            shard_mapping_entry3,
        ]
        self.queries.append(query)
        self.answers.append(answer)

        # ======================================================
        # Query 2:
        query = ShardedWeightDesc(
            key="d1",
            local_shape=(1, 4, 2),
            global_shape=(1, 4, 2),
            global_offset=(0, 0, 0),
        )
        # d1[:, :, 0:1] <--- s0[:, :, 2:3].transpose([2, 0, 1])
        src_sharded_weight_desc0 = ShardedWeightDesc(
            key="s0",
            local_shape=(4, 1, 1),
            global_shape=(4, 1, 3),
            global_offset=(0, 0, 2),
        )
        dst_sharded_weight_desc0 = ShardedWeightDesc(
            key="d1",
            local_shape=(1, 4, 1),
            global_shape=(1, 4, 2),
            global_offset=(0, 0, 0),
        )

        # d1[:, :, 1:2] <--- s1[:, :, 2:3].transpose([2, 0, 1])
        src_sharded_weight_desc1 = ShardedWeightDesc(
            key="s1",
            local_shape=(4, 1, 1),
            global_shape=(4, 1, 3),
            global_offset=(0, 0, 2),
        )
        dst_sharded_weight_desc1 = ShardedWeightDesc(
            key="d1",
            local_shape=(1, 4, 1),
            global_shape=(1, 4, 2),
            global_offset=(0, 0, 1),
        )

        shard_mapping_entry0 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc0,
            source_slice=src_sharded_weight_desc0,
            postprocess_list=["bfloat16", "[2, 0, 1]", "float32"],
        )
        shard_mapping_entry1 = ShardMappingEntry(
            target_slice=dst_sharded_weight_desc1,
            source_slice=src_sharded_weight_desc1,
            postprocess_list=["bfloat16", "[2, 0, 1]", "float32"],
        )
        answer = [shard_mapping_entry0, shard_mapping_entry1]
        self.queries.append(query)
        self.answers.append(answer)


if __name__ == '__main__':
    unittest.main()
