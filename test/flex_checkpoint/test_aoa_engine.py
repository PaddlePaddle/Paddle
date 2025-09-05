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


class TestAOAEngine(unittest.TestCase):
    def test_aoa_spilt_merge(self):
        # ------------------------------------------------------
        # 1. Define source tensor shards (s0 and s1).
        # Each is a (2,2) tensor, fully covering its global shape.
        #
        #  s0 (2,2):         s1 (2,2):
        #  +----+----+       +----+----+
        #  |    |    |       |    |    |
        #  +----+----+       +----+----+
        #  |    |    |       |    |    |
        #  +----+----+       +----+----+
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

        # ------------------------------------------------------
        # 2. Define destination tensor shards (d0 and d1).
        # Both are (1,4) tensors, i.e., a single row with 4 columns.
        #
        #  d0 (1,4):       d1 (1,4):
        #  +--+--+--+--+   +--+--+--+--+
        #  |  |  |  |  |   |  |  |  |  |
        #  +--+--+--+--+   +--+--+--+--+
        d0 = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 4),
            global_shape=(1, 4),
            global_offset=(0, 0),
        )
        d1 = ShardedWeightDesc(
            key="d1",
            local_shape=(1, 4),
            global_shape=(1, 4),
            global_offset=(0, 0),
        )

        # ------------------------------------------------------
        # 3. Record the shard info for sources and destinations
        source_state_shard_info = {
            "s0": [s0],
            "s1": [s1],
        }
        destination_state_shard_info = {
            "d0": [d0],
            "d1": [d1],
        }

        # ------------------------------------------------------
        # 4. AOA statements define axis mapping for concatenation and splitting:
        #    - "s" is formed by concatenating s0 and s1 along axis 1 (columns).
        #    - d0 and d1 are obtained by splitting "s" along axis 0 (rows).
        aoa_statements = [
            "s0, s1 -> s, axis = 1 \n",
            "s -> d0, d1, axis = 0 \n",
        ]

        # ------------------------------------------------------
        # 5. Create the AOAEngine with this configuration
        aoa_engine = AOAEngine(
            aoa_config={"aoa_statements": aoa_statements},
            source_state_shard_info=source_state_shard_info,
            destination_state_shard_info=destination_state_shard_info,
        )

        queries = []
        answers = []

        # ======================================================
        # Query 1: Find source for the first half of d0 (columns 0-1)
        # d0 shard: key="d0", local_shape=(1,2), global_shape=(1,4), global_offset=(0,0)
        # Covers d0[:, 0:2]
        #
        #  d0 (1,4):
        #  +------+------+------+------+
        #  |(0,0) |(0,1) |      |      |
        #  +------+------+------+------+
        #
        # This region is mapped from s0, row 0, columns 0-1
        query = ShardedWeightDesc(
            key="d0",
            local_shape=(1, 2),
            global_shape=(1, 4),
            global_offset=(0, 0),
        )
        src_sharded_weight_desc = ShardedWeightDesc(
            key="s0",
            local_shape=(1, 2),
            global_shape=(2, 2),
            global_offset=(0, 0),
        )
        shard_mapping_entry = ShardMappingEntry(
            target_slice=query,
            source_slice=src_sharded_weight_desc,
            postprocess_list=None,
        )
        answer = [shard_mapping_entry]
        queries.append(query)
        answers.append(answer)

        # ======================================================
        # Query 2: Find source for the second half of d1 (columns 2-3)
        # d1 shard: key="d1", local_shape=(1,2), global_shape=(1,4), global_offset=(0,2)
        # Covers d1[:, 2:4]
        #
        #  d1 (1,4):
        #  +------+------+------+------+
        #  |      |      |(0,2)|(0,3)|
        #  +------+------+------+------+
        #
        # This region is mapped from s1, row 1, columns 0-1
        query = ShardedWeightDesc(
            key="d1",
            local_shape=(1, 2),
            global_shape=(1, 4),
            global_offset=(0, 2),
        )
        src_sharded_weight_desc = ShardedWeightDesc(
            key="s1",
            local_shape=(1, 2),
            global_shape=(2, 2),
            global_offset=(1, 0),
        )
        shard_mapping_entry = ShardMappingEntry(
            target_slice=query,
            source_slice=src_sharded_weight_desc,
            postprocess_list=None,
        )
        answer = [shard_mapping_entry]
        queries.append(query)
        answers.append(answer)

        # ======================================================
        # Query 3: Find sources for the entire d1 (full row)
        # d1 shard: key="d1", local_shape=(1,4), global_shape=(1,4), global_offset=(0,0)
        # Layout: covers all columns
        #
        #  d1 (1,4):
        #  +------+------+------+------+
        #  | s0   | s0   | s1   | s1   |
        #  |(0,0) |(0,1) |(0,2) |(0,3) |
        #  +------+------+------+------+
        # The first two columns come from s0, the last two from s1.
        #
        # Source slices:
        #  s0, local_shape=(1,2), global_shape=(2,2), global_offset=(1,0)
        #      +----+----+
        #      |(1,0)|(1,1)|   <- used for d1 (0,0)-(0,1)
        #      +----+----+
        #
        #  s1, local_shape=(1,2), global_shape=(2,2), global_offset=(1,0)
        #      +----+----+
        #      |(1,0)|(1,1)|   <- used for d1 (0,2)-(0,3)
        #      +----+----+
        #
        # The answer consists of two mapping entries:
        # 1. d1[:, 0:2] <-- s0[1, :]
        # 2. d1[:, 2:4] <-- s1[1, :]
        query = ShardedWeightDesc(
            key="d1",
            local_shape=(1, 4),
            global_shape=(1, 4),
            global_offset=(0, 0),
        )

        # d1[:, 0:2] <--- s0[1, :]
        src_sharded_weight_desc0 = ShardedWeightDesc(
            key="s0",
            local_shape=(1, 2),
            global_shape=(2, 2),
            global_offset=(1, 0),  # row 1, columns 0:2
        )
        dst_sharded_weight_desc0 = ShardedWeightDesc(
            key="d1",
            local_shape=(1, 2),
            global_shape=(1, 4),
            global_offset=(0, 0),
        )
        # Visual mapping:
        # d1 (0,0)-(0,1) <--- s0 (1,0)-(1,1)
        #  +------+------+------+------+
        #  |==s0==|==s0==|      |      |
        #  +------+------+------+------+
        src_sharded_weight_desc1 = ShardedWeightDesc(
            key="s1",
            local_shape=(1, 2),
            global_shape=(2, 2),
            global_offset=(1, 0),
        )
        dst_sharded_weight_desc1 = ShardedWeightDesc(
            key="d1",
            local_shape=(1, 2),
            global_shape=(1, 4),
            global_offset=(0, 2),
        )
        # Visual mapping:
        # d1 (0,2)-(0,3) <--- s1 (1,0)-(1,1)
        #  +------+------+------+------+
        #  |      |      |==s1==|==s1==|
        #  +------+------+------+------+

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
        queries.append(query)
        answers.append(answer)
        # Visual answer summary:
        # d1 (row 0):
        #  +------+------+------+------+
        #  |==s0==|==s0==|==s1==|==s1==|
        #  +------+------+------+------+
        #   ^      ^      ^      ^
        #   |      |      |      |
        #   |______|      |______|
        #    from s0       from s1

        # ------------------------------------------------------
        # 6. Run the queries and check results
        for idx in range(len(queries)):
            query = queries[idx]
            answer = answers[idx]
            result = aoa_engine.find_shard_sources(query)
            self.assertEqual(result, answer)


if __name__ == '__main__':
    unittest.main()
