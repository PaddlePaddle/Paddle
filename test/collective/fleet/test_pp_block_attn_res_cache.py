# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE_2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import paddle
from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
    PipelineParallelWithInterleave,
)


class FakeHcg:
    def __init__(self, pp_size, stage_id):
        self._pp_size = pp_size
        self._stage_id = stage_id

    def get_pipe_parallel_world_size(self):
        return self._pp_size

    def get_stage_id(self):
        return self._stage_id


class FakePipe:
    """Minimal stand-in exposing only the state the block cache helpers touch."""

    def __init__(self, pp_size, vpp_size, stage_id):
        self._hcg = FakeHcg(pp_size, stage_id)
        self.num_model_chunks = vpp_size
        self._block_cache = {}
        self._block_cache_meta = {}
        self._recv_block_cache_meta_for_vpp = [None] * vpp_size

    merge = PipelineParallelWithInterleave._merge_block_cache
    update = PipelineParallelWithInterleave._update_block_cache


def make_block(value):
    block = paddle.full([2, 2], value, dtype='float32')
    block.stop_gradient = False
    return block


class TestBlockAttnResCache(unittest.TestCase):
    def test_merge_is_noop_without_cache_and_recv(self):
        pipe = FakePipe(pp_size=2, vpp_size=2, stage_id=0)
        input_tensor_dict = {"hidden": make_block(1.0)}
        pipe.merge(0, 0, input_tensor_dict)
        self.assertNotIn("blocks", input_tensor_dict)
        self.assertEqual(pipe._block_cache_meta, {})

    def test_merge_concats_cache_before_received_blocks(self):
        pipe = FakePipe(pp_size=2, vpp_size=2, stage_id=0)
        cached = [make_block(1.0)]
        pipe._block_cache[0] = cached
        received = [make_block(2.0)]
        input_tensor_dict = {"hidden": make_block(0.0), "blocks": received}
        # meta layout is [chunk0_stage0, chunk0_stage1, chunk1_stage0, ...]
        pipe._recv_block_cache_meta_for_vpp[0] = [0, 0, 3, 0, 3]

        pipe.merge(0, 0, input_tensor_dict)

        merged = input_tensor_dict["blocks"]
        self.assertEqual(len(merged), 2)
        self.assertIs(merged[0], cached[0])
        self.assertIs(merged[1], received[0])
        # stage 1 counts come from the received meta, stage 0 counts add recv len
        meta = pipe._block_cache_meta[0]
        self.assertEqual(meta, [1, 3, 1, 3])

    def test_update_caches_blocks_and_trims_output(self):
        pipe = FakePipe(pp_size=2, vpp_size=2, stage_id=0)
        recv_block = make_block(1.0)
        produced = make_block(2.0)
        output_tensor = {
            "hidden": make_block(0.0),
            "blocks": [recv_block, produced],
        }

        meta_to_send = pipe.update(0, 0, output_tensor, merged_len_pre=1)

        cached = pipe._block_cache[0]
        self.assertEqual(len(cached), 2)
        # received blocks are cached by reference, produced ones by a detached copy
        self.assertIs(cached[0], recv_block)
        self.assertTrue(recv_block.pp_block_cached)
        self.assertIsNot(cached[1], produced)
        self.assertIs(produced.pp_cc_ref, cached[1])
        self.assertFalse(cached[1].stop_gradient)

        self.assertEqual(meta_to_send[0], 0)
        self.assertEqual(meta_to_send[1:], pipe._block_cache_meta[0])
        # only the blocks the next chunk still misses are kept for sending
        self.assertEqual(len(output_tensor["blocks"]), 1)
        self.assertIs(output_tensor["blocks"][0], produced)

    def test_update_returns_none_for_plain_tensor_output(self):
        pipe = FakePipe(pp_size=2, vpp_size=2, stage_id=0)
        self.assertIsNone(pipe.update(0, 0, make_block(1.0), 0))
        self.assertIsNone(pipe.update(0, 0, {"hidden": make_block(1.0)}, 0))


if __name__ == "__main__":
    unittest.main()
