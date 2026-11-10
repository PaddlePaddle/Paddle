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

"""Save a tensor with one sharding and load it, with another sharding, into
targets that live on CUDA pinned memory (as optimizer states do with
tensorwise offload). After the load every target must hold the loaded values
and keep its pinned place and storage (data_ptr), including targets that are
non-contiguous views of a pinned parent.
"""

import os
import shutil

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight

FLAG = "FLAGS_pinned_memory_as_cpu_backend"
GLOBAL_SHAPE = (8, 6)


def _under_flag(fn):
    # Optimizer.sharded_state_dict() builds its views with this flag on.
    old = paddle.get_flags([FLAG])[FLAG]
    paddle.set_flags({FLAG: True})
    try:
        return fn()
    finally:
        paddle.set_flags({FLAG: old})


class TestPinnedLoadReshard:
    def __init__(self):
        dist.init_parallel_env()
        self.rank = dist.get_rank()
        self.base_dir = os.environ.get("FLEX_PINNED_TEST_DIR", "/tmp")
        self.full = (
            np.random.default_rng(7)
            .standard_normal(GLOBAL_SHAPE)
            .astype("float32")
        )

    def _ckpt_dir(self, tag):
        path = os.path.join(self.base_dir, f"pinned_load_{tag}")
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path)
        dist.barrier()
        return path

    def _save(self, tag, layout, dtype):
        r = self.rank
        if layout == "col":  # tensor parallel split along dim 1
            region, offset = (slice(None), slice(3 * r, 3 * r + 3)), (0, 3 * r)
        else:  # split along dim 0
            region, offset = (slice(4 * r, 4 * r + 4), slice(None)), (4 * r, 0)
        local = paddle.to_tensor(self.full[region].copy()).astype(dtype)
        path = self._ckpt_dir(tag)
        dist.save_state_dict(
            {
                "w": ShardedWeight(
                    "w", local, tuple(local.shape), GLOBAL_SHAPE, offset
                )
            },
            path,
        )
        dist.barrier()
        return path

    def _check(self, tag, save_layout, target, dtype, comm_method):
        r = self.rank
        path = self._save(tag, save_layout, dtype)
        init = np.full(GLOBAL_SHAPE, -7.0, dtype="float32")
        if target == "row_tensor":
            # Row shard as its own pinned tensor; loading a column-split
            # checkpoint writes non-contiguous slices of it.
            region = (slice(4 * r, 4 * r + 4), slice(None))
            parent = paddle.to_tensor(init[region]).astype(dtype).pin_memory()
            local, offset, expected = parent, (4 * r, 0), self.full[region]
        else:
            # Non-contiguous column view of a full pinned parent.
            region = (slice(None), slice(3 * r, 3 * r + 3))
            parent = paddle.to_tensor(init).astype(dtype).pin_memory()
            local = _under_flag(
                lambda: paddle.slice(parent, [1], [3 * r], [3 * r + 3])
            )
            offset = (0, 3 * r)
            expected = init.copy()
            expected[region] = self.full[region]
        ptr = parent.data_ptr()
        dist.load_state_dict(
            {
                "w": ShardedWeight(
                    "w", local, tuple(local.shape), GLOBAL_SHAPE, offset
                )
            },
            path,
            comm_method=comm_method,
        )
        name = f"{tag} {dtype}"
        assert parent.place.is_cuda_pinned_place(), (
            f"{name}: left pinned memory"
        )
        assert parent.data_ptr() == ptr, f"{name}: storage reallocated"
        atol = 1e-2 if dtype == "bfloat16" else 0
        np.testing.assert_allclose(
            parent.cpu().astype("float32").numpy(),
            expected,
            atol=atol,
            err_msg=name,
        )

    def run_test_case(self):
        cases = [
            ("col_to_row", "col", "row_tensor"),
            ("row_to_col_view", "row", "col_view"),
            ("col_to_col_view", "col", "col_view"),
        ]
        for comm_method in ("broadcast", "send_recv", "grouped_send_recv"):
            for dtype in ("float32", "bfloat16"):
                for tag, save_layout, target in cases:
                    self._check(
                        f"{comm_method}_{tag}",
                        save_layout,
                        target,
                        dtype,
                        comm_method,
                    )


if __name__ == "__main__":
    TestPinnedLoadReshard().run_test_case()
