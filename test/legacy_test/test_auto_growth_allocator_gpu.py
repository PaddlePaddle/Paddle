# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid

import paddle
from paddle import base

MiB = 1 << 20


def _run_test_case(plan, flags, cuda_visible_devices="0"):
    script = os.path.join(
        os.path.dirname(__file__), "auto_growth_allocator_gpu.py"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["FLAGS_JSON"] = json.dumps(flags)
    env.setdefault("PYTHONUNBUFFERED", "1")

    keep = os.environ.get("AG_KEEP_OUT", "").strip()
    if keep:
        if keep == "1":
            out_dir = os.path.join(os.getcwd(), "_ag_out")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir, f"ag_{os.getpid()}_{uuid.uuid4().hex}.json"
            )
        elif keep.endswith(".json"):
            os.makedirs(
                os.path.dirname(os.path.abspath(keep)) or ".", exist_ok=True
            )
            out_path = os.path.abspath(keep)
        else:
            out_dir = os.path.abspath(keep)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir, f"ag_{os.getpid()}_{uuid.uuid4().hex}.json"
            )
    else:
        fd, out_path = tempfile.mkstemp(prefix="ag_", suffix=".json")
        os.close(fd)

    log_path = out_path + ".log" if keep else None

    cmd = [
        sys.executable,
        script,
        "--plan",
        json.dumps(plan),
        "--out",
        out_path,
    ]
    if log_path:
        cmd += ["--log", log_path]

    if env.get("AG_TEE", "") == "1":
        p = subprocess.run(cmd, env=env, text=True)
    else:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if p.returncode != 0:
        raise RuntimeError(
            f"probe failed:\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not keep:
        try:
            os.remove(out_path)
            if log_path:
                os.remove(log_path)
        except Exception:
            pass
    else:
        sys.stderr.write(f"[AG_KEEP_OUT] {out_path}\n")
        if log_path:
            sys.stderr.write(f"[AG_KEEP_OUT] {log_path}\n")

    return data


class TestAllocatorFlagsWithSubprocess(unittest.TestCase):
    def setUp(self):
        if base.is_compiled_with_cuda():
            paddle.set_flags(
                {
                    'FLAGS_allocator_strategy': 'auto_growth',
                    'FLAGS_use_cuda_malloc_async_allocator': 0,
                }
            )

    def test_memory_pool_flags(self):
        if not base.is_compiled_with_cuda():
            return
        flags = {
            "FLAGS_small_pool_size_in_mb": 1,
            "FLAGS_auto_growth_chunk_size_in_mb": 10,  # ignored because FLAGS_small_pool_size_in_mb > 0
            "FLAGS_small_pool_auto_growth_chunk_size_in_mb": 2,
            "FLAGS_large_pool_auto_growth_chunk_size_in_mb": 16,
            "FLAGS_small_pool_pre_alloc_in_mb": 2,
            "FLAGS_large_pool_pre_alloc_in_mb": 20,
        }
        plan = [
            {"op": "init"},
            {"op": "alloc_small", "mb_per_block": 0.5, "blocks": 7},
        ]
        out = _run_test_case(plan, flags)

        a0, a1 = out["allocated"][0], out["allocated"][1]
        r0, r1 = out["reserved"][0], out["reserved"][1]

        self.assertEqual(a1, int(3.5 * MiB))
        self.assertEqual(r0, int(22 * MiB))
        self.assertEqual(r1, r0 + int(2 * MiB), msg=f"r0={r0}, r1={r1}")

    def test_large_pool_growth_override_16mb(self):
        if not base.is_compiled_with_cuda():
            return
        flags = {
            "FLAGS_small_pool_size_in_mb": 1,
            "FLAGS_small_pool_auto_growth_chunk_size_in_mb": 0,
            "FLAGS_large_pool_auto_growth_chunk_size_in_mb": 16,
            "FLAGS_small_pool_pre_alloc_in_mb": 0,
            "FLAGS_large_pool_pre_alloc_in_mb": 6,
        }
        plan = [
            {"op": "init"},
            {"op": "alloc_large", "mb": 8},
        ]
        out = _run_test_case(plan, flags)

        r0, r1 = out["reserved"][0], out["reserved"][1]
        self.assertEqual(r1, r0 + int(16 * MiB), msg=f"r0={r0}, r1={r1}")

    def test_single_pool(self):
        if not base.is_compiled_with_cuda():
            return
        flags = {
            "FLAGS_small_pool_size_in_mb": 0,
            "FLAGS_small_pool_auto_growth_chunk_size_in_mb": 2,
            "FLAGS_large_pool_auto_growth_chunk_size_in_mb": 4,
            "FLAGS_auto_growth_chunk_size_in_mb": 10,
            "FLAGS_small_pool_pre_alloc_in_mb": 2,
            "FLAGS_large_pool_pre_alloc_in_mb": 6,
        }
        plan = [
            {"op": "init"},
            {"op": "alloc_small", "mb_per_block": 0.5, "blocks": 1},
            {"op": "alloc_large", "mb": 10},
        ]
        out = _run_test_case(plan, flags)

        a0, a1, a2 = (
            out["allocated"][0],
            out["allocated"][1],
            out["allocated"][2],
        )
        r0, r1, r2 = out["reserved"][0], out["reserved"][1], out["reserved"][2]

        self.assertEqual(a1, int(0.5 * MiB))
        self.assertEqual(a2, int(10.5 * MiB))
        self.assertEqual(r0, int(10 * MiB), msg=f"r0={r0}")
        self.assertEqual(r1, int(10 * MiB), msg=f"r1={r1}")
        self.assertEqual(r2, int(20 * MiB), msg=f"r2={r2}")

    def test_memory_limit(self):
        if not base.is_compiled_with_cuda():
            return
        flags = {
            "FLAGS_gpu_memory_limit_mb": 10,
        }
        plan = [
            {"op": "try_alloc", "mb": 5},
            {"op": "try_alloc", "mb": 20},
        ]
        out = _run_test_case(plan, flags)
        self.assertEqual(out["try_alloc_ok"][0], True)
        self.assertEqual(out["try_alloc_ok"][1], False)

    def test_auto_growth_allocator_v2(self):
        if not base.is_compiled_with_cuda():
            return
        flags = {
            "FLAGS_use_auto_growth_v2": True,
            "FLAGS_large_pool_pre_alloc_in_mb": 6,
        }
        plan = [
            {"op": "init"},
            {"op": "alloc_large", "mb": 20},
        ]
        out = _run_test_case(plan, flags)
        r0 = out["reserved"][0]
        self.assertLessEqual(r0, int(6 * MiB), msg=f"r0={r0}")

    def test_trace_flag(self):
        if not base.is_compiled_with_cuda():
            return
        flags = {
            "FLAGS_small_pool_size_in_mb": 1,
            "FLAGS_large_pool_pre_alloc_in_mb": 5,
            "FLAGS_free_idle_chunk": True,
            "FLAGS_free_when_no_cache_hit": True,
            "FLAGS_print_allocator_trace_info": True,
        }
        plan = [
            {"op": "init"},
            {"op": "alloc_small", "mb": 1},
        ]
        out = _run_test_case(plan, flags)


if __name__ == "__main__":
    unittest.main()
