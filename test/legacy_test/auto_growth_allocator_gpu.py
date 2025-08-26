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

import argparse
import json
import os
import sys

MiB = 1 << 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="JSON array of ops")
    parser.add_argument(
        "--out", required=True, help="path to write JSON result"
    )
    args = parser.parse_args()

    flags_json = os.environ.get("FLAGS_JSON")
    if flags_json:
        try:
            cfg = json.loads(flags_json)
            for k, v in cfg.items():
                os.environ[k] = str(v)
        except Exception as e:
            _write_out(
                args.out,
                {
                    "error": f"parse FLAGS_JSON failed: {e}",
                    "reserved": [],
                    "allocated": [],
                    "try_alloc_ok": [],
                },
            )
            sys.exit(1)

    import paddle
    from paddle import base

    result = {
        "reserved": [],
        "allocated": [],
        "try_alloc_ok": [],
        "error": "",
    }

    if not base.is_compiled_with_cuda():
        _write_out(args.out, result)
        return

    def max_reserved():
        return int(paddle.device.cuda.max_memory_reserved())

    def max_allocated():
        return int(paddle.device.cuda.max_memory_allocated())

    try:
        plan = json.loads(args.plan)
    except Exception as e:
        result["error"] = f"parse plan failed: {e}"
        _write_out(args.out, result)
        sys.exit(2)

    holds = []

    def append_stats():
        result["reserved"].append(max_reserved())
        result["allocated"].append(max_allocated())

    try:
        for step in plan:
            op = step.get("op")
            if op == "init":
                _ = paddle.rand([1])
            elif op == "alloc_small":
                mb_per_block = float(
                    step.get("mb_per_block", step.get("mb", 0.5))
                )
                blocks = int(step.get("blocks", 1))
                elems = max(1, int((mb_per_block * MiB) // 4))  # float32
                for _ in range(blocks):
                    holds.append(paddle.rand([elems]))
            elif op == "alloc_large":
                mb = float(step.get("mb", 8))
                elems = max(1, int((mb * MiB) // 4))
                holds.append(paddle.rand([elems]))
            elif op == "try_alloc":
                mb = float(step.get("mb", 0))
                elems = max(1, int((mb * MiB) // 4))
                ok = True
                try:
                    holds.append(paddle.rand([elems]))
                except Exception:
                    ok = False
                result["try_alloc_ok"].append(ok)
            append_stats()

        _write_out(args.out, result)
    except Exception as e:
        result["error"] = f"runtime error: {e}"
        _write_out(args.out, result)
        sys.exit(3)


def _write_out(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
