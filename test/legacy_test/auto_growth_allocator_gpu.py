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

MiB = 1 << 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="JSON array of ops")
    args = parser.parse_args()

    flags_json = os.environ.get("FLAGS_JSON")
    if flags_json:
        cfg = json.loads(flags_json)
        for k, v in cfg.items():
            os.environ[k] = str(v)

    import paddle
    from paddle import base

    result = {
        "device": "none",
        "reserved": [],
        "allocated": [],
        "try_alloc_ok": [],
    }

    if not base.is_compiled_with_cuda():
        print(json.dumps(result))
        return

    result["device"] = "cuda"

    def max_reserved():
        return int(paddle.device.cuda.max_memory_reserved())

    def max_allocated():
        return int(paddle.device.cuda.max_memory_allocated())

    plan = json.loads(args.plan)
    holds = []

    for step in plan:
        op = step.get("op")
        if op == "init":
            _ = paddle.rand([1])
        elif op == "alloc_small":
            mb_per_block = float(step.get("mb_per_block", 0.5))
            blocks = int(step.get("blocks", 4))
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
        else:
            pass

        result["reserved"].append(max_reserved())
        result["allocated"].append(max_allocated())

    print(json.dumps(result))


if __name__ == "__main__":
    main()
