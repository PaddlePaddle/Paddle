#!/usr/bin/env python3
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
"""Split build/all_ut_list into one shard's build/ut_list.

Usage:
    shard_ut_list.py <PADDLE_ROOT> <shard_index> <num_shards>

group_case_for_parallel.py reads build/ut_list with the highest priority, so
writing a shard-filtered ut_list makes the whole grouping (single_card /
multiple_card / exclusive / no_parallel) run only this shard's tests. The
split is a stable hash on the test name, so every shard runs a disjoint,
collectively exhaustive subset regardless of execution order.
"""

import hashlib
import os
import sys


def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: %s <PADDLE_ROOT> <shard_index> <num_shards>" % sys.argv[0])
    root = sys.argv[1]
    shard_index = int(sys.argv[2])
    num_shards = int(sys.argv[3])
    if not (0 <= shard_index < num_shards):
        sys.exit("shard_index %d out of range [0, %d)" % (shard_index, num_shards))

    all_list = os.path.join(root, 'build', 'all_ut_list')
    ut_list = os.path.join(root, 'build', 'ut_list')
    with open(all_list) as f:
        tests = [line.strip() for line in f if line.strip()]

    shard_tests = [
        t
        for t in tests
        if int(hashlib.md5(t.encode('utf-8')).hexdigest(), 16) % num_shards
        == shard_index
    ]
    if not shard_tests:
        # An empty ut_list makes group_case_for_parallel.py fall back to
        # all_ut_list (running the whole suite). Fail loudly instead.
        sys.exit(
            "shard %d has 0 tests out of %d; "
            "refusing to write an empty ut_list" % (shard_index, len(tests))
        )

    with open(ut_list, 'w') as f:
        f.write('\n'.join(shard_tests) + '\n')
    print(
        "shard %d/%d: %d/%d tests"
        % (shard_index, num_shards, len(shard_tests), len(tests))
    )


if __name__ == '__main__':
    main()
