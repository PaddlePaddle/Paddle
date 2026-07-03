#!/usr/bin/env python

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
"""usage: gcda_clean.py pull_id."""

import os
import sys
import time

from github import Github

paddle_root = os.getenv("PADDLE_ROOT")


def get_pull(pull_id):
    """Get pull.

    Args:
        pull_id (int): Pull id.

    Returns:
        github.PullRequest.PullRequest
    """
    token = os.getenv("GITHUB_API_TOKEN")
    github = Github(token, timeout=60)
    idx = 1
    while idx < 4:
        try:
            repo = github.get_repo("PaddlePaddle/Paddle")
        except Exception as e:
            print(e)
            print(f"get_repo error, retry {idx} times after {idx * 10} secs.")
        else:
            break
        idx += 1
        time.sleep(idx * 10)
    pull = repo.get_pull(pull_id)

    return pull


def get_files(pull_id):
    """Get files.

    Args:
        pull_id (int): Pull id.

    Returns:
       iterable: The generator will yield every filename.
    """
    pull = get_pull(pull_id)

    for file in pull.get_files():
        yield file.filename


def normalize_gcda_path(gcda_path):
    """Map a generated gcda path back to the changed-source gcda path."""
    parts = []
    for part in gcda_path.split(os.sep):
        if part == "CMakeFiles" or part.endswith(".dir"):
            continue
        if part == "__":
            parts.append("..")
        else:
            parts.append(part)
    return os.path.normpath(os.sep.join(parts))


def clean(pull_id):
    """Clean.

    Args:
        pull_id (int): Pull id.

    Returns:
        None.
    """
    changed = set()

    for file in get_files(pull_id):
        changed.add(
            os.path.normpath(os.path.join(paddle_root, "build", f"{file}.gcda"))
        )

    for parent, dirs, files in os.walk(os.path.join(paddle_root, "build")):
        for gcda in files:
            if gcda.endswith(".gcda"):
                # Convert generated paths back to the source gcda path. For
                # example:
                #   paddle/fluid/pybind/CMakeFiles/paddle.dir/__/__/pir/src/core/type_util.cc.gcda
                # becomes:
                #   paddle/pir/src/core/type_util.cc.gcda
                normalized_gcda = normalize_gcda_path(
                    os.path.join(parent, gcda)
                )

                # remove no changed gcda

                if normalized_gcda not in changed:
                    gcda = os.path.join(parent, gcda)
                    os.remove(gcda)


if __name__ == "__main__":
    pull_id = sys.argv[1]
    pull_id = int(pull_id)

    clean(pull_id)
