#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
""" usage: gcda_clean.py pull_id. """

import os
import sys
import time

from github import Github


def get_pull(pull_id):
    """Get pull.

    Args:
        pull_id (int): Pull id.

    Returns:
        github.PullRequest.PullRequest
    """
    token = os.getenv('GITHUB_API_TOKEN')
    github = Github(token, timeout=60)
    idx = 1
    while idx < 4:
        try:
            repo = github.get_repo('PaddlePaddle/Paddle')
        except Exception as e:
            print(e)
            print("get_repo error, retry {} times after {} secs.".format(
                idx, idx * 10))
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


def clean(pull_id):
    """Clean.

    Args:
        pull_id (int): Pull id.

    Returns:
        None.
    """
    changed = []

    for file in get_files(pull_id):
        changed.append('/paddle/build/{}.gcda'.format(file))

    for parent, dirs, files in os.walk('/paddle/build/'):
        for gcda in files:
            if gcda.endswith('.gcda'):
                trimmed = parent

                # convert paddle/fluid/imperative/CMakeFiles/layer.dir/layer.cc.gcda
                # to paddle/fluid/imperative/layer.cc.gcda

                if trimmed.endswith('.dir'):
                    trimmed = os.path.dirname(trimmed)

                if trimmed.endswith('CMakeFiles'):
                    trimmed = os.path.dirname(trimmed)

                # remove no changed gcda

                if os.path.join(trimmed, gcda) not in changed:
                    gcda = os.path.join(parent, gcda)
                    os.remove(gcda)


if __name__ == '__main__':
    pull_id = sys.argv[1]
    pull_id = int(pull_id)

    clean(pull_id)
