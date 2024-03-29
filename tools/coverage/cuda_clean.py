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
""" usage: cuda_clean.py pull_id. """

import os
import sys

from github import Github


def get_pull(pull_id):
    """
    Args:
        pull_id (int): Pull id.

    Returns:
        github.PullRequest.PullRequest: The pull request.
    """
    token = os.getenv('GITHUB_API_TOKEN')
    github = Github(token, timeout=60)
    repo = github.get_repo('PaddlePaddle/Paddle')
    pull = repo.get_pull(pull_id)

    return pull


def get_files(pull_id):
    """
    Args:
        pull_id (int): Pull id.

    Returns:
       iterable: The generator will yield every filename.
    """

    pull = get_pull(pull_id)

    for file in pull.get_files():
        yield file.filename


def clean(pull_id):
    """
    Args:
        pull_id (int): Pull id.

    Returns:
        None.
    """

    changed = []

    for file in get_files(pull_id):
        # changed.append('/paddle/build/{}.gcda'.format(file))
        changed.append(file)

    for parent, dirs, files in os.walk('/paddle/build/'):
        for gcda in files:
            if gcda.endswith('.gcda'):
                file_name = gcda.replace('.gcda', '')
                dir_name_list = parent.replace('/paddle/build/', '').split('/')
                dir_name_list = dir_name_list[:-2]
                dir_name = '/'.join(dir_name_list)
                src_name = dir_name + '/' + file_name

                # remove no changed gcda

                if src_name not in changed:
                    unused_file = parent + '/' + gcda
                    # print unused_file
                    os.remove(gcda)
                else:
                    print(src_name)


if __name__ == '__main__':
    pull_id = sys.argv[1]
    pull_id = int(pull_id)

    clean(pull_id)
