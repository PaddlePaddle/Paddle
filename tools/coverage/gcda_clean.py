#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" usage: gcda_clean.py pull_id. """

import os
import sys

from github import Github


def get_pull(pull_id):
    """Get pull.

    Args:
        pull_id (int): Pull id.

    Returns:
        github.PullRequest.PullRequest
    """
    token = os.getenv('GITHUB_API_TOKEN',
                      'e1f9c3cf211d5c20e65bd9ab7ec07983da284bca')
    github = Github(token, timeout=60)
    repo = github.get_repo('PaddlePaddle/Paddle')
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
