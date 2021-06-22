#! /usr/bin/env python

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
TestCases for check_pr_approval.py
"""
import unittest
import subprocess
import sys


class Test_check_approval(unittest.TestCase):
    def setUp(self):
        self.jsonstr = """
[
  {
    "id": 688077074,
    "node_id": "MDE3OlB1bGxSZXF1ZXN0UmV2aWV3Njg4MDc3MDc0",
    "user": {
      "login": "wadefelix",
      "id": 1306724,
      "node_id": "MDQ6VXNlcjEzMDY3MjQ=",
      "avatar_url": "https://avatars.githubusercontent.com/u/1306724?v=4",
      "gravatar_id": "",
      "url": "https://api.github.com/users/wadefelix",
      "html_url": "https://github.com/wadefelix",
      "followers_url": "https://api.github.com/users/wadefelix/followers",
      "following_url": "https://api.github.com/users/wadefelix/following{/other_user}",
      "gists_url": "https://api.github.com/users/wadefelix/gists{/gist_id}",
      "starred_url": "https://api.github.com/users/wadefelix/starred{/owner}{/repo}",
      "subscriptions_url": "https://api.github.com/users/wadefelix/subscriptions",
      "organizations_url": "https://api.github.com/users/wadefelix/orgs",
      "repos_url": "https://api.github.com/users/wadefelix/repos",
      "events_url": "https://api.github.com/users/wadefelix/events{/privacy}",
      "received_events_url": "https://api.github.com/users/wadefelix/received_events",
      "type": "User",
      "site_admin": false
    },
    "body": "",
    "state": "COMMENTED",
    "html_url": "https://github.com/PaddlePaddle/Paddle/pull/33634#pullrequestreview-688077074",
    "pull_request_url": "https://api.github.com/repos/PaddlePaddle/Paddle/pulls/33634",
    "author_association": "CONTRIBUTOR",
    "_links": {
      "html": {
        "href": "https://github.com/PaddlePaddle/Paddle/pull/33634#pullrequestreview-688077074"
      },
      "pull_request": {
        "href": "https://api.github.com/repos/PaddlePaddle/Paddle/pulls/33634"
      }
    },
    "submitted_at": "2021-06-21T06:47:16Z",
    "commit_id": "8122b5b055f925860e8dc68d1ca8d3d687194471"
  },
  {
    "id": 688092580,
    "node_id": "MDE3OlB1bGxSZXF1ZXN0UmV2aWV3Njg4MDkyNTgw",
    "user": {
      "login": "MingMingShangTian",
      "id": 13469016,
      "node_id": "MDQ6VXNlcjEzNDY5MDE2",
      "avatar_url": "https://avatars.githubusercontent.com/u/13469016?u=a939e72fd80b44317aeb5a6b37c37b80457157a7&v=4",
      "gravatar_id": "",
      "url": "https://api.github.com/users/MingMingShangTian",
      "html_url": "https://github.com/MingMingShangTian",
      "followers_url": "https://api.github.com/users/MingMingShangTian/followers",
      "following_url": "https://api.github.com/users/MingMingShangTian/following{/other_user}",
      "gists_url": "https://api.github.com/users/MingMingShangTian/gists{/gist_id}",
      "starred_url": "https://api.github.com/users/MingMingShangTian/starred{/owner}{/repo}",
      "subscriptions_url": "https://api.github.com/users/MingMingShangTian/subscriptions",
      "organizations_url": "https://api.github.com/users/MingMingShangTian/orgs",
      "repos_url": "https://api.github.com/users/MingMingShangTian/repos",
      "events_url": "https://api.github.com/users/MingMingShangTian/events{/privacy}",
      "received_events_url": "https://api.github.com/users/MingMingShangTian/received_events",
      "type": "User",
      "site_admin": false
    },
    "body": "LGTM",
    "state": "APPROVED",
    "html_url": "https://github.com/PaddlePaddle/Paddle/pull/33634#pullrequestreview-688092580",
    "pull_request_url": "https://api.github.com/repos/PaddlePaddle/Paddle/pulls/33634",
    "author_association": "CONTRIBUTOR",
    "_links": {
      "html": {
        "href": "https://github.com/PaddlePaddle/Paddle/pull/33634#pullrequestreview-688092580"
      },
      "pull_request": {
        "href": "https://api.github.com/repos/PaddlePaddle/Paddle/pulls/33634"
      }
    },
    "submitted_at": "2021-06-21T07:06:36Z",
    "commit_id": "8122b5b055f925860e8dc68d1ca8d3d687194471"
  },
  {
    "id": 689175539,
    "node_id": "MDE3OlB1bGxSZXF1ZXN0UmV2aWV3Njg5MTc1NTM5",
    "user": {
      "login": "pangyoki",
      "id": 26408901,
      "node_id": "MDQ6VXNlcjI2NDA4OTAx",
      "avatar_url": "https://avatars.githubusercontent.com/u/26408901?v=4",
      "gravatar_id": "",
      "url": "https://api.github.com/users/pangyoki",
      "html_url": "https://github.com/pangyoki",
      "followers_url": "https://api.github.com/users/pangyoki/followers",
      "following_url": "https://api.github.com/users/pangyoki/following{/other_user}",
      "gists_url": "https://api.github.com/users/pangyoki/gists{/gist_id}",
      "starred_url": "https://api.github.com/users/pangyoki/starred{/owner}{/repo}",
      "subscriptions_url": "https://api.github.com/users/pangyoki/subscriptions",
      "organizations_url": "https://api.github.com/users/pangyoki/orgs",
      "repos_url": "https://api.github.com/users/pangyoki/repos",
      "events_url": "https://api.github.com/users/pangyoki/events{/privacy}",
      "received_events_url": "https://api.github.com/users/pangyoki/received_events",
      "type": "User",
      "site_admin": false
    },
    "body": "LGTM",
    "state": "APPROVED",
    "html_url": "https://github.com/PaddlePaddle/Paddle/pull/33634#pullrequestreview-689175539",
    "pull_request_url": "https://api.github.com/repos/PaddlePaddle/Paddle/pulls/33634",
    "author_association": "CONTRIBUTOR",
    "_links": {
      "html": {
        "href": "https://github.com/PaddlePaddle/Paddle/pull/33634#pullrequestreview-689175539"
      },
      "pull_request": {
        "href": "https://api.github.com/repos/PaddlePaddle/Paddle/pulls/33634"
      }
    },
    "submitted_at": "2021-06-22T07:53:41Z",
    "commit_id": "3dd01a464d8a088003a831626edfe7a67ee8c6a2"
  }
]
    """

    def test_ids(self):
        cmd = [sys.executable, 'check_pr_approval.py', '1', '26408901']
        subprc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        output, error = subprc.communicate(input=self.jsonstr)
        self.assertEqual('TRUE', output.rstrip())

    def test_logins(self):
        cmd = [sys.executable, 'check_pr_approval.py', '1', 'pangyoki']
        subprc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        output, error = subprc.communicate(input=self.jsonstr)
        self.assertEqual('TRUE', output.rstrip())

    def test_ids_and_logins(self):
        cmd = [
            sys.executable, 'check_pr_approval.py', '2', 'pangyoki', '13469016'
        ]
        subprc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        output, error = subprc.communicate(input=self.jsonstr)
        #self.assertEqual('', error.rstrip())
        self.assertEqual('TRUE', output.rstrip())

    def test_check_with_required_reviewer_not_approved(self):
        cmd = [
            sys.executable, 'check_pr_approval.py', '2', 'wadefelix',
            ' 13469016'
        ]
        subprc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        output, error = subprc.communicate(input=self.jsonstr)
        self.assertEqual('FALSE', output.rstrip())


if __name__ == '__main__':
    unittest.main()
