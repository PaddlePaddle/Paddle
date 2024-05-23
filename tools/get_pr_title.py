# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import sys

import requests

pr_id = os.getenv('GIT_PR_ID')
if not pr_id:
    print('PREC No PR ID')
    sys.exit(0)

response = requests.get(
    f'https://api.github.com/repos/PaddlePaddle/Paddle/pulls/{pr_id}',
    headers={'Accept': 'application/vnd.github.v3+json'},
)

data = json.loads(response.text)

title = data['title']

prefixes = ['【CINN】', '[CINN]', '[cinn]', '【cinn】']
if any(title.startswith(prefix) for prefix in prefixes):
    print('The title starts with cinn')
else:
    print('The title does not start with cinn')
    sys.exit(1)
