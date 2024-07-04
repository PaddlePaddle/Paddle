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
import re
import sys

import requests

SKIP_COVERAGE_CHECKING_LABELS = [
    "cinn",
    "typing",
    "codestyle",
]

SKIP_COVERAGE_CHECKING_REGEX = re.compile(
    rf"[\[【]({'|'.join(SKIP_COVERAGE_CHECKING_LABELS)})[\]】]",
    re.IGNORECASE,
)


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

if match_obj := SKIP_COVERAGE_CHECKING_REGEX.search(title):
    print(f'The title starts with {match_obj.group(0)}')
else:
    print(
        f'The title does not start with {" or ".join(SKIP_COVERAGE_CHECKING_LABELS)}'
    )
    sys.exit(1)
