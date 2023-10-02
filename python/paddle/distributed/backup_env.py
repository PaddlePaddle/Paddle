# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

g_backup_envs = None


def getenv_or_backup(name, default=None):
    global g_backup_envs
    if g_backup_envs is None:
        backup_path = os.getenv('PADDLE_BACKUP_ENV_PATH')
        if backup_path is None:
            g_backup_envs = {}
        else:
            with open(backup_path, 'r') as f:
                g_backup_envs = json.load(f)

    value = os.getenv(name)
    if value is not None:
        return value
    else:
        return g_backup_envs.get(name, default)
