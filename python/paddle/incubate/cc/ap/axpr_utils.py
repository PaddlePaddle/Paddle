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

import os
from contextlib import contextmanager


@contextmanager
def apy_envs(ap_path="", ap_workspace_dir="/tmp/paddle/ap_workspace"):
    import paddle

    ap_sys_path = f"{os.path.dirname(paddle.__file__)}/apy/sys"
    old_ap_path = os.environ.get('AP_PATH')
    old_ap_workspace_dir = os.environ.get('AP_WORKSPACE_DIR')
    os.environ['AP_PATH'] = (
        f"{ap_sys_path}:{ap_path}:{old_ap_path if old_ap_path is not None else ''}"
    )
    os.environ['AP_WORKSPACE_DIR'] = ap_workspace_dir
    old_flags = paddle.get_flags(['FLAGS_enable_ap'])
    flags = dict(old_flags)
    flags['FLAGS_enable_ap'] = True
    paddle.set_flags(flags)
    yield
    if old_ap_path is not None:
        os.environ['AP_PATH'] = old_ap_path
    else:
        del os.environ['AP_PATH']
    if old_ap_workspace_dir is not None:
        os.environ['AP_WORKSPACE_DIR'] = old_ap_workspace_dir
    else:
        del os.environ['AP_WORKSPACE_DIR']
    paddle.set_flags(old_flags)
