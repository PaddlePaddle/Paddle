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

from . import auto_checkpoint as acp
from . import dataloader_auto_checkpoint as dacp

CONST_DACP_TYPE = "dacp"
CONST_ACP_TYPE = "acp"
g_acp_type = None

logger = acp._get_logger(20)


def _auto_checkpoint(exe, program):
    if not acp._can_auto_checkpoint(program):
        return False

    changed = False

    global g_acp_type
    if g_acp_type is None:
        if len(dacp.g_train_epoch_ranges) > 1:
            g_acp_type = CONST_DACP_TYPE
        elif acp.g_train_epoch_range is not None:
            g_acp_type = CONST_ACP_TYPE
        else:
            assert False, "internal error: check acp_type error"
        changed = True

    if changed:
        logger.info("enter auto_checkpoint type:{}".format(g_acp_type))

    if g_acp_type == CONST_DACP_TYPE:
        return dacp._auto_checkpoint(exe, program)

    return acp._auto_checkpoint(exe, program)
