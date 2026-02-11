#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Optimizer Factory."""

__all__ = []
import logging

from paddle.framework import core

OpRole = core.op_proto_and_checker_maker.OpRole
# this dict is for storing info about pull/push sparse ops.
FLEET_GLOBAL_DICT = {
    # global settings
    "enable": False,
    "emb_to_table": {},
    "emb_to_accessor": {},
    "emb_to_size": {},
    # current embedding settings
    "cur_sparse_id": 0,
    "cur_accessor": "",
    "click_name": "",
    "scale_sparse_grad": None,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
