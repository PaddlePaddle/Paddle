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

import os

import paddle


def use_paddle_recall_error():
    val = os.getenv("FLAGS_use_paddle_recall_error", "1").strip().lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")


if use_paddle_recall_error():
    AADIFF_ERROR = "PaddleRecall error(101): AAdiff"
    LOSS_NAN_ERROR = "PaddleRecall error(102): LossNan"
    SHARDING_PAD_NON_ZERO_ERROR = "PaddleRecall error(103): ShardingPadNonZero"
    LOSS_INF_ERROR = "PaddleRecall error(104): LossInf"
else:
    AADIFF_ERROR = "CUDA error(1001)"
    LOSS_NAN_ERROR = "CUDA error(1002)"
    SHARDING_PAD_NON_ZERO_ERROR = "CUDA error(1003)"
    LOSS_INF_ERROR = "CUDA error(1004)"


def check_naninf(tensor):
    if paddle.isfinite(tensor).all().item():
        return None
    elif paddle.isnan(tensor).any().item():
        return LOSS_NAN_ERROR
    else:
        return LOSS_INF_ERROR
