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


import paddle

AADIFF_ERROR = "PaddleRecall error(101): AAdiff"
LOSS_NAN_ERROR = "PaddleRecall error(102): LossNan"
SHARDING_PAD_NON_ZERO_ERROR = "PaddleRecall error(103): ShardingPadNonZero"
LOSS_INF_ERROR = "PaddleRecall error(104): LossInf"


def check_naninf(tensor):
    if paddle.isfinite(tensor).all().item():
        return None
    elif paddle.isnan(tensor).any().item():
        return LOSS_NAN_ERROR
    else:
        return LOSS_INF_ERROR
