# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# default tolerance
TOLERANCE = {
    "float16": {
        "forward": {"rtol": 1e-3, "atol": 1e-3},
        "backward": {"rtol": 1e-3, "atol": 1e-3},
        "prim_backward": {"rtol": 1e-3, "atol": 1e-3},
    },
    "float32": {
        "forward": {"rtol": 1e-6, "atol": 1e-6},
        "backward": {"rtol": 1e-6, "atol": 1e-6},
        "prim_backward": {"rtol": 1e-6, "atol": 1e-6},
    },
    "float64": {
        "forward": {"rtol": 1e-15, "atol": 1e-15},
        "backward": {"rtol": 1e-15, "atol": 1e-15},
        "prim_backward": {"rtol": 1e-15, "atol": 1e-15},
    },
}

# this tolerance is for big composite ops like batch_norm.
SUB_TOLERANCE = {
    "float32": {
        "forward": {"rtol": 1e-5, "atol": 1e-5},
        "backward": {"rtol": 1e-5, "atol": 1e-5},
        "prim_backward": {"rtol": 1e-5, "atol": 1e-5},
    },
    "float64": {
        "forward": {"rtol": 1e-13, "atol": 1e-13},
        "backward": {"rtol": 1e-13, "atol": 1e-13},
        "prim_backward": {"rtol": 1e-13, "atol": 1e-13},
    },
}
