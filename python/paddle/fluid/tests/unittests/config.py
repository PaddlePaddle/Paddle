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

import numpy as np

TOLERANCE = {
    np.dtype('float64'): {
        "jit_comp": {"rtol": 1e-15, "atol": 1e-15},
        "fw_comp": {"rtol": 1e-15, "atol": 1e-15},
        "rev_comp": {"rtol": 1e-15, "atol": 1e-15},
        "cinn": {"rtol": 1e-14, "atol": 1e-14},
    },
    np.dtype('float32'): {
        "jit_comp": {"rtol": 1e-6, "atol": 1e-6},
        "fw_comp": {"rtol": 1e-6, "atol": 1e-6},
        "rev_comp": {"rtol": 1e-6, "atol": 1e-6},
        "cinn": {"rtol": 1e-5, "atol": 1e-5},
    },
    np.dtype('float16'): {
        "jit_comp": {"rtol": 1e-3, "atol": 1e-3},
        "fw_comp": {"rtol": 1e-3, "atol": 1e-3},
        "rev_comp": {"rtol": 1e-3, "atol": 1e-3},
        "cinn": {"rtol": 1e-2, "atol": 1e-2},
    },
    np.dtype('uint16'): {
        "jit_comp": {"rtol": 1e-2, "atol": 1e-2},
        "fw_comp": {"rtol": 1e-2, "atol": 1e-2},
        "rev_comp": {"rtol": 1e-2, "atol": 1e-2},
        "cinn": {"rtol": 1e-1, "atol": 1e-1},
    },
}
