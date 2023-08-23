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
import cinn
import numpy as np
from cinn import runtime


class DataArray:
    """ """

    def __init__(self, data, shape, dtype) -> None:
        self.data = data
        self.shape = shape
        self.dtype = dtype

    def to_numpy(self):
        """
        Convert DataArray to numpy array
        """
        cinn_dtype_to_np_dtype = {cinn.common.Float(32): "float32"}
        if self.dtype.is_float(32, cinn.common.Type.specific_type_t.UNK):
            np_dtype = np.float32
        np_arr = np.empty(self.shape, np_dtype)
        assert np_arr.flags["C_CONTIGUOUS"]
        self.data.copy_to(np_arr)
        return np_arr

    @staticmethod
    def from_numpy(np_array, target=cinn.common.DefaultHostTarget()):
        """ """
        assert isinstance(np_array, np.ndarray)
        data = runtime.cinn_buffer_t(np_array, target)
        dtype_np_to_cinn_common = {
            "float": cinn.common.Float(32),
            "float32": cinn.common.Float(32),
        }
        # TODO(6clc): Support float16,
        dtype_np = str(np_array.dtype).split(".")[-1]
        assert dtype_np in dtype_np_to_cinn_common.keys()

        return DataArray(
            data, np_array.shape, dtype_np_to_cinn_common[dtype_np]
        )
