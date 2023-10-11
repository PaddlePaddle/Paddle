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
from cinn import common, runtime
from cinn.common import BFloat16, Bool, Float, Float16, Int, UInt


class DataArray:
    """
    Provides Python encapsulation of the cinn_buffer_t
    data interface in the CINN RunTime module.
    """

    def __init__(
        self,
        shape: list,
        dtype: common.Type = common.Float(32),
        data: runtime.cinn_buffer_t = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.data = data

    def to_numpy(self):
        """
        Convert DataArray to numpy array
        """
        cinn_dtype_to_np_dtype = {
            # numpy has no 'bfloat16', we use uint16 to hold bfloat16 data, same to Paddle
            BFloat16(): "uint16",
            BFloat16(): "bfloat16",
            Float16(): "float16",
            Float(32): "float32",
            Float(64): "float64",
            Int(8): "int8",
            Int(16): "int16",
            Int(32): "int32",
            Int(64): "int64",
            UInt(8): "uint8",
            # numpy has no 'bfloat16', we use uint16 to hold bfloat16 data, same to Paddle
            # "UInt(16): uint16"
            UInt(32): "uint32",
            UInt(64): "uint64",
            Bool(): "bool",
        }
        for cinn_dtype, np_dtype in cinn_dtype_to_np_dtype.items():
            if isinstance(self.dtype, cinn_dtype):
                np_arr = np.empty(self.shape, np_dtype)
                assert np_arr.flags["C_CONTIGUOUS"]
                self.data.copy_to(np_arr)
                return np_arr

        raise TypeError(f"no support {self._dtype} in CINN")

    @staticmethod
    def from_numpy(np_array, target=common.DefaultHostTarget()):
        """
        Create DataArray form numpy array
        """
        assert isinstance(np_array, np.ndarray)
        data = runtime.cinn_buffer_t(np_array, target)
        dtype_np_to_common = {
            # numpy has no 'bfloat16', we use uint16 to hold bfloat16 data, same to Paddle
            "uint16": BFloat16(),
            "bfloat16": BFloat16(),
            "float16": Float16(),
            "float32": Float(32),
            "float64": Float(64),
            "int8": Int(8),
            "int16": Int(16),
            "int32": Int(32),
            "int64": Int(64),
            "uint8": UInt(8),
            # numpy has no 'bfloat16', we use uint16 to hold bfloat16 data, same to Paddle
            # "uint16": UInt(16),
            "uint32": UInt(32),
            "uint64": UInt(64),
            "bool": Bool(),
        }
        dtype_np = str(np_array.dtype).split(".")[-1]
        assert str(dtype_np) in dtype_np_to_common, (
            str(dtype_np) + " not support in CINN"
        )
        assert dtype_np in dtype_np_to_common.keys()

        return DataArray(np_array.shape, dtype_np_to_common[dtype_np], data)
