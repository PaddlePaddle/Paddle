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

from paddle._typing import (
    DataLayout0D,
    DataLayout1D,
    DataLayout1DVariant,
    DataLayout2D,
    DataLayout3D,
    DataLayoutImage,
    DataLayoutND,
)

d0: DataLayout0D = "NC"
d1_1: DataLayout1D = "NCL"
d1_2: DataLayout1D = "NLC"
d2_1: DataLayout2D = "NCHW"
d2_2: DataLayout2D = "NHCW"
d3_1: DataLayout3D = "NCDHW"
d3_2: DataLayout3D = "NDHWC"
d_v_1: DataLayout1DVariant = "NCW"
d_v_2: DataLayout1DVariant = "NWC"
d_n: DataLayoutND = "NHCW"
d_i_1: DataLayoutImage = "HWC"
d_i_2: DataLayoutImage = "CHW"
