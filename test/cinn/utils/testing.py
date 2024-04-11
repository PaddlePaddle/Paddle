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
from paddle.cinn.ir import IrCompare
from paddle.cinn.runtime import CinnLowerLevelIrJit


def assert_llir_equal(
    llir1, llir2, allow_name_suffix_diff=True, only_compare_structure=True
):
    comparer = IrCompare(allow_name_suffix_diff, only_compare_structure)

    if isinstance(llir1, CinnLowerLevelIrJit):
        llir1_expr = llir1.convert_to_llir().body()
        llir2_expr = llir2.convert_to_llir().body()
    assert comparer.compare(
        llir1_expr, llir2_expr
    ), f'llir1: {llir1} \n llir2: {llir2}'
