# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

No_Check_Set_White_List = []
No_Check_Set_Need_To_Fix_Op_List = [
    'fake_quantize_range_abs_max', 'coalesce_tensor', 'flatten2', 'squeeze2',
    'reshape2', 'transpose2', 'unsqueeze2', 'cross_entropy2'
]
