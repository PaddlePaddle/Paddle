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

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass



@dataclass
class ChunkMetadata:
    local_shape: List[int]
    global_offset: List[int]

@dataclass(frozen=True)
class MetadataIndex:
    param: str
    global_offset: Tuple[int]

@dataclass
class Metadata:
    state_dict_metadata: Dict[str, List[ChunkMetadata]] = None
    storage_metadata: Dict[MetadataIndex, str] = None