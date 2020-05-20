#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define all functions about input & output in this directory 
__all__ = [
    'Dataset',
    'BatchSampler',
    #            'Transform',
    'DataLoader',
    'load',
    'save',
    'load_program_state',
    'set_program_state',
    'load_inference_model',
    'save_inference_model',
    'batch',
    'shuffle',
    'buffered',
    'cache',
    'chain',
    'firstn',
    'compose',
    'map_readers',
    'xmap_readers'
]

from ..fluid.io import DataLoader
from ..fluid.dataloader import Dataset, BatchSampler
from ..fluid.io import load, save, load_program_state, set_program_state, \
        load_inference_model, save_inference_model, batch
from ..reader import shuffle, buffered, cache, chain, firstn, compose, map_readers, xmap_readers
