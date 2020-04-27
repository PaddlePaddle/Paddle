# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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
from hapi import logger
from hapi.configure import Config
from hapi import callbacks
from hapi import datasets
from hapi import distributed
from hapi import download
from hapi import metrics
from hapi import model
from hapi import progressbar
from hapi import text
from hapi import vision
from hapi import loss

logger.setup_logger()

__all__ = [
    'Config', 'callbacks', 'datasets', 'distributed', 'download', 'metrics',
    'model', 'progressbar', 'text', 'vision', 'loss'
]
