#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import io, os
import random
import numpy as np
from paddle.trainer.PyDataProvider2 import *


def initHook(settings, height, width, color, num_class, **kwargs):
    settings.height = height
    settings.width = width
    settings.color = color
    settings.num_class = num_class
    if settings.color:
        settings.data_size = settings.height * settings.width * 3
    else:
        settings.data_size = settings.height * settings.width
    settings.is_infer = kwargs.get('is_infer', False)
    settings.num_samples = kwargs.get('num_samples', 2560)
    if settings.is_infer:
        settings.slots = [dense_vector(settings.data_size)]
    else:
        settings.slots = [dense_vector(settings.data_size), integer_value(1)]


@provider(
    init_hook=initHook, min_pool_size=-1, cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, file_list):
    for i in xrange(settings.num_samples):
        img = np.random.rand(1, settings.data_size).reshape(-1, 1).flatten()
        if settings.is_infer:
            yield img.astype('float32')
        else:
            lab = random.randint(0, settings.num_class - 1)
            yield img.astype('float32'), int(lab)
