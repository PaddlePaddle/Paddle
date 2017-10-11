# Copyright (c) 2016 PaddlePaddle Authors, Inc. All Rights Reserved
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

from paddle.trainer.PyDataProvider2 import *
import sys
import numpy as np
TERM_NUM = 24
FORECASTING_NUM = 24
LABEL_VALUE_NUM = 4


def initHook(settings, file_list, **kwargs):
    """
    Init hook is invoked before process data. It will set obj.slots and store data meta.

    :param settings: global object. It will passed to process routine.
    :type obj: object
    :param file_list: the meta file object, which passed from trainer_config.py,but unused in this function.
    :param kwargs: unused other arguments.
    """
    del kwargs  #unused 

    settings.pool_size = sys.maxint
    #Use a time seires of the past as feature.
    #Dense_vector's expression form is [float,float,...,float]
    settings.input_types = [dense_vector(TERM_NUM)]
    #There are next FORECASTING_NUM fragments you need predict.
    #Every predicted condition at time point has four states.
    for i in range(FORECASTING_NUM):
        settings.input_types.append(integer_value(LABEL_VALUE_NUM))


@provider(
    init_hook=initHook, cache=CacheType.CACHE_PASS_IN_MEM, should_shuffle=True)
def process(settings, file_name):
    with open(file_name) as f:
        #abandon fields name
        f.next()
        for row_num, line in enumerate(f):
            speeds = map(int, line.rstrip('\r\n').split(",")[1:])
            # Get the max index.
            end_time = len(speeds)
            # Scanning and generating samples
            for i in range(TERM_NUM, end_time - FORECASTING_NUM):
                # For dense slot
                pre_spd = map(float, speeds[i - TERM_NUM:i])

                # Integer value need predicting, values start from 0, so every one minus 1.
                fol_spd = [j - 1 for j in speeds[i:i + FORECASTING_NUM]]

                # Predicting label is missing, abandon the sample.
                if -1 in fol_spd:
                    continue
                yield [pre_spd] + fol_spd


def predict_initHook(settings, file_list, **kwargs):
    settings.pool_size = sys.maxint
    settings.input_types = [dense_vector(TERM_NUM)]


@provider(init_hook=predict_initHook, should_shuffle=False)
def process_predict(settings, file_name):
    with open(file_name) as f:
        #abandon fields name
        f.next()
        for row_num, line in enumerate(f):
            speeds = map(int, line.rstrip('\r\n').split(","))
            end_time = len(speeds)
            pre_spd = map(float, speeds[end_time - TERM_NUM:end_time])
            yield pre_spd
