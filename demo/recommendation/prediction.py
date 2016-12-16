#!/bin/env python2
# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from py_paddle import swig_paddle, DataProviderConverter

from common_utils import *
from paddle.trainer.config_parser import parse_config

try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

if __name__ == '__main__':
    model_path = sys.argv[1]
    swig_paddle.initPaddle('--use_gpu=0')
    conf = parse_config("trainer_config.py", "is_predict=1")
    network = swig_paddle.GradientMachine.createFromConfigProto(
        conf.model_config)
    assert isinstance(network, swig_paddle.GradientMachine)
    network.loadParameters(model_path)
    with open('./data/meta.bin', 'rb') as f:
        meta = pickle.load(f)
        headers = [h[1] for h in meta_to_header(meta, 'movie')]
        headers.extend([h[1] for h in meta_to_header(meta, 'user')])
        cvt = DataProviderConverter(headers)
        while True:
            movie_id = int(raw_input("Input movie_id: "))
            user_id = int(raw_input("Input user_id: "))
            movie_meta = meta['movie'][movie_id]  # Query Data From Meta.
            user_meta = meta['user'][user_id]
            data = [movie_id - 1]
            data.extend(movie_meta)
            data.append(user_id - 1)
            data.extend(user_meta)
            print "Prediction Score is %.2f" % (
                (network.forwardTest(cvt.convert([data]))[0]['value'][0][0] + 5)
                / 2)
