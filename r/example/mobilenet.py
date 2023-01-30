#!/usr/bin/env python3.7

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
<<<<<<< HEAD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
=======
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

<<<<<<< HEAD
import numpy as np

from paddle.fluid.core import AnalysisConfig, create_paddle_predictor
=======
import functools
import numpy as np
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import AnalysisPredictor
from paddle.fluid.core import create_paddle_predictor
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def main():
    config = set_config()
    predictor = create_paddle_predictor(config)

    data, result = parse_data()

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])
    shape = (1, 3, 300, 300)
    input_data = data[:-4].astype(np.float32).reshape(shape)
    input_tensor.copy_from_cpu(input_data)

    predictor.zero_copy_run()

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])
    output_data = output_tensor.copy_to_cpu()


def set_config():
    config = AnalysisConfig("")
    config.set_model("model/__model__", "model/__params__")
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.enable_profile()

    return config


def parse_data():
<<<<<<< HEAD
    """parse input and output data"""
=======
    """ parse input and output data """
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    with open('data/data.txt', 'r') as fr:
        data = np.array([float(_) for _ in fr.read().split()])

    with open('data/result.txt', 'r') as fr:
        result = np.array([float(_) for _ in fr.read().split()])

    return (data, result)


if __name__ == "__main__":
    main()
