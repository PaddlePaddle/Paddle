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

from ...fluid import framework


def is_complex(x):
    """
    Return true if the input(x) is a ComplexVariable.
    """
    return isinstance(x, framework.ComplexVariable)


def is_real(x):
    """
    Return true if the input(x) is a real number Variable.
    """
    return isinstance(x, framework.Variable)


def complex_variable_exists(inputs, layer_name):
    for inp in inputs:
        if is_complex(inp):
            return
    err_msg = "At least one inputs of layer complex." if len(inputs) > 1 \
              else "The input of layer complex."
    raise ValueError(err_msg + layer_name +
                     "() must be ComplexVariable, please "
                     "use the layer for real numher instead.")
