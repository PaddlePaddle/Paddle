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

import six


#  str and bytes related functions
def to_literal_str(obj):
    if isinstance(obj, list):
        return [_to_literal_str(item) for item in obj]
    elif isinstance(obj, set):
        return set([_to_literal_str(item) for item in obj])
    else:
        return _to_literal_str(obj)


def _to_literal_str(obj):
    if isinstance(obj, six.binary_type):
        return obj.decode('utf-8')
    elif isinstance(obj, six.text_type):
        return obj
    else:
        return six.u(obj)


def to_bytes(obj):
    if isinstance(obj, list):
        return [_to_bytes(item) for item in obj]
    elif isinstance(obj, set):
        return set([_to_bytes(item) for item in obj])
    else:
        return _to_bytes(obj)


def _to_bytes(obj):
    if isinstance(obj, six.text_type):
        return obj.encode('utf-8')
    elif isinstance(obj, six.binary_type):
        return obj
    else:
        return six.b(obj)


# math related functions
import math


def round(x, d=0):
    """
    Compatible round which act the same behaviour in Python3.

    Args:
        x(float) : The number to round halfway.

    Returns:
        round result of x
    """
    p = 10**d
    return float(math.floor((x * p) + math.copysign(0.5, x))) / p


def floor_division(x, y):
    return x // y
