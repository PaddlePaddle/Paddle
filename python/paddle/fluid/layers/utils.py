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


def convert_to_list(value, n, name):
    """Converts a single integer or iterable of integers into an integer list.

    Arguments:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the list to be returned.
      name: The name of the argument being validated, e.g. "stride" or
        "filter_size". This is only used to format error messages.

    Returns:
      A list of n integers.

    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return [value, ] * n
    else:
        try:
            value_list = list(value)
        except TypeError:
            raise ValueError("The " + name +
                             "'s type must be list or tuple. Received: " + str(
                                 value))
        if len(value_list) != n:
            raise ValueError("The " + name + "'s length must be " + str(n) +
                             ". Received: " + str(value))
        for single_value in value_list:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    "The " + name + "'s type must be a list or tuple of " + str(
                        n) + " integers. Received: " + str(value) + " "
                    "including element " + str(single_value) + " of type" + " "
                    + str(type(single_value)))
        return value_list
