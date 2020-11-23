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

from . import core

__all__ = ['set_tf32', 'allow_tf32']


def set_tf32(on_off):
    """
  Set tf32 switch by users.

  Args:
    on_off: The param passed by usrs, indecating whether activate
    the tf32 acceleration or not.
  """

    return core.set_switch(on_off)


def allow_tf32():
    """
  get the state of tf32 switch.

  Args:
    None
  """

    return core.get_switch()
