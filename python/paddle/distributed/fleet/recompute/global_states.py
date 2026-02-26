# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# 全局状态管理器
_states = {}
_state_idx = 0


def get_states():
    """获取全局状态字典"""
    return _states


def get_state_idx():
    return _state_idx


def reset_state_idx():
    global _state_idx
    _state_idx = 0


def get_states_keys():
    return _states.keys()


def set_state(value, key=None):
    """设置状态值"""
    if key is None:
        global _state_idx
        _states[_state_idx] = value
        _state_idx += 1
    else:
        _states[key] = value


def get_state(key):
    """获取状态值"""
    return _states.pop(key)


def clear_states():
    """清空所有状态"""
    _states.clear()
