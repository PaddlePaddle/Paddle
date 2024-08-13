# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# This file stores the customized function that will be called by the dispatch mechanism.

from ...utils import BreakGraphError, FallbackError


def raise_break_graph_fn(*args, **kwarg):
    raise BreakGraphError("raise by raise_break_graph_fn.")


def raise_not_implement_fn(*args, **kwarg):
    raise FallbackError("raise by raise_break_graph_fn.")


# just a function for operator.in
def operator_in(left, right):
    return left in right


def operator_not_in(left, right):
    return left not in right


def operator_exception_match(left, right):
    pass


def operator_BAD(left, right):
    pass


def operator_is_none(val):
    pass


def operator_is_not_none(val):
    pass


def tensor_numel(x):
    pass
