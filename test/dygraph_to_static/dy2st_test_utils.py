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

from functools import wraps

import numpy as np

from paddle import set_flags, static
from paddle.fluid import core


def test_with_new_ir(func):
    @wraps(func)
    def impl(*args, **kwargs):
        ir_outs = None
        with static.scope_guard(static.Scope()):
            with static.program_guard(static.Program()):
                new_ir_flag = 'FLAGS_enable_new_ir_in_executor'
                set_flags({new_ir_flag: True})
                ir_outs = func(*args, **kwargs)
                set_flags({new_ir_flag: False})
        return ir_outs

    return impl


def test_and_compare_with_new_ir(func):
    @wraps(func)
    def impl(*args, **kwargs):
        outs = func(*args, **kwargs)
        if core._is_bwd_prim_enabled() or core._is_fwd_prim_enabled():
            return outs
        ir_outs = test_with_new_ir(func)(*args, **kwargs)
        if ir_outs is None:
            return outs
        for i in range(len(outs)):
            np.testing.assert_array_equal(
                outs[i],
                ir_outs[i],
                err_msg='Dy2St Unittest Check ('
                + func.__name__
                + ') has diff '
                + '\nExpect '
                + str(outs[i])
                + '\n'
                + 'But Got'
                + str(ir_outs[i]),
            )
        return outs

    return impl
