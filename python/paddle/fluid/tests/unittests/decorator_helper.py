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

import paddle.fluid as fluid

__all__ = ['many_times', 'prog_scope']


def many_times(times):

    def __impl__(fn):

        def __fn__(*args, **kwargs):
            for _ in range(times):
                fn(*args, **kwargs)

        return __fn__

    return __impl__


def prog_scope():

    def __impl__(fn):

        def __fn__(*args, **kwargs):
            prog = fluid.Program()
            startup_prog = fluid.Program()
            scope = fluid.core.Scope()
            with fluid.scope_guard(scope):
                with fluid.program_guard(prog, startup_prog):
                    fn(*args, **kwargs)

        return __fn__

    return __impl__
