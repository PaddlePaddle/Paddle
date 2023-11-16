# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib

from paddle import static


@contextlib.contextmanager
def new_program_scope(main=None, startup=None, scope=None):
    prog = main if main else static.Program()
    startup_prog = startup if startup else static.Program()
    scope = scope if scope else static.core.Scope()
    with static.scope_guard(scope):
        with static.program_guard(prog, startup_prog):
            with static.unique_name.guard():
                yield
