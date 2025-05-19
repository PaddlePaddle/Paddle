# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class DrrPass:

    def make_drr_ctx(self):
        drr_ctx = DrrCtx()  # noqa: F821
        drr_ctx.set_drr_pass_type(self.drr_pass_type())
        drr_ctx.init_source_pattern(self.source_pattern)
        drr_ctx.init_constraint_func(self.constraint)
        drr_ctx.init_result_pattern(self.result_pattern)
        return drr_ctx

    def constraint(self, o, t):
        return True

    def drr_pass_type(self):
        return "access_topo_drr_pass_type"


class register_drr_pass:
    def __init__(self, pass_name, tag):
        self.pass_name = pass_name
        self.tag = tag

    def __call__(self, drr_pass_cls):
        Registry.access_topo_drr_pass(  # noqa: F821
            self.pass_name, self.tag, drr_pass_cls
        )
        return drr_pass_cls
