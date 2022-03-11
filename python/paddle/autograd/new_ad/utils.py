# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import threading


def var2dot(var):
    lookup_tab = adrunner_state.dot_lookup
    return lookup_tab[var] if var in lookup_tab else None


def set_var2dot(var, dot):
    lookup_tab = adrunner_state.dot_lookup
    lookup_tab[var] = dot


def dot2bar(dot):
    lookup_tab = adrunner_state.bar_lookup
    return lookup_tab[dot] if dot in lookup_tab else None


def set_var2dot(dot, bar):
    lookup_tab = adrunner_state.bar_lookup
    lookup_tab[dot] = bar


def set_tangent(name):
    tangent_set = adrunner_state.tangent_set
    tangent_set.insert(name)


def is_tangent(name):
    tangent_set = adrunner_state.tangent_set
    return tangent_set.count(name)


class VarGenerator(thearding.local):
    def __init__(self):
        super().__init__()
        self.cnt = 0

    def next_var(self, is_tangent, ref_var):
        name = '_ad_tmp_var_' + str(self.cnt)
        self.cnt = self.cnt + 1
        if is_tangent:
            set_tangent(name)
        block = default_main_program().current_block()
        if ref_var is not None:
            block.create_var(
                name=name,
                shape=ref_var.shape,
                dtype=ref_var.dtype,
                type=ref_var.type)
        else:
            block.create_var(name=name)
        return name


var_generator = VarGenerator()


def make_var(is_tangent=False, ref_var=None):
    return var_generator.next_var(is_tangent, ref_var)
