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


class NameGen(thearding.local):
    def __init__(self):
        super().__init__()
        self.cnt = 0

    def get_name(self):
        name = 'name_gen_' + str(self.cnt)
        self.cnt = self.cnt + 1
        return name

    def get_var(self, block=None, ref_var=None, shape=None):
        name = self.get_name()
        new_shape = ref_var.shape if shape is None else shape
        block.create_var(
            name=name,
            shape=new_shape,
            dtype=ref_var.dtype,
            type=ref_var.type,
            persistable=False,
            stop_gradient=False)
        return name


name_gen = NameGen()
