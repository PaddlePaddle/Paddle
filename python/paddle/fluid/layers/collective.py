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

from __future__ import print_function
from ..layer_helper import LayerHelper, unique_name


def _allreduce(x, out=None, reduce_type="sum", sync_mode=False):
    helper = LayerHelper("allreduce", **locals())
    # Convert string reduce type to op int type
    red_typ_int = 0
    if reduce_type == "sum":
        red_typ_int = 0
    elif reduce_type == "prod":
        red_typ_int = 1
    elif reduce_type == "max":
        red_typ_int = 2
    elif reduce_type == "min":
        red_typ_int = 3
    else:
        raise TypeError("reduce type can only be [sum|prod|max|min]")

    if out is None:
        out = helper.create_variable(
            name=unique_name.generate(".".join([x.name, 'tmp'])),
            shape=x.shape,
            dtype=x.dtype,
            type=x.type,
            persistable=x.persistable,
            stop_gradient=True)
    helper.append_op(
        type='allreduce',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={"reduce_type": red_typ_int,
               "sync_mode": sync_mode})
    return out
