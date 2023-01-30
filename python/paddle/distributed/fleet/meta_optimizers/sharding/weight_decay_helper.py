# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_VAR_KEY

__all__ = []


<<<<<<< HEAD
class WeightDecayHelper:
=======
class WeightDecayHelper(object):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        pass

    def _is_weight_decay_op(self, op):
<<<<<<< HEAD
        return op.desc.has_attr("op_namescope") and op.desc.attr(
            "op_namescope"
        ).startswith("/regularization")
=======
        return op.desc.has_attr("op_namescope") \
            and op.desc.attr("op_namescope").startswith("/regularization")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def prune_weight_decay(self, block, shard):
        for idx, op in reversed(list(enumerate(block.ops))):
            if not self._is_weight_decay_op(op):
                continue
            if OP_ROLE_VAR_KEY not in op.attr_names:
                raise ValueError(
                    "The Weight Dacay op should hold op_role_var attribute"
<<<<<<< HEAD
                    "but the {} op does not hold op_role_var".format(op.type)
                )
=======
                    "but the {} op does not hold op_role_var".format(op.type))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
            if not shard.has_param(op_role_var[0]):
                block._remove_op(idx, sync=False)
        block._sync_with_cpp()
