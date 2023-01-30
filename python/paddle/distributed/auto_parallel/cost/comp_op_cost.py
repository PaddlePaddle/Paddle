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
# limitations under the License

<<<<<<< HEAD
from .base_cost import CompOpCost, register_op_cost
=======
from .base_cost import Cost, register_op_cost, CompOpCost, _g_op_cost_factory
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


@register_op_cost
class AdamOpCost(CompOpCost):
    OP_TYPE = "adam"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(AdamOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ArgsortOpCost(CompOpCost):
    OP_TYPE = "argsort"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ArgsortOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class AssignOpCost(CompOpCost):
    OP_TYPE = "assign"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(AssignOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class AssignValueOpCost(CompOpCost):
    OP_TYPE = "assign_value"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(AssignValueOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class BeamSearchOpCost(CompOpCost):
    OP_TYPE = "beam_search"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(BeamSearchOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class BeamSearchDecodeOpCost(CompOpCost):
    OP_TYPE = "beam_search_decode"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(BeamSearchDecodeOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class CastOpCost(CompOpCost):
    OP_TYPE = "cast"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(CastOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ConcatOpCost(CompOpCost):
    OP_TYPE = "concat"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ConcatOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class DropoutOpCost(CompOpCost):
    OP_TYPE = "dropout"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(DropoutOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class DropoutGradOpCost(CompOpCost):
    OP_TYPE = "dropout_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(DropoutGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ElementwiseAddOpCost(CompOpCost):
    OP_TYPE = "elementwise_add"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ElementwiseAddOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ElementwiseAddGradOpCost(CompOpCost):
    OP_TYPE = "elementwise_add_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ElementwiseAddGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ElementwiseDivOpCost(CompOpCost):
    OP_TYPE = "elementwise_div"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ElementwiseDivOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ElementwiseDivGradOpCost(CompOpCost):
    OP_TYPE = "elementwise_div_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ElementwiseDivGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ElementwiseMulOpCost(CompOpCost):
    OP_TYPE = "elementwise_mul"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ElementwiseMulOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ElementwiseMulGradOpCost(CompOpCost):
    OP_TYPE = "elementwise_mul_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ElementwiseMulGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ElementwiseSubOpCost(CompOpCost):
    OP_TYPE = "elementwise_sub"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ElementwiseSubOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ElementwiseSubGradOpCost(CompOpCost):
    OP_TYPE = "elementwise_sub_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ElementwiseSubGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class EqualOpCost(CompOpCost):
    OP_TYPE = "equal"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(EqualOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class EmbeddingOpCost(CompOpCost):
    OP_TYPE = "c_embedding"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(EmbeddingOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class EmbeddingGradOpCost(CompOpCost):
    OP_TYPE = "c_embedding_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(EmbeddingGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class FillConstantOpCost(CompOpCost):
    OP_TYPE = "fill_constant"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(FillConstantOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class FillConstantBatchSizeLikeOpCost(CompOpCost):
    OP_TYPE = "fill_constant_batch_size_like"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(FillConstantBatchSizeLikeOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class FusedSoftmaxMaskUpperTriangleOpCost(CompOpCost):
    OP_TYPE = "fused_softmax_mask_upper_triangle"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(FusedSoftmaxMaskUpperTriangleOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class FusedSoftmaxMaskUpperTriangleGradOpCost(CompOpCost):
    OP_TYPE = "fused_softmax_mask_upper_triangle_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(FusedSoftmaxMaskUpperTriangleGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class GatherOpCost(CompOpCost):
    OP_TYPE = "gather"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(GatherOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class GeluOpCost(CompOpCost):
    OP_TYPE = "gelu"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(GeluOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class GeluGradOpCost(CompOpCost):
    OP_TYPE = "gelu_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(GeluGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class GreaterEqualOpCost(CompOpCost):
    OP_TYPE = "greater_equal"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(GreaterEqualOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class IncrementOpCost(CompOpCost):
    OP_TYPE = "increment"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(IncrementOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class IsEmptyOpCost(CompOpCost):
    OP_TYPE = "is_empty"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(IsEmptyOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class LayerNormOpCost(CompOpCost):
    OP_TYPE = "layer_norm"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(LayerNormOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class LayerNormGradOpCost(CompOpCost):
    OP_TYPE = "layer_norm_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(LayerNormGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class LessThanOpCost(CompOpCost):
    OP_TYPE = "less_than"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(LessThanOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class LogicalNotOpCost(CompOpCost):
    OP_TYPE = "logical_not"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(LogicalNotOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class LogicalAndOpCost(CompOpCost):
    OP_TYPE = "logical_and"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(LogicalAndOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class LodResetOpCost(CompOpCost):
    OP_TYPE = "lod_reset"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(LodResetOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class LogOpCost(CompOpCost):
    OP_TYPE = "log"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(LogOpCost, self).__init__(op=op, op_desc=op_desc, cluster=cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class LookupTableV2OpCost(CompOpCost):
    OP_TYPE = "lookup_table_v2"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(LookupTableV2OpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class LookupTableV2GradOpCost(CompOpCost):
    OP_TYPE = "lookup_table_v2_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(LookupTableV2GradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class MatmulOpCost(CompOpCost):
    OP_TYPE = "matmul"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(MatmulOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class MatmulGradOpCost(CompOpCost):
    OP_TYPE = "matmul_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(MatmulGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class MatmulV2OpCost(CompOpCost):
    OP_TYPE = "matmul_v2"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(MatmulV2OpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class MatmulV2GradOpCost(CompOpCost):
    OP_TYPE = "matmul_v2_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(MatmulV2GradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class MemcpyOpCost(CompOpCost):
    OP_TYPE = "memcpy"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(MemcpyOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class MulOpCost(CompOpCost):
    OP_TYPE = "mul"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(MulOpCost, self).__init__(op=op, op_desc=op_desc, cluster=cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class MulGradOpCost(CompOpCost):
    OP_TYPE = "mul_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(MulGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class OneHotOpCost(CompOpCost):
    OP_TYPE = "one_hot"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(OneHotOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ReadFromArrayOpCost(CompOpCost):
    OP_TYPE = "read_from_array"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ReadFromArrayOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ReduceSumOpCost(CompOpCost):
    OP_TYPE = "reduce_sum"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ReduceSumOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ReduceSumGradOpCost(CompOpCost):
    OP_TYPE = "reduce_sum_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ReduceSumGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class Reshape2OpCost(CompOpCost):
    OP_TYPE = "reshape2"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(Reshape2OpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class Reshape2GradOpCost(CompOpCost):
    OP_TYPE = "reshape2_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(Reshape2GradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ReduceMeanOpCost(CompOpCost):
    OP_TYPE = "reduce_mean"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ReduceMeanOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ReduceMeanGradOpCost(CompOpCost):
    OP_TYPE = "reduce_mean_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ReduceMeanGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class SamplingIdOpCost(CompOpCost):
    OP_TYPE = "sampling_id"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(SamplingIdOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class ScaleOpCost(CompOpCost):
    OP_TYPE = "scale"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(ScaleOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class SliceOpCost(CompOpCost):
    OP_TYPE = "slice"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(SliceOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class SoftmaxOpCost(CompOpCost):
    OP_TYPE = "softmax"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(SoftmaxOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class SoftmaxGradOpCost(CompOpCost):
    OP_TYPE = "softmax_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(SoftmaxGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class SoftmaxWithCrossEntropyOpCost(CompOpCost):
    OP_TYPE = "softmax_with_cross_entropy"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(SoftmaxWithCrossEntropyOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class SoftmaxWithCrossEntropyGradOpCost(CompOpCost):
    OP_TYPE = "softmax_with_cross_entropy_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(SoftmaxWithCrossEntropyGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class SplitOpCost(CompOpCost):
    OP_TYPE = "split"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(SplitOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class Squeeze2OpCost(CompOpCost):
    OP_TYPE = "squeeze2"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(Squeeze2OpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class SquareOpCost(CompOpCost):
    OP_TYPE = "square"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(SquareOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class SquareGradOpCost(CompOpCost):
    OP_TYPE = "square_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(SquareGradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class SumOpCost(CompOpCost):
    OP_TYPE = "sum"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(SumOpCost, self).__init__(op=op, op_desc=op_desc, cluster=cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class TopKOpCost(CompOpCost):
    OP_TYPE = "top_k"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(TopKOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class Transpose2OpCost(CompOpCost):
    OP_TYPE = "transpose2"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(Transpose2OpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class Transpose2GradOpCost(CompOpCost):
    OP_TYPE = "transpose2_grad"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(Transpose2GradOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class Unsqueeze2OpCost(CompOpCost):
    OP_TYPE = "unsqueeze2"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(Unsqueeze2OpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0


@register_op_cost
class WriteToArrayOpCost(CompOpCost):
    OP_TYPE = "write_to_array"

    def __init__(self, op=None, op_desc=None, cluster=None):
<<<<<<< HEAD
        super().__init__(op=op, op_desc=op_desc, cluster=cluster)
=======
        super(WriteToArrayOpCost, self).__init__(
            op=op, op_desc=op_desc, cluster=cluster
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # For a concrete COMP OP, the calc_time and calc_flops function need to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0
