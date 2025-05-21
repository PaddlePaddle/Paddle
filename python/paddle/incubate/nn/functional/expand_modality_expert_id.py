from __future__ import annotations
from typing import TYPE_CHECKING
from paddle import _C_ops
import paddle
# from ....framework import LayerHelper, in_dynamic_or_pir_mode
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper

if TYPE_CHECKING:
    from paddle import Tensor

def expand_modality_expert_id(
    expert_id: Tensor, 
    num_expert_per_modality: int, 
    group_size: int, 
    modality_offset: int, 
    is_group_expert: bool,
    name: str | None = None
) -> Tensor:
    """
    Args:
        expert_id:
        num_expert_per_modality: 
        group_size:
        modality_offset:
        is_group_expert:
    
    Returns:
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.expand_modality_expert_id(expert_id, num_expert_per_modality, group_size, modality_offset, is_group_expert)
    helper = LayerHelper('expand_modality_expert_id', **locals())
    expert_id_out = helper.create_variable_for_type_inference(dtype=expert_id.dtype)
    inputs = {
      'expert_id': expert_id
    }
    attrs = {
      'num_expert_per_modality': num_expert_per_modality,
      'group_size': group_size,
      'modality_offset': modality_offset,
      'is_group_expert': is_group_expert
    }
    helper.append_op(type='expand_modality_expert_id', inputs=inputs, attrs=attrs, outputs={'expert_id_out': expert_id_out})
    return y
