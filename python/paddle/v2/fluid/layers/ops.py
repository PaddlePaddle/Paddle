from ..registry import register_layer
__all__ = [
    'mean', 'mul', 'dropout', 'reshape', 'sigmoid', 'scale', 'transpose',
    'sigmoid_cross_entropy_with_logits', 'elementwise_add', 'elementwise_div',
    'elementwise_sub', 'elementwise_mul', 'clip', 'abs'
]

for _OP in set(__all__):
    globals()[_OP] = register_layer(_OP)
