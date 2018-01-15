from ..registry import register_layer

__activations__ = [
    'abs', 'tanh', 'sigmoid', 'relu', 'sqrt', 'ceil', 'floor', 'log', 'round',
    'pow'
]

__all__ = [
    'mean', 'mul', 'reshape', 'scale', 'transpose',
    'sigmoid_cross_entropy_with_logits', 'elementwise_add', 'elementwise_div',
    'elementwise_sub', 'elementwise_mul', 'clip', 'clip_by_norm',
    'sequence_softmax', 'reduce_sum'
] + __activations__

for _OP in set(__all__):
    globals()[_OP] = register_layer(_OP)
