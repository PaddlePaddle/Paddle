from ..registry import register_layer

__activations__ = [
    'abs',
    'ceil',
    'exp',
    'floor',
    'log',
    'relu',
    'round',
    'sigmoid',
    'sqrt',
    'square',
    'tanh',
]

__all__ = [
    'mean',
    'mul',
    'dropout',
    'reshape',
    'scale',
    'transpose',
    'sigmoid_cross_entropy_with_logits',
    'elementwise_add',
    'elementwise_div',
    'elementwise_sub',
    'elementwise_mul',
    'clip',
    'sequence_softmax',
] + __activations__

for _OP in set(__all__):
    globals()[_OP] = register_layer(_OP)
