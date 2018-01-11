from ..registry import register_layer

__activations__ = [
    'abs', 'tanh', 'sigmoid', 'relu', 'sqrt', 'ceil', 'floor', 'log', 'round'
]

__all__ = [
    'mean',
    'mul',
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
