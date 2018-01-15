from ..registry import register_layer

__activations__ = [
    'sigmoid',
    'logsigmoid',
    'exp',
    'relu',
    'tanh',
    'tanh_shrink',
    'softshrink',
    'sqrt',
    'abs',
    'ceil',
    'floor',
    'round',
    'reciprocal',
    'log',
    'square',
    'softplus',
    'softsign',
    'brelu',
    'leaky_relu',
    'soft_relu',
    'elu',
    'relu6',
    'pow',
    'stanh',
    'hard_shrink',
    'thresholded_relu',
    'hard_sigmoid',
    'swish',
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
