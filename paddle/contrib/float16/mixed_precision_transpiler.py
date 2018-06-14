"""
https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/
Use half-precision to training neural-network with tiny accuracy loss even without accuracy loss.
pros:
1. Shorten the training or inference time
2. Decrease the required amount of memory
cons:
may loss some precision. Please keep an eye open on the convergence result.
"""

import numpy as np
import paddle.fluid.core as core
from paddle.fluid.framework import Program
from paddle.fluid.executor import global_scope
from float16_transpiler import Float16Transpiler


class Float16TrainingTranspiler(object):
    """
    Fp16Training transpiler take a program, and generate a support fp16 training 
    successful training of DNNs with half precision need three pre: accumulation of FP16 products into FP32; loss scaling; and an FP32 master copy of weights
    """
    def __init__(self, program):
        assert isinstance(program, Program), "Expect argument of Program, but get %s" %(type(program))
        self._program = program
