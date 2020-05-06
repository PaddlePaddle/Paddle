#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..helper import is_complex, is_real, complex_variable_exists
from ....fluid.framework import ComplexVariable
from ....fluid import layers

__all__ = ['matmul', ]


def matmul(x, y, transpose_x=False, transpose_y=False, alpha=1.0, name=None):
    """
    Applies matrix multiplication to two complex number tensors. See the 
    detailed description in :ref:`api_fluid_layers_matmul`.

    Args:
        x (ComplexVariable|Variable): The first input, can be a ComplexVariable 
            with data type complex64 or complex128, or a Variable with data type 
            float32 or float64.
        y (ComplexVariable|Variable): The second input, can be a ComplexVariable 
            with data type complex64 or complex128, or a Variable with data type 
            float32 or float64.
        transpose_x (bool): Whether to transpose :math:`x` before multiplication.
        transpose_y (bool): Whether to transpose :math:`y` before multiplication.
        alpha (float): The scale of output. Default 1.0.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.
   
    Returns:
        ComplexVariable: The product result, with the same data type as inputs.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.fluid.dygraph as dg
            with dg.guard():
                x = np.array([[1.0 + 1j, 2.0 + 1j], [3.0+1j, 4.0+1j]])
                y = np.array([1.0 + 1j, 1.0 + 1j])
                x_var = dg.to_variable(x)
                y_var = dg.to_variable(y)
                result = paddle.complex.matmul(x_var, y_var)
                print(result.numpy())
                # [1.+5.j 5.+9.j]         
    """
    # x = a + bi, y = c + di
    # mm(x, y) = mm(a, c) - mm(b, d) + (mm(a, d) + mm(b, c))i
    complex_variable_exists([x, y], "matmul")
    a, b = (x.real, x.imag) if is_complex(x) else (x, None)
    c, d = (y.real, y.imag) if is_complex(y) else (y, None)
    ac = layers.matmul(a, c, transpose_x, transpose_y, alpha, name)
    if is_real(b) and is_real(d):
        bd = layers.matmul(b, d, transpose_x, transpose_y, alpha, name)
        real = ac - bd
        imag = layers.matmul(a, d, transpose_x, transpose_y, alpha, name) + \
               layers.matmul(b, c, transpose_x, transpose_y, alpha, name)
    elif is_real(b):
        real = ac
        imag = layers.matmul(b, c, transpose_x, transpose_y, alpha, name)
    else:
        real = ac
        imag = layers.matmul(a, d, transpose_x, transpose_y, alpha, name)
    return ComplexVariable(real, imag)
