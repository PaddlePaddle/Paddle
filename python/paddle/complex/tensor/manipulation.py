#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from ..helper import is_complex, is_real, complex_variable_exists
from ...fluid.framework import ComplexVariable
from ...fluid import layers

__all__ = ['transpose']


def transpose(x, perm, name=None):
    """
    Permute the data dimensions for complex number :attr:`input` according to `perm`. 
    
    See :ref:`api_fluid_layers_transpose` for the real number API. 
    
    Args:
        x (ComplexVariable): The input n-D ComplexVariable with data type 
            complex64 or complex128.
        perm (list): Permute the input according to the value of perm.
        name (str): The name of this layer. It is optional.

    Returns:
        ComplexVariable: A transposed n-D ComplexVariable, with the same data type as :attr:`input`.

    Examples:
        .. code-block:: python
 
            import paddle
            import numpy as np
            import paddle.fluid.dygraph as dg
 
            with dg.guard():
                a = np.array([[1.0 + 1.0j, 2.0 + 1.0j], [3.0+1.0j, 4.0+1.0j]])
                x = dg.to_variable(a)
                y = paddle.complex.transpose(x, [1, 0])
                print(y.numpy())
                # [[1.+1.j 3.+1.j]
                #  [2.+1.j 4.+1.j]]
    """
    complex_variable_exists([x], "transpose")
    real = layers.transpose(x.real, perm, name)
    imag = layers.transpose(x.imag, perm, name)
    return ComplexVariable(real, imag)
