# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle

from .. import functional as F
from ..layer.common import Linear
from ..layer.conv import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose

if TYPE_CHECKING:
    from typing_extensions import Never

    from paddle import Tensor
    from paddle.nn import Layer

__all__ = []


def normal_(x: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    temp_value = paddle.normal(mean, std, shape=x.shape)
    paddle.assign(temp_value, x)
    return x


class SpectralNorm:
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(
        self,
        name: str = 'weight',
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(
                'Expected n_power_iterations to be positive, but '
                f'got n_power_iterations={n_power_iterations}'
            )
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight: Tensor) -> Tensor:
        weight_mat = weight
        if self.dim != 0:
            # transpose dim to front
            weight_mat = weight_mat.transpose(
                [self.dim]
                + [d for d in range(weight_mat.dim()) if d != self.dim]
            )

        height = weight_mat.shape[0]

        return weight_mat.reshape([height, -1])

    def compute_weight(self, layer: Layer, do_power_iteration: bool) -> Tensor:
        weight = getattr(layer, self.name + '_orig')
        u = getattr(layer, self.name + '_u')
        v = getattr(layer, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with paddle.no_grad():
                for _ in range(self.n_power_iterations):
                    paddle.assign(
                        F.normalize(
                            paddle.matmul(
                                weight_mat,
                                u,
                                transpose_x=True,
                                transpose_y=False,
                            ),
                            axis=0,
                            epsilon=self.eps,
                        ),
                        v,
                    )

                    paddle.assign(
                        F.normalize(
                            paddle.matmul(weight_mat, v),
                            axis=0,
                            epsilon=self.eps,
                        ),
                        u,
                    )
                if self.n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = paddle.dot(u, paddle.mv(weight_mat, v))
        weight = weight / sigma
        return weight

    def __call__(self, layer: Layer, inputs: Never) -> None:
        setattr(
            layer,
            self.name,
            self.compute_weight(layer, do_power_iteration=layer.training),
        )

    @staticmethod
    def apply(
        layer: Layer, name: str, n_power_iterations: int, dim: int, eps: float
    ) -> SpectralNorm:
        for k, hook in layer._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    f"the same parameter {name}"
                )

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = layer._parameters[name]

        with paddle.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.shape

            # randomly initialize u and v
            u = layer.create_parameter([h])
            u = normal_(u, 0.0, 1.0)
            v = layer.create_parameter([w])
            v = normal_(v, 0.0, 1.0)
            u = F.normalize(u, axis=0, epsilon=fn.eps)
            v = F.normalize(v, axis=0, epsilon=fn.eps)

        # delete fn.name form parameters, otherwise you can not set attribute
        del layer._parameters[fn.name]
        layer.add_parameter(fn.name + "_orig", weight)
        # still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an Parameter and
        # gets added as a parameter. Instead, we register weight * 1.0 as a plain
        # attribute.
        setattr(layer, fn.name, weight * 1.0)
        layer.register_buffer(fn.name + "_u", u)
        layer.register_buffer(fn.name + "_v", v)
        layer.register_forward_pre_hook(fn)
        return fn


def spectral_norm(
    layer: Layer,
    name: str = 'weight',
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: int | None = None,
) -> Layer:
    r"""
    Applies spectral normalization to a parameter according to the
    following Calculation:

    Step 1:
    Generate vector U in shape of [H], and V in shape of [W].
    While H is the :attr:`dim` th dimension of the input weights,
    and W is the product result of remaining dimensions.

    Step 2:
    :attr:`n_power_iterations` should be a positive integer, do following
    calculations with U and V for :attr:`power_iters` rounds.

    .. math::

        \mathbf{v} := \frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}

        \mathbf{u} := \frac{\mathbf{W} \mathbf{v}}{\|\mathbf{W} \mathbf{v}\|_2}

    Step 3:
    Calculate :math:`\sigma(\mathbf{W})` and normalize weight values.

    .. math::

        \sigma(\mathbf{W}) = \mathbf{u}^{T} \mathbf{W} \mathbf{v}

        \mathbf{W} = \frac{\mathbf{W}}{\sigma(\mathbf{W})}


    Refer to `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .

    Parameters:
        layer(Layer): Layer of paddle, which has weight.
        name(str, optional): Name of the weight parameter. Default: 'weight'.
        n_power_iterations(int, optional): The number of power iterations to calculate spectral norm. Default: 1.
        eps(float, optional): The epsilon for numerical stability in calculating norms. Default: 1e-12.
        dim(int|None, optional): The index of dimension which should be permuted to the first before reshaping Input(Weight) to matrix, it should be set as 0 if Input(Weight) is the weight of fc layer, and should be set as 1 if Input(Weight) is the weight of conv layer. Default: None.

    Returns:
        Layer, the original layer with the spectral norm hook.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.nn import Conv2D
            >>> from paddle.nn.utils import spectral_norm
            >>> paddle.seed(2023)
            >>> conv = Conv2D(3, 1, 3)
            >>> sn_conv = spectral_norm(conv)
            >>> print(sn_conv)
            Conv2D(3, 1, kernel_size=[3, 3], data_format=NCHW)
            >>> # Conv2D(3, 1, kernel_size=[3, 3], data_format=NCHW)
            >>> print(sn_conv.weight)
            Tensor(shape=[1, 3, 3, 3], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[[[ 0.01668976,  0.30305523,  0.11405435],
               [-0.06765547, -0.50396705, -0.40925547],
               [ 0.47344422,  0.03628403,  0.45277366]],
              [[-0.15177251, -0.16305730, -0.15723954],
               [-0.28081197, -0.09183260, -0.08081978],
               [-0.40895155,  0.18298769, -0.29325116]],
              [[ 0.21819633, -0.01822380, -0.50351536],
               [-0.06262003,  0.17713565,  0.20517939],
               [ 0.16659889, -0.14333329,  0.05228264]]]])

    """

    if dim is None:
        if isinstance(
            layer, (Conv1DTranspose, Conv2DTranspose, Conv3DTranspose, Linear)
        ):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(layer, name, n_power_iterations, dim, eps)
    return layer
