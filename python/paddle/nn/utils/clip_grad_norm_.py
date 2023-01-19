#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

__all__ = []


def clip_grad_norm_(
    parameters: paddle.Tensor,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> paddle.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    This API can only run in dynamic mode, not static mode.

    Args:
        parameters (paddle.Tensor): Tensors or a single Tensor
            that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    Example:
        .. code-block:: python
            import paddle

            x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
            max_norm = float(5.0)
            linear = paddle.nn.Linear(in_features=10, out_features=10,
                                      weight_attr=paddle.ParamAttr(need_clip=True),
                                      bias_attr=paddle.ParamAttr(need_clip=False))
            out = linear(x)
            loss = paddle.mean(out)
            loss.backward()

            clip_grad_norm_(linear.parameters(), max_norm)

            sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters())
            sdg.step()
    """
    if not paddle.in_dynamic_mode():
        raise RuntimeError('this API can only run in dynamic mode.')

    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]

    support_norm_type = [float("inf"), 0, 1, 2]
    if norm_type not in support_norm_type:
        raise ValueError(f'norm_type only support {support_norm_type}')

    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return paddle.to_tensor(0.0)
    if norm_type == float("inf"):
        norms = [g.detach().abs().max() for g in grads]
        total_norm = (
            norms[0] if len(norms) == 1 else paddle.max(paddle.stack(norms))
        )
    else:
        total_norm = paddle.linalg.norm(
            paddle.stack(
                [paddle.linalg.norm(g.detach(), norm_type) for g in grads]
            ),
            norm_type,
        )

    if error_if_nonfinite and paddle.logical_or(
        total_norm.isnan(), total_norm.isinf()
    ):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`'
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional.
    clip_coef_clamped = paddle.clip(clip_coef, max=1.0)

    for _, p in enumerate(parameters):
        g = p.grad
        if g is not None:
            p.grad = paddle.multiply(x=g, y=clip_coef_clamped)
    return total_norm
