#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from paddle import Tensor

    MaskModSignature: TypeAlias = Callable[
        [Tensor, Tensor, Tensor, Tensor], Tensor
    ]

__all__ = ["or_masks", "and_masks"]


def or_masks(*mask_mods: MaskModSignature) -> MaskModSignature:
    """
    Return a mask function that computes the union of provided mask functions.

    Args:
        *mask_mods (Callable): Mask functions with signature
            ``mask_mod(b, h, q_idx, kv_idx)``.

    Returns:
        Callable: A mask function that applies logical OR to all mask results.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> from paddle.nn.attention.flex_attention import or_masks

            >>> def mask_a(b, h, q_idx, kv_idx):
            ...     return q_idx >= kv_idx

            >>> def mask_b(b, h, q_idx, kv_idx):
            ...     return h == 0

            >>> b = paddle.to_tensor([0])
            >>> h = paddle.to_tensor([1])
            >>> q_idx = paddle.to_tensor([2])
            >>> kv_idx = paddle.to_tensor([3])
            >>> mask = or_masks(mask_a, mask_b)
            >>> print(mask(b, h, q_idx, kv_idx))
            Tensor(shape=[1], dtype=bool, place=Place(cpu), stop_gradient=True,
                   [False])
    """
    if not all(callable(arg) for arg in mask_mods):
        raise RuntimeError(
            f"All inputs should be callable mask_mods: {mask_mods}"
        )

    def or_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        result = b.new_zeros((), dtype='bool')
        for mask in mask_mods:
            result = result | mask(b, h, q_idx, kv_idx)
        return result

    return or_mask


def and_masks(*mask_mods: MaskModSignature) -> MaskModSignature:
    """
    Return a mask function that computes the intersection of provided mask functions.

    Args:
        *mask_mods (Callable): Mask functions with signature
            ``mask_mod(b, h, q_idx, kv_idx)``.

    Returns:
        Callable: A mask function that applies logical AND to all mask results.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> from paddle.nn.attention.flex_attention import and_masks

            >>> def mask_a(b, h, q_idx, kv_idx):
            ...     return q_idx >= kv_idx

            >>> def mask_b(b, h, q_idx, kv_idx):
            ...     return h == 0

            >>> b = paddle.to_tensor([0])
            >>> h = paddle.to_tensor([0])
            >>> q_idx = paddle.to_tensor([2])
            >>> kv_idx = paddle.to_tensor([1])
            >>> mask = and_masks(mask_a, mask_b)
            >>> print(mask(b, h, q_idx, kv_idx))
            Tensor(shape=[1], dtype=bool, place=Place(cpu), stop_gradient=True,
                   [True])
    """
    if not all(callable(arg) for arg in mask_mods):
        raise RuntimeError(
            f"All inputs should be callable mask_mods: {mask_mods}"
        )

    def and_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        result = b.new_ones((), dtype='bool')
        for mask in mask_mods:
            result = result & mask(b, h, q_idx, kv_idx)
        return result

    return and_mask
