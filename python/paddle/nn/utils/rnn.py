# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

if TYPE_CHECKING:
    from paddle import Tensor

__all__: list[str] = []


def pad_sequence(
    sequences: list[Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
    padding_side: str = 'right',
) -> Tensor:
    r"""Pad a list of variable length Tensors with ``padding_value``.

    ``pad_sequence`` stacks a list of Tensors along a new dimension, and pads
    them to equal length. ``sequences`` can be a list of sequences with size
    ``L x *``, where ``L`` is the length of the sequence and ``*`` is any
    number of dimensions (including 0). If ``batch_first`` is ``False``, the
    output is of size ``T x B x *``, and ``B x T x *`` otherwise, where ``B``
    is the batch size (the number of elements in ``sequences``), ``T`` is the
    length of the longest sequence.

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where ``T`` is the length of the longest sequence. This function
        assumes trailing dimensions and type of all the Tensors in sequences
        are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): if ``True``, the output will be in
            ``B x T x *`` format, ``T x B x *`` otherwise. Default: ``False``.
        padding_value (float, optional): value for padded elements.
            Default: ``0.0``.
        padding_side (str, optional): the side to pad the sequences on,
            either ``'right'`` or ``'left'``. Default: ``'right'``.

    Returns:
        Tensor: Tensor of size ``T x B x *`` if ``batch_first`` is ``False``,
        or ``B x T x *`` otherwise.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> a = paddle.ones([25, 300])
            >>> b = paddle.ones([22, 300])
            >>> c = paddle.ones([15, 300])
            >>> padded = paddle.nn.utils.pad_sequence([a, b, c])
            >>> print(padded.shape)
            paddle.Size([25, 3, 300])
            >>> padded = paddle.nn.utils.pad_sequence([a, b, c], batch_first=True)
            >>> print(padded.shape)
            paddle.Size([3, 25, 300])
    """
    if not isinstance(sequences, list):
        raise TypeError(
            f"pad_sequence expects a list of Tensors, but got {type(sequences)}"
        )
    if padding_side not in ('right', 'left'):
        raise ValueError(
            f"padding_side must be 'right' or 'left', but got '{padding_side}'"
        )

    max_len = max(seq.shape[0] for seq in sequences)
    trailing_dims = sequences[0].shape[1:]
    dtype = sequences[0].dtype

    padded_seqs = []
    for seq in sequences:
        length = seq.shape[0]
        if length == max_len:
            padded_seqs.append(seq)
        else:
            pad_size = [max_len - length, *list(trailing_dims)]
            padding = paddle.full(pad_size, padding_value, dtype=dtype)
            if padding_side == 'right':
                padded_seqs.append(paddle.concat([seq, padding], axis=0))
            else:
                padded_seqs.append(paddle.concat([padding, seq], axis=0))

    out = paddle.stack(padded_seqs, axis=0)

    if not batch_first:
        # Transpose from B x T x * to T x B x *
        perm = [1, 0, *list(range(2, len(out.shape)))]
        out = out.transpose(perm)

    return out


def unpad_sequence(
    padded_sequences: Tensor,
    lengths: Tensor,
    batch_first: bool = False,
) -> list[Tensor]:
    r"""Unpad a padded Tensor into a list of variable length Tensors.

    ``unpad_sequence`` unstacks a padded Tensor into a list of variable length
    Tensors.

    Args:
        padded_sequences (Tensor): padded sequences.
        lengths (Tensor): length of original (unpadded) sequences.
        batch_first (bool, optional): whether batch dimension is first or not.
            Default: ``False``.

    Returns:
        list[Tensor]: a list of Tensor objects with original lengths.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> a = paddle.ones([25, 300])
            >>> b = paddle.ones([22, 300])
            >>> c = paddle.ones([15, 300])
            >>> sequences = [a, b, c]
            >>> padded = paddle.nn.utils.pad_sequence(sequences)
            >>> lengths = paddle.to_tensor([v.shape[0] for v in sequences])
            >>> unpadded = paddle.nn.utils.unpad_sequence(padded, lengths)
            >>> paddle.allclose(sequences[0], unpadded[0]).item()
            True
            >>> paddle.allclose(sequences[1], unpadded[1]).item()
            True
            >>> paddle.allclose(sequences[2], unpadded[2]).item()
            True
    """
    if not batch_first:
        # Transpose from T x B x * to B x T x *
        perm = [1, 0, *list(range(2, len(padded_sequences.shape)))]
        padded_sequences = padded_sequences.transpose(perm)

    unpadded = []
    for seq, length in zip(padded_sequences, lengths):
        length_val = length.item()
        unpadded.append(seq[:length_val])

    return unpadded
