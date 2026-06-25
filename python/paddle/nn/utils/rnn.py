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

from collections.abc import Iterable
from typing import TYPE_CHECKING

import paddle

if TYPE_CHECKING:
    from paddle import Tensor

__all__ = [
    "PackedSequence",
    "invert_permutation",
    "pack_padded_sequence",
    "pad_packed_sequence",
    "pad_sequence",
    "unpad_sequence",
    "pack_sequence",
    "unpack_sequence",
]


def invert_permutation(permutation: Tensor | None) -> Tensor | None:
    """Returns the inverse of ``permutation``.

    This is useful for converting between sorted and unsorted indices in
    a :class:`~nn.utils.rnn.PackedSequence`.

    Args:
        permutation (Tensor|None): a 1-D tensor of indices to invert.

    Returns:
        Tensor|None: the inverse permutation tensor, or None if input is None.

    Examples:
        >>> import paddle
    """
    if permutation is None:
        return None
    # Use paddle.scatter instead of scatter_ for better static mode support
    output = paddle.scatter(
        paddle.zeros_like(permutation),
        permutation,
        paddle.arange(
            0,
            permutation.numel(),
            dtype=permutation.dtype,
            device=permutation.place,
        ),
        overwrite=True,
    )
    return output


class PackedSequence:
    """Holds the data and batch sizes of a packed sequence.

    PackedSequence is used to represent a packed sequence, which is typically
    produced by ``pack_padded_sequence`` and consumed by ``pad_packed_sequence``.

    Args:
        data (Tensor): The packed data tensor.
        batch_sizes (Tensor): A tensor containing the batch size at each step.
        sorted_indices (Tensor|None, optional): The indices used to sort the sequences.
        unsorted_indices (Tensor|None, optional): The indices to restore the original order.

    Examples:
        .. code-block:: pycon

            >>> import paddle

    """

    def __init__(
        self,
        data: Tensor,
        batch_sizes: Tensor,
        sorted_indices: Tensor | None = None,
        unsorted_indices: Tensor | None = None,
    ):
        self.data = data
        self.batch_sizes = batch_sizes
        self.sorted_indices = sorted_indices
        self.unsorted_indices = unsorted_indices

    @property
    def is_pinned(self) -> bool:
        return (
            self.data.place.is_cuda_pinned_place()
            or self.data.place.is_xpu_pinned_place()
        )

    def to(self, *args, **kwargs) -> PackedSequence:
        data = self.data.to(*args, **kwargs)
        if data is self.data:
            return self
        # Only convert indices to same device as data, not dtype
        target_device = data.place
        sorted_indices = (
            self.sorted_indices.to(target_device)
            if self.sorted_indices is not None
            else None
        )
        unsorted_indices = (
            self.unsorted_indices.to(target_device)
            if self.unsorted_indices is not None
            else None
        )
        return PackedSequence(
            data, self.batch_sizes, sorted_indices, unsorted_indices
        )

    def cuda(self) -> PackedSequence:
        return self.to(device="gpu")

    def cpu(self) -> PackedSequence:
        return self.to(device="cpu")

    def __repr__(self) -> str:
        return (
            f"PackedSequence(data={self.data}, batch_sizes={self.batch_sizes}, "
            f"sorted_indices={self.sorted_indices}, unsorted_indices={self.unsorted_indices})"
        )

    def pin_memory(self) -> PackedSequence:
        return PackedSequence(
            self.data.pin_memory(),
            self.batch_sizes,
            self.sorted_indices.pin_memory()
            if self.sorted_indices is not None
            else None,
            self.unsorted_indices.pin_memory()
            if self.unsorted_indices is not None
            else None,
        )

    @property
    def is_cuda(self) -> bool:
        return self.data.is_cuda

    def double(self) -> PackedSequence:
        return self.to(dtype=paddle.float64)

    def float(self) -> PackedSequence:
        return self.to(dtype=paddle.float32)

    def half(self) -> PackedSequence:
        return self.to(dtype=paddle.float16)

    def long(self) -> PackedSequence:
        return self.to(dtype=paddle.int64)

    def int(self) -> PackedSequence:
        return self.to(dtype=paddle.int32)

    def short(self) -> PackedSequence:
        return self.to(dtype=paddle.int16)

    def char(self) -> PackedSequence:
        return self.to(dtype=paddle.int8)

    def byte(self) -> PackedSequence:
        return self.to(dtype=paddle.uint8)


def pack_padded_sequence(
    input: Tensor,
    lengths: Tensor | list[int],
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    r"""Packs a Tensor containing padded sequences of variable length.

    This function packs a Tensor containing padded sequences into a PackedSequence
    object, which can be used as input to a recurrent neural network.

    Args:
        input (Tensor): The padded sequence tensor. Shape is ``T x B x *`` if
            ``batch_first`` is False, or ``B x T x *`` if ``batch_first`` is True,
            where ``T`` is the length of the longest sequence, ``B`` is the batch size.
        lengths (Tensor|list[int]): The lengths of each sequence in the batch.
        batch_first (bool, optional): If True, the input is expected to be in
            ``B x T x *`` format. Default: False.
        enforce_sorted (bool, optional): If True, the input is expected to contain
            sequences sorted by length in descending order. Default: True.

    Returns:
        PackedSequence: A PackedSequence object containing the packed data.

    Examples:
        .. code-block:: pycon

            >>> import paddle

    """
    if batch_first:
        input = input.transpose([1, 0, *range(2, len(input.shape))])

    if isinstance(lengths, paddle.Tensor):
        lengths = lengths.tolist()

    batch_size = input.shape[1]

    if len(lengths) != batch_size:
        raise ValueError(
            f"Length of lengths ({len(lengths)}) does not match batch size ({batch_size})"
        )

    sorted_indices = None
    unsorted_indices = None

    if not enforce_sorted:
        sorted_lengths = sorted(
            enumerate(lengths), key=lambda x: x[1], reverse=True
        )
        sorted_indices = paddle.to_tensor(
            [i for i, _ in sorted_lengths], place=input.place
        )
        unsorted_indices = paddle.argsort(sorted_indices)
        lengths = [l for _, l in sorted_lengths]
        # Use index_select to reorder along batch dimension (axis=1)
        input = paddle.index_select(input, sorted_indices, axis=1)

    packed_data_list = []
    batch_sizes_list = []

    # num_steps may be different from actual input shape[0] after sorting
    # We need to iterate over the actual sequence length
    actual_num_steps = input.shape[0]
    for step in range(actual_num_steps):
        batch_size_at_step = sum(1 for l in lengths if l > step)
        if batch_size_at_step > 0:
            packed_data_list.append(input[step, :batch_size_at_step])
            batch_sizes_list.append(batch_size_at_step)

    packed_data = paddle.concat(packed_data_list, axis=0)
    batch_sizes = paddle.to_tensor(batch_sizes_list, dtype="int64")

    return PackedSequence(
        packed_data, batch_sizes, sorted_indices, unsorted_indices
    )


def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: int | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Pads a packed sequence to a Tensor of padded sequences.

    This function is the inverse of ``pack_padded_sequence``. It takes a PackedSequence
    and returns a padded Tensor and a list of lengths.

    Args:
        sequence (PackedSequence): The packed sequence to pad.
        batch_first (bool, optional): If True, the output will be in ``B x T x *``
            format. Default: False.
        padding_value (float, optional): The value to use for padding. Default: 0.0.
        total_length (int|None, optional): If not None, the output will be padded to
            this length. Default: None.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - The padded sequence tensor.
            - A tensor of sequence lengths.

    Examples:
        .. code-block:: pycon

            >>> import paddle

    """
    if not isinstance(sequence, PackedSequence):
        raise TypeError(f"Expected PackedSequence, got {type(sequence)}")

    data = sequence.data
    batch_sizes = sequence.batch_sizes.tolist()
    unsorted_indices = sequence.unsorted_indices

    max_seq_len = len(batch_sizes)
    max_batch_size = batch_sizes[0]

    if total_length is not None:
        if total_length < max_seq_len:
            raise ValueError(
                f"total_length ({total_length}) must be >= max sequence length ({max_seq_len})"
            )

    trailing_dims = list(data.shape[1:])

    if total_length is not None and total_length > max_seq_len:
        output = paddle.full(
            [total_length, max_batch_size, *trailing_dims],
            padding_value,
            dtype=data.dtype,
            device=data.place,
        )
    else:
        output = paddle.full(
            [max_seq_len, max_batch_size, *trailing_dims],
            padding_value,
            dtype=data.dtype,
            device=data.place,
        )

    data_offset = 0
    for step, batch_size in enumerate(batch_sizes):
        output[step, :batch_size] = data[data_offset : data_offset + batch_size]
        data_offset += batch_size

    # Calculate lengths from batch_sizes
    # batch_sizes is in descending order, e.g., [3, 2, 1] means:
    # - First time step has 3 sequences
    # - Second time step has 2 sequences
    # - Third time step has 1 sequence
    # This means sequence lengths are [3, 2, 1] in sorted order
    lengths_list = []
    for i in range(max_batch_size):
        # Find the length of the i-th sequence (in sorted order)
        # It's the number of time steps where batch_sizes > i
        seq_len = sum(1 for bs in batch_sizes if bs > i)
        lengths_list.append(seq_len)
    lengths = paddle.to_tensor(lengths_list, dtype="int64", place=data.place)

    if unsorted_indices is not None:
        output = output[:, unsorted_indices]
        lengths = lengths[unsorted_indices]

    if batch_first:
        output = output.transpose([1, 0, *range(2, len(output.shape))])

    return output, lengths


def pad_sequence(
    sequences: Iterable[Tensor],
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
            >>> padded = paddle.nn.utils.rnn.pad_sequence([a, b, c])
            >>> print(padded.shape)
            paddle.Size([25, 3, 300])
            >>> padded = paddle.nn.utils.rnn.pad_sequence([a, b, c], batch_first=True)
            >>> print(padded.shape)
            paddle.Size([3, 25, 300])

    """
    if not isinstance(sequences, Iterable):
        raise TypeError(
            f"pad_sequence expects an iterable of Tensors, but got {type(sequences)}"
        )
    sequences = tuple(sequences)
    for seq in sequences:
        if not isinstance(seq, paddle.Tensor):
            raise TypeError(
                f"pad_sequence expects an iterable of Tensors, but got element of type {type(seq)}"
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
            >>> padded = paddle.nn.utils.rnn.pad_sequence(sequences)
            >>> lengths = paddle.to_tensor([v.shape[0] for v in sequences])
            >>> unpadded = paddle.nn.utils.rnn.unpad_sequence(padded, lengths)
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


def pack_sequence(
    sequences: list[Tensor],
    enforce_sorted: bool = True,
) -> PackedSequence:
    r"""Packs a list of variable length Tensors.

    Consecutive call of the next functions: ``pad_sequence``, ``pack_padded_sequence``.

    ``sequences`` should be a list of Tensors of size ``L x *``, where `L` is
    the length of a sequence and `*` is any number of trailing dimensions,
    including ``0``.

    For unsorted sequences, use `enforce_sorted = False`. If ``enforce_sorted``
    is ``True``, the sequences should be sorted in the order of decreasing length.
    ``enforce_sorted = True`` is only necessary for ONNX export.

    Args:
        sequences (list[Tensor]): A list of sequences of decreasing length.
        enforce_sorted (bool, optional): if ``True``, checks that the input
            contains sequences sorted by length in a decreasing order. If
            ``False``, this condition is not checked. Default: ``True``.

    Returns:
        PackedSequence: a PackedSequence object.

    Examples:
        >>> import paddle

    """
    lengths = paddle.to_tensor([v.shape[0] for v in sequences])
    return pack_padded_sequence(
        pad_sequence(sequences), lengths, enforce_sorted=enforce_sorted
    )


def unpack_sequence(packed_sequences: PackedSequence) -> list[Tensor]:
    r"""Unpack PackedSequence into a list of variable length Tensors.

    ``packed_sequences`` should be a PackedSequence object.

    Args:
        packed_sequences (PackedSequence): A PackedSequence object.

    Returns:
        list[Tensor]: a list of Tensor objects.

    Examples:
        >>> import paddle

    """
    padded_sequences, lengths = pad_packed_sequence(
        packed_sequences, batch_first=True
    )
    unpacked_sequences = unpad_sequence(
        padded_sequences, lengths, batch_first=True
    )
    return unpacked_sequences
