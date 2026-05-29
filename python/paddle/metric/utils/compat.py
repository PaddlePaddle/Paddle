# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


def pack_padded_sequence(
    input: paddle.Tensor,
    lengths: paddle.Tensor,
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    Paddle 实现的 pack_padded_sequence，对齐 PyTorch 接口。

    Args:
        input (paddle.Tensor): 填充后的批量序列。
        lengths (paddle.Tensor): 每个序列的原始长度，1D 张量。
        batch_first (bool): 若为 True，input 形状为 [batch, seq_len, *]；否则为 [seq_len, batch, *]。
        enforce_sorted (bool): 若为 True，要求 lengths 按降序排列。

    Returns:
        Tuple: 包含 (packed_data, batch_sizes, sorted_indices, unsorted_indices)
    """
    # 转换为 64 位整数以避免索引错误
    if not isinstance(lengths, paddle.Tensor):
        lengths = paddle.to_tensor(lengths, dtype=paddle.int64)
    lengths = lengths.astype(paddle.int64)

    # 处理维度
    if batch_first:
        input = input.transpose([1, 0, *range(2, input.ndim)])  # [T, B, *]

    max_seq_len, batch_size = input.shape[:2]

    # 验证长度
    if (
        (lengths <= 0).any()
        or (lengths > max_seq_len).any()
        or lengths.numel() == 0
    ):
        raise ValueError(
            "Lengths values must be between 1 and max_seq_len, and numel of lengths cannot be zero"
        )

    # 处理排序
    if enforce_sorted:
        if not paddle.all(lengths[1:] <= lengths[:-1]):
            raise ValueError(
                "Lengths are not sorted in non-increasing order. Set enforce_sorted=False."
            )
        sorted_indices = paddle.arange(batch_size, dtype=paddle.int64)
    else:
        lengths_sorted, sorted_indices = (
            paddle.sort(lengths, descending=True),
            paddle.argsort(lengths, descending=True),
        )
        # 重新排列输入
        input = input[:, sorted_indices]
        lengths = lengths_sorted

    # 生成 batch_sizes (按时间步的有效批量大小)
    batch_sizes = []
    for t in range(max_seq_len):
        # 统计长度大于当前时间步 t 的样本数
        bs = (lengths > t).sum().item()
        batch_sizes.append(bs)
        if bs == 0:
            break  # 后续都是0，提前终止

    batch_sizes = paddle.to_tensor(batch_sizes, dtype=paddle.int64)

    # 生成掩码并提取有效数据
    mask = paddle.arange(max_seq_len).unsqueeze(1) < lengths.unsqueeze(0)
    # 展平维度以便提取
    flat_input = input.reshape([max_seq_len, batch_size, -1])  # [T, B, D]
    packed_data = paddle.masked_select(flat_input, mask.unsqueeze(-1)).reshape(
        [-1, flat_input.shape[-1]]
    )
    # 生成逆序索引（用于解包时恢复原顺序）
    unsorted_indices = (
        paddle.argsort(sorted_indices)
        if not enforce_sorted
        else paddle.arange(batch_size, dtype=paddle.int64)
    )
    return packed_data, batch_sizes, sorted_indices, unsorted_indices


def pad_packed_sequence(
    sequence: tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: int | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    Paddle 实现的 pad_packed_sequence，对齐 PyTorch 接口。

    Args:
        sequence (Tuple): pack_padded_sequence 的输出元组。
        batch_first (bool): 输出是否为 [batch, seq_len, *] 格式。
        padding_value (float): 填充值。
        total_length (Optional[int]): 输出序列的总长度，若未指定则为最大长度。

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor]: (paded_sequence, lengths)
    """
    packed_data, batch_sizes, sorted_indices, unsorted_indices = sequence
    max_seq_len = len(batch_sizes)
    batch_size = len(sorted_indices)
    data_dim = packed_data.shape[1:] if packed_data.ndim > 1 else ()

    # 处理 total_length
    if total_length is not None:
        if total_length < max_seq_len:
            raise ValueError(
                "total_length must be >= the maximum sequence length in the batch"
            )
        max_seq_len = total_length

    # 初始化输出张量
    output_shape = [max_seq_len, batch_size, *list(data_dim)]
    padded = paddle.full(output_shape, padding_value, dtype=packed_data.dtype)

    # 计算每个时间步的起始和结束索引
    cum_sizes = paddle.concat(
        [paddle.to_tensor([0]), paddle.cumsum(batch_sizes, axis=0)]
    )

    # 回填数据
    for t in range(len(batch_sizes)):
        bs = batch_sizes[t].item()
        if bs == 0:
            break
        start = cum_sizes[t].item()
        end = cum_sizes[t + 1].item()
        padded[t, :bs] = packed_data[start:end]

    # 生成原始长度（基于 batch_sizes）
    lengths = paddle.zeros([batch_size], dtype=paddle.int64)
    for t in reversed(range(len(batch_sizes))):
        bs = batch_sizes[t].item()
        if bs == 0:
            continue
        # 对于当前时间步有数据的样本，其长度至少为 t+1
        lengths[:bs] = paddle.maximum(lengths[:bs], paddle.to_tensor(t + 1))

    # 恢复原始批次顺序
    padded = padded[:, unsorted_indices]
    lengths = lengths[unsorted_indices]

    # 转换输出格式
    if batch_first:
        padded = padded.transpose([1, 0, *range(2, padded.ndim)])

    return padded, lengths
