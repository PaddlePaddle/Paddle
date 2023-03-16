# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle


def BertSparseMask(batch_size, num_heads, seq_len):
    init_alphas = 1e-3 * paddle.randn((1, seq_len, 2))
    init_alphas = paddle.to_tensor(init_alphas)
    logits = init_alphas

    tau = 1
    dim = -1
    hard = True
    eps = 1e-5

    gumbels = -(
        paddle.empty_like(logits).exponential_() + eps
    ).log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    softmax = paddle.nn.Softmax(dim)
    y_soft = softmax(gumbels)

    if hard:
        # Straight through.
        index = y_soft.argmax(axis=dim, keepdim=True)
        y_hard = paddle.zeros_like(logits).put_along_axis_(index, 1.0, dim)
        mask = y_hard - y_soft + y_soft
    else:
        # Reparametrization trick.
        mask = y_soft

    mask = mask[:, :, 0].unsqueeze(2)  # (12, 128, 1)
    mask = mask.expand((1, seq_len, seq_len))
    mask = paddle.triu(mask)
    mask = paddle.rot90(mask, 1, (1, 2))
    mask_tri = paddle.zeros((1, seq_len, seq_len))
    mask_tri[:, 0] = mask[:, 0]
    for i in range(1, seq_len):
        mask_tri[:, i, i:] = mask[:, i, :-i]
    masks = mask_tri + paddle.transpose(paddle.triu(mask_tri, 1), (0, 2, 1))
    mask = paddle.repeat_interleave(masks, batch_size * num_heads, 0)
    return mask


def ErineSparseMask(batch_size, num_heads, seq_len):
    # seq_len >= 1024 will get batter performance
    num_wnd = (seq_len + 1023) / 1024
    glb = 32 * num_wnd
    wnd = 16 * num_wnd
    pad_len = (seq_len + wnd - 1) // wnd * wnd
    mask = np.zeros([pad_len, pad_len])
    mask[:glb, :glb] = 1
    pos = np.repeat(list(range((pad_len - glb) // wnd)), wnd)
    local_visible_part = pos[None, :] == pos[:, None]
    local_visible_part |= np.roll(local_visible_part, wnd, axis=1)
    local_visible_part |= np.roll(local_visible_part, -wnd, axis=1)
    mask[glb:, glb:][local_visible_part] = 1
    if pad_len > seq_len:
        mask = mask[:seq_len, :seq_len]
    mask = np.repeat(mask[None, :, :], [batch_size * num_heads], axis=0)
    return paddle.to_tensor(mask)


mask_configs = {
    "erine": ErineSparseMask,
    "bert": BertSparseMask,
}


def get_sparse_mask(mask_name):
    return mask_configs[mask_name]
