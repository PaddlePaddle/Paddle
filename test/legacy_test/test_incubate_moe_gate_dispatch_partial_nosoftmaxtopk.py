# ruff: noqa: C419
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

import unittest

import paddle
from paddle.incubate.nn.functional import (
    moe_gate_dispatch,
    moe_gate_dispatch_partial_nosoftmaxtopk,
)


def test_moe_dispatch_partial_nosoftmaxtopk_nonepad_op():

    s, d, e = 4, 100, 8
    k, cap = 4, 3
    local_expert_num = 2

    # x = paddle.randn([s, d])
    # gate_logits = paddle.randn([s, e])
    x = paddle.arange(1, s + 1).unsqueeze(-1).expand([s, d]).astype("bfloat16")
    x_ = x.clone().detach()

    t = (
        (paddle.arange(0, e)).unsqueeze(0)
        + paddle.arange(0, -s, -1).unsqueeze(-1)
    ) % e
    gate_logits = (1 / (t + 1)).astype("float32")
    # gate_logits = F.softmax(paddle.randn([s,e]),-1).astype('float32')
    gate_logits_ = gate_logits.clone().detach()
    s = x.shape[0]
    d = x.shape[1]
    e = gate_logits.shape[1]
    x.stop_gradient = False
    x_.stop_gradient = False
    gate_logits.stop_gradient = False
    gate_logits_.stop_gradient = False
    print(f"gate_logits:{gate_logits}")

    def check_ascend(index_rev, chunks):
        for idx in index_rev.split(chunks.tolist()):
            if len(idx) > 2:
                assert (paddle.diff(idx) >= 0).all(), (index_rev,)

    ys, comm, scatter_idx = [], [], []
    for ilocal_expert in range(0, e, local_expert_num):
        combine_weihgts, expert_id = gate_logits.topk(k=k, axis=1)
        (
            y,
            combine_weihgts,
            scatter_index,
            scatter_index_rev,
            expert_offset,
            expert_num_local,
        ) = moe_gate_dispatch_partial_nosoftmaxtopk(
            x,
            combine_weihgts,
            expert_id.astype("int32"),
            k=k,
            capacity=cap,
            num_experts=gate_logits.shape[-1],
            use_pad=False,
            expert_start_index=ilocal_expert,
            expert_end_index=ilocal_expert + local_expert_num,  # k  # cap
            reverse_token_drop=False,
        )
        check_ascend(scatter_index_rev, expert_num_local)
        print(f"y:{y.mean(-1)}")
        print(f"combine_weihgts:{combine_weihgts}")
        print(f"expert_num_local:{expert_num_local}")
        print(f"scatter_index:{scatter_index.transpose([1,0])}")
        print(f"scatter_index_rev:{scatter_index_rev}")

        ys.append(y)
        comm.append(combine_weihgts)
        scatter_idx.append(scatter_index)

    comm_sum = paddle.stack(comm).sum(0)
    ys_sum = paddle.concat(ys)

    y_, combine_weihgts_, scatter_index_, expert_offset_, expert_id_ = (
        moe_gate_dispatch(
            x_,
            gate_logits_,
            None,
            k=k,
            capacity=cap,
            use_pad=True,  # k  # cap
        )
    )
    valid_y = y_.sum(-1) > 0.0
    y_2 = y_[valid_y].squeeze()

    print(
        f"""
    y: {ys_sum.astype("float32").mean(axis=-1)}
    y_: {y_2.astype("float32").mean(axis=-1)}

    comm-weight: {comm_sum}
    comm-weight_: {combine_weihgts_}

    expert_id:{expert_id}
    scatter_index:{scatter_index}
    scatter_index_rev: {scatter_index_rev}
    expert_num_global:{expert_offset}
    expert_num_local:{expert_num_local}
    """
    )

    print("<<< begin backward>>>")

    assert combine_weihgts_.shape == combine_weihgts.shape, (
        combine_weihgts_.shape,
        combine_weihgts.shape,
    )

    dysum, dcombine_weights_sum = paddle.ones_like(ys_sum), paddle.randn(
        comm_sum.shape
    ).astype(comm_sum.dtype)
    dy_, dcombine_weights_ = paddle.ones_like(y_), paddle.ones_like(
        combine_weihgts_
    )
    dy_[~valid_y] = 0

    y_shapes = [len(y) for y in ys]
    for dyy, yy, commm in zip(
        paddle.split(dysum, y_shapes),
        ys,
        comm,
    ):
        print(f"dyy:{dyy.shape}, {yy.shape} {commm.shape}")
        paddle.autograd.backward([yy, commm], [dyy, dcombine_weights_sum])
        print(x.grad.astype("float32").mean(axis=-1))
    print(f"bwd original:{y_.shape} {dy_.shape}")
    paddle.autograd.backward([y_, combine_weihgts_], [dy_, dcombine_weights_])

    print(x_.grad.astype("float32").mean(axis=-1))

    print(
        f"""
    x: {x.grad.astype('float32').mean(axis=-1)}
    x_: {x_.grad.astype('float32').mean(axis=-1)}
    """
    )


def test_moe_ops_partial_nosoftmaxtopk_w_reverse_token_drop():

    S, E, D = 3, 4, 3
    k = 2
    capacity = 2
    x = (paddle.arange(S) + 1).unsqueeze(-1).expand([S, D]).astype("bfloat16")
    cw = paddle.randn([S, k])
    eid = paddle.to_tensor(
        [[0, 1], [0, 1], [0, 2]], dtype="int32"
    )  # 1  # 2  # 3
    (
        y,
        cw_,
        idx,
        idx_rev,
        num_ex_global,
        num_ex_local,
    ) = moe_gate_dispatch_partial_nosoftmaxtopk(
        x, cw, eid, k, capacity, E, False, 0, 2, reverse_token_drop=True
    )

    y0, y1 = y.split([i for i in num_ex_local.tolist() if i > 0])
    assert y0[:, 0].astype("int32").tolist() == [2, 3], y0[:, 0]
    assert y1[:, 0].astype("int32").tolist() == [1, 2]


def test_moe_ops_partial_nosoftmax_topk_empty_output():

    S, E, D = 3, 4, 3
    k = 2
    capacity = 2
    x = (paddle.arange(S) + 1).unsqueeze(-1).expand([S, D]).astype("bfloat16")
    cw = paddle.randn([S, k])
    paddle.device.synchronize()
    eid = paddle.to_tensor(
        [[0, 1], [0, 1], [0, 2]], dtype="int32"
    )  # 1  # 2  # 3
    (
        y,
        cw_,
        idx,
        idx_rev,
        num_ex_global,
        num_ex_local,
    ) = moe_gate_dispatch_partial_nosoftmaxtopk(
        x, cw, eid, k, capacity, E, False, 3, 4, reverse_token_drop=True
    )
    assert all([i == 0 for i in num_ex_local.tolist()]), num_ex_local


class TestAddition(unittest.TestCase):

    def test_moe_dispatch_partial_nosoftmaxtopk_nonepad_op(self):
        test_moe_dispatch_partial_nosoftmaxtopk_nonepad_op()

    def test_moe_ops_partial_nosoftmaxtopk_w_reverse_token_drop(self):
        test_moe_ops_partial_nosoftmaxtopk_w_reverse_token_drop()

    def test_moe_ops_partial_nosoftmax_topk_empty_output(self):
        test_moe_ops_partial_nosoftmax_topk_empty_output()


if __name__ == "__main__":
    unittest.main()
