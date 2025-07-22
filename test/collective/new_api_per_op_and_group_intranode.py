# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed as dist
from paddle.base import core
from paddle.distributed.communication.group import Group


def test_reducescatter(ep_group: Group, mode: str):
    m, n = 4096, 8192

    local_rank = dist.get_rank(ep_group)

    num_local_ranks = dist.get_world_size(ep_group)

    send_tensors = [
        paddle.ones(shape=[m, n], dtype=paddle.float32) * (i + 1)
        for i in range(num_local_ranks)
    ]

    recv_tensor = paddle.zeros(shape=[m, n], dtype=paddle.float32)

    dist.reduce_scatter(recv_tensor, send_tensors, group=ep_group)

    expected_tensor = (
        paddle.ones(shape=[m, n], dtype=paddle.float32)
        * (local_rank + 1)
        * num_local_ranks
    )

    assert paddle.allclose(
        recv_tensor, expected_tensor
    ), f"rank {local_rank}: reduce_scatter validation failed"

    if local_rank == 0:
        print(f'[Algo {mode}] primitive reducescatter... passed')


def test_alltoall(ep_group: Group, mode: str):
    m, n = 4096, 8192

    local_rank = dist.get_rank(ep_group)

    num_local_ranks = dist.get_world_size(ep_group)

    send_tensors = [
        paddle.ones(shape=[m, n], dtype=paddle.float32) * (i + 1)
        for i in range(num_local_ranks)
    ]

    recv_tensors = [
        paddle.zeros(shape=[m, n], dtype=paddle.float32)
        for _ in range(num_local_ranks)
    ]

    dist.alltoall(recv_tensors, send_tensors, group=ep_group)
    expected_tensor = paddle.ones(shape=[m, n], dtype=paddle.float32) * (
        local_rank + 1
    )

    for i in range(num_local_ranks):
        assert paddle.allclose(
            recv_tensors[i], expected_tensor
        ), f"rank {local_rank}: alltoall validation failed"

    if local_rank == 0:
        print(f'[Algo {mode}] primitive alltoall... passed')


def test_scatter(ep_group: Group, mode: str):
    local_rank = dist.get_rank(ep_group)

    num_local_ranks = dist.get_world_size(ep_group)

    m, n = 4096, 8192

    if local_rank == 0:

        scatter_list = [
            paddle.ones(shape=[m, n], dtype=paddle.float32) * (i + 1)
            for i in range(num_local_ranks)
        ]
    else:
        scatter_list = []

    recv_tensor = paddle.zeros(shape=[m, n], dtype=paddle.float32)
    dist.scatter(recv_tensor, scatter_list, src=0, group=ep_group)

    expected = paddle.ones(shape=[m, n], dtype=paddle.float32) * (
        local_rank + 1
    )
    assert paddle.allclose(
        recv_tensor, expected
    ), f"rank {local_rank}: scatter validation failed"

    if local_rank == 0:
        print(f'[Algo {mode}] primitive scatter... passed')


def test_reduce(ep_group: Group, mode: str):
    m, n = 4096, 8192

    local_rank = dist.get_rank(ep_group)

    num_local_ranks = dist.get_world_size(ep_group)

    x = paddle.ones(shape=[m, n], dtype=paddle.float32) * (local_rank + 1)
    gbl_x = x.clone()

    dist.reduce(gbl_x, dst=0, group=ep_group)

    if local_rank == 0:

        res = paddle.ones(shape=[m, n], dtype=paddle.float32) * (
            num_local_ranks * (num_local_ranks + 1) / 2
        )
        assert paddle.allclose(
            gbl_x, res
        ), f"rank {local_rank}: reduce validation failed"
        print(f'[Algo {mode}] primitive reduce... passed')


def test_all_gather(ep_group: Group, mode: str):
    local_rank = dist.get_rank(ep_group)

    num_local_ranks = dist.get_world_size(ep_group)

    m, n = 4096, (8192 // num_local_ranks) * num_local_ranks
    assert n % num_local_ranks == 0

    x = paddle.ones(shape=[m, n], dtype=paddle.float32) * (local_rank + 1)

    tensor_list = [
        paddle.zeros(shape=[m, n], dtype=paddle.float32)
        for _ in range(num_local_ranks)
    ]
    dist.all_gather(tensor_list, x, group=ep_group)

    for i in range(num_local_ranks):
        expected = paddle.ones(shape=[m, n], dtype=paddle.float32) * (i + 1)
        assert paddle.allclose(
            tensor_list[i], expected
        ), f"rank {local_rank}: allgather validation failed"

    if local_rank == 0:
        print(f'[Algo {mode}] primitive allgather... passed')


def test_broadcast(ep_group: Group, mode: str):
    m, n = 4096, 8192

    local_rank = dist.get_rank(ep_group)

    num_local_ranks = dist.get_world_size(ep_group)

    if local_rank == 0:
        x = paddle.ones(shape=[m, n], dtype=paddle.float32) * 10
    else:
        x = paddle.zeros(shape=[m, n], dtype=paddle.float32)

    gbl_x = x.clone()
    dist.broadcast(gbl_x, src=0, group=ep_group)

    res = paddle.ones(shape=[m, n], dtype=paddle.float32) * 10
    assert paddle.allclose(
        gbl_x, res
    ), f"rank {local_rank}: broadcast validation failed"

    if local_rank == 0:
        print(f'[Algo {mode}] primitive broadcast... passed')


def test_all_reduce(ep_group: Group, mode: str):
    m, n = 4096, 8192

    local_rank = dist.get_rank(ep_group)

    num_local_ranks = dist.get_world_size(ep_group)

    x = paddle.ones(shape=[m, n], dtype=paddle.float32) * (local_rank + 1)
    gbl_x = x.clone()

    dist.all_reduce(gbl_x, group=ep_group)
    res = paddle.ones(shape=[m, n], dtype=paddle.float32) * (
        num_local_ranks * (num_local_ranks + 1) / 2
    )

    assert paddle.allclose(
        gbl_x, res
    ), f"rank {local_rank}: all reduce validation failed"

    if local_rank == 0:
        print(f'[Algo {mode}] primitive allreduce... passed')


def test_primitive():

    dist.init_parallel_env()

    ranks = [0, 1]

    proto_list = ["LL", "LL128", "Simple"]

    ll_size = list(range(32768, 1000 * 1024, 1024 * 250))
    ll128_size = list(range(32768 * 4, 6 * 1024 * 1024, 3 * 1024 * 1024 // 2))
    simple_size = list(range(32768 * 4, 5 * 1024 * 1024, 1024 * 1024))

    nchannels = [8, 16, 24, 32]

    nccl_config1 = [
        core.NCCLConfig.create(
            commName="tp_comm",
            ll_buffsize=ll_size[j],
            ll128_buffsize=-1,
            simple_buffsize=-1,
            buffsize_align=1024,
            nchannels=nchannels[i],
            algoStr="Tree",
            protoStr="LL",
        )
        for i in range(len(nchannels))
        for j in range(len(ll_size))
    ]

    nccl_config2 = [
        core.NCCLConfig.create(
            commName="tp_comm",
            ll_buffsize=-1,
            ll128_buffsize=ll128_size[j],
            simple_buffsize=-1,
            buffsize_align=1024,
            nchannels=nchannels[i],
            algoStr="Tree",
            protoStr="LL128",
        )
        for i in range(len(nchannels))
        for j in range(len(ll128_size))
    ]

    nccl_config3 = [
        core.NCCLConfig.create(
            commName="tp_comm",
            ll_buffsize=-1,
            ll128_buffsize=-1,
            simple_buffsize=simple_size[j],
            buffsize_align=1024,
            nchannels=nchannels[i],
            algoStr="Tree",
            protoStr="Simple",
        )
        for i in range(len(nchannels))
        for j in range(len(simple_size))
    ]

    nccl_config4 = [
        core.NCCLConfig.create(
            commName="tp_comm",
            ll_buffsize=ll_size[j],
            ll128_buffsize=-1,
            simple_buffsize=-1,
            buffsize_align=1024,
            nchannels=nchannels[i],
            algoStr="Ring",
            protoStr="LL",
        )
        for i in range(len(nchannels))
        for j in range(len(ll_size))
    ]

    nccl_config5 = [
        core.NCCLConfig.create(
            commName="tp_comm",
            ll_buffsize=-1,
            ll128_buffsize=ll128_size[j],
            simple_buffsize=-1,
            buffsize_align=1024,
            nchannels=nchannels[i],
            algoStr="Ring",
            protoStr="LL128",
        )
        for i in range(len(nchannels))
        for j in range(len(ll128_size))
    ]

    nccl_config6 = [
        core.NCCLConfig.create(
            commName="tp_comm",
            ll_buffsize=-1,
            ll128_buffsize=-1,
            simple_buffsize=simple_size[j],
            buffsize_align=1024,
            nchannels=nchannels[i],
            algoStr="Ring",
            protoStr="Simple",
        )
        for i in range(len(nchannels))
        for j in range(len(simple_size))
    ]

    # nccl_config7 = [
    #     core.NCCLConfig.create(
    #         commName="tp_comm",
    #         ll_buffsize=-1,
    #         ll128_buffsize=-1,
    #         simple_buffsize=simple_size[j],
    #         buffsize_align=1024,
    #         nchannels=nchannels[i],
    #         algoStr="NVLS",
    #         protoStr="Simple",
    #     )
    #     for i in range(len(nchannels))
    #     for j in range(len(simple_size))
    # ]

    config_len1 = len(nccl_config1)
    config_len2 = len(nccl_config2)
    config_len3 = len(nccl_config3)
    config_len4 = len(nccl_config4)
    config_len5 = len(nccl_config5)
    config_len6 = len(nccl_config6)
    # config_len7 = len(nccl_config7)

    ep_group1 = [
        paddle.distributed.new_group(ranks, nccl_config=nccl_config1[i])
        for i in range(config_len1)
    ]

    ep_group2 = [
        paddle.distributed.new_group(ranks, nccl_config=nccl_config2[i])
        for i in range(config_len2)
    ]

    ep_group3 = [
        paddle.distributed.new_group(ranks, nccl_config=nccl_config3[i])
        for i in range(config_len3)
    ]

    ep_group4 = [
        paddle.distributed.new_group(ranks, nccl_config=nccl_config4[i])
        for i in range(config_len4)
    ]

    ep_group5 = [
        paddle.distributed.new_group(ranks, nccl_config=nccl_config5[i])
        for i in range(config_len5)
    ]

    ep_group6 = [
        paddle.distributed.new_group(ranks, nccl_config=nccl_config6[i])
        for i in range(config_len6)
    ]

    # ep_group7 = [
    #     paddle.distributed.new_group(ranks, nccl_config=nccl_config7[i])
    #     for i in range(config_len7)
    # ]

    # ep_groups = [paddle.distributed.new_group(ranks) for _ in range(7)]

    for i in range(config_len1):
        test_all_reduce(ep_group1[i], "Tree LL")

    for i in range(config_len2):
        test_all_reduce(ep_group2[i], "Tree LL128")

    for i in range(config_len3):
        test_all_reduce(ep_group3[i], "Tree Simple")

    for i in range(config_len4):
        test_all_reduce(ep_group4[i], "Ring LL")
        test_broadcast(ep_group4[i], "Ring LL")
        test_all_gather(ep_group4[i], "Ring LL")
        test_reduce(ep_group4[i], "Ring LL")
        test_reducescatter(ep_group4[i], "Ring LL")

    for i in range(config_len5):
        test_all_reduce(ep_group5[i], "Ring LL128")
        test_broadcast(ep_group5[i], "Ring LL128")
        test_all_gather(ep_group5[i], "Ring LL128")
        test_reduce(ep_group5[i], "Ring LL128")
        test_reducescatter(ep_group5[i], "Ring LL128")

    for i in range(config_len6):
        test_all_reduce(ep_group6[i], "Ring Simple")
        test_broadcast(ep_group6[i], "Ring Simple")
        test_all_gather(ep_group6[i], "Ring Simple")
        test_reduce(ep_group6[i], "Ring Simple")
        test_reducescatter(ep_group6[i], "Ring Simple")

    # for i in range(config_len7):
    #     test_all_reduce(ep_group7[i], "NVLS Simple")
    #     test_all_gather(ep_group7[i], "NVLS Simple")
    #     test_reducescatter(ep_group7[i], "NVLS Simple")


class TestCollectiveDeepEPNewAPIIntranode(unittest.TestCase):
    def test(self):
        test_primitive()


if __name__ == "__main__":
    unittest.main()
