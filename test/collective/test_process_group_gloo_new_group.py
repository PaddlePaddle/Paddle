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

import multiprocessing
import os
import socket
import time
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.base import core


def _run_worker(rank, endpoints):
    os.environ.update(
        PADDLE_TRAINER_ID=str(rank),
        PADDLE_TRAINERS_NUM=str(len(endpoints)),
        PADDLE_CURRENT_ENDPOINT=endpoints[rank],
        PADDLE_TRAINER_ENDPOINTS=','.join(endpoints),
        PADDLE_DISTRI_BACKEND='gloo',
        GLOO_SOCKET_IFNAME='lo',
    )
    paddle.set_device('cpu')
    dist.init_parallel_env()
    groups = []
    for ranks in ([0, 1, 2, 3], [1, 3], [0, 2], [0, 1, 2, 3]) * 3:
        # Leave the preceding group's rendezvous entries visible while other
        # ranks start the next handshake.
        if rank == ranks[0]:
            time.sleep(0.2)
        group = dist.new_group(ranks, backend='gloo')
        groups.append(group)
        if rank in ranks:
            tensor = paddle.to_tensor([rank + 1.0])
            dist.all_reduce(tensor, group=group)
            np.testing.assert_array_equal(
                tensor.numpy(), [sum(peer + 1 for peer in ranks)]
            )
            tensor = paddle.to_tensor([rank + 1.0])
            dist.broadcast(tensor, src=ranks[-1], group=group)
            np.testing.assert_array_equal(tensor.numpy(), [ranks[-1] + 1.0])
        # Each new group must also leave existing groups usable.
        tensor = paddle.to_tensor([rank + 1.0])
        dist.all_reduce(tensor)
        np.testing.assert_array_equal(tensor.numpy(), [10.0])
    for group in groups:
        if rank in group.ranks:
            dist.barrier(group=group)
    dist.barrier()


@unittest.skipUnless(
    hasattr(core, 'ProcessGroupGloo'), 'Paddle is not compiled with Gloo'
)
class TestProcessGroupGlooNewGroup(unittest.TestCase):
    def test_new_groups(self):
        sockets = []
        processes = []
        try:
            for _ in range(4):
                sock = socket.socket()
                sock.bind(('127.0.0.1', 0))
                sockets.append(sock)
            endpoints = [
                f'127.0.0.1:{sock.getsockname()[1]}' for sock in sockets
            ]
            for sock in sockets:
                sock.close()

            ctx = multiprocessing.get_context('spawn')
            for rank in range(4):
                process = ctx.Process(
                    target=_run_worker, args=(rank, endpoints)
                )
                process.start()
                processes.append(process)
            deadline = time.monotonic() + 60
            for process in processes:
                process.join(timeout=max(0, deadline - time.monotonic()))
            self.assertEqual(
                [process.exitcode for process in processes], [0] * 4
            )
        finally:
            for sock in sockets:
                sock.close()
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()


if __name__ == '__main__':
    unittest.main()
