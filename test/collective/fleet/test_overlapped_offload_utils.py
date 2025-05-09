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

import numpy as np

import paddle
from paddle.distributed.fleet.meta_parallel import (
    RROOQueue,
    get_rroo_buffer_pool_manager,
    get_rroo_queue_manager,
)


class TestRROO(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)

    def test_rroo_put_and_get(self):
        rroo_buffer_pool_manager = get_rroo_buffer_pool_manager()

        for do_rroo in [True, False]:
            rroo_queue = RROOQueue(acc_num=2, do_rroo=do_rroo)

            cuda_data0_acc_0 = paddle.randn([4096, 4096])
            cuda_data1_acc_0 = paddle.randn([4096, 4096])

            cuda_data0_acc_1 = paddle.randn([4096, 4096])
            cuda_data1_acc_1 = paddle.randn([4096, 4096])

            # fwd acc0
            rroo_queue.offload()
            rroo_queue.put(cuda_data0_acc_0.clone())
            rroo_queue.wait_and_release()

            rroo_queue.offload()
            rroo_queue.put(cuda_data1_acc_0.clone())
            rroo_queue.wait_and_release()

            # fwd acc1
            rroo_queue.offload()
            rroo_queue.put(cuda_data0_acc_1.clone())
            rroo_queue.wait_and_release()

            rroo_queue.offload()
            rroo_queue.put(cuda_data1_acc_1.clone())
            rroo_queue.wait_and_release()

            rroo_queue.offload()
            rroo_queue.wait_and_release()

            # bwd acc0
            rroo_queue.reload()
            rroo_queue.wait_and_release()
            a = rroo_queue.get()

            rroo_queue.reload()
            rroo_queue.wait_and_release()
            b = rroo_queue.get()

            # bwd acc1
            rroo_queue.reload()
            rroo_queue.wait_and_release()
            c = rroo_queue.get()

            rroo_queue.reload()
            rroo_queue.wait_and_release()
            d = rroo_queue.get()

            # check
            np.testing.assert_array_equal(
                rroo_queue.empty(),
                True,
            )
            np.testing.assert_array_equal(
                rroo_buffer_pool_manager.is_all_memory_free(),
                True,
            )

            np.testing.assert_array_equal(
                cuda_data0_acc_0._md5sum(),
                a._md5sum(),
            )
            np.testing.assert_array_equal(
                cuda_data1_acc_0._md5sum(),
                b._md5sum(),
            )
            np.testing.assert_array_equal(
                cuda_data0_acc_1._md5sum(),
                c._md5sum(),
            )
            np.testing.assert_array_equal(
                cuda_data1_acc_1._md5sum(),
                d._md5sum(),
            )

            np.testing.assert_array_equal(
                cuda_data0_acc_0.shape,
                a.shape,
            )
            np.testing.assert_array_equal(
                cuda_data1_acc_0.shape,
                b.shape,
            )
            np.testing.assert_array_equal(
                cuda_data0_acc_1.shape,
                c.shape,
            )
            np.testing.assert_array_equal(
                cuda_data1_acc_1.shape,
                d.shape,
            )

    def test_rroo_queue_manager(self):
        chunk_num = 4
        acc_num = 5
        rroo_queue_manager = get_rroo_queue_manager()
        rroo_queue_manager.init(chunk_num, acc_num)

        rroo_buffer_pool_manager = get_rroo_buffer_pool_manager()

        for split_factor in range(1, chunk_num):

            queue_list = [None for _ in range(chunk_num)]
            data_list = [
                [paddle.randn([4096, 4096]) for _ in range(acc_num)]
                for _ in range(chunk_num)
            ]
            # init
            for chunk_id in range(chunk_num):
                rroo_queue_manager.set_cur_chunk_id(chunk_id)
                queue_list[chunk_id] = rroo_queue_manager.create_rroo_queue(
                    split_factor=split_factor
                )

            # forward
            for chunk_id in range(chunk_num):
                rroo_queue_manager.set_cur_chunk_id(chunk_id)

                for acc_id in range(acc_num):
                    rroo_queue_manager.offload()

                    rroo_queue = queue_list[chunk_id]
                    rroo_queue.put(data_list[chunk_id][acc_id].clone())

                    rroo_queue_manager.wait_and_release()

            # backward
            for chunk_id in range(chunk_num - 1, -1, -1):
                rroo_queue_manager.set_cur_chunk_id(chunk_id)

                for acc_id in range(acc_num):
                    rroo_queue_manager.reload()

                    rroo_queue = queue_list[chunk_id]
                    data = rroo_queue.get()
                    np.testing.assert_array_equal(
                        data._md5sum(),
                        data_list[chunk_id][acc_id]._md5sum(),
                    )
                    np.testing.assert_array_equal(
                        data.shape,
                        data_list[chunk_id][acc_id].shape,
                    )
                    rroo_queue_manager.wait_and_release()

            np.testing.assert_array_equal(
                rroo_queue_manager.empty(),
                True,
            )
            np.testing.assert_array_equal(
                rroo_buffer_pool_manager.is_all_memory_free(),
                True,
            )


if __name__ == '__main__':
    unittest.main()
