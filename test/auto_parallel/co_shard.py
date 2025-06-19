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
import paddle.distributed as dist


class TestCoShard:
    def basic_interface_case(self):
        shard = dist.Shard(0, shard_order=0)
        np.testing.assert_equal(str(shard), "Shard(dim=0, shard_order=0)")

        shard = dist.Shard(0, split_factor=2)
        np.testing.assert_equal(str(shard), "Shard(dim=0, split_factor=2)")

    def run_test_case_0(self):
        a = paddle.to_tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])

        placements = [
            dist.Shard(0, shard_order=0),
            dist.Shard(0, shard_order=1),
        ]
        input = dist.shard_tensor(a, mesh, placements)

        idx = dist.get_rank()
        np.testing.assert_equal(
            input._local_value().numpy().flatten(), a[idx].numpy().flatten()
        )

        reshard_placements = [dist.Replicate(), dist.Replicate()]
        out = dist.reshard(input, mesh, reshard_placements)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a.numpy().flatten()
        )

        reshard_placements = [dist.Shard(0), dist.Replicate()]
        out = dist.reshard(input, mesh, reshard_placements)
        new_idx = idx // 2 * 2
        np.testing.assert_equal(
            out._local_value().numpy().flatten(),
            a[new_idx : new_idx + 2].numpy().flatten(),
        )

        reshard_placements = [dist.Replicate(), dist.Shard(0)]
        out = dist.reshard(input, mesh, reshard_placements)
        new_idx = idx % 2 * 2
        np.testing.assert_equal(
            out._local_value().numpy().flatten(),
            a[new_idx : new_idx + 2].numpy().flatten(),
        )

    def run_test_case_1(self):
        a = paddle.to_tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        placements = [
            dist.Shard(0, shard_order=1),
            dist.Shard(0, shard_order=0),
        ]
        input = dist.shard_tensor(a, mesh, placements)

        idx = dist.get_rank()
        new_idx = idx % 2 * 2 + idx // 2
        np.testing.assert_equal(
            input._local_value().numpy().flatten(), a[new_idx].numpy().flatten()
        )

        reshard_placements = [dist.Replicate(), dist.Replicate()]
        out = dist.reshard(input, mesh, reshard_placements)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a.numpy().flatten()
        )

        reshard_placements = [dist.Shard(0), dist.Replicate()]
        out = dist.reshard(input, mesh, reshard_placements)
        new_idx = idx // 2 * 2
        np.testing.assert_equal(
            out._local_value().numpy().flatten(),
            a[new_idx : new_idx + 2].numpy().flatten(),
        )

        reshard_placements = [dist.Replicate(), dist.Shard(0)]
        out = dist.reshard(input, mesh, reshard_placements)
        new_idx = idx % 2 * 2
        np.testing.assert_equal(
            out._local_value().numpy().flatten(),
            a[new_idx : new_idx + 2].numpy().flatten(),
        )

    def run_test_case_2(self):
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])

        # dense tensor
        a = paddle.to_tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

        placements = [dist.Shard(0, split_factor=2), dist.Replicate()]
        # distributed tensor
        input = dist.shard_tensor(a, mesh, placements)

        idx = dist.get_rank()
        if idx == 0 or idx == 1:
            golden = np.array([[1, 2], [5, 6]])
        else:
            golden = np.array([[3, 4], [7, 8]])
        np.testing.assert_equal(
            input._local_value().numpy().flatten(), golden.flatten()
        )

        reshard_placements = [dist.Replicate(), dist.Replicate()]
        out = dist.reshard(input, mesh, reshard_placements)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a.numpy().flatten()
        )

        reshard_placements = [dist.Shard(0), dist.Replicate()]
        out = dist.reshard(input, mesh, reshard_placements)
        new_idx = idx // 2 * 2
        np.testing.assert_equal(
            out._local_value().numpy().flatten(),
            a[new_idx : new_idx + 2].numpy().flatten(),
        )

        reshard_placements = [dist.Replicate(), dist.Shard(0)]
        out = dist.reshard(input, mesh, reshard_placements)
        new_idx = idx % 2 * 2
        np.testing.assert_equal(
            out._local_value().numpy().flatten(),
            a[new_idx : new_idx + 2].numpy().flatten(),
        )

    def run_test_case_3(self):
        a = paddle.to_tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        placements = [dist.Shard(0), dist.Shard(1)]
        input = dist.shard_tensor(a, mesh, placements)

        reshard_placements = [
            dist.Shard(0, shard_order=0),
            dist.Shard(0, shard_order=1),
        ]
        out = dist.reshard(input, mesh, reshard_placements)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(),
            a[dist.get_rank()].numpy().flatten(),
        )
        np.testing.assert_equal(
            str(out.placements[0]), "Shard(dim=0, shard_order=0)"
        )
        np.testing.assert_equal(
            str(out.placements[1]), "Shard(dim=0, shard_order=1)"
        )

    def run_test_case_4(self):
        a = paddle.to_tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype='float32')
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        placements = [dist.Shard(0), dist.Shard(1)]
        input = dist.shard_tensor(a, mesh, placements)

        out = paddle.reshape(input, [-1])
        np.testing.assert_equal(out.shape, [8])
        np.testing.assert_equal(
            str(out.placements[0]), "Shard(dim=0, shard_order=0)"
        )
        np.testing.assert_equal(
            str(out.placements[1]), "Shard(dim=0, shard_order=1)"
        )
        np.testing.assert_equal(
            out._local_value().numpy(), a[dist.get_rank()].numpy().flatten()
        )

        relu_out = paddle.nn.ReLU()(out)
        np.testing.assert_equal(
            str(relu_out.placements[0]), "Shard(dim=0, shard_order=0)"
        )
        np.testing.assert_equal(
            str(relu_out.placements[1]), "Shard(dim=0, shard_order=1)"
        )

    def run_test_case_main(self):
        self.basic_interface_case()
        self.run_test_case_0()
        self.run_test_case_1()
        self.run_test_case_2()
        self.run_test_case_3()
        self.run_test_case_4()


if __name__ == '__main__':
    TestCoShard().run_test_case_main()
