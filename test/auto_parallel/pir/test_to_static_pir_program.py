# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import Replicate, Shard
from paddle.io import DataLoader

BATCH_SIZE = 4
BATCH_NUM = 40
IMAGE_SIZE = 16
CLASS_NUM = 8
np.random.seed(2024)
paddle.seed(2024)


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class DemoNet(nn.Layer):
    def __init__(self, mesh, shard=True):
        super().__init__()
        self._mesh = mesh
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, bias_attr=False)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, bias_attr=False)
        self.relu_0 = nn.ReLU()
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.shard = shard
        # shard the weights of this layer
        if self.shard:
            self.linear_0.weight = dist.shard_tensor(
                self.linear_0.weight,
                self._mesh,
                [Shard(1)],
                stop_gradient=False,
            )
            self.linear_1.weight = dist.shard_tensor(
                self.linear_1.weight,
                self._mesh,
                [Shard(0)],
                stop_gradient=False,
            )
        else:
            self.linear_0.weight = dist.shard_tensor(
                self.linear_0.weight,
                self._mesh,
                [Replicate()],
                stop_gradient=False,
            )
            self.linear_1.weight = dist.shard_tensor(
                self.linear_1.weight,
                self._mesh,
                [Replicate()],
                stop_gradient=False,
            )

    def forward(self, x):
        x.stop_gradient = False
        out = self.relu_0(x)  # triggle backward partial allreduce
        out = self.linear_0(out)
        out = self.relu_1(out)
        out = self.linear_1(out)
        out = self.relu_2(out)  # triggle forward partial allreduce
        out = paddle.cast(out, 'float32')
        return out


def create_data_loader(
    batch_size=BATCH_SIZE,
    batch_num=BATCH_NUM,
    image_size=IMAGE_SIZE,
    class_num=CLASS_NUM,
):
    nsamples = batch_size * batch_num
    images = np.random.rand(nsamples, image_size).astype('float32')
    labels = np.random.rand(nsamples, class_num).astype('float32')
    dataset = RandomDataset(images, labels, nsamples)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader


class TestToStaticPirProgramTrain(unittest.TestCase):
    def test_to_static_program(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        layer = DemoNet(mesh)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
        dist_model = dist.to_static(layer, dist_loader, loss_fn, opt)

        # dist_model.train()
        mode = "train"
        dist_model.train()
        main_program = dist_model._engine._pir_dist_main_progs["train"]

        relu_idx = 0
        matmul_idx = 0
        data_idx = 0
        matmul_grad_idx = 0
        sgd_idx = 0
        ops = main_program.global_block().ops

        backward_op_list = [
            "pd_op.sgd_",
            "pd_op.sgd_",
            "pd_op.relu_grad",
            "pd_op.all_reduce",
            "pd_op.matmul_grad",
            "pd_op.relu_grad",
            "pd_op.matmul_grad",
            "pd_op.relu_grad",
            "pd_op.cast",
            "pd_op.subtract_grad",
            "pd_op.square_grad",
            "pd_op.mean_grad",
        ]
        index = -1
        for op_name in backward_op_list:
            self.assertEqual(ops[index].name(), op_name)
            index = index - 1

        for op in ops:
            # skip shadow_output
            if op.num_results() == 0:
                continue
            tensor = op.result(0)
            # while tensor's stop_gradient is true, the corresponding grad tensor is initialized.
            if not tensor.initialized():
                continue
            self.assertTrue(tensor.is_dist_dense_tensor_type())
            self.assertEqual(tensor.dist_attr().process_mesh.shape, [2])
            self.assertEqual(
                tensor.dist_attr().process_mesh.process_ids, [0, 1]
            )

            if op.name() == 'pd_op.data':
                if data_idx != 0:
                    self.assertEqual(tensor.dist_attr().dims_mapping, [-1, -1])
                    self.assertEqual(tensor.dist_attr().partial_dims, set())
                data_idx += 1
            elif op.name() == 'builtin.parameter':
                self.assertTrue(tensor.is_dense_tensor_type())
                self.assertTrue(tensor.is_dist_dense_tensor_type())
                self.assertTrue(tensor.is_dist_dense_tensor_type())
                self.assertEqual(tensor.dist_attr().process_mesh.shape, [2])
                self.assertEqual(
                    tensor.dist_attr().process_mesh.process_ids, [0, 1]
                )
                if tensor.shape == [IMAGE_SIZE, IMAGE_SIZE]:
                    self.assertEqual(tensor.dist_attr().dims_mapping, [-1, 0])
                elif tensor.shape == [IMAGE_SIZE, CLASS_NUM]:
                    self.assertEqual(tensor.dist_attr().dims_mapping, [0, -1])
                self.assertEqual(tensor.dist_attr().partial_dims, set())
            if op.name() == 'pd_op.relu':
                if relu_idx == 0:
                    self.assertEqual(tensor.dist_attr().dims_mapping, [-1, -1])
                    self.assertEqual(tensor.dist_attr().partial_dims, set())
                    self.assertEqual(
                        tensor._local_shape, [BATCH_SIZE, IMAGE_SIZE]
                    )
                elif relu_idx == 1:
                    self.assertEqual(tensor.dist_attr().dims_mapping, [-1, 0])
                    self.assertEqual(tensor.dist_attr().partial_dims, set())
                    self.assertEqual(
                        tensor._local_shape, [BATCH_SIZE, IMAGE_SIZE // 2]
                    )
                elif relu_idx == 2:
                    self.assertEqual(tensor.dist_attr().dims_mapping, [-1, -1])
                    self.assertEqual(tensor.dist_attr().partial_dims, set())
                    self.assertEqual(
                        tensor._local_shape, [BATCH_SIZE, CLASS_NUM]
                    )
                relu_idx += 1
            if op.name() == 'pd_op.matmul':
                if matmul_idx == 0:
                    self.assertEqual(tensor.dist_attr().dims_mapping, [-1, 0])
                    self.assertEqual(tensor.dist_attr().partial_dims, set())
                    self.assertEqual(
                        tensor._local_shape, [BATCH_SIZE, IMAGE_SIZE // 2]
                    )
                elif matmul_idx == 1:
                    self.assertEqual(tensor.dist_attr().dims_mapping, [-1, -1])
                    self.assertEqual(tensor.dist_attr().partial_dims, {0})
                    self.assertEqual(
                        tensor._local_shape, [BATCH_SIZE, CLASS_NUM]
                    )
                matmul_idx += 1
            if op.name() == 'pd_op.matmul_grad':
                if matmul_grad_idx == 0:
                    self.assertEqual(tensor.dist_attr().dims_mapping, [-1, 0])
                    self.assertEqual(tensor.dist_attr().partial_dims, set())
                    self.assertEqual(
                        tensor._local_shape, [BATCH_SIZE, CLASS_NUM]
                    )
                elif matmul_grad_idx == 1:
                    self.assertEqual(tensor.dist_attr().dims_mapping, [-1, -1])
                    self.assertEqual(tensor.dist_attr().partial_dims, {0})
                    self.assertEqual(
                        tensor._local_shape, [BATCH_SIZE, IMAGE_SIZE]
                    )
                matmul_grad_idx += 1
            if op.name() == 'pd_op.sgd_':
                if sgd_idx == 0:
                    self.assertEqual(tensor.dist_attr().dims_mapping, [0, -1])
                    self.assertEqual(tensor.dist_attr().partial_dims, set())
                    self.assertEqual(
                        tensor._local_shape, [IMAGE_SIZE // 2, CLASS_NUM]
                    )
                elif sgd_idx == 1:
                    self.assertEqual(tensor.dist_attr().dims_mapping, [-1, 0])
                    self.assertEqual(tensor.dist_attr().partial_dims, set())
                    self.assertEqual(
                        tensor._local_shape, [IMAGE_SIZE, IMAGE_SIZE // 2]
                    )
                sgd_idx += 1

        # dist_model.train()
        # for batch_id, (image, label) in enumerate(dist_loader()):
        #     loss = dist_model(image, label)


if __name__ == "__main__":
    unittest.main()
