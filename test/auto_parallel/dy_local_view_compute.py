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

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.api import (
    dtensor_from_local,
    dtensor_to_local,
)


class TestLocalViewCompute:
    def __init__(self):
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def run_test_cases(self):
        self.test_local_view_compute()

    def masked_lm_loss_func(self, pred, label, ignored_idx=-100):
        pred_sub = pred[:, 0:1]  # shape [B,1]
        label_float = paddle.cast(label, 'float32')  # shape [B,1]

        raw_loss = paddle.abs(pred_sub - label_float)

        lossmask = label != ignored_idx
        lossmask_ = lossmask.reshape([-1]).cast('float32')
        raw_loss_flat = raw_loss.reshape([-1]).cast('float32')

        masked_lm_loss_sum = paddle.sum(raw_loss_flat * lossmask_)
        valid_count = paddle.sum(lossmask_)

        loss = masked_lm_loss_sum / (valid_count + 1e-8)
        return loss

    def local_view_compute(self, local_pred, local_label):
        # do not use dist.shard_tensor here
        local_pred = local_pred + 1
        local_loss = self.masked_lm_loss_func(
            local_pred, local_label, ignored_idx=-100
        )

        return local_loss

    def test_local_view_compute(self):
        dist.init_parallel_env()
        cur_rank = dist.get_rank()

        # prepare data and label for mask_lm_loss
        if cur_rank == 0:
            pred = paddle.to_tensor([[1.0, 2.0], [4.0, 4.0]], dtype='float32')
            label = paddle.to_tensor([[1], [3]], dtype='int64')
        elif cur_rank == 1:
            pred = paddle.to_tensor([[2.0, 2.0], [7.0, 8.0]], dtype='float32')
            label = paddle.to_tensor([[2], [-100]], dtype='int64')

        local_result = self.local_view_compute(pred.clone(), label.clone())

        dist_pred = dist.shard_tensor(pred, self._mesh, [dist.Replicate()])
        dist_label = dist.shard_tensor(label, self._mesh, [dist.Replicate()])

        local_pred = dtensor_to_local(dist_pred)
        local_label = dtensor_to_local(dist_label)

        local_pred = local_pred + 1
        local_loss = self.masked_lm_loss_func(
            local_pred, local_label, ignored_idx=-100
        )

        assert local_result == local_loss, "local_result != local_loss"

        tensor_list = []
        dist.all_gather(tensor_list, local_loss)
        loss_sum = paddle.sum(paddle.stack(tensor_list))
        dist_loss = dtensor_from_local(
            local_loss, self._mesh, [dist.Partial(dist.ReduceType.kRedSum)]
        )

        assert loss_sum == dist_loss, "loss_sum != dist_loss"


if __name__ == '__main__':
    TestLocalViewCompute().run_test_cases()
