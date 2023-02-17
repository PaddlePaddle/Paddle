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

import platform
import time
import unittest

import numpy as np
from bert import Bert, BertPretrainingCriterion, create_pretraining_dataset

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

SEED = 2023
BATCH_SIZE = 2

if core.is_compiled_with_cuda():
    paddle.set_flags({'FLAGS_cudnn_deterministic': True})


def train(to_static, enable_prim, enable_cinn):
    if core.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    fluid.core._set_prim_all_enabled(
        enable_prim and platform.system() == 'Linux'
    )

    np.random.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)

    train_data_loader = create_pretraining_dataset(
        20, {}, batch_size=BATCH_SIZE, worker_init=None
    )

    bert = Bert()
    criterion = BertPretrainingCriterion()
    if to_static:
        # input_sepc = [
        #     InputSpec(shape=(-1, -1), dtype=paddle.int64, name='input_ids'),
        #     InputSpec(shape=(-1, -1), dtype=paddle.int64, name='segment_ids'),
        #     None,
        #     InputSpec(shape=(-1, 1, 1, -1), dtype=paddle.float32, name='input_mask'),
        #     InputSpec(shape=(-1,), dtype=paddle.int32, name='masked_lm_positions'),
        # ]
        input_sepc = None
        build_strategy = paddle.static.BuildStrategy()
        if enable_cinn:
            build_strategy.build_cinn_pass = True
        bert = paddle.jit.to_static(
            bert, input_sepc, build_strategy=build_strategy
        )

    optimizer = fluid.optimizer.Adam(parameter_list=bert.parameters())

    losses = []
    for step, batch in enumerate(train_data_loader):
        start_time = time.time()
        (
            input_ids,
            segment_ids,
            input_mask,
            masked_lm_positions,
            masked_lm_labels,
            next_sentence_labels,
            masked_lm_scale,
        ) = batch

        prediction_scores, seq_relationship_score = bert(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            masked_positions=masked_lm_positions,
        )

        loss = criterion(
            prediction_scores,
            seq_relationship_score,
            masked_lm_labels,
            next_sentence_labels,
            masked_lm_scale,
        )

        loss.backward()
        optimizer.minimize(loss)
        bert.clear_gradients()
        losses.append(loss)

        print(
            "step: {}, loss: {}, batch_cost: {:.5}".format(
                step,
                loss.numpy(),
                time.time() - start_time,
            )
        )
        if step >= 9:
            break
    return losses


class TestResnet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dy2st = train(to_static=True, enable_prim=False, enable_cinn=False)

    def test_prim(self):
        dy2st_prim = train(to_static=True, enable_prim=True, enable_cinn=False)

    #     # NOTE: Now dy2st is equal to dy2st_prim. With the splitting of kernels, the threshold here may need to be adjusted
    #     # np.testing.assert_allclose(self.dy2st, dy2st_prim, rtol=1e-6)

    @unittest.skipIf(
        not paddle.is_compiled_with_cinn(), "padle is not compiled with CINN"
    )
    def test_cinn(self):
        dy2st_cinn = train(to_static=True, enable_prim=False, enable_cinn=True)

    #     # TODO(0x45f): The following is only temporary thresholds, and the final thresholds needs to be discussed
    #     # np.testing.assert_allclose(self.dy2st[0:2], dy2st_cinn[0:2], rtol=1e-3)
    #     # np.testing.assert_allclose(self.dy2st, dy2st_cinn, rtol=1e-1)

    @unittest.skipIf(
        not paddle.is_compiled_with_cinn(), "padle is not compiled with CINN"
    )
    def test_prim_cinn(self):
        dy2st_prim_cinn = train(
            to_static=True, enable_prim=True, enable_cinn=True
        )

    # #     # TODO(0x45f): The following is only temporary thresholds, and the final thresholds need to be discussed
    # #     # np.testing.assert_allclose(
    # #     #     self.dy2st[0:2], dy2st_prim_cinn[0:2], rtol=1e-2
    # #     # )
    # #     # np.testing.assert_allclose(self.dy2st, dy2st_prim_cinn, rtol=1e-1)


if __name__ == '__main__':
    unittest.main()
