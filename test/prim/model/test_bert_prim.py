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

import os
import time
import unittest

import numpy as np
from bert import Bert, BertPretrainingCriterion, create_pretraining_dataset

import paddle
from paddle import base
from paddle.base import core
from paddle.dataset.common import DATA_HOME, download

SEED = 2023
BATCH_SIZE = 2

URL = 'https://paddle-ci.gz.bcebos.com/prim_cinn/bert_training_data.npz'
MODULE_NAME = 'test_bert_prim_cinn'
MD5SUM = '71e730ee8d7aa77a215b7e898aa089af'
SAVE_NAME = 'bert_training_data.npz'


DY2ST_PRIM_GT = [
    11.144556999206543,
    10.343620300292969,
    10.330279350280762,
    10.276118278503418,
    10.222086906433105,
    10.194628715515137,
    10.14902114868164,
    10.096250534057617,
    10.104615211486816,
    9.985644340515137,
]

if core.is_compiled_with_cuda():
    paddle.set_flags({'FLAGS_cudnn_deterministic': True})


def train(to_static, enable_prim, enable_cinn):
    if core.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    base.core._set_prim_all_enabled(enable_prim)

    np.random.seed(SEED)
    paddle.seed(SEED)
    # paddle.framework.random._manual_program_seed(SEED)

    train_data_loader = create_pretraining_dataset(
        os.path.join(DATA_HOME, MODULE_NAME, SAVE_NAME),
        20,
        {},
        batch_size=BATCH_SIZE,
        worker_init=None,
    )

    # Now only apply dy2st for encoder
    bert = Bert(to_static, enable_cinn)
    criterion = BertPretrainingCriterion()

    optimizer = paddle.optimizer.Adam(parameters=bert.parameters())

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
        losses.append(loss.numpy().item())

        print(
            f"step: {step}, loss: {loss.numpy()}, batch_cost: {time.time() - start_time:.5}"
        )
        if step >= 9:
            break
    print(losses)
    return losses


class TestBert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        download(URL, MODULE_NAME, MD5SUM, SAVE_NAME)

    def tearDown(self):
        paddle.set_flags({'FLAGS_deny_cinn_ops': ''})

    @unittest.skipIf(
        not (paddle.is_compiled_with_cinn() and paddle.is_compiled_with_cuda()),
        "paddle is not compiled with CINN and CUDA",
    )
    def test_prim(self):
        dy2st_prim = train(to_static=True, enable_prim=True, enable_cinn=False)
        np.testing.assert_allclose(dy2st_prim, DY2ST_PRIM_GT, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
