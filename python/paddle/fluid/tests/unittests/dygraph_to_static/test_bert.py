# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import time
import unittest

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator

from bert_dygraph_model import PretrainModelLayer
from bert_utils import get_bert_config, get_feed_data_reader

program_translator = ProgramTranslator()
place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace(
)
SEED = 2020
STEP_NUM = 10
PRINT_STEP = 2


def train(bert_config, data_reader):
    with fluid.dygraph.guard(place):
        fluid.default_main_program().random_seed = SEED
        fluid.default_startup_program().random_seed = SEED

        data_loader = fluid.io.DataLoader.from_generator(
            capacity=50, iterable=True)
        data_loader.set_batch_generator(
            data_reader.data_generator(), places=place)

        bert = PretrainModelLayer(
            config=bert_config, weight_sharing=False, use_fp16=False)

        optimizer = fluid.optimizer.Adam(parameter_list=bert.parameters())
        step_idx = 0
        speed_list = []
        for input_data in data_loader():
            src_ids, pos_ids, sent_ids, input_mask, mask_label, mask_pos, labels = input_data
            next_sent_acc, mask_lm_loss, total_loss = bert(
                src_ids=src_ids,
                position_ids=pos_ids,
                sentence_ids=sent_ids,
                input_mask=input_mask,
                mask_label=mask_label,
                mask_pos=mask_pos,
                labels=labels)
            total_loss.backward()
            optimizer.minimize(total_loss)
            bert.clear_gradients()

            acc = np.mean(np.array(next_sent_acc.numpy()))
            loss = np.mean(np.array(total_loss.numpy()))
            ppl = np.mean(np.exp(np.array(mask_lm_loss.numpy())))

            if step_idx % PRINT_STEP == 0:
                if step_idx == 0:
                    print("Step: %d, loss: %f, ppl: %f, next_sent_acc: %f" %
                          (step_idx, loss, ppl, acc))
                    avg_batch_time = time.time()
                else:
                    speed = PRINT_STEP / (time.time() - avg_batch_time)
                    speed_list.append(speed)
                    print(
                        "Step: %d, loss: %f, ppl: %f, next_sent_acc: %f, speed: %.3f steps/s"
                        % (step_idx, loss, ppl, acc, speed))
                    avg_batch_time = time.time()

            step_idx += 1
            if step_idx == STEP_NUM:
                break
        return loss, ppl


def train_dygraph(bert_config, data_reader):
    program_translator.enable(False)
    return train(bert_config, data_reader)


def train_static(bert_config, data_reader):
    program_translator.enable(True)
    return train(bert_config, data_reader)


class TestBert(unittest.TestCase):
    def setUp(self):
        self.bert_config = get_bert_config()
        self.data_reader = get_feed_data_reader(self.bert_config)

    def test_train(self):
        static_loss, static_ppl = train_static(self.bert_config,
                                               self.data_reader)
        dygraph_loss, dygraph_ppl = train_dygraph(self.bert_config,
                                                  self.data_reader)
        self.assertTrue(
            np.allclose(static_loss, static_loss),
            msg="static_loss: {} \n static_loss: {}".format(static_loss,
                                                            dygraph_loss))
        self.assertTrue(
            np.allclose(static_ppl, dygraph_ppl),
            msg="static_ppl: {} \n dygraph_ppl: {}".format(static_ppl,
                                                           dygraph_ppl))


if __name__ == '__main__':
    unittest.main()
