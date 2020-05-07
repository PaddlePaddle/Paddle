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
from bert_dygraph_model import PretrainModelLayer
from bert_utils import get_bert_config, get_feed_data_reader

trainer_count = 1
place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace(
)
SEED = 2020
STEP_NUM = 10
PRINT_STEP = 2


def create_model(bert_config):
    input_fields = {
        'names': [
            'src_ids', 'pos_ids', 'sent_ids', 'input_mask', 'mask_label',
            'mask_pos', 'labels'
        ],
        'shapes': [[None, None], [None, None], [None, None], [None, None, 1],
                   [None, 1], [None, 1], [None, 1]],
        'dtypes':
        ['int64', 'int64', 'int64', 'float32', 'int64', 'int64', 'int64'],
        'lod_levels': [0, 0, 0, 0, 0, 0, 0],
    }

    inputs = [
        fluid.data(
            name=input_fields['names'][i],
            shape=input_fields['shapes'][i],
            dtype=input_fields['dtypes'][i],
            lod_level=input_fields['lod_levels'][i])
        for i in range(len(input_fields['names']))
    ]

    (src_ids, pos_ids, sent_ids, input_mask, mask_label, mask_pos,
     labels) = inputs

    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=inputs, capacity=50, iterable=True)

    bert = PretrainModelLayer(
        config=bert_config, weight_sharing=False, use_fp16=False)

    next_sent_acc, mask_lm_loss, total_loss = bert(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        mask_label=mask_label,
        mask_pos=mask_pos,
        labels=labels)

    return data_loader, next_sent_acc, mask_lm_loss, total_loss


def train_static(bert_config, data_reader):
    train_program = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_program, startup_prog):
        train_program.random_seed = SEED
        startup_prog.random_seed = SEED
        with fluid.unique_name.guard():
            train_data_loader, next_sent_acc, mask_lm_loss, total_loss = \
                create_model(bert_config=bert_config)
            optimizer = fluid.optimizer.Adam()
            optimizer.minimize(total_loss)

    train_data_loader.set_batch_generator(
        data_reader.data_generator(), places=place)
    step_idx = 0
    speed_list = []
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    for feed_dict in train_data_loader:
        cost = []
        lm_cost = []
        acc = []
        fetch_list = [next_sent_acc, mask_lm_loss, total_loss]
        outputs = exe.run(feed=feed_dict,
                          fetch_list=fetch_list,
                          program=train_program)
        each_next_acc, each_mask_lm_cost, each_total_cost = outputs

        acc.extend(each_next_acc)
        lm_cost.extend(each_mask_lm_cost)
        cost.extend(each_total_cost)

        loss = np.mean(np.array(cost))
        ppl = np.mean(np.exp(np.array(lm_cost)))

        if step_idx % PRINT_STEP == 0:
            if step_idx == 0:
                print("step: %d, loss: %f, ppl: %f, next_sent_acc: %f" %
                      (step_idx, loss, ppl, np.mean(np.array(acc))))
                avg_batch_time = time.time()
            else:
                speed = PRINT_STEP / (time.time() - avg_batch_time)
                speed_list.append(speed)
                print(
                    "step: %d, loss: %f, ppl: %f, next_sent_acc: %f, speed: %.3f steps/s"
                    % (step_idx, loss, ppl, np.mean(np.array(acc)), speed))
                avg_batch_time = time.time()

        step_idx += 1
        if step_idx == STEP_NUM:
            break
    return loss, ppl


def train_dygraph(bert_config, data_reader):
    with fluid.dygraph.guard(place):
        fluid.default_main_program().random_seed = SEED
        fluid.default_startup_program().random_seed = SEED
        # define data loader
        data_loader = fluid.io.DataLoader.from_generator(
            capacity=50, iterable=True)
        data_loader.set_batch_generator(
            data_reader.data_generator(), places=place)

        # define model
        bert = PretrainModelLayer(
            config=bert_config, weight_sharing=False, use_fp16=False)

        # define optimizer
        optimizer = fluid.optimizer.Adam(parameter_list=bert.parameters())
        step_idx = 0
        speed_list = []
        for input_data in data_loader():
            cost = []
            lm_cost = []
            acc = []
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

            each_next_acc = next_sent_acc.numpy()
            each_mask_lm_cost = mask_lm_loss.numpy()
            each_total_cost = total_loss.numpy()

            acc.extend(each_next_acc)
            lm_cost.extend(each_mask_lm_cost)
            cost.extend(each_total_cost)

            loss = np.mean(np.array(cost))
            ppl = np.mean(np.exp(np.array(lm_cost)))

            if step_idx % PRINT_STEP == 0:
                if step_idx == 0:
                    print("step: %d, loss: %f, ppl: %f, next_sent_acc: %f" %
                          (step_idx, loss, ppl, np.mean(np.array(acc))))
                    avg_batch_time = time.time()
                else:
                    speed = PRINT_STEP / (time.time() - avg_batch_time)
                    speed_list.append(speed)
                    print(
                        "step: %d, loss: %f, ppl: %f, next_sent_acc: %f, speed: %.3f steps/s"
                        % (step_idx, loss, ppl, np.mean(np.array(acc)), speed))
                    avg_batch_time = time.time()

            step_idx += 1
            if step_idx == STEP_NUM:
                break
        return loss, ppl


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
