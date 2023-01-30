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

import os
import tempfile
import time
import unittest

import numpy as np
<<<<<<< HEAD
from seq2seq_dygraph_model import AttentionModel, BaseModel
from seq2seq_utils import Seq2SeqModelHyperParams, get_data_iter

import paddle
import paddle.fluid as fluid
from paddle.nn import ClipGradByGlobalNorm

place = (
    fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
)
=======
import paddle.fluid as fluid
from paddle.fluid.clip import GradientClipByGlobalNorm
from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator

from seq2seq_dygraph_model import BaseModel, AttentionModel
from seq2seq_utils import Seq2SeqModelHyperParams
from seq2seq_utils import get_data_iter

place = fluid.CUDAPlace(
    0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
program_translator = ProgramTranslator()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
STEP_NUM = 10
PRINT_STEP = 2


def prepare_input(batch):
    src_ids, src_mask, tar_ids, tar_mask = batch
    src_ids = src_ids.reshape((src_ids.shape[0], src_ids.shape[1]))
    in_tar = tar_ids[:, :-1]
    label_tar = tar_ids[:, 1:]

    in_tar = in_tar.reshape((in_tar.shape[0], in_tar.shape[1]))
    label_tar = label_tar.reshape((label_tar.shape[0], label_tar.shape[1], 1))
    inputs = [src_ids, in_tar, label_tar, src_mask, tar_mask]
    return inputs, np.sum(tar_mask)


def train(args, attn_model=False):
    with fluid.dygraph.guard(place):
        fluid.default_startup_program().random_seed = 2020
        fluid.default_main_program().random_seed = 2020

        if attn_model:
<<<<<<< HEAD
            model = AttentionModel(
                args.hidden_size,
                args.src_vocab_size,
                args.tar_vocab_size,
                args.batch_size,
                num_layers=args.num_layers,
                init_scale=args.init_scale,
                dropout=args.dropout,
            )
        else:
            model = BaseModel(
                args.hidden_size,
                args.src_vocab_size,
                args.tar_vocab_size,
                args.batch_size,
                num_layers=args.num_layers,
                init_scale=args.init_scale,
                dropout=args.dropout,
            )

        gloabl_norm_clip = ClipGradByGlobalNorm(args.max_grad_norm)
        optimizer = fluid.optimizer.SGD(
            args.learning_rate,
            parameter_list=model.parameters(),
            grad_clip=gloabl_norm_clip,
        )
=======
            model = AttentionModel(args.hidden_size,
                                   args.src_vocab_size,
                                   args.tar_vocab_size,
                                   args.batch_size,
                                   num_layers=args.num_layers,
                                   init_scale=args.init_scale,
                                   dropout=args.dropout)
        else:
            model = BaseModel(args.hidden_size,
                              args.src_vocab_size,
                              args.tar_vocab_size,
                              args.batch_size,
                              num_layers=args.num_layers,
                              init_scale=args.init_scale,
                              dropout=args.dropout)

        gloabl_norm_clip = GradientClipByGlobalNorm(args.max_grad_norm)
        optimizer = fluid.optimizer.SGD(args.learning_rate,
                                        parameter_list=model.parameters(),
                                        grad_clip=gloabl_norm_clip)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        model.train()
        train_data_iter = get_data_iter(args.batch_size)

        batch_times = []
        for batch_id, batch in enumerate(train_data_iter):
            total_loss = 0
            word_count = 0.0
            batch_start_time = time.time()
            input_data_feed, word_num = prepare_input(batch)
            input_data_feed = [
                fluid.dygraph.to_variable(np_inp) for np_inp in input_data_feed
            ]
            word_count += word_num
            loss = model(input_data_feed)
            loss.backward()
            optimizer.minimize(loss)
            model.clear_gradients()
            total_loss += loss * args.batch_size
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            if batch_id % PRINT_STEP == 0:
                print(
                    "Batch:[%d]; Time: %.5f s; loss: %.5f; total_loss: %.5f; word num: %.5f; ppl: %.5f"
<<<<<<< HEAD
                    % (
                        batch_id,
                        batch_time,
                        loss.numpy(),
                        total_loss.numpy(),
                        word_count,
                        np.exp(total_loss.numpy() / word_count),
                    )
                )
=======
                    % (batch_id, batch_time, loss.numpy(), total_loss.numpy(),
                       word_count, np.exp(total_loss.numpy() / word_count)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            if attn_model:
                # NOTE: Please see code of AttentionModel.
                # Because diff exits if call while_loop in static graph, only run 4 batches to pass the test temporarily.
                if batch_id + 1 >= 4:
                    break
            else:
                if batch_id + 1 >= STEP_NUM:
                    break

<<<<<<< HEAD
        model_path = (
            args.attn_model_path if attn_model else args.base_model_path
        )
=======
        model_path = args.attn_model_path if attn_model else args.base_model_path
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        model_dir = os.path.join(model_path)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
<<<<<<< HEAD
        paddle.save(model.state_dict(), model_dir + '.pdparams')
=======
        fluid.save_dygraph(model.state_dict(), model_dir)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return loss.numpy()


def infer(args, attn_model=False):
    with fluid.dygraph.guard(place):

        if attn_model:
<<<<<<< HEAD
            model = AttentionModel(
                args.hidden_size,
                args.src_vocab_size,
                args.tar_vocab_size,
                args.batch_size,
                beam_size=args.beam_size,
                num_layers=args.num_layers,
                init_scale=args.init_scale,
                dropout=0.0,
                mode='beam_search',
            )
        else:
            model = BaseModel(
                args.hidden_size,
                args.src_vocab_size,
                args.tar_vocab_size,
                args.batch_size,
                beam_size=args.beam_size,
                num_layers=args.num_layers,
                init_scale=args.init_scale,
                dropout=0.0,
                mode='beam_search',
            )

        model_path = (
            args.attn_model_path if attn_model else args.base_model_path
        )
        state_dict = paddle.load(model_path + '.pdparams')
=======
            model = AttentionModel(args.hidden_size,
                                   args.src_vocab_size,
                                   args.tar_vocab_size,
                                   args.batch_size,
                                   beam_size=args.beam_size,
                                   num_layers=args.num_layers,
                                   init_scale=args.init_scale,
                                   dropout=0.0,
                                   mode='beam_search')
        else:
            model = BaseModel(args.hidden_size,
                              args.src_vocab_size,
                              args.tar_vocab_size,
                              args.batch_size,
                              beam_size=args.beam_size,
                              num_layers=args.num_layers,
                              init_scale=args.init_scale,
                              dropout=0.0,
                              mode='beam_search')

        model_path = args.attn_model_path if attn_model else args.base_model_path
        state_dict, _ = fluid.dygraph.load_dygraph(model_path)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        model.set_dict(state_dict)
        model.eval()
        train_data_iter = get_data_iter(args.batch_size, mode='infer')
        for batch_id, batch in enumerate(train_data_iter):
            input_data_feed, word_num = prepare_input(batch)
            input_data_feed = [
                fluid.dygraph.to_variable(np_inp) for np_inp in input_data_feed
            ]
            outputs = model.beam_search(input_data_feed)
            break

        return outputs.numpy()


class TestSeq2seq(unittest.TestCase):
<<<<<<< HEAD
    def setUp(self):
        self.args = Seq2SeqModelHyperParams
        self.temp_dir = tempfile.TemporaryDirectory()
        self.args.base_model_path = os.path.join(
            self.temp_dir.name, self.args.base_model_path
        )
        self.args.attn_model_path = os.path.join(
            self.temp_dir.name, self.args.attn_model_path
        )
        self.args.reload_model = os.path.join(
            self.temp_dir.name, self.args.reload_model
        )
=======

    def setUp(self):
        self.args = Seq2SeqModelHyperParams
        self.temp_dir = tempfile.TemporaryDirectory()
        self.args.base_model_path = os.path.join(self.temp_dir.name,
                                                 self.args.base_model_path)
        self.args.attn_model_path = os.path.join(self.temp_dir.name,
                                                 self.args.attn_model_path)
        self.args.reload_model = os.path.join(self.temp_dir.name,
                                              self.args.reload_model)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_dygraph(self, mode="train", attn_model=False):
<<<<<<< HEAD
        paddle.jit.enable_to_static(False)
=======
        program_translator.enable(False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if mode == "train":
            return train(self.args, attn_model)
        else:
            return infer(self.args, attn_model)

    def run_static(self, mode="train", attn_model=False):
<<<<<<< HEAD
        paddle.jit.enable_to_static(True)
=======
        program_translator.enable(True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if mode == "train":
            return train(self.args, attn_model)
        else:
            return infer(self.args, attn_model)

    def _test_train(self, attn_model=False):
        dygraph_loss = self.run_dygraph(mode="train", attn_model=attn_model)
        static_loss = self.run_static(mode="train", attn_model=attn_model)
        result = np.allclose(dygraph_loss, static_loss)
<<<<<<< HEAD
        self.assertTrue(
            result,
            msg="\ndygraph_loss = {} \nstatic_loss = {}".format(
                dygraph_loss, static_loss
            ),
        )
=======
        self.assertTrue(result,
                        msg="\ndygraph_loss = {} \nstatic_loss = {}".format(
                            dygraph_loss, static_loss))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _test_predict(self, attn_model=False):
        pred_dygraph = self.run_dygraph(mode="test", attn_model=attn_model)
        pred_static = self.run_static(mode="test", attn_model=attn_model)
        result = np.allclose(pred_static, pred_dygraph)
<<<<<<< HEAD
        self.assertTrue(
            result,
            msg="\npred_dygraph = {} \npred_static = {}".format(
                pred_dygraph, pred_static
            ),
        )
=======
        self.assertTrue(result,
                        msg="\npred_dygraph = {} \npred_static = {}".format(
                            pred_dygraph, pred_static))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_base_model(self):
        self._test_train(attn_model=False)
        self._test_predict(attn_model=False)

    def test_attn_model(self):
        self._test_train(attn_model=True)
        # TODO(liym27): add predict
        # self._test_predict(attn_model=True)


if __name__ == '__main__':
<<<<<<< HEAD
    unittest.main()
=======
    # switch into new eager mode
    with fluid.framework._test_eager_guard():
        unittest.main()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
