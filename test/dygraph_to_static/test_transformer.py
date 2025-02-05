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

import logging
import os
import tempfile
import time
import unittest

import numpy as np
import transformer_util as util
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
    test_default_and_pir,
)
from transformer_dygraph_model import (
    CrossEntropyCriterion,
    Transformer,
    position_encoding_init,
)

import paddle

trainer_count = 1
place = (
    paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
)
SEED = 10
STEP_NUM = 10


def train_dygraph(args, batch_generator):
    if SEED is not None:
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
    # define data loader

    train_loader = paddle.io.DataLoader(
        batch_generator, batch_size=None, places=place
    )

    # define model
    transformer = paddle.jit.to_static(
        Transformer(
            args.src_vocab_size,
            args.trg_vocab_size,
            args.max_length + 1,
            args.n_layer,
            args.n_head,
            args.d_key,
            args.d_value,
            args.d_model,
            args.d_inner_hid,
            args.prepostprocess_dropout,
            args.attention_dropout,
            args.relu_dropout,
            args.preprocess_cmd,
            args.postprocess_cmd,
            args.weight_sharing,
            args.bos_idx,
            args.eos_idx,
        )
    )
    # define loss
    criterion = CrossEntropyCriterion(args.label_smooth_eps)
    # define optimizer
    learning_rate = paddle.optimizer.lr.NoamDecay(
        args.d_model, args.warmup_steps, args.learning_rate
    )
    # define optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=float(args.eps),
        parameters=transformer.parameters(),
    )
    # the best cross-entropy value with label smoothing
    loss_normalizer = -(
        (1.0 - args.label_smooth_eps) * np.log(1.0 - args.label_smooth_eps)
        + args.label_smooth_eps
        * np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20)
    )
    ce_time = []
    ce_ppl = []
    avg_loss = []
    step_idx = 0
    for pass_id in range(args.epoch):
        pass_start_time = time.time()
        batch_id = 0
        for input_data in train_loader():
            (
                src_word,
                src_pos,
                src_slf_attn_bias,
                trg_word,
                trg_pos,
                trg_slf_attn_bias,
                trg_src_attn_bias,
                lbl_word,
                lbl_weight,
            ) = input_data
            logits = transformer(
                src_word,
                src_pos,
                src_slf_attn_bias,
                trg_word,
                trg_pos,
                trg_slf_attn_bias,
                trg_src_attn_bias,
            )
            sum_cost, avg_cost, token_num = criterion(
                logits, lbl_word, lbl_weight
            )
            avg_cost.backward()
            optimizer.minimize(avg_cost)
            transformer.clear_gradients()
            if step_idx % args.print_step == 0:
                total_avg_cost = avg_cost.numpy() * trainer_count
                avg_loss.append(float(total_avg_cost))
                if step_idx == 0:
                    logging.info(
                        f"step_idx: {step_idx}, epoch: {pass_id}, batch: {batch_id}, avg loss: {total_avg_cost:f}, "
                        f"normalized loss: {total_avg_cost - loss_normalizer:f}, ppl: {np.exp([min(total_avg_cost, 100)]).item():f}"
                    )
                    avg_batch_time = time.time()
                else:
                    logging.info(
                        f"step_idx: {step_idx}, epoch: {pass_id}, batch: {batch_id}, avg loss: {total_avg_cost:f}, "
                        f"normalized loss: {total_avg_cost - loss_normalizer:f}, "
                        f"ppl: {np.exp([min(total_avg_cost, 100)]).item():f}, "
                        f"speed: {args.print_step / (time.time() - avg_batch_time):.2f} steps/s"
                    )
                    ce_ppl.append(np.exp([min(total_avg_cost, 100)]))
                    avg_batch_time = time.time()
            batch_id += 1
            step_idx += 1
            if step_idx == STEP_NUM:
                if args.save_dygraph_model_path:
                    model_dir = os.path.join(args.save_dygraph_model_path)
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    paddle.save(
                        transformer.state_dict(),
                        os.path.join(model_dir, "transformer") + '.pdparams',
                    )
                    paddle.save(
                        optimizer.state_dict(),
                        os.path.join(model_dir, "transformer") + '.pdparams',
                    )
                break
        time_consumed = time.time() - pass_start_time
        ce_time.append(time_consumed)
    return np.array(avg_loss)


def predict_dygraph(args, batch_generator):
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)

    # define data loader
    test_loader = paddle.io.DataLoader(
        batch_generator, batch_size=None, places=place
    )

    # define model
    transformer = paddle.jit.to_static(
        Transformer(
            args.src_vocab_size,
            args.trg_vocab_size,
            args.max_length + 1,
            args.n_layer,
            args.n_head,
            args.d_key,
            args.d_value,
            args.d_model,
            args.d_inner_hid,
            args.prepostprocess_dropout,
            args.attention_dropout,
            args.relu_dropout,
            args.preprocess_cmd,
            args.postprocess_cmd,
            args.weight_sharing,
            args.bos_idx,
            args.eos_idx,
        )
    )

    # load the trained model
    model_dict, _ = util.load_dygraph(
        os.path.join(args.save_dygraph_model_path, "transformer")
    )
    # to avoid a longer length than training, reset the size of position
    # encoding to max_length
    model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model
    )
    model_dict["decoder.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model
    )
    transformer.load_dict(model_dict)

    # set evaluate mode
    transformer.eval()

    step_idx = 0
    speed_list = []
    for input_data in test_loader():
        (
            src_word,
            src_pos,
            src_slf_attn_bias,
            trg_word,
            trg_src_attn_bias,
        ) = input_data
        seq_ids, seq_scores = paddle.jit.to_static(
            transformer.beam_search(
                src_word,
                src_pos,
                src_slf_attn_bias,
                trg_word,
                trg_src_attn_bias,
                bos_id=args.bos_idx,
                eos_id=args.eos_idx,
                beam_size=args.beam_size,
                max_len=args.max_out_len,
            )
        )
        seq_ids = seq_ids.numpy()
        seq_scores = seq_scores.numpy()
        if step_idx % args.print_step == 0:
            if step_idx == 0:
                logging.info(
                    f"Dygraph Predict: step_idx: {step_idx}, 1st seq_id: {seq_ids[0][0][0]}, 1st seq_score: {seq_scores[0][0]:.2f}"
                )
                avg_batch_time = time.time()
            else:
                speed = args.print_step / (time.time() - avg_batch_time)
                speed_list.append(speed)
                logging.info(
                    f"Dygraph Predict: step_idx: {step_idx}, 1st seq_id: {seq_ids[0][0][0]}, 1st seq_score: {seq_scores[0][0]:.2f}, speed: {speed:.3f} steps/s"
                )
                avg_batch_time = time.time()

        step_idx += 1
        if step_idx == STEP_NUM:
            break
    logging.info(
        f"Dygraph Predict:  avg_speed: {np.mean(speed_list):.4f} steps/s"
    )
    return seq_ids, seq_scores


class TestTransformer(Dy2StTestBase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def prepare(self, mode='train'):
        args = util.ModelHyperParams()
        args.save_dygraph_model_path = os.path.join(
            self.temp_dir.name, args.save_dygraph_model_path
        )
        args.save_static_model_path = os.path.join(
            self.temp_dir.name, args.save_static_model_path
        )
        args.inference_model_dir = os.path.join(
            self.temp_dir.name, args.inference_model_dir
        )
        args.output_file = os.path.join(self.temp_dir.name, args.output_file)
        batch_generator = util.get_feed_data_reader(args, mode)
        if mode == 'train':
            batch_generator = util.TransedWMT16TrainDataSet(
                batch_generator, args.batch_size * (args.epoch + 1)
            )
        else:
            batch_generator = util.TransedWMT16TestDataSet(
                batch_generator, args.batch_size * (args.epoch + 1)
            )
        return args, batch_generator

    def _test_train(self):
        args, batch_generator = self.prepare(mode='train')
        static_avg_loss = train_dygraph(args, batch_generator)
        with enable_to_static_guard(False):
            dygraph_avg_loss = train_dygraph(args, batch_generator)
        np.testing.assert_allclose(
            static_avg_loss, dygraph_avg_loss, rtol=1e-05
        )

    def _test_predict(self):
        args, batch_generator = self.prepare(mode='test')
        static_seq_ids, static_scores = predict_dygraph(args, batch_generator)
        with enable_to_static_guard(False):
            dygraph_seq_ids, dygraph_scores = predict_dygraph(
                args, batch_generator
            )

        np.testing.assert_allclose(static_seq_ids, dygraph_seq_ids, rtol=1e-05)
        np.testing.assert_allclose(static_scores, dygraph_scores, rtol=1e-05)

    @test_default_and_pir
    def test_check_result(self):
        self._test_train()
        # TODO(zhangliujie) fix predict fail due to precision misalignment
        # self._test_predict()


if __name__ == '__main__':
    unittest.main()
