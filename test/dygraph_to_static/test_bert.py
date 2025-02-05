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
from bert_dygraph_model import PretrainModelLayer
from bert_utils import get_bert_config, get_feed_data_reader
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
    test_sot_only,
)
from predictor_utils import PredictorTools

import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import unique_name
from paddle.framework import use_pir_api
from paddle.jit.pir_translated_layer import PIR_INFER_MODEL_SUFFIX
from paddle.jit.translated_layer import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX

place = (
    paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
)
SEED = 2020
STEP_NUM = 10
PRINT_STEP = 2


class FakeBertDataset(paddle.io.Dataset):
    def __init__(self, data_reader, steps):
        self.src_ids = []
        self.pos_ids = []
        self.sent_ids = []
        self.input_mask = []
        self.mask_label = []
        self.mask_pos = []
        self.labels = []
        self.data_reader = data_reader

        self._generate_fake_data(1 * (steps + 1))

    def _generate_fake_data(self, length):
        for i, data in enumerate(self.data_reader.data_generator()()):
            if i >= length:
                break
            self.src_ids.append(data[0])
            self.pos_ids.append(data[1])
            self.sent_ids.append(data[2])
            self.input_mask.append(data[3])
            self.mask_label.append(data[4])
            self.mask_pos.append(data[5])
            self.labels.append(data[6])

    def __getitem__(self, idx):
        return [
            self.src_ids[idx],
            self.pos_ids[idx],
            self.sent_ids[idx],
            self.input_mask[idx],
            self.mask_label[idx],
            self.mask_pos[idx],
            self.labels[idx],
        ]

    def __len__(self):
        return len(self.src_ids)


class TestBert(Dy2StTestBase):
    def setUp(self):
        self.bert_config = get_bert_config()
        self.data_reader = get_feed_data_reader(self.bert_config)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_save_dir = os.path.join(self.temp_dir.name, 'inference')
        self.model_save_prefix = os.path.join(self.model_save_dir, 'bert')
        self.model_filename = 'bert' + INFER_MODEL_SUFFIX
        self.pir_model_filename = 'bert' + PIR_INFER_MODEL_SUFFIX
        self.params_filename = 'bert' + INFER_PARAMS_SUFFIX
        self.dy_state_dict_save_path = os.path.join(
            self.temp_dir.name, 'bert.dygraph'
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def train(self, bert_config, data_reader, to_static):
        with unique_name.guard():
            paddle.seed(SEED)

            fake_dataset = FakeBertDataset(data_reader, STEP_NUM)
            data_loader = paddle.io.DataLoader(
                fake_dataset, places=place, batch_size=None
            )

            bert = PretrainModelLayer(
                config=bert_config, weight_sharing=False, use_fp16=False
            )

            optimizer = paddle.optimizer.Adam(parameters=bert.parameters())
            step_idx = 0
            speed_list = []
            for input_data in data_loader():
                (
                    src_ids,
                    pos_ids,
                    sent_ids,
                    input_mask,
                    mask_label,
                    mask_pos,
                    labels,
                ) = input_data

                next_sent_acc, mask_lm_loss, total_loss = bert(
                    src_ids=src_ids,
                    position_ids=pos_ids,
                    sentence_ids=sent_ids,
                    input_mask=input_mask,
                    mask_label=mask_label,
                    mask_pos=mask_pos,
                    labels=labels,
                )
                total_loss.backward()
                optimizer.minimize(total_loss)
                bert.clear_gradients()

                acc = np.mean(np.array(next_sent_acc.numpy()))
                loss = np.mean(np.array(total_loss.numpy()))
                ppl = np.mean(np.exp(np.array(mask_lm_loss.numpy())))

                if step_idx % PRINT_STEP == 0:
                    if step_idx == 0:
                        print(
                            f"Step: {step_idx}, loss: {loss:f}, ppl: {ppl:f}, next_sent_acc: {acc:f}"
                        )
                        avg_batch_time = time.time()
                    else:
                        speed = PRINT_STEP / (time.time() - avg_batch_time)
                        speed_list.append(speed)
                        print(
                            f"Step: {step_idx}, loss: {loss:f}, ppl: {ppl:f}, next_sent_acc: {acc:f}, speed: {speed:.3f} steps/s"
                        )
                        avg_batch_time = time.time()

                step_idx += 1
                if step_idx == STEP_NUM:
                    if to_static:
                        paddle.jit.save(bert, self.model_save_prefix)
                    else:
                        paddle.save(
                            bert.state_dict(),
                            self.dy_state_dict_save_path + '.pdparams',
                        )
                    break
            return loss, ppl

    def train_dygraph(self, bert_config, data_reader):
        with enable_to_static_guard(False):
            return self.train(bert_config, data_reader, False)

    def train_static(self, bert_config, data_reader):
        return self.train(bert_config, data_reader, True)

    def predict_static(self, data):
        paddle.enable_static()
        exe = base.Executor(place)
        if use_pir_api():
            model_filename = self.pir_model_filename
        else:
            model_filename = self.model_filename

        # load inference model
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.io.load_inference_model(
            self.model_save_dir,
            executor=exe,
            model_filename=model_filename,
            params_filename=self.params_filename,
        )
        pred_res = exe.run(
            inference_program,
            feed=dict(zip(feed_target_names, data)),
            fetch_list=fetch_targets,
        )

        paddle.disable_static()
        return pred_res

    def predict_dygraph(self, bert_config, data):
        with enable_to_static_guard(False):
            with unique_name.guard():
                bert = PretrainModelLayer(
                    config=bert_config, weight_sharing=False, use_fp16=False
                )
                model_dict = paddle.load(
                    self.dy_state_dict_save_path + '.pdparams'
                )

                bert.set_dict(model_dict)
                bert.eval()

                input_vars = [paddle.to_tensor(x) for x in data]
                (
                    src_ids,
                    pos_ids,
                    sent_ids,
                    input_mask,
                    mask_label,
                    mask_pos,
                    labels,
                ) = input_vars
                pred_res = bert(
                    src_ids=src_ids,
                    position_ids=pos_ids,
                    sentence_ids=sent_ids,
                    input_mask=input_mask,
                    mask_label=mask_label,
                    mask_pos=mask_pos,
                    labels=labels,
                )
                pred_res = [var.numpy() for var in pred_res]

                return pred_res

    def predict_dygraph_jit(self, data):
        bert = paddle.jit.load(self.model_save_prefix)
        bert.eval()

        (
            src_ids,
            pos_ids,
            sent_ids,
            input_mask,
            mask_label,
            mask_pos,
            labels,
        ) = data
        pred_res = bert(
            src_ids,
            pos_ids,
            sent_ids,
            input_mask,
            mask_label,
            mask_pos,
            labels,
        )
        pred_res = [var.numpy() for var in pred_res]

        return pred_res

    def predict_analysis_inference(self, data):
        if use_pir_api():
            model_filename = self.pir_model_filename
        else:
            model_filename = self.model_filename

        output = PredictorTools(
            self.model_save_dir, model_filename, self.params_filename, data
        )
        out = output()
        return out

    def test_train(self):
        static_loss, static_ppl = self.train_static(
            self.bert_config, self.data_reader
        )
        dygraph_loss, dygraph_ppl = self.train_dygraph(
            self.bert_config, self.data_reader
        )
        np.testing.assert_allclose(static_loss, dygraph_loss, rtol=1e-05)
        np.testing.assert_allclose(static_ppl, dygraph_ppl, rtol=1e-05)

        self.verify_predict()

    @test_sot_only
    def test_train_composite(self):
        core._set_prim_backward_enabled(True)
        # core._add_skip_comp_ops("layer_norm")
        static_loss, static_ppl = self.train_static(
            self.bert_config, self.data_reader
        )
        core._set_prim_backward_enabled(False)
        # core._add_skip_comp_ops("layer_norm")
        dygraph_loss, dygraph_ppl = self.train_dygraph(
            self.bert_config, self.data_reader
        )
        np.testing.assert_allclose(static_loss, dygraph_loss, rtol=1e-05)
        np.testing.assert_allclose(static_ppl, dygraph_ppl, rtol=1e-05)

    def verify_predict(self):
        for data in self.data_reader.data_generator()():
            dygraph_pred_res = self.predict_dygraph(self.bert_config, data)
            static_pred_res = self.predict_static(data)
            dygraph_jit_pred_res = self.predict_dygraph_jit(data)
            predictor_pred_res = self.predict_analysis_inference(data)

            for dy_res, st_res, dy_jit_res, predictor_res in zip(
                dygraph_pred_res,
                static_pred_res,
                dygraph_jit_pred_res,
                predictor_pred_res,
            ):
                np.testing.assert_allclose(
                    st_res,
                    dy_res,
                    rtol=1e-05,
                    err_msg=f'dygraph_res: {dy_res[~np.isclose(st_res, dy_res)]},\n static_res: {st_res[~np.isclose(st_res, dy_res)]}',
                )
                np.testing.assert_allclose(
                    st_res,
                    dy_jit_res,
                    rtol=1e-05,
                    err_msg=f'dygraph_jit_res: {dy_jit_res[~np.isclose(st_res, dy_jit_res)]},\n static_res: {st_res[~np.isclose(st_res, dy_jit_res)]}',
                )
                np.testing.assert_allclose(
                    st_res,
                    predictor_res,
                    rtol=1e-05,
                    err_msg=f'dygraph_jit_res_predictor: {predictor_res[~np.isclose(st_res, predictor_res)]},\n static_res: {st_res[~np.isclose(st_res, predictor_res)]}',
                )
            break


if __name__ == '__main__':
    unittest.main()
