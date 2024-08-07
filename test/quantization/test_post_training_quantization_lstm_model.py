#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
import os
import random
import struct
import sys
import tempfile
import time
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.dataset.common import download
from paddle.static.quantization import PostTrainingQuantization

paddle.enable_static()

random.seed(0)
np.random.seed(0)


class TestPostTrainingQuantization(unittest.TestCase):
    def setUp(self):
        self.download_path = 'int8/download'
        self.cache_folder = os.path.expanduser(
            '~/.cache/paddle/dataset/' + self.download_path
        )
        self.root_path = tempfile.TemporaryDirectory()
        self.int8_model_path = os.path.join(
            self.root_path.name, "post_training_quantization"
        )
        try:
            os.system("mkdir -p " + self.int8_model_path)
        except Exception as e:
            print(f"Failed to create {self.int8_model_path} due to {e}")
            sys.exit(-1)

    def tearDown(self):
        self.root_path.cleanup()

    def cache_unzipping(self, target_folder, zip_path):
        if not os.path.exists(target_folder):
            cmd = (
                f'mkdir {target_folder} && tar xf {zip_path} -C {target_folder}'
            )
            os.system(cmd)

    def download_model(self, data_url, data_md5, folder_name):
        download(data_url, self.download_path, data_md5)
        file_name = data_url.split('/')[-1]
        zip_path = os.path.join(self.cache_folder, file_name)
        print(f'Data is downloaded at {zip_path}')

        data_cache_folder = os.path.join(self.cache_folder, folder_name)
        self.cache_unzipping(data_cache_folder, zip_path)
        return data_cache_folder

    def get_batch_reader(self, data_path, place):
        def reader():
            with open(data_path, 'rb') as in_file:
                while True:
                    plen = in_file.read(4)
                    if plen is None or len(plen) != 4:
                        break

                    all_len = struct.unpack('i', plen)[0]
                    label_len = all_len & 0xFFFF
                    seq_len = (all_len >> 16) & 0xFFFF

                    label = in_file.read(4 * label_len)
                    label = np.frombuffer(label, dtype=np.int32).reshape(
                        [len(label) // 4]
                    )
                    if label.shape[0] != 1 or label[0] > 6350:
                        continue

                    feat = in_file.read(4 * seq_len * 8)
                    feat = np.frombuffer(feat, dtype=np.float32).reshape(
                        [len(feat) // 4 // 8, 8]
                    )
                    lod_feat = [feat.shape[0]]

                    minputs = base.create_lod_tensor(feat, [lod_feat], place)
                    yield [minputs]

        return reader

    def get_simple_reader(self, data_path, place):
        def reader():
            with open(data_path, 'rb') as in_file:
                while True:
                    plen = in_file.read(4)
                    if plen is None or len(plen) != 4:
                        break

                    all_len = struct.unpack('i', plen)[0]
                    label_len = all_len & 0xFFFF
                    seq_len = (all_len >> 16) & 0xFFFF

                    label = in_file.read(4 * label_len)
                    label = np.frombuffer(label, dtype=np.int32).reshape(
                        [len(label) // 4]
                    )
                    if label.shape[0] != 1 or label[0] > 6350:
                        continue

                    feat = in_file.read(4 * seq_len * 8)
                    feat = np.frombuffer(feat, dtype=np.float32).reshape(
                        [len(feat) // 4 // 8, 8]
                    )
                    lod_feat = [feat.shape[0]]

                    minputs = base.create_lod_tensor(feat, [lod_feat], place)
                    yield minputs, label

        return reader

    def run_program(
        self,
        model_path,
        model_filename,
        params_filename,
        data_path,
        infer_iterations,
    ):
        print("test model path:" + model_path)
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        [
            infer_program,
            feed_dict,
            fetch_targets,
        ] = paddle.static.load_inference_model(
            model_path,
            exe,
            model_filename=model_filename,
            params_filename=params_filename,
        )

        val_reader = self.get_simple_reader(data_path, place)

        all_num = 0
        right_num = 0
        periods = []
        for batch_id, (data, label) in enumerate(val_reader()):
            t1 = time.time()
            cls_out, ctc_out = exe.run(
                infer_program,
                feed={feed_dict[0]: data},
                fetch_list=fetch_targets,
                return_numpy=False,
            )
            t2 = time.time()
            periods.append(t2 - t1)

            cls_out = np.array(cls_out).reshape(-1)
            out_cls_label = np.argmax(cls_out)

            all_num += 1
            if out_cls_label == label[0]:
                right_num += 1

            if (batch_id + 1) == infer_iterations:
                break

        latency = np.average(periods)
        acc = right_num / all_num
        return (latency, acc)

    def generate_quantized_model(
        self,
        model_path,
        model_filename,
        params_filename,
        data_path,
        algo="KL",
        round_type="round",
        quantizable_op_type=["conv2d"],
        is_full_quantize=False,
        is_use_cache_file=False,
        is_optimize_model=False,
        batch_size=10,
        batch_nums=10,
        onnx_format=False,
    ):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        scope = paddle.static.global_scope()
        batch_generator = self.get_batch_reader(data_path, place)

        ptq = PostTrainingQuantization(
            executor=exe,
            model_dir=model_path,
            model_filename=model_filename,
            params_filename=params_filename,
            batch_generator=batch_generator,
            batch_nums=batch_nums,
            algo=algo,
            quantizable_op_type=quantizable_op_type,
            round_type=round_type,
            is_full_quantize=is_full_quantize,
            optimize_model=is_optimize_model,
            onnx_format=onnx_format,
            is_use_cache_file=is_use_cache_file,
        )
        ptq.quantize()
        if onnx_format:
            ptq._clip_extra = False
        ptq.save_quantized_model(self.int8_model_path)

    def run_test(
        self,
        model_name,
        model_filename,
        params_filename,
        model_url,
        model_md5,
        data_name,
        data_url,
        data_md5,
        algo,
        round_type,
        quantizable_op_type,
        is_full_quantize,
        is_use_cache_file,
        is_optimize_model,
        diff_threshold,
        infer_iterations,
        quant_iterations,
        onnx_format=False,
    ):
        fp32_model_path = self.download_model(model_url, model_md5, model_name)
        fp32_model_path = os.path.join(fp32_model_path, model_name)

        data_path = self.download_model(data_url, data_md5, data_name)
        data_path = os.path.join(data_path, data_name)

        print(
            f"Start FP32 inference for {model_name} on {infer_iterations} samples ..."
        )
        (fp32_latency, fp32_acc) = self.run_program(
            fp32_model_path,
            model_filename,
            params_filename,
            data_path,
            infer_iterations,
        )

        print(
            f"Start post training quantization for {model_name} on {quant_iterations} samples ..."
        )
        self.generate_quantized_model(
            fp32_model_path,
            model_filename,
            params_filename,
            data_path,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            10,
            quant_iterations,
            onnx_format,
        )

        print(
            f"Start INT8 inference for {model_name} on {infer_iterations} samples ..."
        )
        (int8_latency, int8_acc) = self.run_program(
            self.int8_model_path,
            'model.pdmodel',
            'model.pdiparams',
            data_path,
            infer_iterations,
        )

        print(f"---Post training quantization of {algo} method---")
        print(
            f"FP32 {model_name}: batch_size {1}, latency {fp32_latency} s, acc {fp32_acc}."
        )
        print(
            f"INT8 {model_name}: batch_size {1}, latency {int8_latency} s, acc1 {int8_acc}.\n"
        )
        sys.stdout.flush()

        delta_value = fp32_acc - int8_acc
        self.assertLess(delta_value, diff_threshold)


class TestPostTrainingAvgForLSTM(TestPostTrainingQuantization):
    def test_post_training_avg(self):
        model_name = "nlp_lstm_fp32_model"
        model_url = "https://paddle-inference-dist.cdn.bcebos.com/int8/unittest_model_data/nlp_lstm_fp32_model_combined.tar.gz"
        model_md5 = "5b47cd7ba2afcf24120d9727ed3f05a7"
        data_name = "quant_lstm_input_data"
        data_url = "https://paddle-inference-dist.cdn.bcebos.com/int8/unittest_model_data/quant_lstm_input_data.tar.gz"
        data_md5 = "add84c754e9b792fea1fbd728d134ab7"
        algo = "avg"
        round_type = "round"
        quantizable_op_type = ["mul", "lstm"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        diff_threshold = 0.02
        infer_iterations = 100
        quant_iterations = 10
        self.run_test(
            model_name,
            'model.pdmodel',
            'model.pdiparams',
            model_url,
            model_md5,
            data_name,
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            infer_iterations,
            quant_iterations,
        )


class TestPostTrainingAvgForLSTMONNXFormat(TestPostTrainingQuantization):
    def not_test_post_training_avg_onnx_format(self):
        model_name = "nlp_lstm_fp32_model"
        model_url = "https://paddle-inference-dist.cdn.bcebos.com/int8/unittest_model_data/nlp_lstm_fp32_model_combined.tar.gz"
        model_md5 = "5b47cd7ba2afcf24120d9727ed3f05a7"
        data_name = "quant_lstm_input_data"
        data_url = "https://paddle-inference-dist.cdn.bcebos.com/int8/unittest_model_data/quant_lstm_input_data.tar.gz"
        data_md5 = "add84c754e9b792fea1fbd728d134ab7"
        algo = "avg"
        round_type = "round"
        quantizable_op_type = ["mul", "lstm"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        diff_threshold = 0.02
        infer_iterations = 100
        quant_iterations = 10
        onnx_format = True
        self.run_test(
            model_name,
            'model.pdmodel',
            'model.pdiparams',
            model_url,
            model_md5,
            data_name,
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            infer_iterations,
            quant_iterations,
            onnx_format=onnx_format,
        )


if __name__ == '__main__':
    unittest.main()
