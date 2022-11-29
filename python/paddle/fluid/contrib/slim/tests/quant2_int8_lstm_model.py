# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import numpy as np
import struct
import sys
import time
import unittest
from paddle import fluid
from paddle.fluid.core import AnalysisConfig, create_paddle_predictor
from save_quant_model import transform_and_save_int8_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fp32_model', type=str, default='', help='A path to a FP32 model.'
    )
    parser.add_argument(
        '--quant_model', type=str, default='', help='A path to a quant model.'
    )
    parser.add_argument('--infer_data', type=str, default='', help='Data file.')
    parser.add_argument(
        '--warmup_iter',
        type=int,
        default=1,
        help='Number of the first iterations to skip in performance statistics.',
    )
    parser.add_argument(
        '--acc_diff_threshold',
        type=float,
        default=0.01,
        help='Accepted accuracy difference threshold.',
    )
    parser.add_argument(
        '--num_threads', type=int, default=1, help='Number of threads.'
    )
    parser.add_argument(
        '--mkldnn_cache_capacity',
        type=int,
        default=0,
        help='Mkldnn cache capacity. The default value in Python API is 15, which can slow down int8 models. Default 0 means unlimited cache.',
    )

    test_args, args = parser.parse_known_args(namespace=unittest)
    return test_args, sys.argv[:1] + args


class TestLstmModelPTQ(unittest.TestCase):
    def get_warmup_tensor(self, data_path, place):
        data = []
        with open(data_path, 'rb') as in_f:
            while True:
                plen = in_f.read(4)
                if plen is None or len(plen) != 4:
                    break

                alllen = struct.unpack('i', plen)[0]
                label_len = alllen & 0xFFFF
                seq_len = (alllen >> 16) & 0xFFFF

                label = in_f.read(4 * label_len)
                label = np.frombuffer(label, dtype=np.int32).reshape(
                    [len(label) // 4]
                )
                feat = in_f.read(4 * seq_len * 8)
                feat = np.frombuffer(feat, dtype=np.float32).reshape(
                    [len(feat) // 4 // 8, 8]
                )
                lod_feat = [feat.shape[0]]
                minputs = fluid.create_lod_tensor(feat, [lod_feat], place)

                infer_data = fluid.core.PaddleTensor()
                infer_data.lod = minputs.lod()
                infer_data.data = fluid.core.PaddleBuf(np.array(minputs))
                infer_data.shape = minputs.shape()
                infer_data.dtype = fluid.core.PaddleDType.FLOAT32
                infer_label = fluid.core.PaddleTensor()
                infer_label.data = fluid.core.PaddleBuf(np.array(label))
                infer_label.shape = label.shape
                infer_label.dtype = fluid.core.PaddleDType.INT32
                data.append([infer_data, infer_label])
        warmup_data = data[:1]
        inputs = data[1:]
        return warmup_data, inputs

    def set_config(
        self,
        model_path,
        num_threads,
        mkldnn_cache_capacity,
        warmup_data=None,
        use_analysis=False,
        enable_ptq=False,
    ):
        config = AnalysisConfig(model_path)
        config.set_cpu_math_library_num_threads(num_threads)
        if use_analysis:
            config.disable_gpu()
            config.switch_use_feed_fetch_ops(True)
            config.switch_ir_optim(True)
            config.enable_mkldnn()
            config.disable_mkldnn_fc_passes()  # fc passes caused dnnl error
            config.set_mkldnn_cache_capacity(mkldnn_cache_capacity)
            if enable_ptq:
                # This pass to work properly, must be added before fc_fuse_pass
                config.pass_builder().insert_pass(5, "fc_lstm_fuse_pass")
                config.enable_quantizer()
                config.quantizer_config().set_quant_data(warmup_data)
                config.quantizer_config().set_quant_batch_size(1)
        return config

    def run_program(
        self,
        model_path,
        data_path,
        num_threads,
        mkldnn_cache_capacity,
        warmup_iter,
        use_analysis=False,
        enable_ptq=False,
    ):
        place = fluid.CPUPlace()
        warmup_data, inputs = self.get_warmup_tensor(data_path, place)
        warmup_data = [item[0] for item in warmup_data]
        config = self.set_config(
            model_path,
            num_threads,
            mkldnn_cache_capacity,
            warmup_data,
            use_analysis,
            enable_ptq,
        )

        predictor = create_paddle_predictor(config)
        data = [item[0] for item in inputs]
        label = np.array([item[1] for item in inputs])

        all_hz_num = 0
        ok_hz_num = 0
        all_ctc_num = 0
        ok_ctc_num = 0

        dataset_size = len(data)
        start = time.time()
        for i in range(dataset_size):
            if i == warmup_iter:
                start = time.time()
            hz_out, ctc_out = predictor.run([data[i]])
            np_hz_out = np.array(hz_out.data.float_data()).reshape(-1)
            np_ctc_out = np.array(ctc_out.data.int64_data()).reshape(-1)

            out_hz_label = np.argmax(np_hz_out)

            this_label = label[i]
            this_label_data = np.array(this_label.data.int32_data()).reshape(-1)
            if this_label.shape[0] == 1:
                all_hz_num += 1
                best = this_label_data[0]
                if out_hz_label == best:
                    ok_hz_num += 1

                if this_label_data[0] <= 6350:
                    all_ctc_num += 1
                    if (
                        np_ctc_out.shape[0] == 1
                        and np_ctc_out.all() == this_label_data.all()
                    ):
                        ok_ctc_num += 1
            else:
                all_ctc_num += 1
                if (
                    np_ctc_out.shape[0] == this_label.shape[0]
                    and np_ctc_out.all() == this_label_data.all()
                ):
                    ok_ctc_num += 1

            if all_ctc_num > 1000 or all_hz_num > 1000:
                break

        end = time.time()
        fps = (dataset_size - warmup_iter) / (end - start)
        hx_acc = ok_hz_num / all_hz_num
        ctc_acc = ok_ctc_num / all_ctc_num
        return hx_acc, ctc_acc, fps

    def test_lstm_model(self):
        if not fluid.core.is_compiled_with_mkldnn():
            return

        fp32_model = test_case_args.fp32_model
        assert (
            fp32_model
        ), 'The FP32 model path cannot be empty. Please, use the --fp32_model option.'
        quant_model = test_case_args.quant_model
        assert (
            quant_model
        ), 'The quant model path cannot be empty. Please, use the --quant_model option.'
        infer_data = test_case_args.infer_data
        assert (
            infer_data
        ), 'The dataset path cannot be empty. Please, use the --infer_data option.'
        num_threads = test_case_args.num_threads
        mkldnn_cache_capacity = test_case_args.mkldnn_cache_capacity
        warmup_iter = test_case_args.warmup_iter
        acc_diff_threshold = test_case_args.acc_diff_threshold

        (fp32_hx_acc, fp32_ctc_acc, fp32_fps) = self.run_program(
            fp32_model,
            infer_data,
            num_threads,
            mkldnn_cache_capacity,
            warmup_iter,
            False,
            False,
        )

        (int8_hx_acc, int8_ctc_acc, int8_fps) = self.run_program(
            fp32_model,
            infer_data,
            num_threads,
            mkldnn_cache_capacity,
            warmup_iter,
            True,
            True,
        )

        quant_model_save_path = quant_model + "_int8"
        # transform model to quant2
        transform_and_save_int8_model(
            quant_model, quant_model_save_path, "fusion_lstm,concat"
        )

        (quant_hx_acc, quant_ctc_acc, quant_fps) = self.run_program(
            quant_model_save_path,
            infer_data,
            num_threads,
            mkldnn_cache_capacity,
            warmup_iter,
            True,
            False,
        )

        print(
            "FP32: fps {0}, hx_acc {1}, ctc_acc {2}".format(
                fp32_fps, fp32_hx_acc, fp32_ctc_acc
            )
        )

        print(
            "PTQ_INT8: fps {0}, hx_acc {1}, ctc_acc {2}".format(
                int8_fps, int8_hx_acc, int8_ctc_acc
            )
        )

        print(
            "QUANT2_INT8: fps {0}, hx_acc {1}, ctc_acc {2}".format(
                quant_fps, quant_hx_acc, quant_ctc_acc
            )
        )

        sys.stdout.flush()

        self.assertLess(fp32_hx_acc - int8_hx_acc, acc_diff_threshold)
        self.assertLess(fp32_ctc_acc - int8_ctc_acc, acc_diff_threshold)
        self.assertLess(fp32_hx_acc - quant_hx_acc, acc_diff_threshold)
        self.assertLess(fp32_ctc_acc - quant_ctc_acc, acc_diff_threshold)


if __name__ == "__main__":
    global test_case_args
    test_case_args, remaining_args = parse_args()
    unittest.main(argv=remaining_args)
