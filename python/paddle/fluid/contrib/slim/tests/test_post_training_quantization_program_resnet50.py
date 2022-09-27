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
import sys
import time
import paddle
import random
import unittest
import functools
import contextlib
import numpy as np
import paddle.fluid as fluid
from PIL import Image, ImageEnhance
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantizationProgram
from test_post_training_quantization_mobilenetv1 import TestPostTrainingQuantization

paddle.enable_static()

random.seed(0)
np.random.seed(0)

THREAD = 1
DATA_DIM = 224
BUF_SIZE = 102400
DATA_DIR = 'data/ILSVRC2012'

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def process_image(sample, mode, color_jitter, rotate):
    img_path = sample[0]
    img = Image.open(img_path)
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=DATA_DIM, center=True)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std
    return img, sample[1]


def _reader_creator(file_list,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    data_dir=DATA_DIR):

    def reader():
        with open(file_list) as flist:
            full_lines = [line.strip() for line in flist]
            if shuffle:
                np.random.shuffle(full_lines)
            lines = full_lines

            for line in lines:
                img_path, label = line.split()
                img_path = os.path.join(data_dir, img_path)
                if not os.path.exists(img_path):
                    continue
                yield img_path, int(label)

    mapper = functools.partial(process_image,
                               mode=mode,
                               color_jitter=color_jitter,
                               rotate=rotate)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def val(data_dir=DATA_DIR):
    file_list = os.path.join(data_dir, 'val_list.txt')
    return _reader_creator(file_list, 'val', shuffle=False, data_dir=data_dir)


class TestPostTrainingQuantizationProgram(TestPostTrainingQuantization):

    def run_program(self, model_path, batch_size, infer_iterations):
        image_shape = [3, 224, 224]
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        [infer_program, feed_dict, fetch_targets] = \
            fluid.io.load_inference_model(model_path, exe)
        val_reader = paddle.batch(val(), batch_size)
        iterations = infer_iterations
        test_info = []
        cnt = 0
        periods = []
        for batch_id, data in enumerate(val_reader()):
            image = np.array([x[0].reshape(image_shape)
                              for x in data]).astype("float32")
            label = np.array([x[1] for x in data]).astype("int64")
            label = label.reshape([-1, 1])

            t1 = time.time()
            _, acc1, _ = exe.run(infer_program,
                                 feed={
                                     feed_dict[0]: image,
                                     feed_dict[1]: label
                                 },
                                 fetch_list=fetch_targets)
            t2 = time.time()
            period = t2 - t1
            periods.append(period)

            test_info.append(np.mean(acc1) * len(data))
            cnt += len(data)

            if (batch_id + 1) % 100 == 0:
                print("{0} images,".format(batch_id + 1))
                sys.stdout.flush()
            if (batch_id + 1) == iterations:
                break

        throughput = cnt / np.sum(periods)
        latency = np.average(periods)
        acc1 = np.sum(test_info) / cnt
        [infer_program, feed_dict, fetch_targets] = \
                    fluid.io.load_inference_model(model_path, exe)
        return (throughput, latency, acc1, infer_program, feed_dict,
                fetch_targets)

    def generate_quantized_model(
        self,
        program,
        quantizable_op_type,
        feed_list,
        fetch_list,
        algo="KL",
        round_type="round",
        is_full_quantize=False,
        is_use_cache_file=False,
        is_optimize_model=False,
        onnx_format=False,
    ):
        try:
            os.system("mkdir " + self.int8_model)
        except Exception as e:
            print("Failed to create {} due to {}".format(
                self.int8_model, str(e)))
            sys.exit(-1)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.global_scope()
        val_reader = val()
        same_scale_tensor_list = [[
            'batch_norm_3.tmp_2#/#1', 'batch_norm_4.tmp_2#*#1'
        ], ['batch_norm_27.tmp_2', 'batch_norm_26.tmp_2'],
                                  [
                                      'test_scale_name_not_in_scale_dict1',
                                      'test_scale_name_not_in_scale_dict2'
                                  ],
                                  [
                                      'test_scale_name_not_in_scale_dict1#/#1',
                                      'test_scale_name_not_in_scale_dict2#/#1'
                                  ]]
        ptq = PostTrainingQuantizationProgram(
            executor=exe,
            program=program,
            sample_generator=val_reader,
            batch_nums=10,
            algo=algo,
            quantizable_op_type=quantizable_op_type,
            round_type=round_type,
            is_full_quantize=is_full_quantize,
            optimize_model=is_optimize_model,
            onnx_format=onnx_format,
            is_use_cache_file=is_use_cache_file,
            feed_list=feed_list,
            fetch_list=fetch_list,
            same_scale_tensor_list=same_scale_tensor_list)
        ptq.quantize()
        ptq.save_quantized_model(self.int8_model)

    def run_test(self,
                 model,
                 algo,
                 round_type,
                 data_urls,
                 data_md5s,
                 quantizable_op_type,
                 is_full_quantize,
                 is_use_cache_file,
                 is_optimize_model,
                 diff_threshold,
                 onnx_format=False):
        infer_iterations = self.infer_iterations
        batch_size = self.batch_size
        sample_iterations = self.sample_iterations

        model_cache_folder = self.download_data(data_urls, data_md5s, model)

        print("Start FP32 inference for {0} on {1} images ...".format(
            model, infer_iterations * batch_size))
        (fp32_throughput, fp32_latency, fp32_acc1, infer_program, feed_dict,
         fetch_targets) = self.run_program(
             os.path.join(model_cache_folder, "model"), batch_size,
             infer_iterations)
        print("Start INT8 post training quantization for {0} on {1} images ...".
              format(model, sample_iterations * batch_size))
        self.generate_quantized_model(infer_program, quantizable_op_type,
                                      feed_dict, fetch_targets, algo,
                                      round_type, is_full_quantize,
                                      is_use_cache_file, is_optimize_model,
                                      onnx_format)

        print("Start INT8 inference for {0} on {1} images ...".format(
            model, infer_iterations * batch_size))
        (int8_throughput, int8_latency, int8_acc1, _, _,
         _) = self.run_program(self.int8_model, batch_size, infer_iterations)

        print("---Post training quantization of {} method---".format(algo))
        print(
            "FP32 {0}: batch_size {1}, throughput {2} images/second, latency {3} second, accuracy {4}."
            .format(model, batch_size, fp32_throughput, fp32_latency,
                    fp32_acc1))
        print(
            "INT8 {0}: batch_size {1}, throughput {2} images/second, latency {3} second, accuracy {4}.\n"
            .format(model, batch_size, int8_throughput, int8_latency,
                    int8_acc1))
        sys.stdout.flush()

        delta_value = fp32_acc1 - int8_acc1
        self.assertLess(delta_value, diff_threshold)


class TestPostTrainingProgramAbsMaxForResnet50(
        TestPostTrainingQuantizationProgram):

    def test_post_training_abs_max_resnet50(self):
        model = "ResNet-50"
        algo = "abs_max"
        round_type = "round"
        data_urls = [
            'http://paddle-inference-dist.bj.bcebos.com/int8/resnet50_int8_model.tar.gz'
        ]
        data_md5s = ['4a5194524823d9b76da6e738e1367881']
        quantizable_op_type = ["conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        diff_threshold = 0.025
        self.run_test(model, algo, round_type, data_urls, data_md5s,
                      quantizable_op_type, is_full_quantize, is_use_cache_file,
                      is_optimize_model, diff_threshold)


if __name__ == '__main__':
    unittest.main()
