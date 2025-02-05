#   copyright (c) 2022 paddlepaddle authors. all rights reserved.
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
import functools
import os
import random
import sys
import time
import unittest

import numpy as np
from PIL import Image

import paddle
from paddle import base
from paddle.dataset.common import download
from paddle.static.quantization import PostTrainingQuantization

paddle.enable_static()

random.seed(0)
np.random.seed(0)

DATA_DIM = 224
THREAD = 1
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
    if center:
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


def _reader_creator(
    file_list,
    mode,
    shuffle=False,
    color_jitter=False,
    rotate=False,
    data_dir=DATA_DIR,
):
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

    mapper = functools.partial(
        process_image, mode=mode, color_jitter=color_jitter, rotate=rotate
    )

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def val(data_dir=DATA_DIR):
    file_list = os.path.join(data_dir, 'val_list.txt')
    return _reader_creator(file_list, 'val', shuffle=False, data_dir=data_dir)


class TestPostTrainingQuantization(unittest.TestCase):
    def setUp(self):
        self.int8_download = 'int8/download'
        self.cache_folder = os.path.expanduser(
            '~/.cache/paddle/dataset/' + self.int8_download
        )
        self.data_cache_folder = ''
        data_urls = []
        data_md5s = []
        if os.environ.get('DATASET') == 'full':
            data_urls.append(
                'https://paddle-inference-dist.bj.bcebos.com/int8/ILSVRC2012_img_val.tar.gz.partaa'
            )
            data_md5s.append('60f6525b0e1d127f345641d75d41f0a8')
            data_urls.append(
                'https://paddle-inference-dist.bj.bcebos.com/int8/ILSVRC2012_img_val.tar.gz.partab'
            )
            data_md5s.append('1e9f15f64e015e58d6f9ec3210ed18b5')
            self.data_cache_folder = self.download_data(
                data_urls, data_md5s, "full_data", False
            )
        else:
            data_urls.append(
                'http://paddle-inference-dist.bj.bcebos.com/int8/calibration_test_data.tar.gz'
            )
            data_md5s.append('1b6c1c434172cca1bf9ba1e4d7a3157d')
            self.data_cache_folder = self.download_data(
                data_urls, data_md5s, "small_data", False
            )

        # reader/decorator.py requires the relative path to the data folder
        if not os.path.exists("./data/ILSVRC2012"):
            cmd = 'rm -rf {0} && ln -s {1} {0}'.format(
                "data", self.data_cache_folder
            )
            os.system(cmd)

        self.batch_size = 1 if os.environ.get('DATASET') == 'full' else 50
        self.sample_iterations = (
            50 if os.environ.get('DATASET') == 'full' else 2
        )
        self.infer_iterations = (
            50000 if os.environ.get('DATASET') == 'full' else 2
        )

        self.int8_model = "post_training_quantization"
        print("self.int8_model: ", self.int8_model)

    def tearDown(self):
        cmd = 'rm -rf post_training_quantization'
        os.system(cmd)

    def cache_unzipping(self, target_folder, zip_path):
        if not os.path.exists(target_folder):
            cmd = (
                f'mkdir {target_folder} && tar xf {zip_path} -C {target_folder}'
            )
            os.system(cmd)

    def download_data(self, data_urls, data_md5s, folder_name, is_model=True):
        data_cache_folder = os.path.join(self.cache_folder, folder_name)
        zip_path = ''
        if os.environ.get('DATASET') == 'full':
            file_names = []
            for i in range(0, len(data_urls)):
                download(data_urls[i], self.int8_download, data_md5s[i])
                file_names.append(data_urls[i].split('/')[-1])

            zip_path = os.path.join(
                self.cache_folder, 'full_imagenet_val.tar.gz'
            )
            if not os.path.exists(zip_path):
                cat_command = 'cat'
                for file_name in file_names:
                    cat_command += ' ' + os.path.join(
                        self.cache_folder, file_name
                    )
                cat_command += ' > ' + zip_path
                os.system(cat_command)

        if os.environ.get('DATASET') != 'full' or is_model:
            download(data_urls[0], self.int8_download, data_md5s[0])
            file_name = data_urls[0].split('/')[-1]
            zip_path = os.path.join(self.cache_folder, file_name)

        print(f'Data is downloaded at {zip_path}')
        self.cache_unzipping(data_cache_folder, zip_path)
        return data_cache_folder

    def download_model(self):
        pass

    def run_program(
        self, model_path, batch_size, infer_iterations, is_quantized_model=False
    ):
        image_shape = [3, 224, 224]
        config = paddle.inference.Config(model_path)
        config.disable_gpu()
        config.enable_mkldnn()
        config.switch_ir_optim()
        config.set_cpu_math_library_num_threads(1)
        config.disable_glog_info()
        if is_quantized_model:
            config.enable_mkldnn_int8()
        predictor = paddle.inference.create_predictor(config)

        input_names = predictor.get_input_names()
        image_tensor = predictor.get_input_handle(input_names[0])
        label_tensor = predictor.get_input_handle(input_names[1])

        output_names = predictor.get_output_names()
        acc_tensor = predictor.get_output_handle("accuracy_0.tmp_0")

        val_reader = paddle.batch(val(), batch_size)
        iterations = infer_iterations

        test_info = []
        cnt = 0
        periods = []
        for batch_id, data in enumerate(val_reader()):
            image = np.array([x[0].reshape(image_shape) for x in data]).astype(
                "float32"
            )
            label = np.array([x[1] for x in data]).astype("int64")
            label = label.reshape([-1, 1])

            t1 = time.time()
            image_tensor.copy_from_cpu(image)
            label_tensor.copy_from_cpu(label)
            predictor.run()
            acc1 = acc_tensor.copy_to_cpu()

            t2 = time.time()
            period = t2 - t1
            periods.append(period)

            test_info.append(np.mean(acc1) * len(data))
            cnt += len(data)

            if (batch_id + 1) % 100 == 0:
                print(f"{batch_id + 1} images,")
                sys.stdout.flush()
            if (batch_id + 1) == iterations:
                break

        throughput = cnt / np.sum(periods)
        latency = np.average(periods)
        acc1 = np.sum(test_info) / cnt
        return (throughput, latency, acc1)

    def generate_quantized_model(
        self,
        model_path,
        quantizable_op_type,
        algo="KL",
        round_type="round",
        is_full_quantize=False,
        is_use_cache_file=False,
        is_optimize_model=False,
        onnx_format=False,
    ):
        place = base.CPUPlace()
        exe = base.Executor(place)
        val_reader = val()

        ptq = PostTrainingQuantization(
            executor=exe,
            sample_generator=val_reader,
            model_dir=model_path,
            algo=algo,
            quantizable_op_type=quantizable_op_type,
            round_type=round_type,
            is_full_quantize=is_full_quantize,
            optimize_model=is_optimize_model,
            onnx_format=onnx_format,
            is_use_cache_file=is_use_cache_file,
        )
        ptq.quantize()
        ptq.save_quantized_model(self.int8_model)

    def run_test(
        self,
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
        onnx_format=True,
    ):
        infer_iterations = self.infer_iterations
        batch_size = self.batch_size
        sample_iterations = self.sample_iterations

        model_cache_folder = self.download_data(data_urls, data_md5s, model)

        print(
            f"Start INT8 post training quantization for {model} on {sample_iterations * batch_size} images ..."
        )
        self.generate_quantized_model(
            os.path.join(model_cache_folder, "model"),
            quantizable_op_type,
            algo,
            round_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            onnx_format,
        )

        print(
            f"Start FP32 inference for {model} on {infer_iterations * batch_size} images ..."
        )
        (fp32_throughput, fp32_latency, fp32_acc1) = self.run_program(
            os.path.join(model_cache_folder, "model"),
            batch_size,
            infer_iterations,
        )

        print(
            f"Start INT8 inference for {model} on {infer_iterations * batch_size} images ..."
        )
        (int8_throughput, int8_latency, int8_acc1) = self.run_program(
            self.int8_model,
            batch_size,
            infer_iterations,
            is_quantized_model=True,
        )

        print(f"---Post training quantization of {algo} method---")
        print(
            f"FP32 {model}: batch_size {batch_size}, throughput {fp32_throughput} images/second, latency {fp32_latency} second, accuracy {fp32_acc1}."
        )
        print(
            f"INT8 {model}: batch_size {batch_size}, throughput {int8_throughput} images/second, latency {int8_latency} second, accuracy {int8_acc1}.\n"
        )
        sys.stdout.flush()

        delta_value = int8_latency - fp32_latency
        self.assertLess(delta_value, diff_threshold)


class TestMKLDNNInt8ForResnet50AvgONNXFormat(TestPostTrainingQuantization):
    def test_onnx_format_avg_resnet50(self):
        model = "resnet50"
        algo = "avg"
        round_type = "round"
        data_urls = [
            'http://paddle-inference-dist.bj.bcebos.com/int8/mobilenetv1_int8_model.tar.gz'
        ]
        data_md5s = ['13892b0716d26443a8cdea15b3c6438b']
        quantizable_op_type = [
            "conv2d",
            "depthwise_conv2d",
            "mul",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        diff_threshold = 0
        self.run_test(
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
            onnx_format=True,
        )


if __name__ == '__main__':
    unittest.main()
