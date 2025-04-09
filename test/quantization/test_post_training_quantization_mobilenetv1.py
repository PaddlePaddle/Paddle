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
import logging
import os
import random
import sys
import tempfile
import time
import unittest

import numpy as np
from PIL import Image

import paddle
from paddle.dataset.common import download
from paddle.io import Dataset
from paddle.static.log_helper import get_logger
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

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center is True:
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


class ImageNetDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, shuffle=False, need_label=False):
        super().__init__()
        self.need_label = need_label
        self.data_dir = data_dir
        val_file_list = os.path.join(data_dir, 'val_list.txt')
        with open(val_file_list) as flist:
            lines = [line.strip() for line in flist]
            if shuffle:
                np.random.shuffle(lines)
            self.data = [line.split() for line in lines]

    def __getitem__(self, index):
        sample = self.data[index]
        data_path = os.path.join(self.data_dir, sample[0])
        data, label = process_image(
            [data_path, sample[1]], mode='val', color_jitter=False, rotate=False
        )
        if self.need_label:
            return data, np.array([label]).astype('int64')
        else:
            return data

    def __len__(self):
        return len(self.data)


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
        self.infer_iterations = (
            50000 if os.environ.get('DATASET') == 'full' else 2
        )

        self.root_path = tempfile.TemporaryDirectory()
        self.int8_model = os.path.join(
            self.root_path.name, "post_training_quantization"
        )

    def tearDown(self):
        self.root_path.cleanup()

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

        _logger.info(f'Data is downloaded at {zip_path}')
        self.cache_unzipping(data_cache_folder, zip_path)
        return data_cache_folder

    def download_model(self):
        pass

    def run_program(
        self,
        model_path,
        model_filename,
        params_filename,
        batch_size,
        infer_iterations,
    ):
        image_shape = [3, 224, 224]
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
            pred = exe.run(
                infer_program,
                feed={feed_dict[0]: image},
                fetch_list=fetch_targets,
            )
            t2 = time.time()
            period = t2 - t1
            periods.append(period)

            pred = np.array(pred[0])
            sort_array = pred.argsort(axis=1)
            top_1_pred = sort_array[:, -1:][:, ::-1]
            top_1 = np.mean(label == top_1_pred)

            test_info.append(np.mean(top_1) * len(data))
            cnt += len(data)

            if (batch_id + 1) % 100 == 0:
                _logger.info(f"{batch_id + 1} images,")
                sys.stdout.flush()
            if (batch_id + 1) == iterations:
                break

        throughput = cnt / np.sum(periods)
        latency = np.average(periods)
        acc1 = np.sum(test_info) / cnt
        return (throughput, latency, acc1, feed_dict)

    def generate_quantized_model(
        self,
        model_path,
        model_filename,
        params_filename,
        quantizable_op_type,
        batch_size,
        algo="KL",
        round_type="round",
        is_full_quantize=False,
        is_use_cache_file=False,
        is_optimize_model=False,
        batch_nums=1,
        onnx_format=False,
        deploy_backend=None,
        feed_name="inputs",
    ):
        try:
            os.system("mkdir " + self.int8_model)
        except Exception as e:
            _logger.info(f"Failed to create {self.int8_model} due to {e}")
            sys.exit(-1)

        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        image = paddle.static.data(
            name=feed_name[0], shape=[None, 3, 224, 224], dtype='float32'
        )
        feed_list = [image]
        if len(feed_name) == 2:
            label = paddle.static.data(
                name='label', shape=[None, 1], dtype='int64'
            )
            feed_list.append(label)

        val_dataset = ImageNetDataset(need_label=len(feed_list) == 2)
        data_loader = paddle.io.DataLoader(
            val_dataset,
            places=place,
            feed_list=feed_list,
            drop_last=False,
            return_list=False,
            batch_size=2,
            shuffle=False,
        )

        ptq = PostTrainingQuantization(
            executor=exe,
            data_loader=data_loader,
            model_dir=model_path,
            model_filename=model_filename,
            params_filename=params_filename,
            batch_size=batch_size,
            batch_nums=batch_nums,
            algo=algo,
            quantizable_op_type=quantizable_op_type,
            round_type=round_type,
            is_full_quantize=is_full_quantize,
            optimize_model=is_optimize_model,
            onnx_format=onnx_format,
            is_use_cache_file=is_use_cache_file,
            deploy_backend=deploy_backend,
        )
        ptq.quantize()
        ptq.save_quantized_model(
            self.int8_model,
            model_filename=model_filename,
            params_filename=params_filename,
        )

    def run_test(
        self,
        model,
        model_filename,
        params_filename,
        algo,
        round_type,
        data_urls,
        data_md5s,
        data_name,
        quantizable_op_type,
        is_full_quantize,
        is_use_cache_file,
        is_optimize_model,
        diff_threshold,
        onnx_format=False,
        batch_nums=1,
        deploy_backend=None,
    ):
        infer_iterations = self.infer_iterations
        batch_size = self.batch_size

        model_cache_folder = self.download_data(data_urls, data_md5s, model)
        model_path = os.path.join(model_cache_folder, data_name)
        _logger.info(
            f"Start FP32 inference for {model} on {infer_iterations * batch_size} images ..."
        )
        (
            fp32_throughput,
            fp32_latency,
            fp32_acc1,
            feed_name,
        ) = self.run_program(
            model_path,
            model_filename,
            params_filename,
            batch_size,
            infer_iterations,
        )

        self.generate_quantized_model(
            model_path,
            model_filename,
            params_filename,
            quantizable_op_type,
            batch_size,
            algo,
            round_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            batch_nums,
            onnx_format,
            deploy_backend,
            feed_name,
        )

        _logger.info(
            f"Start INT8 inference for {model} on {infer_iterations * batch_size} images ..."
        )
        (int8_throughput, int8_latency, int8_acc1, _) = self.run_program(
            self.int8_model,
            model_filename,
            params_filename,
            batch_size,
            infer_iterations,
        )

        _logger.info(f"---Post training quantization of {algo} method---")
        _logger.info(
            f"FP32 {model}: batch_size {batch_size}, throughput {fp32_throughput} images/second, latency {fp32_latency} second, accuracy {fp32_acc1}."
        )
        _logger.info(
            f"INT8 {model}: batch_size {batch_size}, throughput {int8_throughput} images/second, latency {int8_latency} second, accuracy {int8_acc1}.\n"
        )
        sys.stdout.flush()

        delta_value = fp32_acc1 - int8_acc1
        self.assertLess(delta_value, diff_threshold)


class TestPostTrainingKLForMobilenetv1(TestPostTrainingQuantization):
    def test_post_training_kl_mobilenetv1(self):
        model = "MobileNet-V1"
        algo = "KL"
        round_type = "round"
        data_urls = [
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
        ]
        data_md5s = ['5ee2b1775b11dc233079236cdc216c2e']
        quantizable_op_type = [
            "conv2d",
            "depthwise_conv2d",
            "mul",
            "pool2d",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.025
        batch_nums = 2
        self.run_test(
            model,
            'inference.pdmodel',
            'inference.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            "MobileNetV1_infer",
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
        )


class TestPostTrainingavgForMobilenetv1(TestPostTrainingQuantization):
    def test_post_training_avg_mobilenetv1(self):
        model = "MobileNet-V1"
        algo = "avg"
        round_type = "round"
        data_urls = [
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
        ]
        data_md5s = ['5ee2b1775b11dc233079236cdc216c2e']
        quantizable_op_type = [
            "conv2d",
            "depthwise_conv2d",
            "mul",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.025
        self.run_test(
            model,
            'inference.pdmodel',
            'inference.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            "MobileNetV1_infer",
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_nums=2,
        )


class TestPostTrainingavgForPwganCsmsc(TestPostTrainingQuantization):
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
                'https://paddlespeech.bj.bcebos.com/tmp/csmsc_voc1.npy'
            )
            data_md5s.append('47950146167ca8d885a78d71e74f1a2b')
            self.data_cache_folder = self.download_data(
                data_urls, data_md5s, "full_data", False
            )
        else:
            data_urls.append(
                'https://paddlespeech.bj.bcebos.com/tmp/csmsc_voc1.npy'
            )
            data_md5s.append('47950146167ca8d885a78d71e74f1a2b')
            self.data_cache_folder = self.download_data(
                data_urls, data_md5s, "small_data", False
            )

        # reader/decorator.py requires the relative path to the data folder
        if not os.path.exists("./data/BZNSYP"):
            cmd = 'rm -rf {0} && ln -s {1} {0}'.format(
                "data", self.data_cache_folder
            )
            os.system(cmd)

        self.batch_size = 1 if os.environ.get('DATASET') == 'full' else 50
        self.infer_iterations = (
            50000 if os.environ.get('DATASET') == 'full' else 2
        )

        self.root_path = tempfile.TemporaryDirectory()
        self.int8_model = os.path.join(
            self.root_path.name, "post_training_quantization"
        )

    def cache_unzipping(self, target_folder, zip_path):
        if not os.path.exists(target_folder):
            if zip_path.endswith('.tar.gz'):
                cmd = f'mkdir {target_folder} && tar xf {zip_path} -C {target_folder}'
            elif zip_path.endswith('.zip'):
                cmd = f'mkdir {target_folder} && unzip -o {zip_path} -d {target_folder}'
            else:
                cmd = f'mkdir {target_folder}'
            os.system(cmd)

    def generate_quantized_model(
        self,
        model_path,
        model_filename,
        params_filename,
        quantizable_op_type,
        batch_size,
        algo="avg",
        round_type="round",
        is_full_quantize=False,
        is_use_cache_file=False,
        is_optimize_model=False,
        batch_nums=1,
        onnx_format=False,
        deploy_backend=None,
        feed_name="inputs",
    ):
        try:
            os.system("mkdir " + self.int8_model)
        except Exception as e:
            _logger.info(f"Failed to create {self.int8_model} due to {e}")
            sys.exit(-1)

        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        val_dataset = "~/.cache/paddle/dataset/int8/download/csmsc_voc1.npy"
        data_loader = paddle.io.DataLoader(
            val_dataset,
            places=place,
            drop_last=False,
            batch_size=2,
        )
        ptq = PostTrainingQuantization(
            executor=exe,
            data_loader=data_loader,
            model_dir=model_path,
            model_filename=model_filename,
            params_filename=params_filename,
            batch_size=batch_size,
            batch_nums=batch_nums,
            algo=algo,
            quantizable_op_type=quantizable_op_type,
            round_type=round_type,
            is_full_quantize=is_full_quantize,
            optimize_model=is_optimize_model,
            onnx_format=onnx_format,
            is_use_cache_file=is_use_cache_file,
            deploy_backend=deploy_backend,
        )
        ptq.quantize()
        ptq.save_quantized_model(
            self.int8_model,
            model_filename=model_filename,
            params_filename=params_filename,
        )


class TestPostTraininghistForNoneShape(TestPostTrainingavgForPwganCsmsc):
    def test_post_training_avg_pwgancsmsc(self):
        model = "pwg_baker_static_0.4"
        algo = "avg"
        round_type = "round"
        data_urls = [
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_static_0.4.zip'
        ]
        data_md5s = ['e3504aed9c5a290be12d1347836d2742']
        quantizable_op_type = [
            "conv2d",
            "depthwise_conv2d",
            "mul",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.05
        self.run_test(
            model,
            'pwgan_csmsc.pdmodel',
            'pwgan_csmsc.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            "pwg_baker_static_0.4",
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_nums=2,
        )


class TestPostTraininghistForMobilenetv1(TestPostTrainingQuantization):
    def test_post_training_hist_mobilenetv1(self):
        model = "MobileNet-V1"
        algo = "hist"
        round_type = "round"
        data_urls = [
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
        ]
        data_md5s = ['5ee2b1775b11dc233079236cdc216c2e']
        quantizable_op_type = [
            "conv2d",
            "depthwise_conv2d",
            "mul",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.03
        batch_nums = 1
        self.run_test(
            model,
            'inference.pdmodel',
            'inference.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            "MobileNetV1_infer",
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_nums=batch_nums,
        )


class TestPostTrainingAbsMaxForMobilenetv1(TestPostTrainingQuantization):
    def test_post_training_abs_max_mobilenetv1(self):
        model = "MobileNet-V1"
        algo = "abs_max"
        round_type = "round"
        data_urls = [
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
        ]
        data_md5s = ['5ee2b1775b11dc233079236cdc216c2e']
        quantizable_op_type = [
            "conv2d",
            "mul",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        # The accuracy diff of post-training quantization (abs_max) maybe bigger
        diff_threshold = 0.05
        self.run_test(
            model,
            'inference.pdmodel',
            'inference.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            "MobileNetV1_infer",
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
        )


class TestPostTrainingAvgONNXFormatForMobilenetv1(TestPostTrainingQuantization):
    def test_post_training_onnx_format_mobilenetv1(self):
        model = "MobileNet-V1"
        algo = "emd"
        round_type = "round"
        data_urls = [
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
        ]
        data_md5s = ['5ee2b1775b11dc233079236cdc216c2e']
        quantizable_op_type = [
            "conv2d",
            "depthwise_conv2d",
            "mul",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        onnx_format = True
        diff_threshold = 0.05
        batch_nums = 1
        self.run_test(
            model,
            'inference.pdmodel',
            'inference.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            "MobileNetV1_infer",
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            onnx_format=onnx_format,
            batch_nums=batch_nums,
        )


class TestPostTrainingAvgONNXFormatForMobilenetv1TensorRT(
    TestPostTrainingQuantization
):
    def test_post_training_onnx_format_mobilenetv1_tensorrt(self):
        model = "MobileNet-V1"
        algo = "KL"
        round_type = "round"
        data_urls = [
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
        ]
        data_md5s = ['5ee2b1775b11dc233079236cdc216c2e']
        quantizable_op_type = [
            "conv2d",
            "depthwise_conv2d",
            "mul",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        onnx_format = True
        diff_threshold = 0.05
        batch_nums = 12
        deploy_backend = "tensorrt"
        self.run_test(
            model,
            'inference.pdmodel',
            'inference.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            "MobileNetV1_infer",
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            onnx_format=onnx_format,
            batch_nums=batch_nums,
            deploy_backend=deploy_backend,
        )


class TestPostTrainingKLONNXFormatForMobilenetv1MKLDNN(
    TestPostTrainingQuantization
):
    def test_post_training_onnx_format_mobilenetv1_mkldnn(self):
        model = "MobileNet-V1"
        algo = "ptf"
        round_type = "round"
        data_urls = [
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
        ]
        data_md5s = ['5ee2b1775b11dc233079236cdc216c2e']
        quantizable_op_type = [
            "conv2d",
            "depthwise_conv2d",
            "mul",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = False
        onnx_format = True
        diff_threshold = 0.05
        batch_nums = 12
        deploy_backend = "mkldnn"
        self.run_test(
            model,
            'inference.pdmodel',
            'inference.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            "MobileNetV1_infer",
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            onnx_format=onnx_format,
            batch_nums=batch_nums,
            deploy_backend=deploy_backend,
        )


class TestPostTrainingAvgONNXFormatForMobilenetv1ARMCPU(
    TestPostTrainingQuantization
):
    def test_post_training_onnx_format_mobilenetv1_armcpu(self):
        model = "MobileNet-V1"
        algo = "avg"
        round_type = "round"
        data_urls = [
            'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
        ]
        data_md5s = ['5ee2b1775b11dc233079236cdc216c2e']
        quantizable_op_type = [
            "conv2d",
            "depthwise_conv2d",
            "mul",
        ]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        onnx_format = True
        diff_threshold = 0.05
        batch_nums = 1
        deploy_backend = "arm"
        self.run_test(
            model,
            'inference.pdmodel',
            'inference.pdiparams',
            algo,
            round_type,
            data_urls,
            data_md5s,
            "MobileNetV1_infer",
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            onnx_format=onnx_format,
            batch_nums=batch_nums,
            deploy_backend=deploy_backend,
        )


if __name__ == '__main__':
    unittest.main()
