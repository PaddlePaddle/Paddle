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
import unittest
import os
import numpy as np
import time
import sys
import random
import paddle
import paddle.fluid as fluid
import functools
import contextlib
from paddle.dataset.common import download
from PIL import Image, ImageEnhance
import math
import paddle.fluid.contrib.int8_inference.utility as int8_utility

random.seed(0)
np.random.seed(0)

DATA_DIM = 224

THREAD = 1
BUF_SIZE = 102400

DATA_DIR = 'data/ILSVRC2012'

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


# TODO(guomingz): Remove duplicated code from resize_short, crop_image, process_image, _reader_creator
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

    mapper = functools.partial(
        process_image, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def val(data_dir=DATA_DIR):
    file_list = os.path.join(data_dir, 'val_list.txt')
    return _reader_creator(file_list, 'val', shuffle=False, data_dir=data_dir)


class TestCalibrationForResnet50(unittest.TestCase):
    def setUp(self):
        self.int8_download = 'int8/download'
        self.cache_folder = os.path.expanduser('~/.cache/paddle/dataset/' +
                                               self.int8_download)

        data_urls = []
        data_md5s = []
        self.data_cache_folder = ''
        if os.environ.get('DATASET') == 'full':
            data_urls.append(
                'https://paddle-inference-dist.bj.bcebos.com/int8/ILSVRC2012_img_val.tar.gz.partaa'
            )
            data_md5s.append('60f6525b0e1d127f345641d75d41f0a8')
            data_urls.append(
                'https://paddle-inference-dist.bj.bcebos.com/int8/ILSVRC2012_img_val.tar.gz.partab'
            )
            data_md5s.append('1e9f15f64e015e58d6f9ec3210ed18b5')
            self.data_cache_folder = self.download_data(data_urls, data_md5s,
                                                        "full_data", False)
        else:
            data_urls.append(
                'http://paddle-inference-dist.bj.bcebos.com/int8/calibration_test_data.tar.gz'
            )
            data_md5s.append('1b6c1c434172cca1bf9ba1e4d7a3157d')
            self.data_cache_folder = self.download_data(data_urls, data_md5s,
                                                        "small_data", False)

        # reader/decorator.py requires the relative path to the data folder
        cmd = 'rm -rf {0} && ln -s {1} {0}'.format("data",
                                                   self.data_cache_folder)
        os.system(cmd)

        self.batch_size = 1 if os.environ.get('DATASET') == 'full' else 50
        self.sample_iterations = 50 if os.environ.get(
            'DATASET') == 'full' else 1
        self.infer_iterations = 50000 if os.environ.get(
            'DATASET') == 'full' else 1

    def cache_unzipping(self, target_folder, zip_path):
        if not os.path.exists(target_folder):
            cmd = 'mkdir {0} && tar xf {1} -C {0}'.format(target_folder,
                                                          zip_path)
            os.system(cmd)

    def download_data(self, data_urls, data_md5s, folder_name, is_model=True):
        data_cache_folder = os.path.join(self.cache_folder, folder_name)
        zip_path = ''
        if os.environ.get('DATASET') == 'full':
            file_names = []
            for i in range(0, len(data_urls)):
                download(data_urls[i], self.int8_download, data_md5s[i])
                file_names.append(data_urls[i].split('/')[-1])

            zip_path = os.path.join(self.cache_folder,
                                    'full_imagenet_val.tar.gz')
            if not os.path.exists(zip_path):
                cat_command = 'cat'
                for file_name in file_names:
                    cat_command += ' ' + os.path.join(self.cache_folder,
                                                      file_name)
                cat_command += ' > ' + zip_path
                os.system(cat_command)

        if os.environ.get('DATASET') != 'full' or is_model:
            download(data_urls[0], self.int8_download, data_md5s[0])
            file_name = data_urls[0].split('/')[-1]
            zip_path = os.path.join(self.cache_folder, file_name)

        print('Data is downloaded at {0}').format(zip_path)
        self.cache_unzipping(data_cache_folder, zip_path)
        return data_cache_folder

    def download_model(self):
        # resnet50 fp32 data
        data_urls = [
            'http://paddle-inference-dist.bj.bcebos.com/int8/resnet50_int8_model.tar.gz'
        ]
        data_md5s = ['4a5194524823d9b76da6e738e1367881']
        self.model_cache_folder = self.download_data(data_urls, data_md5s,
                                                     "resnet50_fp32")
        self.model = "ResNet-50"
        self.algo = "direct"

    def run_program(self, model_path, generate_int8=False, algo='direct'):
        image_shape = [3, 224, 224]

        fluid.memory_optimize(fluid.default_main_program())

        exe = fluid.Executor(fluid.CPUPlace())

        [infer_program, feed_dict,
         fetch_targets] = fluid.io.load_inference_model(model_path, exe)

        t = fluid.transpiler.InferenceTranspiler()
        t.transpile(infer_program, fluid.CPUPlace())

        val_reader = paddle.batch(val(), self.batch_size)
        iterations = self.infer_iterations

        if generate_int8:
            int8_model = os.path.join(os.getcwd(), "calibration_out")
            iterations = self.sample_iterations

            if os.path.exists(int8_model):
                os.system("rm -rf " + int8_model)
                os.system("mkdir " + int8_model)

            calibrator = int8_utility.Calibrator(
                program=infer_program,
                pretrained_model=model_path,
                algo=algo,
                exe=exe,
                output=int8_model,
                feed_var_names=feed_dict,
                fetch_list=fetch_targets)

        test_info = []
        cnt = 0
        periods = []
        for batch_id, data in enumerate(val_reader()):
            image = np.array(
                [x[0].reshape(image_shape) for x in data]).astype("float32")
            label = np.array([x[1] for x in data]).astype("int64")
            label = label.reshape([-1, 1])
            running_program = calibrator.sampling_program.clone(
            ) if generate_int8 else infer_program.clone()

            t1 = time.time()
            _, acc1, _ = exe.run(
                running_program,
                feed={feed_dict[0]: image,
                      feed_dict[1]: label},
                fetch_list=fetch_targets)
            t2 = time.time()
            period = t2 - t1
            periods.append(period)

            if generate_int8:
                calibrator.sample_data()

            test_info.append(np.mean(acc1) * len(data))
            cnt += len(data)

            if (batch_id + 1) % 100 == 0:
                print("{0} images,".format(batch_id + 1))
                sys.stdout.flush()

            if (batch_id + 1) == iterations:
                break

        if generate_int8:
            calibrator.save_int8_model()

            print(
                "Calibration is done and the corresponding files are generated at {}".
                format(os.path.abspath("calibration_out")))
        else:
            throughput = cnt / np.sum(periods)
            latency = np.average(periods)
            acc1 = np.sum(test_info) / cnt
            return (throughput, latency, acc1)

    def test_calibration(self):
        self.download_model()
        print("Start FP32 inference for {0} on {1} images ...").format(
            self.model, self.infer_iterations * self.batch_size)
        (fp32_throughput, fp32_latency,
         fp32_acc1) = self.run_program(self.model_cache_folder + "/model")
        print("Start INT8 calibration for {0} on {1} images ...").format(
            self.model, self.sample_iterations * self.batch_size)
        self.run_program(
            self.model_cache_folder + "/model", True, algo=self.algo)
        print("Start INT8 inference for {0} on {1} images ...").format(
            self.model, self.infer_iterations * self.batch_size)
        (int8_throughput, int8_latency,
         int8_acc1) = self.run_program("calibration_out")
        delta_value = fp32_acc1 - int8_acc1
        self.assertLess(delta_value, 0.01)
        print(
            "FP32 {0}: batch_size {1}, throughput {2} images/second, latency {3} second, accuracy {4}".
            format(self.model, self.batch_size, fp32_throughput, fp32_latency,
                   fp32_acc1))
        print(
            "INT8 {0}: batch_size {1}, throughput {2} images/second, latency {3} second, accuracy {4}".
            format(self.model, self.batch_size, int8_throughput, int8_latency,
                   int8_acc1))
        sys.stdout.flush()


class TestCalibrationForMobilenetv1(TestCalibrationForResnet50):
    def download_model(self):
        # mobilenetv1 fp32 data
        data_urls = [
            'http://paddle-inference-dist.bj.bcebos.com/int8/mobilenetv1_int8_model.tar.gz'
        ]
        data_md5s = ['13892b0716d26443a8cdea15b3c6438b']
        self.model_cache_folder = self.download_data(data_urls, data_md5s,
                                                     "mobilenetv1_fp32")
        self.model = "MobileNet-V1"
        self.algo = "KL"


if __name__ == '__main__':
    unittest.main()
