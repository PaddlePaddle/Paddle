#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
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

import paddle
import unittest
import os
import tarfile
import struct
import numpy as np
import paddle.fluid as fluid
from mobilenet import MobileNet
from paddle.fluid.contrib.slim.core import Compressor
from paddle.fluid.contrib.slim.graph import GraphWrapper
from paddle.dataset.common import download


class TestInferQuantizeStrategy(unittest.TestCase):
    """
    Test API of Post Training quantization strategy for int8 with MKLDNN.
    """

    def setUp(self):
        self.cache_folder = os.path.expanduser(
            '~/.cache/paddle/dataset/int8/download')
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        self.mobilenetv1 = 'mobilenetv1_int8_model.tar.gz'
        self.small_data = 'imagenet_val_100_tail.tar.gz'
        self.url_base = 'http://paddle-inference-dist.bj.bcebos.com/int8/'
        urls = [
            self.url_base + 'imagenet_val_100_tail.tar.gz',
            self.url_base + 'mobilenetv1_int8_model.tar.gz'
        ]
        md5s = [
            'b6e05365252f12f75e7daf3fcbc62e96',
            '13892b0716d26443a8cdea15b3c6438b'
        ]
        for url, md5 in zip(urls, md5s):
            download(url, self.cache_folder, md5)
            file_name = url.split('/')[-1]
            tar_file = tarfile.open(os.path.join(self.cache_folder, file_name))
            target_dir = file_name.split('.')[0]
            target_dir = os.path.join(self.cache_folder, target_dir)
            tar_file.extractall(target_dir)

    def _reader_creator(self, data_file='data.bin', cycle=False):
        def reader():
            with open(data_file, 'rb') as fp:
                num = fp.read(8)
                num = struct.unpack('q', num)[0]
                imgs_offset = 8
                img_ch = 3
                img_w = 224
                img_h = 224
                img_pixel_size = 4
                img_size = img_ch * img_h * img_w * img_pixel_size
                label_size = 8
                labels_offset = imgs_offset + num * img_size
                step = 0

                while step < num:
                    fp.seek(imgs_offset + img_size * step)
                    img = fp.read(img_size)
                    img = struct.unpack_from('{}f'.format(img_ch * img_w *
                                                          img_h), img)
                    img = np.array(img)
                    img.shape = (img_ch, img_w, img_h)
                    fp.seek(labels_offset + label_size * step)
                    label = fp.read(label_size)
                    label = struct.unpack('q', label)[0]
                    yield img, int(label)
                    step += 1
                    if cycle and step == num:
                        step = 0

        return reader

    def test_infer_quant(self):
        config_path = './quantization/config_mkldnn_int8.yaml'
        with open(config_path, 'r+') as fp:
            data = fp.read()
        model_path = os.path.join(self.cache_folder,
                                  'mobilenetv1_int8_model/model')
        data = data.replace('MODEL_PATH', model_path)
        with open(config_path, 'w') as fp:
            fp.write(data)

        data_path = os.path.join(self.cache_folder,
                                 'imagenet_val_100_tail/data.bin')
        test_reader = paddle.batch(
            self._reader_creator(data_path, True), batch_size=1)
        com_pass = Compressor(
            None,
            None,
            None,
            train_reader=None,
            train_feed_list=[],
            train_fetch_list=[],
            eval_program=None,
            eval_reader=test_reader,
            eval_feed_list=[],
            eval_fetch_list=[],
            teacher_programs=[],
            checkpoint_path='./checkpoints',
            train_optimizer=None,
            distiller_optimizer=None)
        com_pass.config(config_path)
        eval_graph = com_pass.run()


if __name__ == '__main__':
    unittest.main()
