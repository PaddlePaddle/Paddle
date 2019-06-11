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

import unittest
import os
import sys
import argparse
import shutil
import logging
import struct
import six
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.slim.core import Compressor
from paddle.fluid.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument(
        '--infer_model',
        type=str,
        default='',
        help='infer_model is used to load an original fp32 ProgramDesc with fp32 weights'
    )
    parser.add_argument('--infer_data', type=str, default='', help='data file')
    parser.add_argument(
        '--int8_model_save_path',
        type=str,
        default='./output',
        help='infer_data is used to save an int8 ProgramDesc with fp32 weights')
    parser.add_argument(
        '--warmup_batch_size',
        type=int,
        default=100,
        help='batch size for quantization warmup')
    parser.add_argument(
        '--accuracy_diff_threshold',
        type=float,
        default=0.01,
        help='accepted accuracy drop threshold.')

    test_args, args = parser.parse_known_args(namespace=unittest)

    return test_args, sys.argv[:1] + args


class TestMKLDNNPostTrainingQuantStrategy(unittest.TestCase):
    """
    Test API of Post Training quantization strategy for int8 with MKL-DNN.
    """

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

    def _update_config_file(self, fp32_model_path, output_path):
        config_path = './quantization/config_mkldnn_int8.yaml'
        new_config_path = './quantization/temp.yaml'
        shutil.copy(config_path, new_config_path)

        with open(new_config_path, 'r+') as fp:
            data = fp.read()
        data = data.replace('MODEL_PATH', fp32_model_path)
        data = data.replace('OUTPUT_PATH', output_path)
        with open(new_config_path, 'w') as fp:
            fp.write(data)

        return new_config_path

    def _predict(self, test_reader=None, model_path=None):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        inference_scope = fluid.executor.global_scope()
        with fluid.scope_guard(inference_scope):
            if os.path.exists(os.path.join(model_path, '__model__')):
                [inference_program, feed_target_names,
                 fetch_targets] = fluid.io.load_inference_model(model_path, exe)
            else:
                [inference_program, feed_target_names,
                 fetch_targets] = fluid.io.load_inference_model(
                     model_path, exe, 'model', 'params')

            dshape = [3, 224, 224]
            top1 = 0.0
            top5 = 0.0
            total_samples = 0
            for batch_id, data in enumerate(test_reader()):
                if six.PY2:
                    images = map(lambda x: x[0].reshape(dshape), data)
                if six.PY3:
                    images = list(map(lambda x: x[0].reshape(dshape), data))
                images = np.array(images).astype('float32')
                labels = np.array([x[1] for x in data]).astype("int64")
                labels = labels.reshape([-1, 1])
                out = exe.run(inference_program,
                              feed={
                                  feed_target_names[0]: images,
                                  feed_target_names[1]: labels
                              },
                              fetch_list=fetch_targets)
                top1 += np.sum(out[1]) * len(data)
                top5 += np.sum(out[2]) * len(data)
                total_samples += len(data)
                if (batch_id + 1) % 100 == 0:
                    _logger.info('{} images have been predicted'.format(
                        total_samples))
            return top1 / total_samples, top5 / total_samples

    def _warmup(self, reader=None, config_path=''):
        com_pass = Compressor(
            place=None,
            scope=None,
            train_program=None,
            train_reader=None,
            train_feed_list=[],
            train_fetch_list=[],
            eval_program=None,
            eval_reader=reader,
            eval_feed_list=[],
            eval_fetch_list=[],
            teacher_programs=[],
            checkpoint_path='',
            train_optimizer=None,
            distiller_optimizer=None)
        com_pass.config(config_path)
        com_pass.run()

    def test_compression(self):
        if not fluid.core.is_compiled_with_mkldnn():
            return

        int8_model_path = test_case_args.int8_model_save_path
        data_path = test_case_args.infer_data
        fp32_model_path = test_case_args.infer_model
        batch_size = test_case_args.batch_size

        warmup_batch_size = test_case_args.warmup_batch_size
        accuracy_diff_threshold = test_case_args.accuracy_diff_threshold

        _logger.info(
            'FP32 & INT8 prediction run: batch_size {0}, warmup batch size {1}.'.
            format(batch_size, warmup_batch_size))

        #warmup dataset, only use the first batch data
        warmup_reader = paddle.batch(
            self._reader_creator(data_path, False),
            batch_size=warmup_batch_size)
        config_path = self._update_config_file(fp32_model_path, int8_model_path)
        self._warmup(warmup_reader, config_path)

        _logger.info('--- INT8 prediction start ---')
        val_reader = paddle.batch(
            self._reader_creator(data_path, False), batch_size=batch_size)
        int8_model_result = self._predict(val_reader, int8_model_path)
        _logger.info('--- FP32 prediction start ---')
        val_reader = paddle.batch(
            self._reader_creator(data_path, False), batch_size=batch_size)
        fp32_model_result = self._predict(val_reader, fp32_model_path)

        _logger.info('--- comparing outputs ---')
        _logger.info('Avg top1 INT8 accuracy: {0:.4f}'.format(int8_model_result[
            0]))
        _logger.info('Avg top1 FP32 accuracy: {0:.4f}'.format(fp32_model_result[
            0]))
        _logger.info('Accepted accuracy drop threshold: {0}'.format(
            accuracy_diff_threshold))
        assert fp32_model_result[0] - int8_model_result[
            0] <= accuracy_diff_threshold


if __name__ == '__main__':
    global test_case_args
    test_case_args, remaining_args = parse_args()
    unittest.main(argv=remaining_args)
