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
import logging
import struct
import six
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import TransformForMkldnnPass
from paddle.fluid import core

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--qat_model', type=str, default='', help='A path to a QAT model.')
    parser.add_argument('--infer_data', type=str, default='', help='Data file.')
    parser.add_argument(
        '--out_diff_threshold',
        type=float,
        default=1e-05,
        help='Accepted output difference threshold.')

    test_args, args = parser.parse_known_args(namespace=unittest)

    return test_args, sys.argv[:1] + args


class TestQatInt8Comparison(unittest.TestCase):
    """
    Test for output comparison of QAT FP32 and INT8 inference with MKLDNN.
    Performed on the first image from the dataset.
    """

    def _reader_creator(self, data_file='data.bin'):
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

                fp.seek(imgs_offset)
                img = fp.read(img_size)
                img = struct.unpack_from('{}f'.format(img_ch * img_w * img_h),
                                         img)
                img = np.array(img)
                img.shape = (img_ch, img_w, img_h)
                yield img

        return reader

    def _predict(self,
                 test_reader=None,
                 model_path=None,
                 transform_to_int8=False):
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

            if (transform_to_int8):
                graph = IrGraph(
                    core.Graph(inference_program.desc), for_test=True)
                mkldnn_int8_pass = TransformForMkldnnPass(
                    scope=inference_scope, place=place)
                mkldnn_int8_pass.apply(graph)
                inference_program = graph.to_program()

            dshape = [3, 224, 224]
            for data in test_reader():
                if six.PY2:
                    images = map(lambda x: x.reshape(dshape), data)
                if six.PY3:
                    images = list(map(lambda x: x.reshape(dshape), data))
                images = np.array(images).astype('float32')
                out = exe.run(inference_program,
                              feed={feed_target_names[0]: images},
                              fetch_list=fetch_targets)

            return out[0][0]

    def test_graph_transformation(self):
        if not fluid.core.is_compiled_with_mkldnn():
            return

        qat_model_path = test_case_args.qat_model
        data_path = test_case_args.infer_data
        out_diff_threshold = test_case_args.out_diff_threshold

        _logger.info('QAT FP32 & INT8 prediction run.')
        _logger.info('QAT model: {0}'.format(qat_model_path))
        _logger.info('Dataset: {0}'.format(data_path))
        _logger.info('Output diff threshold: {0}.'.format(out_diff_threshold))

        _logger.info('--- QAT FP32 prediction start ---')
        val_reader = paddle.batch(self._reader_creator(data_path), batch_size=1)
        fp32_model_result = self._predict(
            val_reader, qat_model_path, transform_to_int8=False)
        #  _logger.info('out fp32: {0}'.format(fp32_model_result))

        _logger.info('--- QAT INT8 prediction start ---')
        val_reader = paddle.batch(self._reader_creator(data_path), batch_size=1)
        int8_model_result = self._predict(
            val_reader, qat_model_path, transform_to_int8=True)
        #  _logger.info('out int8: {0}'.format(int8_model_result))

        _logger.info('--- Comparing outputs ---')
        no_of_values = len(fp32_model_result)
        no_of_different_vales = no_of_values - np.sum(
            np.isclose(
                fp32_model_result,
                int8_model_result,
                rtol=0,
                atol=out_diff_threshold))
        max_abs_diff = np.max(
            np.absolute(
                np.array(fp32_model_result) - np.array(int8_model_result)))
        _logger.info('Accepted diff threshold: {0}'.format(out_diff_threshold))
        _logger.info('Number of values: {0}'.format(no_of_values))
        _logger.info('Number of different values: {0}'.format(
            no_of_different_vales))
        _logger.info('Max absolute diff: {0}'.format(max_abs_diff))
        assert np.allclose(
            fp32_model_result,
            int8_model_result,
            rtol=0,
            atol=out_diff_threshold)


if __name__ == '__main__':
    global test_case_args
    test_case_args, remaining_args = parse_args()
    unittest.main(argv=remaining_args)
