#   copyright (c) 2020 paddlepaddle authors. all rights reserved.
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
import time
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import Qat2Int8MkldnnPass
from paddle.fluid import core

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument(
        '--skip_batch_num',
        type=int,
        default=0,
        help='Number of the first minibatches to skip in performance statistics.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='If used, the graph of QAT model is drawn.')
    parser.add_argument(
        '--qat_model', type=str, default='', help='A path to a QAT model.')
    parser.add_argument(
        '--fp32_model',
        type=str,
        default='',
        help='A path to an FP32 model. If empty, the QAT model will be used for FP32 inference.'
    )
    parser.add_argument('--infer_data', type=str, default='', help='Data file.')
    parser.add_argument(
        '--labels', type=str, default='', help='File with labels.')
    parser.add_argument(
        '--batch_num',
        type=int,
        default=0,
        help='Number of batches to process. 0 or less means whole dataset. Default: 0.'
    )
    parser.add_argument(
        '--acc_diff_threshold',
        type=float,
        default=0.01,
        help='Accepted accuracy difference threshold.')
    parser.add_argument(
        '--ops_to_quantize',
        type=str,
        default='',
        help='A comma separated list of operators to quantize. Only quantizable operators are taken into account. If the option is not used, an attempt to quantize all quantizable operators will be made.'
    )

    test_args, args = parser.parse_known_args(namespace=unittest)

    return test_args, sys.argv[:1] + args


class QatInt8NLPComparisonTest(unittest.TestCase):
    """
    Test for accuracy comparison of QAT FP32 and INT8 NLP inference.
    """

    def _reader_creator(self, data_file=None, labels_file=None):
        assert data_file, "The dataset file is missing."
        assert labels_file, "The labels file is missing."

        def reader():
            with open(data_file, 'r') as df:
                with open(labels_file, 'r') as lf:
                    data_lines = df.readlines()
                    labels_lines = lf.readlines()
                    assert len(data_lines) == len(
                        labels_lines
                    ), "The number of labels does not match the length of the dataset."

                    for i in range(len(data_lines)):
                        data_fields = data_lines[i].split(';')
                        assert len(
                            data_fields
                        ) >= 2, "The number of data fields in the dataset is less than 2"
                        buffers = []
                        shape = []
                        for j in range(2):
                            data = data_fields[j].split(':')
                            assert len(
                                data
                            ) >= 2, "Size of data in the dataset is less than 2"
                            # Shape is stored under index 0, while data under 1
                            shape = data[0].split()
                            shape.pop(0)
                            shape_np = np.array(shape).astype("int64")
                            buffer_i = data[1].split()
                            buffer_np = np.array(buffer_i).astype("int64")
                            buffer_np.shape = tuple(shape_np)
                            buffers.append(buffer_np)
                        label = labels_lines[i]
                        yield buffers[0], buffers[1], int(label)

        return reader

    def _get_batch_correct(self, batch_output=None, labels=None):
        total = len(batch_output)
        assert total > 0, "The batch output is empty."
        correct = 0
        for n, output in enumerate(batch_output[0]):
            max_idx = np.where(output == output.max())
            if max_idx == labels[n]:
                correct += 1
        return correct

    def _predict(self,
                 test_reader=None,
                 model_path=None,
                 batch_size=1,
                 batch_num=1,
                 skip_batch_num=0,
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

            graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
            if (self._debug):
                graph.draw('.', 'qat_orig', graph.all_op_nodes())
            if (transform_to_int8):
                transform_to_mkldnn_int8_pass = Qat2Int8MkldnnPass(
                    self._quantized_ops,
                    _scope=inference_scope,
                    _place=place,
                    _core=core,
                    _debug=self._debug)
                graph = transform_to_mkldnn_int8_pass.apply(graph)

            inference_program = graph.to_program()

            total_correct = 0
            total_samples = 0
            batch_times = []
            ppses = []  # predictions per second
            iters = 0
            infer_start_time = time.time()
            for data in test_reader():
                if batch_num > 0 and iters >= batch_num:
                    break
                if iters == skip_batch_num:
                    total_samples = 0
                    infer_start_time = time.time()
                input0 = np.array([x[0] for x in data]).astype('int64')
                input1 = np.array([x[1] for x in data]).astype('int64')
                labels = np.array([x[2] for x in data]).astype('int64')

                start = time.time()
                out = exe.run(inference_program,
                              feed={
                                  feed_target_names[0]: input0,
                                  feed_target_names[1]: input1
                              },
                              fetch_list=fetch_targets)
                batch_time = (time.time() - start) * 1000  # in miliseconds
                batch_times.append(batch_time)
                batch_correct = self._get_batch_correct(out, labels)
                batch_len = len(data)
                total_samples += batch_len
                total_correct += batch_correct
                batch_acc = float(batch_correct) / float(batch_len)
                pps = batch_len / batch_time * 1000
                ppses.append(pps)
                latency = batch_time / batch_len
                iters += 1
                appx = ' (warm-up)' if iters <= skip_batch_num else ''
                _logger.info(
                    'batch {0}{4}, acc: {1:.4f}, latency: {2:.4f} ms, predictions per sec: {3:.2f}'
                    .format(iters, batch_acc, latency, pps, appx))

            # Postprocess benchmark data
            infer_total_time = time.time() - infer_start_time
            batch_latencies = batch_times[skip_batch_num:]
            batch_latency_avg = np.average(batch_latencies)
            latency_avg = batch_latency_avg / batch_size
            ppses = ppses[skip_batch_num:]
            pps_avg = np.average(ppses)
            acc_avg = float(np.sum(total_correct)) / float(total_samples)
            _logger.info('Total inference run time: {:.2f} s'.format(
                infer_total_time))

            return acc_avg, pps_avg, latency_avg

    def _summarize_performance(self, fp32_pps, fp32_lat, int8_pps, int8_lat):
        _logger.info('--- Performance summary ---')
        _logger.info(
            'FP32: avg predictions per sec: {0:.2f}, avg latency: {1:.4f} ms'.
            format(fp32_pps, fp32_lat))
        _logger.info(
            'INT8: avg predictions per sec: {0:.2f}, avg latency: {1:.4f} ms'.
            format(int8_pps, int8_lat))

    def _compare_accuracy(self, fp32_acc, int8_acc, threshold):
        _logger.info('--- Accuracy summary ---')
        _logger.info(
            'Accepted accuracy drop threshold: {0}. (condition: (FP32_acc - INT8_acc) <= threshold)'
            .format(threshold))
        _logger.info('FP32: avg accuracy: {0:.6f}'.format(fp32_acc))
        _logger.info('INT8: avg accuracy: {0:.6f}'.format(int8_acc))
        # Random outputs give accuracy about 0.33, we assume valid accuracy to be at least 0.5
        assert fp32_acc > 0.5
        assert int8_acc > 0.5
        assert fp32_acc - int8_acc <= threshold

    def test_graph_transformation(self):
        if not fluid.core.is_compiled_with_mkldnn():
            return

        qat_model_path = test_case_args.qat_model
        assert qat_model_path, 'The QAT model path cannot be empty. Please, use the --qat_model option.'
        fp32_model_path = test_case_args.fp32_model if test_case_args.fp32_model else qat_model_path
        data_path = test_case_args.infer_data
        assert data_path, 'The dataset path cannot be empty. Please, use the --infer_data option.'
        labels_path = test_case_args.labels
        batch_size = test_case_args.batch_size
        batch_num = test_case_args.batch_num
        skip_batch_num = test_case_args.skip_batch_num
        acc_diff_threshold = test_case_args.acc_diff_threshold
        self._debug = test_case_args.debug
        self._quantized_ops = set()
        if len(test_case_args.ops_to_quantize) > 0:
            self._quantized_ops = set(test_case_args.ops_to_quantize.split(','))

        _logger.info('FP32 & QAT INT8 prediction run.')
        _logger.info('QAT model: {0}'.format(qat_model_path))
        _logger.info('FP32 model: {0}'.format(fp32_model_path))
        _logger.info('Dataset: {0}'.format(data_path))
        _logger.info('Labels: {0}'.format(labels_path))
        _logger.info('Batch size: {0}'.format(batch_size))
        _logger.info('Batch number: {0}'.format(batch_num))
        _logger.info('Accuracy drop threshold: {0}.'.format(acc_diff_threshold))
        _logger.info('Quantized ops: {0}.'.format(self._quantized_ops))

        _logger.info('--- FP32 prediction start ---')
        val_reader = paddle.batch(
            self._reader_creator(data_path, labels_path), batch_size=batch_size)
        fp32_acc, fp32_pps, fp32_lat = self._predict(
            val_reader,
            fp32_model_path,
            batch_size,
            batch_num,
            skip_batch_num,
            transform_to_int8=False)
        _logger.info('FP32: avg accuracy: {0:.6f}'.format(fp32_acc))
        _logger.info('--- QAT INT8 prediction start ---')
        val_reader = paddle.batch(
            self._reader_creator(data_path, labels_path), batch_size=batch_size)
        int8_acc, int8_pps, int8_lat = self._predict(
            val_reader,
            qat_model_path,
            batch_size,
            batch_num,
            skip_batch_num,
            transform_to_int8=True)
        _logger.info('INT8: avg accuracy: {0:.6f}'.format(int8_acc))

        self._summarize_performance(fp32_pps, fp32_lat, int8_pps, int8_lat)
        self._compare_accuracy(fp32_acc, int8_acc, acc_diff_threshold)


if __name__ == '__main__':
    global test_case_args
    test_case_args, remaining_args = parse_args()
    unittest.main(argv=remaining_args)
