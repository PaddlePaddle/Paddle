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
import numpy as np
import time
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import Quant2Int8MkldnnPass
from paddle.fluid import core

paddle.enable_static()

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
    parser.add_argument('--quant_model',
                        type=str,
                        default='',
                        help='A path to a Quant model.')
    parser.add_argument(
        '--fp32_model',
        type=str,
        default='',
        help=
        'A path to an FP32 model. If empty, the Quant model will be used for FP32 inference.'
    )
    parser.add_argument('--infer_data', type=str, default='', help='Data file.')
    parser.add_argument('--labels',
                        type=str,
                        default='',
                        help='File with labels.')
    parser.add_argument(
        '--batch_num',
        type=int,
        default=0,
        help=
        'Number of batches to process. 0 or less means whole dataset. Default: 0.'
    )
    parser.add_argument('--acc_diff_threshold',
                        type=float,
                        default=0.01,
                        help='Accepted accuracy difference threshold.')
    parser.add_argument(
        '--ops_to_quantize',
        type=str,
        default='',
        help=
        'A comma separated list of operators to quantize. Only quantizable operators are taken into account. If the option is not used, an attempt to quantize all quantizable operators will be made.'
    )
    parser.add_argument(
        '--op_ids_to_skip',
        type=str,
        default='',
        help='A comma separated list of operator ids to skip in quantization.')
    parser.add_argument(
        '--targets',
        type=str,
        default='quant,int8,fp32',
        help=
        'A comma separated list of inference types to run ("int8", "fp32", "quant"). Default: "quant,int8,fp32"'
    )
    parser.add_argument('--debug',
                        action='store_true',
                        help='If used, the graph of Quant model is drawn.')

    test_args, args = parser.parse_known_args(namespace=unittest)

    return test_args, sys.argv[:1] + args


class QuantInt8NLPComparisonTest(unittest.TestCase):
    """
    Test for accuracy comparison of Quant FP32 and INT8 NLP inference.
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
                 target='quant'):
        assert target in ['quant', 'int8', 'fp32']
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        inference_scope = fluid.executor.global_scope()
        with fluid.scope_guard(inference_scope):
            if os.path.exists(os.path.join(model_path, '__model__')):
                [inference_program, feed_target_names, fetch_targets
                 ] = fluid.io.load_inference_model(model_path, exe)
            else:
                [inference_program, feed_target_names, fetch_targets
                 ] = fluid.io.load_inference_model(model_path, exe, 'model',
                                                   'params')

            graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
            if (self._debug):
                graph.draw('.', 'quant_orig', graph.all_op_nodes())
            if (target != 'quant'):
                quant_transform_pass = Quant2Int8MkldnnPass(
                    self._quantized_ops,
                    _op_ids_to_skip=self._op_ids_to_skip,
                    _scope=inference_scope,
                    _place=place,
                    _core=core,
                    _debug=self._debug)
                if (target == 'int8'):
                    graph = quant_transform_pass.apply(graph)
                else:  # target == fp32
                    graph = quant_transform_pass.prepare_and_optimize_fp32(
                        graph)

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
            _logger.info(
                'Total inference run time: {:.2f} s'.format(infer_total_time))

            return acc_avg, pps_avg, latency_avg

    def _print_performance(self, title, pps, lat):
        _logger.info(
            '{0}: avg predictions per sec: {1:.2f}, avg latency: {2:.4f} ms'.
            format(title, pps, lat))

    def _print_accuracy(self, title, acc):
        _logger.info('{0}: avg accuracy: {1:.6f}'.format(title, acc))

    def _summarize_performance(self, int8_pps, int8_lat, fp32_pps, fp32_lat):
        _logger.info('--- Performance summary ---')
        self._print_performance('INT8', int8_pps, int8_lat)
        if fp32_lat >= 0:
            self._print_performance('FP32', fp32_pps, fp32_lat)

    def _summarize_accuracy(self, quant_acc, int8_acc, fp32_acc):
        _logger.info('--- Accuracy summary ---')
        self._print_accuracy('Quant', quant_acc)
        self._print_accuracy('INT8', int8_acc)
        if fp32_acc >= 0:
            self._print_accuracy('FP32', fp32_acc)

    def _compare_accuracy(self, threshold, quant_acc, int8_acc):
        _logger.info(
            'Accepted accuracy drop threshold: {0}. (condition: (Quant_acc - INT8_acc) <= threshold)'
            .format(threshold))
        # Random outputs give accuracy about 0.33, we assume valid accuracy to be at least 0.5
        assert quant_acc > 0.5
        assert int8_acc > 0.5
        assert quant_acc - int8_acc <= threshold

    def _strings_from_csv(self, string):
        return set(s.strip() for s in string.split(','))

    def _ints_from_csv(self, string):
        return set(map(int, string.split(',')))

    def test_graph_transformation(self):
        if not fluid.core.is_compiled_with_mkldnn():
            return

        quant_model_path = test_case_args.quant_model
        assert quant_model_path, 'The Quant model path cannot be empty. Please, use the --quant_model option.'
        data_path = test_case_args.infer_data
        assert data_path, 'The dataset path cannot be empty. Please, use the --infer_data option.'
        fp32_model_path = test_case_args.fp32_model
        labels_path = test_case_args.labels
        batch_size = test_case_args.batch_size
        batch_num = test_case_args.batch_num
        skip_batch_num = test_case_args.skip_batch_num
        acc_diff_threshold = test_case_args.acc_diff_threshold
        self._debug = test_case_args.debug

        self._quantized_ops = set()
        if test_case_args.ops_to_quantize:
            self._quantized_ops = self._strings_from_csv(
                test_case_args.ops_to_quantize)

        self._op_ids_to_skip = set([-1])
        if test_case_args.op_ids_to_skip:
            self._op_ids_to_skip = self._ints_from_csv(
                test_case_args.op_ids_to_skip)

        self._targets = self._strings_from_csv(test_case_args.targets)
        assert self._targets.intersection(
            {'quant', 'int8', 'fp32'}
        ), 'The --targets option, if used, must contain at least one of the targets: "quant", "int8", "fp32".'

        _logger.info('Quant & INT8 prediction run.')
        _logger.info('Quant model: {}'.format(quant_model_path))
        if fp32_model_path:
            _logger.info('FP32 model: {}'.format(fp32_model_path))
        _logger.info('Dataset: {}'.format(data_path))
        _logger.info('Labels: {}'.format(labels_path))
        _logger.info('Batch size: {}'.format(batch_size))
        _logger.info('Batch number: {}'.format(batch_num))
        _logger.info('Accuracy drop threshold: {}.'.format(acc_diff_threshold))
        _logger.info(
            'Quantized ops: {}.'.format(','.join(self._quantized_ops) if self.
                                        _quantized_ops else 'all quantizable'))
        _logger.info('Op ids to skip quantization: {}.'.format(
            ','.join(map(str, self._op_ids_to_skip)
                     ) if test_case_args.op_ids_to_skip else 'none'))
        _logger.info('Targets: {}.'.format(','.join(self._targets)))

        if 'quant' in self._targets:
            _logger.info('--- Quant prediction start ---')
            val_reader = paddle.batch(self._reader_creator(
                data_path, labels_path),
                                      batch_size=batch_size)
            quant_acc, quant_pps, quant_lat = self._predict(val_reader,
                                                            quant_model_path,
                                                            batch_size,
                                                            batch_num,
                                                            skip_batch_num,
                                                            target='quant')
            self._print_performance('Quant', quant_pps, quant_lat)
            self._print_accuracy('Quant', quant_acc)

        if 'int8' in self._targets:
            _logger.info('--- INT8 prediction start ---')
            val_reader = paddle.batch(self._reader_creator(
                data_path, labels_path),
                                      batch_size=batch_size)
            int8_acc, int8_pps, int8_lat = self._predict(val_reader,
                                                         quant_model_path,
                                                         batch_size,
                                                         batch_num,
                                                         skip_batch_num,
                                                         target='int8')
            self._print_performance('INT8', int8_pps, int8_lat)
            self._print_accuracy('INT8', int8_acc)

        fp32_acc = fp32_pps = fp32_lat = -1
        if 'fp32' in self._targets and fp32_model_path:
            _logger.info('--- FP32 prediction start ---')
            val_reader = paddle.batch(self._reader_creator(
                data_path, labels_path),
                                      batch_size=batch_size)
            fp32_acc, fp32_pps, fp32_lat = self._predict(val_reader,
                                                         fp32_model_path,
                                                         batch_size,
                                                         batch_num,
                                                         skip_batch_num,
                                                         target='fp32')
            self._print_performance('FP32', fp32_pps, fp32_lat)
            self._print_accuracy('FP32', fp32_acc)

        if {'int8', 'fp32'}.issubset(self._targets):
            self._summarize_performance(int8_pps, int8_lat, fp32_pps, fp32_lat)
        if {'int8', 'quant'}.issubset(self._targets):
            self._summarize_accuracy(quant_acc, int8_acc, fp32_acc)
            self._compare_accuracy(acc_diff_threshold, quant_acc, int8_acc)


if __name__ == '__main__':
    global test_case_args
    test_case_args, remaining_args = parse_args()
    unittest.main(argv=remaining_args)
