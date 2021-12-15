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

import numpy as np
import unittest
import abc
import os
import enum
import time
import logging
import shutil
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import NumpyArrayInitializer
from paddle.fluid.core import PassVersionChecker
import paddle.fluid.core as core
from paddle import compat as cpt
import paddle.inference as paddle_infer
from typing import Optional, List, Callable, Dict, Any, Set
from program_config import TensorConfig, OpConfig, ProgramConfig, create_fake_model, create_quant_model

import hypothesis
from hypothesis import given, settings, seed, reproduce_failure
import hypothesis.strategies as st

logging.basicConfig(level=logging.INFO, format="%(message)s")

settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=hypothesis.HealthCheck.all(),
    deadline=None,
    print_blob=True,
    derandomize=True,
    report_multiple_bugs=False)
settings.register_profile(
    "dev",
    max_examples=1000,
    suppress_health_check=hypothesis.HealthCheck.all(),
    deadline=None,
    print_blob=True,
    derandomize=True,
    report_multiple_bugs=False)
if float(os.getenv('TEST_NUM_PERCENT_CASES', default='1.0')) < 1 or \
    os.getenv('HYPOTHESIS_TEST_PROFILE', 'dev') == 'ci':
    settings.load_profile("ci")
else:
    settings.load_profile("dev")


class IgnoreReasons(enum.Enum):
    # Paddle not support, but trt support, we need to add the feature.
    TRT_NOT_IMPLEMENTED = 0
    # TRT not support.
    TRT_NOT_SUPPORT = 1
    # Accuracy is abnormal after enabling pass.
    PASS_ACCURACY_ERROR = 2
    # Accuracy is abnormal after enabling mkldnn.
    MKLDNN_ACCURACY_ERROR = 3


# TODO(wilber): just for backward compatible
SkipReasons = IgnoreReasons


class AutoScanTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        np.random.seed(1024)
        paddle.enable_static()
        super(AutoScanTest, self).__init__(*args, **kwargs)
        self.ignore_cases = []
        abs_dir = os.path.abspath(os.path.dirname(__file__))
        self.cache_dir = os.path.join(abs_dir,
                                      str(self.__module__) + '_cache_dir')
        self.available_passes_in_framework = set()
        self.num_ran_programs = 0
        self.num_invalid_programs = 0
        self.num_ignore_tests = 0
        self.num_predictor_kinds = 0

    @abc.abstractmethod
    def sample_program_configs(self):
        '''
        Generate all config with the combination of different Input tensor shape and
        different Attr values.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def sample_predictor_configs(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_ignore_check_case(
            self,
            teller: [Callable[[ProgramConfig, paddle_infer.Config], bool]],
            reason: IgnoreReasons,
            note: str):
        self.ignore_cases.append((teller, reason, note))

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def run_test_config(self, model, params, prog_config, pred_config,
                        feed_data) -> Dict[str, np.ndarray]:
        '''
        Test a single case.
        '''
        pred_config.set_model_buffer(model, len(model), params, len(params))
        predictor = paddle_infer.create_predictor(pred_config)
        self.available_passes_in_framework = self.available_passes_in_framework | set(
            pred_config.pass_builder().all_passes())
        for name, _ in prog_config.inputs.items():
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(feed_data[name]['data'])
            if feed_data[name]['lod'] is not None:
                input_tensor.set_lod(feed_data[name]['lod'])
        predictor.run()
        result = {}
        for out_name, o_name in zip(prog_config.outputs,
                                    predictor.get_output_names()):
            result[out_name] = predictor.get_output_handle(o_name).copy_to_cpu()
        return result

    @abc.abstractmethod
    def assert_tensors_near(self,
                            atol: float,
                            rtol: float,
                            tensor: Dict[str, np.array],
                            baseline: Dict[str, np.array]):
        for key, arr in tensor.items():
            self.assertTrue(
                baseline[key].shape == arr.shape,
                "The output shapes are not equal, the baseline shape is " +
                str(baseline[key].shape) + ', but got ' + str(arr.shape))
            self.assertTrue(
                np.allclose(
                    baseline[key], arr, atol=atol, rtol=rtol),
                "Output has diff. ")

    @abc.abstractmethod
    def run_test(self, quant=False):
        raise NotImplementedError

    def generate_op_config(self,
                           ops_config: List[Dict[str, Any]]) -> List[OpConfig]:
        ops = []
        for i in range(len(ops_config)):
            op_config = ops_config[i]
            ops.append(
                OpConfig(
                    type=op_config['op_type'],
                    inputs=op_config['op_inputs'],
                    outputs=op_config['op_outputs'],
                    attrs=op_config['op_attrs']))
        return ops

    @abc.abstractmethod
    def ignore_log(self, msg: str):
        logging.warning("SKIP: " + msg)

    @abc.abstractmethod
    def fail_log(self, msg: str):
        logging.error("FAIL: " + msg)

    @abc.abstractmethod
    def success_log(self, msg: str):
        logging.info("SUCCESS: " + msg)

    @abc.abstractmethod
    def create_inference_config(self,
                                passes: Optional[List[str]]=None,
                                use_gpu: bool=False,
                                use_mkldnn: bool=False,
                                ir_optim: Optional[bool]=None):
        config = paddle_infer.Config()
        config.switch_ir_debug(True)
        config.set_optim_cache_dir(self.cache_dir)
        config.disable_glog_info()
        if ir_optim is not None:
            config.switch_ir_optim(ir_optim)
        if use_gpu:
            config.enable_use_gpu(100, 0)
        if use_mkldnn:
            config.enable_mkldnn()
        if passes is not None:
            config.pass_builder().set_passes(passes)
            self.passes = passes
        return config


class MkldnnAutoScanTest(AutoScanTest):
    def __init__(self, *args, **kwargs):
        super(MkldnnAutoScanTest, self).__init__(*args, **kwargs)

    def run_test(self, quant=False, *args, **kwargs):
        status = True

        for prog_config in self.sample_program_configs(*args, **kwargs):
            # if program is invalid, we should skip that cases.
            if not self.is_program_valid(prog_config):
                continue

            model, params = create_fake_model(prog_config)
            if quant:
                model, params = create_quant_model(model, params)

            feed_data = {}
            for name, tensor_config in prog_config.inputs.items():
                feed_data[name] = {
                    'data': tensor_config.data,
                    'lod': tensor_config.lod
                }
            results: List[Dict[str, np.ndarray]] = []

            # baseline: cpu no ir_optim run
            base_config = self.create_inference_config(ir_optim=False)
            logging.info('RUN program_config: ' + str(prog_config))
            results.append(
                self.run_test_config(model, params, prog_config, base_config,
                                     feed_data))
            self.success_log('RUN_CPU_BASELINE done')

            for pred_config, (
                    atol, rtol) in self.sample_predictor_configs(prog_config):
                # skip info
                ignore_flag = False
                for ignore_info in self.ignore_cases:
                    if ignore_info[0](prog_config, pred_config):
                        ignore_flag = True
                        if ignore_info[
                                1] == IgnoreReasons.MKLDNN_ACCURACY_ERROR:
                            self.ignore_log("[MKLDNN_ACCURACY_ERROR] " +
                                            ignore_info[2] + ' ' + ' vs ' +
                                            self.inference_config_str(
                                                pred_config))
                        else:
                            raise NotImplementedError
                        break

                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                if not os.path.exists(self.cache_dir):
                    os.mkdir(self.cache_dir)

                try:
                    results.append(
                        self.run_test_config(model, params, prog_config,
                                             pred_config, feed_data))
                    self.assert_tensors_near(atol, rtol, results[-1],
                                             results[0])
                except Exception as e:
                    self.fail_log(
                        self.inference_config_str(pred_config) +
                        '\033[1;31m \nERROR INFO: {}\033[0m'.format(str(e)))
                    if not ignore_flag:
                        status = False
                    continue
                self.success_log('RUN predictor_config ' + self.
                                 inference_config_str(pred_config) + ' done')

        self.assertTrue(status)

    def inference_config_str(self, config) -> str:
        dic = {}
        enable_mkldnn = config.mkldnn_enabled()
        dic['use_mkldnn'] = enable_mkldnn
        enable_gpu = config.use_gpu()
        dic['use_gpu'] = enable_gpu
        return str(dic)


class PassAutoScanTest(AutoScanTest):
    def __init__(self, *args, **kwargs):
        super(PassAutoScanTest, self).__init__(*args, **kwargs)
        self.passes = []

    def check_op_version(self):
        status = True
        for pass_name in self.passes:
            if pass_name not in self.available_passes_in_framework:
                continue
            if not PassVersionChecker.IsCompatible(pass_name):
                self.fail_log('{} version check failed.'.format(pass_name))
                status = False
        return status

    def add_ignore_pass_case(self):
        return

    def assert_op_list(self, op_list_after_fusion):
        if not self.passes:
            raise ValueError(
                "In PassAutoScan you should give a valid pass name.")
        last_passed_program = os.path.join(self.cache_dir,
                                           self.passes[-1] + ".pdmodel")
        if not os.path.exists(last_passed_program):
            raise ValueError(
                "Cannot find file {}, please make sure that your pass name is correct".
                format(last_passed_program))
        model_bytes = paddle.static.load_from_file(last_passed_program)
        pg = paddle.static.deserialize_program(model_bytes)
        main_block = pg.desc.block(0)
        after_op_list = list()
        for i in range(main_block.op_size()):
            if main_block.op(i).type() in ["feed", "fetch"]:
                continue
            after_op_list.append(main_block.op(i).type())
        self.assertTrue(
            op_list_after_fusion == after_op_list,
            "Expected operator list after fusion is {}, but now it's {}".format(
                op_list_after_fusion, after_op_list), )

    def run_and_statis(self,
                       quant=False,
                       max_examples=100,
                       reproduce=None,
                       min_success_num=25,
                       max_duration=180,
                       passes=None,
                       use_gpu_run_baseline=False):
        if os.getenv('HYPOTHESIS_TEST_PROFILE', 'ci') == "dev":
            max_examples *= 10
            min_success_num *= 10
            # while at ce phase, there's no limit on time
            max_duration = -1
        start_time = time.time()
        settings.register_profile(
            "ci",
            max_examples=max_examples,
            suppress_health_check=hypothesis.HealthCheck.all(),
            deadline=None,
            print_blob=True,
            derandomize=True,
            report_multiple_bugs=False, )
        settings.load_profile("ci")
        assert passes is not None, "Parameter of passes must be defined in function run_and_statis."
        self.passes = passes

        self.add_ignore_pass_case()

        def program_generator(draw):
            return self.sample_program_config(draw)

        def run_test(prog_config):
            return self.run_test(
                quant=quant,
                prog_configs=[prog_config],
                use_gpu_run_baseline=use_gpu_run_baseline)

        generator = st.composite(program_generator)
        loop_func = given(generator())(run_test)
        if reproduce is not None:
            loop_func = reproduce(loop_func)
        logging.info("Start to running test of {}".format(type(self)))
        loop_func()
        logging.info(
            "===================Statistical Information===================")
        logging.info("Number of Generated Programs: {}".format(
            self.num_ran_programs + self.num_invalid_programs))
        logging.info("Number of Invalid Programs: {}".format(
            self.num_invalid_programs))
        logging.info("Number of Ran Programs: {}".format(self.num_ran_programs))
        logging.info("Number of Ignore Tests: {}".format(self.num_ignore_tests))
        successful_ran_programs = int(self.num_ran_programs -
                                      self.num_ignore_tests / max(
                                          self.num_predictor_kinds, 1))
        logging.info(
            "Number of successfully ran programs approximately equal to {}".
            format(successful_ran_programs))
        if successful_ran_programs < min_success_num:
            logging.warning(
                "satisfied_programs = ran_programs - num_ignore_tests / num_predictor_kinds"
            )
            logging.error(
                "At least {} programs need to ran successfully, but now only about {} programs satisfied.".
                format(min_success_num, successful_ran_programs))
            assert False
        used_time = time.time() - start_time
        if max_duration > 0 and used_time > max_duration:
            logging.error(
                "The duration exceeds {} seconds, if this is neccessary, try to set a larger number for parameter `max_duration`.".
                format(max_duration))
            assert False

    def run_test(self,
                 quant=False,
                 prog_configs=None,
                 use_gpu_run_baseline=False):
        status = True

        for prog_config in prog_configs:
            # if program is invalid, we should skip that cases.
            if not self.is_program_valid(prog_config):
                self.num_invalid_programs += 1
                continue
            self.num_ran_programs += 1
            model, params = create_fake_model(prog_config)
            if quant:
                model, params = create_quant_model(model, params)

            feed_data = {}
            for name, tensor_config in prog_config.inputs.items():
                feed_data[name] = {
                    'data': tensor_config.data,
                    'lod': tensor_config.lod
                }
            results: List[Dict[str, np.ndarray]] = []

            # baseline: cpu no ir_optim run

            base_config = self.create_inference_config(
                ir_optim=False, use_gpu=use_gpu_run_baseline)
            logging.info('RUN program_config: ' + str(prog_config))
            results.append(
                self.run_test_config(model, params, prog_config, base_config,
                                     feed_data))
            self.success_log('RUN_CPU_BASELINE done')

            self.num_predictor_kinds = 0
            for pred_config, op_list, (
                    atol, rtol) in self.sample_predictor_configs(prog_config):
                self.num_predictor_kinds += 1
                # skip info
                ignore_flag = False
                for ignore_info in self.ignore_cases:
                    if ignore_info[0](prog_config, pred_config):
                        ignore_flag = True
                        self.num_ignore_tests += 1
                        if ignore_info[1] == IgnoreReasons.PASS_ACCURACY_ERROR:
                            self.ignore_log("[PASS_ACCURACY_ERROR] " +
                                            ignore_info[2] + ' ' + ' vs ' +
                                            self.inference_config_str(
                                                pred_config))
                        else:
                            raise NotImplementedError
                        break

                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                if not os.path.exists(self.cache_dir):
                    os.mkdir(self.cache_dir)

                try:
                    results.append(
                        self.run_test_config(model, params, prog_config,
                                             pred_config, feed_data))
                    self.assert_tensors_near(atol, rtol, results[-1],
                                             results[0])
                    if not ignore_flag:
                        self.assert_op_list(op_list)

                except Exception as e:
                    self.fail_log(
                        self.inference_config_str(pred_config) +
                        '\033[1;31m \nERROR INFO: {}\033[0m'.format(str(e)))
                    if not ignore_flag:
                        status = False
                    continue
                self.success_log('RUN predictor_config ' + self.
                                 inference_config_str(pred_config) + ' done')

        status = self.check_op_version() and status
        self.assertTrue(status)

    def inference_config_str(self, config) -> str:
        dic = {}
        enable_mkldnn = config.mkldnn_enabled()
        dic['use_mkldnn'] = enable_mkldnn
        enable_gpu = config.use_gpu()
        dic['use_gpu'] = enable_gpu
        if not self.passes:
            dic['passes'] = self.passes

        enable_trt = config.tensorrt_engine_enabled()
        trt_precison = config.tensorrt_precision_mode()
        trt_dynamic_shape = config.tensorrt_dynamic_shape_enabled()
        if enable_trt:
            dic['use_trt'] = True
            dic['trt_precision'] = trt_precison
            dic['use_dynamic_shape'] = trt_dynamic_shape
        else:
            dic['use_trt'] = False
        return str(dic)

    def create_trt_inference_config(self) -> paddle_infer.Config:
        config = paddle_infer.Config()
        config.disable_glog_info()
        config.enable_use_gpu(100, 0)
        config.set_optim_cache_dir(self.cache_dir)
        config.switch_ir_debug()
        return config


class TrtLayerAutoScanTest(AutoScanTest):
    class TensorRTParam:
        '''
        TensorRT subgraph engine parameters. 
        '''

        def __init__(self, workspace_size, max_batch_size, min_subgraph_size,
                     precision, use_static, use_calib_mode):
            self.workspace_size = workspace_size
            self.max_batch_size = max_batch_size
            self.min_subgraph_size = min_subgraph_size
            self.precision = precision
            self.use_static = use_static
            self.use_calib_mode = use_calib_mode

    class DynamicShapeParam:
        '''
         Prepare TensorRT subgraph engine dynamic shape parameters. 
         '''

        def __init__(self, min_input_shape, max_input_shape, opt_input_shape,
                     disable_trt_plugin_fp16):
            self.min_input_shape = min_input_shape
            self.max_input_shape = max_input_shape
            self.opt_input_shape = opt_input_shape
            self.disable_trt_plugin_fp16 = disable_trt_plugin_fp16

    def __init__(self, *args, **kwargs):
        super(TrtLayerAutoScanTest, self).__init__(*args, **kwargs)
        self.trt_param = self.TensorRTParam(
            workspace_size=1024,
            max_batch_size=4,
            min_subgraph_size=0,
            precision=paddle_infer.PrecisionType.Float32,
            use_static=True,
            use_calib_mode=False)
        self.dynamic_shape = self.DynamicShapeParam({}, {}, {}, False)
        self.num_percent_cases = float(
            os.getenv(
                'TEST_NUM_PERCENT_CASES', default='1.0'))
        # Choose different tests by week
        np.random.seed(int(time.strftime("%W")))

    def create_inference_config(self, use_trt=True) -> paddle_infer.Config:
        config = paddle_infer.Config()
        config.disable_glog_info()
        config.enable_use_gpu(100, 0)
        config.set_optim_cache_dir(self.cache_dir)
        if use_trt:
            config.switch_ir_debug()
            config.enable_tensorrt_engine(
                max_batch_size=self.trt_param.max_batch_size,
                workspace_size=self.trt_param.workspace_size,
                min_subgraph_size=self.trt_param.min_subgraph_size,
                precision_mode=self.trt_param.precision,
                use_static=self.trt_param.use_static,
                use_calib_mode=self.trt_param.use_calib_mode)
            if len(self.dynamic_shape.min_input_shape
                   ) != 0 and self.dynamic_shape.min_input_shape.keys(
                   ) == self.dynamic_shape.max_input_shape.keys(
                   ) and self.dynamic_shape.min_input_shape.keys(
                   ) == self.dynamic_shape.opt_input_shape.keys():
                config.set_trt_dynamic_shape_info(
                    self.dynamic_shape.min_input_shape,
                    self.dynamic_shape.max_input_shape,
                    self.dynamic_shape.opt_input_shape,
                    self.dynamic_shape.disable_trt_plugin_fp16)
        return config

    def assert_op_size(self, trt_engine_num, paddle_op_num):
        last_passed_program = os.path.join(
            self.cache_dir, 'transpose_flatten_concat_fuse_pass.pdmodel')
        model_bytes = paddle.static.load_from_file(last_passed_program)
        pg = paddle.static.deserialize_program(model_bytes)
        main_block = pg.desc.block(0)
        op_size = main_block.op_size()
        op_types = [
            main_block.op(i).type() == 'tensorrt_engine' for i in range(op_size)
        ]
        trt_engine_size = sum(op_types)
        paddle_op_size = op_size - trt_engine_size
        self.assertTrue(trt_engine_size == trt_engine_num,
                        'trt_engine_num is {}, but got {}!'.format(
                            trt_engine_size, trt_engine_num))
        self.assertTrue(paddle_op_size == paddle_op_num,
                        'paddle_op_num is {}, but got {}!'.format(
                            paddle_op_size, paddle_op_num))

    def inference_config_str(self, config: paddle_infer.Config) -> str:
        dic = {}
        enable_trt = config.tensorrt_engine_enabled()
        trt_precison = config.tensorrt_precision_mode()
        trt_dynamic_shape = config.tensorrt_dynamic_shape_enabled()
        if enable_trt:
            dic['use_trt'] = True
            dic['trt_precision'] = trt_precison
            dic['use_dynamic_shape'] = trt_dynamic_shape
        else:
            dic['use_trt'] = False
        return str(dic)

    def run_test(self, quant=False, *args, **kwargs):
        status = True
        run_flags = []
        for prog_config in self.sample_program_configs(*args, **kwargs):
            # In CI, only run 10% cases
            if np.random.rand() < self.num_percent_cases:
                run_flags.append(True)
            else:
                run_flags.append(False)

        for prog_config, run_flags in zip(
                self.sample_program_configs(*args, **kwargs), run_flags):
            if not run_flags:
                continue

            # if program is invalid, we should skip that cases.
            if not self.is_program_valid(prog_config):
                continue

            model, params = create_fake_model(prog_config)
            if quant:
                model, params = create_quant_model(model, params)

            feed_data = {}
            for name, tensor_config in prog_config.inputs.items():
                feed_data[name] = {
                    'data': tensor_config.data,
                    'lod': tensor_config.lod
                }

            results: List[Dict[str, np.ndarray]] = []

            # baseline: gpu run
            logging.info('RUN program_config: ' + str(prog_config))
            gpu_config = self.create_inference_config(use_trt=False)
            results.append(
                self.run_test_config(model, params, prog_config, gpu_config,
                                     feed_data))
            self.success_log('RUN_GPU_BASELINE done')

            for pred_config, nodes_num, threshold in self.sample_predictor_configs(
                    prog_config):

                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)

                if isinstance(threshold, float):
                    atol = threshold
                    rtol = 1e-8
                elif isinstance(threshold, list) or isinstance(threshold,
                                                               tuple):
                    atol = threshold[0]
                    rtol = threshold[1]
                else:
                    raise NotImplementedError

                if quant and pred_config.tensorrt_precision_mode(
                ) != paddle_infer.PrecisionType.Int8:
                    continue
                if pred_config.tensorrt_precision_mode(
                ) == paddle_infer.PrecisionType.Int8 and not quant:
                    continue

                ignore_flag = False
                for ignore_info in self.ignore_cases:
                    if ignore_info[0](prog_config, pred_config):
                        ignore_flag = True
                        if ignore_info[1] == IgnoreReasons.TRT_NOT_IMPLEMENTED:
                            self.ignore_log("[TRT_NOT_IMPLEMENTED] " +
                                            ignore_info[2] + ' ' + ' vs ' +
                                            self.inference_config_str(
                                                pred_config))
                        elif ignore_info[1] == IgnoreReasons.TRT_NOT_SUPPORT:
                            self.ignore_log("[TRT_NOT_SUPPORT] " + ignore_info[
                                2] + ' ' + ' vs ' + self.inference_config_str(
                                    pred_config))
                        else:
                            raise NotImplementedError
                        break

                try:
                    pred_config_deserialize = paddle_infer.Config(pred_config)
                    results.append(
                        self.run_test_config(model, params, prog_config,
                                             pred_config, feed_data))
                    self.assert_tensors_near(atol, rtol, results[-1],
                                             results[0])
                    if not ignore_flag:
                        self.assert_op_size(nodes_num[0], nodes_num[1])
                    # deserialize test
                    if nodes_num[0] > 0:
                        self.run_test_config(model, params, prog_config,
                                             pred_config_deserialize, feed_data)
                except Exception as e:
                    self.fail_log(
                        str(prog_config) + ' vs ' + self.inference_config_str(
                            pred_config) +
                        '\033[1;31m \nERROR INFO: {}\033[0m'.format(str(e)))
                    if not ignore_flag:
                        status = False
                    continue
                self.success_log('RUN predictor_config ' + self.
                                 inference_config_str(pred_config) + ' done')

        self.assertTrue(status)

    # TODO(wilber): just for backward compatible
    def add_skip_case(
            self,
            teller: [Callable[[ProgramConfig, paddle_infer.Config], bool]],
            reason: IgnoreReasons,
            note: str):
        self.ignore_cases.append((teller, reason, note))
