#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

from . import core
from .data_feeder import convert_dtype
from sys import version_info
from paddle.fluid import framework
from .framework import Variable
from .wrapped_decorator import signature_safe_contextmanager
import os
import six
import json
import inspect
import importlib
import numpy as np

__all__ = [
    'cuda_profiler', 'reset_profiler', 'profiler', 'start_profiler',
    'stop_profiler'
]

NVPROF_CONFIG = [
    "gpustarttimestamp",
    "gpuendtimestamp",
    "gridsize3d",
    "threadblocksize",
    "streamid",
    "enableonstart 0",
    "conckerneltrace",
]

not_record_op = [
    'data', 'creat_tensor', 'creat_parameter', 'global_var', 'Optimizer'
]
not_record_param = ['name', 'op_type', 'attrs', 'bias_attr', 'param_attr']

CONTROL_FLOW_OPS = [
    "conditional_block", "switch", "static_rnn", "while", "while_loop", "cond",
    "case", "ifelse", "dynamic_rnn", "switch_case"
]

op_alias = {
    "argmax": "arg_max",
    "argmin": "arg_min",
    "resize_nearest": "nearest_interp",
    "resize_bilinear": "bilinear_interp",
    "sums": "sum",
    "hsigmoid": "hierarchical_sigmoid",
    "sampled_softmax_with_cross_entropy": "sample_logits",
    "max_pool2d_with_index": "adaptive_pool2d",
    "max_pool3d_with_index": "adaptive_pool3d",
    "smooth_l1_loss": "smooth_l1",
    "reshape2": "reshape",
    "unsqueeze2": "unsqueeze",
    "top_k": "topk",
    "global_step_counter": "autoincreased_step_counter",
    "cross_entropy2": "cross_entropy"
}

op_params_json_list = []


class APIStr(object):
    def __init__(self, layer_type, params):
        param_j = self.op_trans(layer_type, params)
        if param_j != None:
            op_params_json_list.append(param_j)

    def import_fluid_module(self, api_name):
        act_api_name = api_name
        try:
            if act_api_name in ["embedding", "ont_hot"]:
                module_name = "paddle.fluid"
            else:
                module_name = "paddle.fluid.layers"
            module = importlib.import_module(module_name)
            return getattr(module, act_api_name)
        except Exception:
            print("Cannot immport %s.%s." % (module_name, act_api_name))
            module = None

    def import_paddle_module(self, api_name):
        act_api_name = api_name
        try:
            module = importlib.import_module("paddle")
            return getattr(module, act_api_name)
        except Exception:
            print("Cannot immport paddle.%s." % (act_api_name))
            module = None

    def is_variable(self, type):
        if isinstance(
                type,
                Variable) or str(type).find('Variable') != -1 or isinstance(
                    type,
                    framework.Parameter) or str(type).find('Parameter') != -1:
            return True

    def op_trans(self, op, params):
        op_json = {}
        params_json = {}
        if op in op_alias.keys():
            op = op_alias[op]
        valid_op = True
        for not_op in not_record_op:
            if op.find(not_op) != -1:
                valid_op = False
        if valid_op:
            op_json["op"] = op
        else:
            print("API_LOG: return as not valid op:", op)
            return None

        if len(params) == 0 or op in CONTROL_FLOW_OPS:
            op_json["param_info"] = ''
        else:
            func = self.import_fluid_module(op)
            if func is None:
                func = self.import_paddle_module(op)

            if func is not None:
                argspec = inspect.getargspec(func)

            for param in params:
                key = param[0]
                values = param[1]
                if key != None and key in argspec.args and key not in not_record_param and not callable(
                        values):
                    data_type = ""
                    dict_temp = {}
                    if hasattr(values, 'dtype'):
                        if self.is_variable(values.dtype):
                            data_type = "Variable"
                        else:
                            data_type = convert_dtype(values.dtype)

                    vtype = self.type_to_string(values, key, op_json["op"])
                    if values is None:
                        dict_temp = {'type': 'string', 'value': str(None)}
                    elif vtype == "Variable":
                        dict_temp = {
                            'type': vtype,
                            'dtype': data_type,
                            'shape': str(list(values.shape))
                        }
                    elif vtype == "dict":
                        dict_temp = None
                    elif vtype == "list" and type(values[0]) is Variable:
                        v = values[0]
                        if hasattr(v, 'dtype'):
                            data_type_l = convert_dtype(v.dtype)
                        for i in range(len(values)):
                            dict_temp[key + str(i)] = {
                                'type': 'Variable',
                                'dtype': data_type_l,
                                'shape': str(list(values[i].shape))
                            }
                        dict_temp['type'] = 'list<Variable>'
                    elif vtype == "VarType":
                        dict_temp = {
                            'type': 'string',
                            'value': convert_dtype(values)
                        }
                    elif vtype == "unicode":
                        dict_temp = {
                            'type': 'string',
                            'value': str(values.encode('utf-8'))
                        }
                    else:
                        dict_temp = {'type': vtype, 'value': str(values)}
                    if dict_temp != None:
                        params_json[key] = dict_temp
                op_json["param_info"] = params_json
        return op_json

    def type_to_string(self, values, key, op):
        vtype = ""
        if self.is_variable(type(values)):
            vtype = "Variable"
        elif type(values) is float:
            vtype = "float"
        elif type(values) is bool:
            vtype = "bool"
        elif type(values) is str:
            vtype = "string"
        elif type(values) is int:
            vtype = "int"
        elif version_info.major == 2 and type(values) is long:
            vtype = "long"
        elif type(values) is list:
            vtype = "list"
        elif type(values) is tuple:
            vtype = "tuple"
        elif type(values) is dict:
            vtype = "dict"
        elif type(values) is np.ndarray:
            vtype = "numpy.ndarray"
        elif str(type(values)).find('NoneType') != -1:
            return vtype
        elif str(type(values)).find('VarType') != -1:
            vtype = 'VarType'
        elif str(type(values)).find('unicode') != -1:
            vtype = 'unicode'
        else:
            print("Unsupported vtype %s (key: %s, op: %s)" %
                  (type(values), key, op))
        return vtype


def API_info_summary():
    info_json_file = os.environ.get('API_INFO_LOG_PATH')
    if info_json_file is not None:
        with open(info_json_file, 'w') as f:
            f.writelines(
                json.dumps(
                    op_params_json_list, sort_keys=True, indent=4))
            f.writelines('\n')


@signature_safe_contextmanager
def cuda_profiler(output_file, output_mode=None, config=None):
    """
    The CUDA profiler.
    
    This fuctions is used to profile CUDA program by CUDA runtime application
    programming interface. The profiling result will be written into
    `output_file`. The users can set the output mode by `output_mode` argument 
    and set the nvidia profiling config by `config` argument. 
    
    After getting the profiling result file, users can use 
    `NVIDIA Visual Profiler <https://developer.nvidia.com/nvidia-visual-profiler>`_ 
    to load this output file to visualize results.

    Args:
        output_file (str) : The output file name, the result will be
            written into this file.
        output_mode (str, optional) : The output mode has Key-Value pair format ('kvp') 
            and Comma separated values format ('csv', default).
        config (list<str>, optional) : Nvidia profile config. Default config is 
            ['gpustarttimestamp', 'gpuendtimestamp', 'gridsize3d', 'threadblocksize', 
            'streamid', 'enableonstart 0', 'conckerneltrace']. For more details, please
            refer to `Compute Command Line Profiler User Guide <https://developer.download.nvidia.cn/compute/DevZone/docs/html/C/doc/Compute_Command_Line_Profiler_User_Guide.pdf>`_ .

    Raises:
        ValueError: If `output_mode` is not in ['kvp', 'csv'].

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.profiler as profiler
            import numpy as np

            epoc = 8
            dshape = [4, 3, 28, 28]
            data = fluid.data(name='data', shape=[None, 3, 28, 28], dtype='float32')
            conv = fluid.layers.conv2d(data, 20, 3, stride=[1, 1], padding=[1, 1])

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            output_file = 'cuda_profiler.txt'
            with profiler.cuda_profiler(output_file, 'csv') as nvprof:
                for i in range(epoc):
                    input = np.random.random(dshape).astype('float32')
                    exe.run(fluid.default_main_program(), feed={'data': input})
            # then use  NVIDIA Visual Profiler (nvvp) to load this output file
            # to visualize results.
    """
    if output_mode is None:
        output_mode = 'csv'
    if output_mode not in ['kvp', 'csv']:
        raise ValueError("The output mode must be 'kvp' or 'csv'.")
    config = NVPROF_CONFIG if config is None else config
    config_file = 'nvprof_config_file'
    with open(config_file, 'wb') as fp:
        fp.writelines([six.b("%s\n" % item) for item in config])
    core.nvprof_init(output_file, output_mode, config_file)
    # Enables profiler collection by the active CUDA profiling tool.
    core.nvprof_start()
    try:
        yield
    # Disables profiler collection.
    finally:
        core.nvprof_stop()
        os.remove(config_file)


def reset_profiler():
    """
    Clear the previous time record. This interface does not work for
    `fluid.profiler.cuda_profiler`, it only works for
    `fluid.profiler.start_profiler`, `fluid.profiler.stop_profiler`,
    and `fluid.profiler.profiler`.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.profiler as profiler
            with profiler.profiler('CPU', 'total', '/tmp/profile'):
                for iter in range(10):
                    if iter == 2:
                        profiler.reset_profiler()
                    # ...
    """
    core.reset_profiler()


def start_profiler(state, tracer_option='Default'):
    """
    Enable the profiler. Uers can use `fluid.profiler.start_profiler` and
    `fluid.profiler.stop_profiler` to profile, which is equal to the usage 
    of `fluid.profiler.profiler` interface.

    Args:
        state (str) : The profiling state, which should be one of 'CPU', 'GPU'
            or 'All'. 'CPU' means only profiling CPU; 'GPU' means profiling
            both CPU and GPU; 'All' means profiling both CPU and GPU, and 
            generates timeline as well.
        tracer_option (str, optional) : tracer_option can be one of ['Default', 'OpDetail', 'AllOpDetail'], it
            can control the profile level and print the different level profile result. `Default` option print 
            the different Op type profiling result and the `OpDetail` option print the detail profiling 
            result of different op types such as compute and data transform, `AllOpDetail` option 
            print the detail profiling result of different op name same as `OpDetail`.

    Raises:
        ValueError: If `state` is not in ['CPU', 'GPU', 'All'] or `tracer_option` 
            is not in ['Default', 'OpDetail', 'AllOpDetail'].

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.profiler as profiler

            profiler.start_profiler('GPU')
            for iter in range(10):
                if iter == 2:
                    profiler.reset_profiler()
                # except each iteration
            profiler.stop_profiler('total', '/tmp/profile')
            
            profiler.start_profiler('GPU', "OpDetail")
            for iter in range(10):
                if iter == 2:
                    profiler.reset_profiler()
                # except each iteration
            profiler.stop_profiler('total', '/tmp/profile')
    """
    if core.is_profiler_enabled():
        return
    if state not in ['CPU', 'GPU', "All"]:
        raise ValueError("The state must be 'CPU' or 'GPU' or 'All'.")
    if state == "GPU":
        prof_state = core.ProfilerState.kCUDA
    elif state == "CPU":
        prof_state = core.ProfilerState.kCPU
    else:
        prof_state = core.ProfilerState.kAll

    if tracer_option not in ['Default', 'OpDetail', 'AllOpDetail']:
        raise ValueError(
            "tracer option must be 'Default', 'OpDetail', 'AllOpDetail'.")
    if tracer_option == "Default":
        prof_tracer_option = core.TracerOption.kDefault
    elif tracer_option == "OpDetail":
        prof_tracer_option = core.TracerOption.kOpDetail
    else:
        prof_tracer_option = core.TracerOption.kAllOpDetail

    core.set_tracer_option(prof_tracer_option)
    core.enable_profiler(prof_state)


def stop_profiler(sorted_key=None, profile_path='/tmp/profile'):
    """
    Stop the profiler. Uers can use `fluid.profiler.start_profiler` and
    `fluid.profiler.stop_profiler` to profile, which is equal to the usage 
    of `fluid.profiler.profiler` interface.

    Args:
        sorted_key (str, optional) : The order of profiling results, which 
            should be one of None, 'calls', 'total', 'max', 'min' or 'ave'.
            Default is None, means the profiling results will be printed
            in the order of first end time of events.
            The `calls` means sorting by the number of calls.
            The `total` means sorting by the total execution time.
            The `max` means sorting by the maximum execution time.
            The `min` means sorting by the minimum execution time.
            The `ave` means sorting by the average execution time.
            and write it into `profile_path`. The default profile_path is `/tmp/profile`. 
        profile_path (str, optional) : If state == 'All', it will generate timeline,

    Raises:
        ValueError: If `sorted_key` is not in
            ['calls', 'total', 'max', 'min', 'ave'].

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.profiler as profiler

            profiler.start_profiler('GPU')
            for iter in range(10):
                if iter == 2:
                    profiler.reset_profiler()
                # except each iteration
            profiler.stop_profiler('total', '/tmp/profile')
    """
    if not core.is_profiler_enabled():
        return
    sorted_key = 'default' if sorted_key is None else sorted_key
    if sorted_key not in ['default', 'calls', 'total', 'max', 'min', 'ave']:
        raise ValueError("The sorted_key must be None or in 'calls', 'total', "
                         "'max', 'min' and 'ave'")
    key_map = {
        'default': core.EventSortingKey.kDefault,
        'calls': core.EventSortingKey.kCalls,
        'total': core.EventSortingKey.kTotal,
        'max': core.EventSortingKey.kMax,
        'min': core.EventSortingKey.kMin,
        'ave': core.EventSortingKey.kAve,
    }
    # TODO(qingqing) : redirect C++ ostream to Python stream.
    # with core.ostream_redirect(stdout=True, stderr=True):
    core.disable_profiler(key_map[sorted_key], profile_path)


@signature_safe_contextmanager
def profiler(state,
             sorted_key=None,
             profile_path='/tmp/profile',
             tracer_option='Default'):
    """
    The profiler interface. Different from `fluid.profiler.cuda_profiler`, 
    this profiler can be used to profile both CPU and GPU program.

    Args:
        state (str) : The profiling state, which should be one of 'CPU', 'GPU'
            or 'All'. 'CPU' means only profiling CPU; 'GPU' means profiling
            both CPU and GPU; 'All' means profiling both CPU and GPU, and 
            generates timeline as well.
        sorted_key (str, optional) : The order of profiling results, which 
            should be one of None, 'calls', 'total', 'max', 'min' or 'ave'.
            Default is None, means the profiling results will be printed
            in the order of first end time of events.
            The `calls` means sorting by the number of calls.
            The `total` means sorting by the total execution time.
            The `max` means sorting by the maximum execution time.
            The `min` means sorting by the minimum execution time.
            The `ave` means sorting by the average execution time.
        profile_path (str, optional) : If state == 'All', it will generate timeline,
            and write it into `profile_path`. The default profile_path is `/tmp/profile`. 
        tracer_option (str, optional) : tracer_option can be one of ['Default', 'OpDetail', 'AllOpDetail'], it
            can control the profile level and print the different level profile result. `Default` option print 
            the different Op type profiling result and the `OpDetail` option print the detail profiling 
            result of different op types such as compute and data transform, `AllOpDetail` option 
            print the detail profiling result of different op name same as `OpDetail`.

    Raises:
        ValueError: If `state` is not in ['CPU', 'GPU', 'All']. If `sorted_key` is
            not in ['calls', 'total', 'max', 'min', 'ave'].

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.profiler as profiler
            import numpy as np

            epoc = 8
            dshape = [4, 3, 28, 28]
            data = fluid.data(name='data', shape=[None, 3, 28, 28], dtype='float32')
            conv = fluid.layers.conv2d(data, 20, 3, stride=[1, 1], padding=[1, 1])

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            with profiler.profiler('CPU', 'total', '/tmp/profile', 'Default') as prof:
                for i in range(epoc):
                    input = np.random.random(dshape).astype('float32')
                    exe.run(fluid.default_main_program(), feed={'data': input})

    Examples Results:

        .. code-block:: text

            #### Examples Results ####
            #### 1) sorted_key = 'total', 'calls', 'max', 'min', 'ave' ####
            # The only difference in 5 sorted_key results is the following sentence: 
            # "Sorted by number of xxx in descending order in the same thread."
            # The reason is that in this example, above 5 columns are already sorted.
            ------------------------->     Profiling Report     <-------------------------

            Place: CPU
            Time unit: ms
            Sorted by total time in descending order in the same thread
            #Sorted by number of calls in descending order in the same thread
            #Sorted by number of max in descending order in the same thread
            #Sorted by number of min in descending order in the same thread
            #Sorted by number of avg in descending order in the same thread

            Event                       Calls       Total       Min.        Max.        Ave.        Ratio.
            thread0::conv2d             8           129.406     0.304303    127.076     16.1758     0.983319
            thread0::elementwise_add    8           2.11865     0.193486    0.525592    0.264832    0.016099
            thread0::feed               8           0.076649    0.006834    0.024616    0.00958112  0.000582432

            #### 2) sorted_key = None  ####
            # Since the profiling results are printed in the order of first end time of Ops,
            # the printed order is feed->conv2d->elementwise_add 
            ------------------------->     Profiling Report     <-------------------------

            Place: CPU
            Time unit: ms
            Sorted by event first end time in descending order in the same thread

            Event                       Calls       Total       Min.        Max.        Ave.        Ratio.
            thread0::feed               8           0.077419    0.006608    0.023349    0.00967738  0.00775934
            thread0::conv2d             8           7.93456     0.291385    5.63342     0.99182     0.795243
            thread0::elementwise_add    8           1.96555     0.191884    0.518004    0.245693    0.196998
    """
    start_profiler(state, tracer_option)
    API_info_summary()
    try:
        yield
    finally:
        stop_profiler(sorted_key, profile_path)
