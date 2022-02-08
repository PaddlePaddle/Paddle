#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import subprocess
import six
import sys
import argparse
from argparse import ArgumentParser, REMAINDER

__all__ = ['perf_analyse']


def _parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser()
    base_group = parser.add_argument_group("Base Parameters")

    base_group.add_argument(
        "--tool",
        type=str,
        default="nsys",
        choices=["nsys"],
        help="Tools for analyzing performance. Currently only supports nsys. Default is nsys."
    )
    base_group.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["model", "op", "all"],
        help="Select the mode to analyze. [model/op/all] mode can be selected. Default is all."
    )
    base_group.add_argument(
        "--profile_start_step",
        type=int,
        help="The number of step to start using profile to analyze performance.")
    base_group.add_argument(
        "--profile_end_step",
        type=int,
        help="The number of step to end using profile to analyze performance.")
    base_group.add_argument(
        "running_script",
        type=str,
        help="The full path to the program/script to analyze dygraph scheduling performance, followed by all the arguments for the running script."
    )
    base_group.add_argument('running_script_args', nargs=REMAINDER)

    nsys_group = parser.add_argument_group("Nsys Parameters")
    # for nsight system
    nsys_group.add_argument(
        "-o",
        "--output",
        type=str,
        default="tmp",
        help="Output report filename. Default is tmp.{qdrep,sqlite}")
    nsys_group.add_argument(
        "-f",
        "--force_overwrite",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to overwrite the log with the same filename. Can be true or false. Default is true."
    )

    return parser.parse_args()


def _parse_string(value):
    # PY2     : PY3
    # unicode : str
    # str     : bytes
    if six.PY3:
        return value
    else:
        return value.encode("utf-8") if isinstance(value, unicode) else value


def _run_command(command, shell=True):
    print("run command: %s" % command)
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell)

    exit_code = None
    stdout = ''
    line = ''
    while exit_code is None or line:
        exit_code = p.poll()
        line = p.stdout.readline().decode('utf-8')
        stdout += line

    return stdout, exit_code


def _nsys_cmd(cmd, args):
    return _run_command(
        "nsys profile -t cuda,nvtx --stats true -o {}.qdrep --capture-range=cudaProfilerApi --stop-on-range-end=true --force-overwrite {} {}".
        format(args.output, args.force_overwrite, cmd))


class NsightRunnerForModelAnalyse(object):
    """
    Use Nsight System tool to analyse performance of Model.
    """

    def run(self, stdout, profile_start_step, profile_end_step):
        """
        parse logs to analyse performance and print the report.
        """
        parse_status, scheduling_time_dict = self.parse_logs(
            stdout.split("\n"), profile_start_step, profile_end_step)
        if parse_status:
            self._print_scheduling_time(scheduling_time_dict)
            return
        print("Parse Error:\n {}".format(stdout))

    def _parse_gpu_time(self, line):
        infos = line.strip().split()
        percent = float(infos[0].replace("%", "")) * 0.01
        gpu_time = float(infos[1].replace(",", "")) * 1E-6
        return gpu_time / percent

    def parse_logs(self, logs, profile_start_step, profile_end_step):
        kernel_line_from = None
        kernel_line_to = None
        memcpy_line_from = None
        memcpy_line_to = None
        nvtx_line_from = None
        total_step_time = 0.0
        step_count = 0
        num_step = profile_end_step - profile_start_step

        scheduling_time_dict = {}

        for i in range(len(logs)):
            line = _parse_string(logs[i])
            if "CUDA Kernel Statistics:" in line:
                kernel_line_from = i
                for j in range(i + 2, len(logs)):
                    if logs[j] == "":
                        kernel_line_to = j
                        break
            if "CUDA Memory Operation Statistics (by time):" in line:
                memcpy_line_from = i
                for j in range(i + 2, len(logs)):
                    if logs[j] == "":
                        memcpy_line_to = j
                        break
            if "NVTX Push-Pop Range Statistics:" in line:
                nvtx_line_from = i
                break

        parse_status = False
        kernel_gpu_time = 0.0
        if kernel_line_from is not None and kernel_line_to is not None:
            kernel_gpu_time = self._parse_gpu_time(logs[kernel_line_from + 4])

        memcpy_gpu_time = 0.0
        if memcpy_line_from is not None and memcpy_line_to is not None:
            memcpy_gpu_time = self._parse_gpu_time(logs[memcpy_line_from + 4])

        total_gpu_time = kernel_gpu_time + memcpy_gpu_time
        scheduling_time_dict['cuda_avg_time'] = total_gpu_time / num_step
        scheduling_time_dict[
            'cuda_kernel_avg_time'] = kernel_gpu_time / num_step
        scheduling_time_dict[
            'cuda_memory_avg_time'] = memcpy_gpu_time / num_step

        # get step_time
        for i in range(nvtx_line_from, len(logs)):
            line = _parse_string(logs[i])
            infos = line.strip().split()
            if not infos:
                continue
            nvtx_range_type = infos[-1]

            # step time
            if nvtx_range_type.isdigit() and int(
                    nvtx_range_type) > profile_start_step and int(
                        nvtx_range_type) < profile_end_step - 1:
                step_count += 1
                step_time = float(infos[1].replace(",", ""))
                total_step_time += step_time

        if step_count:
            scheduling_time_dict[
                'step_time'] = total_step_time / step_count * 1E-6
            scheduling_time_dict['blank_time'] = scheduling_time_dict[
                'step_time'] - scheduling_time_dict['cuda_avg_time']
            scheduling_time_dict['percentage_of_blanks'] = scheduling_time_dict[
                'blank_time'] * 100 / scheduling_time_dict['step_time']
        else:
            scheduling_time_dict['step_time'] = 0.0
            scheduling_time_dict['blank_time'] = 0.0
            scheduling_time_dict['percentage_of_blanks'] = 0.0
        parse_status = True

        return parse_status, scheduling_time_dict

    def _print_scheduling_time(self, time_dict):
        print('\n')
        print('{:*^80}'.format('Model Dygraph Scheduling Profiling Report'))
        print('Time unit: ms\n')

        print('{:=^80}'.format('Step'))
        print('{:70}{:.6}'.format('average time in a step', time_dict[
            'step_time']))
        print('{:=^80}'.format('CUDA'))
        print('{:70}{:.6}'.format('cuda kernel time in a step', time_dict[
            'cuda_kernel_avg_time']))
        print('{:70}{:.6}'.format('cuda memcpy time in a step', time_dict[
            'cuda_memory_avg_time']))
        print('{:-^80}'.format(''))
        print('{:70}{:.6}'.format('average time on cuda side in a step',
                                  time_dict['cuda_avg_time']))
        print('{:=^80}'.format('Blank'))
        print('{:70}{:.6}'.format('blank time in a step', time_dict[
            'blank_time']))
        print('{:70}{:.6}'.format('percentage of blank time (%)', time_dict[
            'percentage_of_blanks']))
        print('\n')


class NsightRunnerForOpAnalyse(object):
    """
    Use Nsight System tool to analyse performance of OP.
    """

    def run(self, stdout, profile_start_step, profile_end_step):
        """
        parse logs to analyse performance and print the report.
        """
        parse_status, op_type_list, scheduling_time_dict = self.parse_logs(
            stdout.split("\n"), profile_start_step, profile_end_step)
        if parse_status:
            self._print_scheduling_time(op_type_list, scheduling_time_dict)
            return
        print("Parse Error:\n {}".format(stdout))

    def _to_float(self, s):
        return float(s.replace(',', ''))

    def _calculate_avg_time_per_op(self, l):
        """
        Within a step, the same OP may be executed multiple times. When the information
         within the OP is analyzed, each OP needs to be statistics separately.
        """
        total_time = self._to_float(l[1])
        max_time = self._to_float(l[5])
        num_calls = self._to_float(l[2]) - 1
        return (total_time - max_time) / num_calls

    def _calculate_avg_time_per_step(self, l, num_step):
        """
        Within a step, the same OP may be executed multiple times. When the influence
         of this OP to the entire step needs to be analysed, the OP needs to be processed
         as a whole in a step. 
        """
        # The same op may appear multiple times within a step.
        total_time = self._to_float(l[1])
        max_time = self._to_float(l[5])
        return (total_time - max_time) / (num_step - 1)

    def _calculate_scheduling_time(self, outside_time, inside_time):
        if outside_time and inside_time:
            return round(outside_time - inside_time, 2)
        return None

    def parse_logs(self, logs, profile_start_step, profile_end_step):
        flag_nvtx_time_start = False
        parse_status = False
        nvtx_time_start_step = 0
        total_step_time = 0.0
        total_op_call_time_per_step = 0.0
        # num step of using profile
        num_step = profile_end_step - profile_start_step
        # Profile data in start_step and end_step may be not correct,
        # so we need to select some reliable data. Number of reliable
        # step data is step_count.
        step_count = 0

        op_type_list = []
        # scheduling time:
        # op_type pybind_imperative_func (imperative_avg_time)
        # op_type (fwd_trace_op_avg_time)
        # op_type compute (fwd_op_compute_avg_time)
        # op_type_grad (bwd_trace_op_avg_time)
        # op_type_grad compute (bwd_op_compute_avg_time)
        _nvtx_meta_data_dict = {}
        scheduling_time_dict = {}

        # get the op_type counted in the profile.
        # get the scheduling list that needs to be analyse.
        for i in range(len(logs)):
            line = _parse_string(logs[i])
            if flag_nvtx_time_start:
                infos = line.strip().split()
                if not infos:
                    continue
                nvtx_range_type = infos[-1]
                if nvtx_range_type == 'pybind_imperative_func' or nvtx_range_type == 'compute':
                    op_type = infos[-2]
                    if op_type not in op_type_list and '_grad' not in op_type:
                        op_type_list.append(op_type)
                        _nvtx_meta_data_dict[op_type +
                                             ' pybind_imperative_func'] = None
                        _nvtx_meta_data_dict[op_type] = None
                        _nvtx_meta_data_dict[op_type + ' compute'] = None
                        _nvtx_meta_data_dict[op_type + '_grad'] = None
                        _nvtx_meta_data_dict[op_type + '_grad compute'] = None
            if not flag_nvtx_time_start and 'NVTX Push-Pop Range Statistics:' in line:
                flag_nvtx_time_start = True
                nvtx_time_start_step = i

        # parse report to get meta scheduling time
        for i in range(nvtx_time_start_step, len(logs)):
            line = _parse_string(logs[i])
            infos = line.strip().split()
            if not infos:
                continue
            nvtx_range_type = infos[-1]
            if nvtx_range_type == 'pybind_imperative_func' or nvtx_range_type == 'compute':
                nvtx_range_type = infos[-2] + ' ' + nvtx_range_type

            # step time
            if nvtx_range_type.isdigit() and int(
                    nvtx_range_type) > profile_start_step and int(
                        nvtx_range_type) < profile_end_step - 1:
                step_count += 1
                step_time = self._to_float(infos[1])
                total_step_time += step_time

            if nvtx_range_type in _nvtx_meta_data_dict:
                avg_time = self._calculate_avg_time_per_op(infos)
                _nvtx_meta_data_dict[nvtx_range_type] = round(avg_time, 2)

                if '_grad' in nvtx_range_type and 'compute' not in nvtx_range_type or 'pybind_imperative_func' in nvtx_range_type:
                    total_op_call_time_per_step += self._calculate_avg_time_per_step(
                        infos, num_step)

        # analyse scheduling time
        scheduling_time_dict['step_time'] = round(
            total_step_time / step_count, 2) if step_count != 0 else None
        scheduling_time_dict['op_call_time_per_step'] = round(
            total_op_call_time_per_step, 2)
        scheduling_time_dict[
            'python_call_time'] = self._calculate_scheduling_time(
                scheduling_time_dict['step_time'],
                scheduling_time_dict['op_call_time_per_step'])
        for op_type in op_type_list:
            tmp_op_time_dict = {}
            tmp_op_time_dict['imperative_avg_time'] = _nvtx_meta_data_dict[
                op_type + ' pybind_imperative_func']
            tmp_op_time_dict['fwd_trace_op_avg_time'] = _nvtx_meta_data_dict[
                op_type]
            tmp_op_time_dict['fwd_op_compute_avg_time'] = _nvtx_meta_data_dict[
                op_type + ' compute']
            tmp_op_time_dict['bwd_trace_op_avg_time'] = _nvtx_meta_data_dict[
                op_type + '_grad']
            tmp_op_time_dict['bwd_op_compute_avg_time'] = _nvtx_meta_data_dict[
                op_type + '_grad compute']

            tmp_op_time_dict[
                'imperative_call_time'] = self._calculate_scheduling_time(
                    tmp_op_time_dict['imperative_avg_time'],
                    tmp_op_time_dict['fwd_trace_op_avg_time'])
            tmp_op_time_dict[
                'fwd_trace_op_call_time'] = self._calculate_scheduling_time(
                    tmp_op_time_dict['fwd_trace_op_avg_time'],
                    tmp_op_time_dict['fwd_op_compute_avg_time'])
            tmp_op_time_dict[
                'bwd_trace_op_call_time'] = self._calculate_scheduling_time(
                    tmp_op_time_dict['bwd_trace_op_avg_time'],
                    tmp_op_time_dict['bwd_op_compute_avg_time'])

            scheduling_time_dict[op_type] = tmp_op_time_dict

        parse_status = True
        return parse_status, op_type_list, scheduling_time_dict

    def _print_scheduling_time(self, op_type_list, time_dict):
        print('\n')
        print('{:*^80}'.format('OP Dygraph Scheduling Profiling Report'))
        print('Time unit: ns\n')
        print('{:^70}  {:^10}'.format('dygraph scheduling process', 'time'))

        for op_type in op_type_list:
            print('\n')
            print('{:=^80}'.format(op_type + ' op'))
            print('{:^80}'.format('[Forward]'))
            print('{:70}'.format(
                'scheduling time of [imperative_op method in Pybind]'),
                  time_dict[op_type]['imperative_call_time'])
            print('{:70}'.format(
                'scheduling time of [TraceOP] other than Run OP'),
                  time_dict[op_type]['fwd_trace_op_call_time'])
            print('{:70}'.format(
                'scheduling time of [OP Compute method] when OP is executed'),
                  time_dict[op_type]['fwd_op_compute_avg_time'])
            print('{:-^80}'.format(''))
            print('{:70}'.format('overall scheduling time of [Forward OP]'),
                  time_dict[op_type]['imperative_avg_time'])
            print('{:^80}'.format('[Backward]'))
            print('{:70}'.format(
                'scheduling time of [TraceOP] other than Run Grad OP'),
                  time_dict[op_type]['bwd_trace_op_call_time'])
            print('{:70}'.format(
                'scheduling time of [OP Compute method] when Grad OP is executed'
            ), time_dict[op_type]['bwd_op_compute_avg_time'])
            print('{:-^80}'.format(''))
            print('{:70}'.format('overall scheduling time of [Backward OP]'),
                  time_dict[op_type]['bwd_trace_op_avg_time'])

        print('\n')
        print('{:=^80}'.format('Summary'))
        print('{:70}'.format(
            'overall scheduling time of [Forward and Backward OP]'),
              time_dict['op_call_time_per_step'])
        print('{:70}'.format(
            '[Python API Call Time] and Overhead of [Pybind Binding C++ OP] etc.'
        ), time_dict['python_call_time'])
        print('{:70}'.format('average time for a [Step]'),
              time_dict['step_time'])
        print('\n')


def perf_analyse():
    """
    Automatically analyze dygraph scheduling performance ``python -m paddle.fluid.perf_analyse``.

    Used to analyze the scheduling overhead during dygraph execution from the report generated by
     the profile tool. Currently only supports ``Nsight System`` performance analysis tool. Make sure
     ``Nsignt System`` tool has been installed on your machine.
    
    Usage:
        .. code-block:: bash

            python -m paddle.fluid.perf_analyse [-h] [--tool {nsys}] [--mode {model,op,all}]
                                        [--profile_start_step PROFILE_START_STEP]
                                        [--profile_end_step PROFILE_END_STEP] [-o OUTPUT]
                                        [-f {true,false}]
                                        running_script ...
    
    Base Parameters:
        - ``--tool``: Tools used to analyze performance. Currently only supports ``Nsight System``
        performance analysis tool. Default is ``nsys``.

        - ``--mode``: Select the mode to analyze. [model/op/all] mode can be selected. Default is ``all``.
        
        -- ``--profile_start_step``: The number of step to start using profile to analyze performance.
        Must be the same as the step set using profile in the model program/script. To analyse more
        accurately, ``profile_end_step`` must be 4 greater than ``profile_start_step``.

        -- ``--profile_end_step``: The number of step to end using profile to analyze performance.
        Must be the same as the step set using profile in the model program/script. To analyse more
        accurately, ``profile_end_step`` must be 4 greater than ``profile_start_step``.
    
    Nsys Parameters:
        -- ``--output`` or ``-o``: Output report filename. Default is tmp.{qdrep,sqlite}.

        -- ``--force_overwrite`` or ``-f``: Whether to overwrite the log with the same filename. Can
        be true or false. Default is true.
    
    Returns:
        If ``model`` mode is selected, the time and proportion of the scheduling overhead (blank
         at the cuda execution side) during the model execution process will be printed out. 
        If ``op`` mode is selected, the scheduling overhead of each stage in the op execution process
         will be printed out. 
        If ``all`` mode is selected, all information including `model` and `op` mode will be printed
         out.
    

    Examples （The execution script is as follows, the script filename is ``test_matmul.py``）:

        .. code-block:: python

            import paddle
            import numpy as np
            from paddle.fluid import core
            import sys

            input1_np = np.random.random([2,2]).astype('float32')
            input2_np = np.random.random([2,2]).astype('float32')

            input1 = paddle.to_tensor(input1_np)
            input2 = paddle.to_tensor(input2_np)

            for i in range(500):
                if i == 10:
                    core.nvprof_start()
                    core.nvprof_enable_record_event()
                    core.nvprof_nvtx_push(str(i))
                if i == 110:
                    paddle.fluid._cuda_synchronize(paddle.fluid.CUDAPlace(0))
                    core.nvprof_nvtx_pop()
                    core.nvprof_stop()
                    sys.exit()
                if i > 10 and i < 110:
                    core.nvprof_nvtx_pop()
                    core.nvprof_nvtx_push(str(i))
                
                m = paddle.matmul(input1, input2)
                n = input1 + m
    

    Examples 1 (model mode):
        .. code-block:: bash

            python -m paddle.fluid.perf_analyse --mode model --profile_start_step 10 --profile_end_step 110 test_matmul.py

            
            # The following information is output: 

            run command: nsys profile -t cuda,nvtx --stats true -o tmp.qdrep --capture-range=cudaProfilerApi --stop-on-range-end=true --force-overwrite true python test_matmul.py


            *******************Model Dygraph Scheduling Profiling Report********************
            Time unit: ms

            ======================================Step======================================
            average time in a step                                                0.122781
            ======================================CUDA======================================
            cuda kernel time in a step                                            0.00339739
            cuda memcpy time in a step                                            0.0
            --------------------------------------------------------------------------------
            average time on cuda side in a step                                   0.00339739
            =====================================Blank======================================
            blank time in a step                                                  0.119383
            percentage of blank time (%)                                          97.233


    Examples 2 (op mode):
        .. code-block:: bash

            python -m paddle.fluid.perf_analyse --mode op --profile_start_step 10 --profile_end_step 110 test_matmul.py

            
            # The following information is output:

            run command: nsys profile -t cuda,nvtx --stats true -o tmp.qdrep --capture-range=cudaProfilerApi --stop-on-range-end=true --force-overwrite true python test_matmul.py


            *********************OP Dygraph Scheduling Profiling Report*********************
            Time unit: ns

                                dygraph scheduling process                           time


            ==================================matmul_v2 op==================================
                                            [Forward]
            scheduling time of [imperative_op method in Pybind]                    11956.07
            scheduling time of [TraceOP] other than Run OP                         21632.19
            scheduling time of [OP Compute method] when OP is executed             36613.59
            --------------------------------------------------------------------------------
            overall scheduling time of [Forward OP]                                70201.85
                                            [Backward]
            scheduling time of [TraceOP] other than Run Grad OP                    None
            scheduling time of [OP Compute method] when Grad OP is executed        None
            --------------------------------------------------------------------------------
            overall scheduling time of [Backward OP]                               None


            ===============================elementwise_add op===============================
                                            [Forward]
            scheduling time of [imperative_op method in Pybind]                    12921.39
            scheduling time of [TraceOP] other than Run OP                         20388.43
            scheduling time of [OP Compute method] when OP is executed             31837.75
            --------------------------------------------------------------------------------
            overall scheduling time of [Forward OP]                                65147.57
                                            [Backward]
            scheduling time of [TraceOP] other than Run Grad OP                    None
            scheduling time of [OP Compute method] when Grad OP is executed        None
            --------------------------------------------------------------------------------
            overall scheduling time of [Backward OP]                               None


            ====================================Summary=====================================
            overall scheduling time of [Forward and Backward OP]                   135349.41
            [Python API Call Time] and Overhead of [Pybind Binding C++ OP] etc.    24619.26
            average time for a [Step]                                              159968.67


    Examples 3 (all mode):
        .. code-block:: bash

            python -m paddle.fluid.perf_analyse --mode all --profile_start_step 10 --profile_end_step 110 test_matmul.py

            
            # output all information including `model` and `op` mode 

    """
    args = _parse_args()
    if not args.profile_start_step:
        raise ValueError(
            "profile_start_step must be set manually and is the same as the profile step set in the script."
        )
    if not args.profile_end_step:
        raise ValueError(
            "profile_end_step must be set manually and is the same as the profile step set in the script."
        )
    if args.profile_end_step - args.profile_start_step < 4:
        raise ValueError(
            "profile_end_step must be 4 greater than profile_start_step.")
    if args.tool == 'nsys':
        cmd = "{} {} {}".format(sys.executable, args.running_script,
                                " ".join(args.running_script_args))
        stdout, exit_code = _nsys_cmd(cmd, args)
        if exit_code == 0:
            if args.mode == 'model' or args.mode == 'all':
                runner = NsightRunnerForModelAnalyse()
                runner.run(stdout, args.profile_start_step,
                           args.profile_end_step)
            if args.mode == 'op' or args.mode == 'all':
                runner = NsightRunnerForOpAnalyse()
                runner.run(stdout, args.profile_start_step,
                           args.profile_end_step)
        else:
            print("Running Error:\n {}".format(stdout))


if __name__ == "__main__":
    perf_analyse()
