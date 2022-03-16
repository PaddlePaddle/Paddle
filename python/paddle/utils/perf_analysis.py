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

__all__ = []


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
        "--task",
        type=str,
        # default="all",
        # choices=["model", "op", "all"],
        default="op",
        choices=["op"],
        help="Select the task to analyze. [model/op/all] task can be selected. Default is all."
    )
    base_group.add_argument(
        "--mode",
        type=str,
        default="fluid",
        # choices=["fluid", "eager", "final"],
        choices=["fluid", "eager"],
        help="Select the mode to analyze. [fluid/eager/final] mode can be selected. Default is fluid."
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
        """
        parse logs to analyse performance.
        """
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

    def run(self, stdout, mode, profile_start_step, profile_end_step):
        """
        parse logs to analyse performance and print the report.
        """
        parse_status, op_type_list, scheduling_time_dict = self.parse_logs(
            stdout.split("\n"), mode, profile_start_step, profile_end_step)
        if parse_status:
            # self._print_scheduling_time(op_type_list, scheduling_time_dict)
            self._print_nvtx_time(op_type_list, scheduling_time_dict)
            return
        print("Parse Error:\n {}".format(stdout))

    def _preprocess_logs(self, logs, op_type_list, _nvtx_meta_data_dict):
        """
        get the op_type counted in the profile.
        get the scheduling list that needs to be analyse.
        """
        flag_nvtx_time_start = False
        nvtx_time_start_step = 0
        # init_map = {'ncalls': 0, 'tottime': 0.0, 'tot_percall': 0.0, 'cumtime': 0.0, 'cum_percall': 0.0}

        for i in range(len(logs)):
            line = _parse_string(logs[i])
            if flag_nvtx_time_start:
                infos = line.strip().split()
                if not infos:
                    continue
                nvtx_range_type = infos[-1]
                if nvtx_range_type in ['pybind_imperative_func', 'compute']:
                    op_type = infos[-2]
                    if op_type not in op_type_list and '_grad' not in op_type:
                        op_type_list.append(op_type)
                        for record_type in [
                                ' pybind_imperative_func', ' dygraph',
                                ' infer_shape', ' compute', ' node_creation',
                                '_grad grad_node', '_grad infer_shape',
                                '_grad compute'
                        ]:
                            _nvtx_meta_data_dict[op_type + record_type] = {
                                'ncalls': 0,
                                'tottime': 0.0,
                                'tot_percall': 0.0,
                                'cumtime': 0.0,
                                'cum_percall': 0.0
                            }
            if not flag_nvtx_time_start and 'NVTX Push-Pop Range Statistics:' in line:
                flag_nvtx_time_start = True
                nvtx_time_start_step = i

        _nvtx_meta_data_dict['Step'] = {
            'ncalls': 0,
            'tottime': 0.0,
            'tot_percall': 0.0,
            'cumtime': 0.0,
            'cum_percall': 0.0
        }
        _nvtx_meta_data_dict['backward'] = {
            'ncalls': 0,
            'tottime': 0.0,
            'tot_percall': 0.0,
            'cumtime': 0.0,
            'cum_percall': 0.0
        }
        _nvtx_meta_data_dict['Accumulation grad_node'] = {
            'ncalls': 0,
            'tottime': 0.0,
            'tot_percall': 0.0,
            'cumtime': 0.0,
            'cum_percall': 0.0
        }

        return nvtx_time_start_step

    def _to_float(self, s):
        return float(s.replace(',', ''))

    def _calculate_avg_time_per_op(self, l):
        """
        Within a step, the same OP may be executed multiple times. When the information
         within the OP is analyzed, each OP needs to be statistics separately.
        """
        total_time = self._to_float(l[1]) * 1E-6
        max_time = self._to_float(l[5]) * 1E-6
        num_calls = self._to_float(l[2]) - 1
        return (total_time - max_time) / num_calls

    def _calculate_avg_time_per_step(self, l, num_step):
        """
        Within a step, the same OP may be executed multiple times. When the influence
         of this OP to the entire step needs to be analysed, the OP needs to be processed
         as a whole in a step. 
        """
        # The same op may appear multiple times within a step.
        total_time = self._to_float(l[1]) * 1E-6
        max_time = self._to_float(l[5]) * 1E-6
        return (total_time - max_time) / (num_step - 1)

    def _calculate_time(self, l, num_step):
        total_time = self._to_float(l[1]) * 1E-6
        max_time = self._to_float(l[5]) * 1E-6
        num_calls = self._to_float(l[2]) - 1
        # calculate result
        ncalls = self._to_float(l[2]) / num_step
        tot_percall = (total_time - max_time) / num_calls
        tottime = tot_percall * ncalls
        return ncalls, tottime, tot_percall

    def _calculate_scheduling_time(self, outside_time, inside_time):
        """
        make sure that neither outside_time nor inside_time is None.
        """
        if outside_time and inside_time:
            return round(outside_time - inside_time, 6)
        return None

    def _get_scheduling_time_from_meta_data(self, op_type, meta_data_dict):
        tmp_op_time_dict = {}
        tmp_op_time_dict['imperative_avg_time'] = meta_data_dict[
            op_type + ' pybind_imperative_func']
        tmp_op_time_dict['fwd_trace_op_avg_time'] = meta_data_dict[op_type]
        tmp_op_time_dict['fwd_op_compute_avg_time'] = meta_data_dict[op_type +
                                                                     ' compute']
        tmp_op_time_dict['bwd_trace_op_avg_time'] = meta_data_dict[op_type +
                                                                   '_grad']
        tmp_op_time_dict['bwd_op_compute_avg_time'] = meta_data_dict[
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
        return tmp_op_time_dict

    def parse_logs(self, logs, mode, profile_start_step, profile_end_step):
        """
        parse logs to analyse performance.
        """
        parse_status = False
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

        # preprocess logs to get op_type appeared in logs and
        # initialize data in _nvtx_meta_data_dict to None.
        nvtx_time_start_step = self._preprocess_logs(logs, op_type_list,
                                                     _nvtx_meta_data_dict)

        # parse report to get meta scheduling time
        for i in range(nvtx_time_start_step, len(logs)):
            line = _parse_string(logs[i])
            infos = line.strip().split()
            if not infos:
                continue
            nvtx_range_type = infos[-1]
            if nvtx_range_type in [
                    'pybind_imperative_func', 'dygraph', 'infer_shape',
                    'compute', 'node_creation'
            ]:
                nvtx_range_type = infos[-2] + ' ' + nvtx_range_type
            if nvtx_range_type == 'grad_node':
                if mode == 'eager':
                    if 'GradNodeAccumulation' in infos[-2]:
                        nvtx_range_type = 'Accumulation grad_node'
                    elif 'GradNode' in infos[-2]:
                        nvtx_range_type = infos[-2].strip().split('GradNode')[
                            1] + '_grad grad_node'
                elif mode == 'fluid':
                    nvtx_range_type = infos[-2] + ' ' + nvtx_range_type
            if nvtx_range_type == 'trace_op' and mode == 'fluid':
                nvtx_range_type = infos[-2] + ' dygraph'

            # step time
            if nvtx_range_type.isdigit() and int(
                    nvtx_range_type) > profile_start_step and int(
                        nvtx_range_type) < profile_end_step - 1:
                step_count += 1
                step_time = self._to_float(infos[1]) * 1E-6
                total_step_time += step_time

            # nvtx time
            if nvtx_range_type in _nvtx_meta_data_dict:
                ncalls, tottime, tot_percall = self._calculate_time(infos,
                                                                    num_step)
                _nvtx_meta_data_dict[nvtx_range_type]['ncalls'] = ncalls
                _nvtx_meta_data_dict[nvtx_range_type]['tottime'] = tottime
                _nvtx_meta_data_dict[nvtx_range_type][
                    'tot_percall'] = tot_percall
                # cumtime will be update later
                _nvtx_meta_data_dict[nvtx_range_type]['cumtime'] = tottime
                # avg_time = self._calculate_avg_time_per_op(infos)
                # _nvtx_meta_data_dict[nvtx_range_type] = round(avg_time, 6)

                # if '_grad' in nvtx_range_type and 'compute' not in nvtx_range_type or 'pybind_imperative_func' in nvtx_range_type:
                #     total_op_call_time_per_step += self._calculate_avg_time_per_step(
                #         infos, num_step)

        _nvtx_meta_data_dict['Step']['ncalls'] = 1
        _nvtx_meta_data_dict['Step'][
            'tottime'] = total_step_time / step_count if step_count != 0 else None
        _nvtx_meta_data_dict['Step']['tot_percall'] = _nvtx_meta_data_dict[
            'Step']['tottime']
        _nvtx_meta_data_dict['Step']['cumtime'] = _nvtx_meta_data_dict['Step'][
            'tottime']

        # calculate cumtime
        if _nvtx_meta_data_dict['Step']['ncalls']:
            _nvtx_meta_data_dict['Step']['cumtime'] -= _nvtx_meta_data_dict[
                'backward']['tottime']
        if _nvtx_meta_data_dict['backward']['ncalls']:
            _nvtx_meta_data_dict['backward']['cumtime'] -= _nvtx_meta_data_dict[
                'Accumulation grad_node']['tottime']

        for op_type in op_type_list:
            if _nvtx_meta_data_dict['Step']['ncalls']:
                _nvtx_meta_data_dict['Step']['cumtime'] -= _nvtx_meta_data_dict[
                    op_type + ' pybind_imperative_func']['tottime']
            if _nvtx_meta_data_dict[op_type + ' pybind_imperative_func'][
                    'ncalls']:
                _nvtx_meta_data_dict[op_type + ' pybind_imperative_func'][
                    'cumtime'] -= _nvtx_meta_data_dict[op_type + ' dygraph'][
                        'tottime']
            for record_type in [' infer_shape', ' compute', ' node_creation']:
                if _nvtx_meta_data_dict[op_type + ' dygraph']['ncalls']:
                    _nvtx_meta_data_dict[op_type + ' dygraph'][
                        'cumtime'] -= _nvtx_meta_data_dict[
                            op_type + record_type]['tottime']
            if _nvtx_meta_data_dict['backward']['ncalls']:
                _nvtx_meta_data_dict['backward'][
                    'cumtime'] -= _nvtx_meta_data_dict[
                        op_type + '_grad grad_node']['tottime']
            for record_type in ['_grad infer_shape', '_grad compute']:
                if _nvtx_meta_data_dict[op_type + '_grad grad_node']['ncalls']:
                    _nvtx_meta_data_dict[op_type + '_grad grad_node'][
                        'cumtime'] -= _nvtx_meta_data_dict[
                            op_type + record_type]['tottime']

        for value in _nvtx_meta_data_dict.values():
            if value['ncalls'] > 0:
                value['cum_percall'] = value['cumtime'] / value['ncalls']

        parse_status = True
        return parse_status, op_type_list, _nvtx_meta_data_dict
        # # analyse scheduling time
        # scheduling_time_dict['step_time'] = round(
        #     total_step_time / step_count, 6) if step_count != 0 else None
        # scheduling_time_dict['op_call_time_per_step'] = round(
        #     total_op_call_time_per_step, 6)
        # scheduling_time_dict[
        #     'python_call_time'] = self._calculate_scheduling_time(
        #         scheduling_time_dict['step_time'],
        #         scheduling_time_dict['op_call_time_per_step'])
        # for op_type in op_type_list:
        #     scheduling_time_dict[
        #         op_type] = self._get_scheduling_time_from_meta_data(
        #             op_type, _nvtx_meta_data_dict)

        # parse_status = True
        # return parse_status, op_type_list, scheduling_time_dict

    def _print_nvtx_time(self, op_type_list, time_dict):
        print(
            '\n\nrecord items:  ncalls  tottime(ms)  tot_percall(ms)  cumtime(ms)   cum_percall(ms)'
        )
        info_types = [
            'ncalls', 'tottime', 'tot_percall', 'cumtime', 'cum_percall'
        ]
        # Step
        step_str = '\n\nStep:  '
        for info_type in info_types:
            step_str += ' {:^20}'.format(time_dict['Step'][info_type])
        print(step_str)

        for op_type in op_type_list:
            fwd_op_str = '\n\n    ' + op_type
            fwd_op_str += '\n    pybind_imperative_func:  '
            for info_type in info_types:
                fwd_op_str += ' {:^20}'.format(time_dict[
                    op_type + ' pybind_imperative_func'][info_type])
            fwd_op_str += '\n        dygraph:  '
            for info_type in info_types:
                fwd_op_str += ' {:^20}'.format(time_dict[op_type + ' dygraph'][
                    info_type])
            for record_type in ['infer_shape', 'compute', 'node_creation']:
                fwd_op_str += '\n            ' + record_type + ':  '
                for info_type in info_types:
                    fwd_op_str += ' {:^20}'.format(time_dict[
                        op_type + ' ' + record_type][info_type])
            print(fwd_op_str)

        backward_str = '\n\n    backward: '
        for info_type in info_types:
            backward_str += ' {:^20}'.format(time_dict['backward'][info_type])
        print(backward_str)

        for op_type in op_type_list:
            bwd_op_str = '\n        ' + op_type + '_grad'
            bwd_op_str += '\n            grad_node:  '
            for info_type in info_types:
                bwd_op_str += ' {:^20}'.format(time_dict[
                    op_type + '_grad grad_node'][info_type])
            for record_type in ['infer_shape', 'compute']:
                bwd_op_str += '\n                ' + record_type + ':  '
                for info_type in info_types:
                    bwd_op_str += ' {:^20}'.format(time_dict[
                        op_type + '_grad ' + record_type][info_type])
            print(bwd_op_str)

        bwd_op_str = '\n        Accumulation grad_node'
        bwd_op_str += '\n            Accumulation grad_node:  '
        for info_type in info_types:
            bwd_op_str += ' {:^20}'.format(time_dict['Accumulation grad_node'][
                info_type])
        print(bwd_op_str)

    def _print_scheduling_time(self, op_type_list, time_dict):
        print('\n')
        print('{:*^80}'.format('OP Dygraph Scheduling Profiling Report'))
        print('Time unit: ms\n')
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


def _perf_analysis():
    """
    Automatically analyze dygraph scheduling performance ``python -m paddle.utils.launch_perf_analysis``.

    Used to analyze the scheduling overhead during dygraph execution from the report generated by
     the profile tool. Currently only supports ``Nsight System`` performance analysis tool. Make sure
     ``Nsignt System`` tool has been installed on your machine.
    
    Usage:
        .. code-block:: bash

            python -m paddle.utils.launch_perf_analysis [-h] [--tool {nsys}] [--mode {model,op,all}]
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

            python -m paddle.utils.launch_perf_analysis --mode model --profile_start_step 10 --profile_end_step 110 test_matmul.py

            
            # The following information is output: 

            run command: nsys profile -t cuda,nvtx --stats true -o tmp.qdrep --capture-range=cudaProfilerApi --stop-on-range-end=true --force-overwrite true python test_matmul.py


            *******************Model Dygraph Scheduling Profiling Report********************
            Time unit: ms

            ======================================Step======================================
            average time in a step                                                0.109621
            ======================================CUDA======================================
            cuda kernel time in a step                                            0.00354393
            cuda memcpy time in a step                                            0.0
            --------------------------------------------------------------------------------
            average time on cuda side in a step                                   0.00354393
            =====================================Blank======================================
            blank time in a step                                                  0.106077
            percentage of blank time (%)                                          96.7671


    Examples 2 (op mode):
        .. code-block:: bash

            python -m paddle.utils.launch_perf_analysis --mode op --profile_start_step 10 --profile_end_step 110 test_matmul.py

            
            # The following information is output:

            run command: nsys profile -t cuda,nvtx --stats true -o tmp.qdrep --capture-range=cudaProfilerApi --stop-on-range-end=true --force-overwrite true python test_matmul.py


            *********************OP Dygraph Scheduling Profiling Report*********************
            Time unit: ms

                                dygraph scheduling process                           time


            ==================================matmul_v2 op==================================
                                            [Forward]
            scheduling time of [imperative_op method in Pybind]                    0.008472
            scheduling time of [TraceOP] other than Run OP                         0.015009
            scheduling time of [OP Compute method] when OP is executed             0.023645
            --------------------------------------------------------------------------------
            overall scheduling time of [Forward OP]                                0.047126
                                            [Backward]
            scheduling time of [TraceOP] other than Run Grad OP                    None
            scheduling time of [OP Compute method] when Grad OP is executed        None
            --------------------------------------------------------------------------------
            overall scheduling time of [Backward OP]                               None


            ===============================elementwise_add op===============================
                                            [Forward]
            scheduling time of [imperative_op method in Pybind]                    0.008631
            scheduling time of [TraceOP] other than Run OP                         0.013468
            scheduling time of [OP Compute method] when OP is executed             0.019308
            --------------------------------------------------------------------------------
            overall scheduling time of [Forward OP]                                0.041407
                                            [Backward]
            scheduling time of [TraceOP] other than Run Grad OP                    None
            scheduling time of [OP Compute method] when Grad OP is executed        None
            --------------------------------------------------------------------------------
            overall scheduling time of [Backward OP]                               None


            ====================================Summary=====================================
            overall scheduling time of [Forward and Backward OP]                   0.088533
            [Python API Call Time] and Overhead of [Pybind Binding C++ OP] etc.    0.021088
            average time for a [Step]                                              0.109621


    Examples 3 (all mode):
        .. code-block:: bash

            python -m paddle.utils.launch_perf_analysis --mode all --profile_start_step 10 --profile_end_step 110 test_matmul.py

            
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
            if args.task in ['model', 'all']:
                runner = NsightRunnerForModelAnalyse()
                runner.run(stdout, args.profile_start_step,
                           args.profile_end_step)
            if args.task in ['op', 'all']:
                runner = NsightRunnerForOpAnalyse()
                runner.run(stdout, args.mode, args.profile_start_step,
                           args.profile_end_step)
        else:
            print("Running Error:\n {}".format(stdout))
