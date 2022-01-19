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

__all__ = ['dygraph_analyse']


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
        default="model",
        choices=["model", "op"],
        help="Select the mode to analyze. [model/op] mode can be selected. Default is model."
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


def _parse_string(value):
    import six
    # PY2     : PY3
    # unicode : str
    # str     : bytes
    if six.PY3:
        return value
    else:
        return value.encode("utf-8") if isinstance(value, unicode) else value


class NsightRunnerForOpAnalyse(object):
    """
    Use Nsight System tool to analyse performance of OP.
    """

    def run(self, cmd, args):
        """
        Run program/script.
        """
        stdout, exit_code = self._nsys_cmd_for_op_analyse(cmd, args)
        if exit_code == 0:
            parse_status, op_type_list, scheduling_time_dict = self._parse_logs(
                stdout.split("\n"), args)
            if parse_status:
                self._print_scheduling_time(op_type_list, scheduling_time_dict)
                return
        print("Running Error:\n {}".format(stdout))

    def _nsys_cmd_for_op_analyse(self, cmd, args):
        return _run_command(
            "nsys profile -t cuda,nvtx --stats true -o {}.qdrep --force-overwrite {} {}".
            format(args.output, args.force_overwrite, cmd))

    def _to_float(self, s):
        return float(s.replace(',', ''))

    def _calculate_avg_time_per_op(self, l):
        total_time = self._to_float(l[1])
        max_time = self._to_float(l[5])
        num_calls = self._to_float(l[2]) - 1
        return (total_time - max_time) / num_calls

    def _calculate_avg_time_per_step(self, l, num_step):
        # The same op may appear multiple times within a step.
        total_time = self._to_float(l[1])
        max_time = self._to_float(l[5])
        return (total_time - max_time) / (num_step - 1)

    def _calculate_scheduling_time(self, outside_time, inside_time):
        if outside_time and inside_time:
            return round(outside_time - inside_time, 2)
        return None

    def _parse_logs(self, logs, args):
        flag_nvtx_time_start = False
        parse_status = False
        nvtx_time_start_step = 0
        total_step_time = 0.0
        total_op_call_time_per_step = 0.0
        num_step = args.profile_end_step - args.profile_start_step
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
                    nvtx_range_type) > args.profile_start_step + 1 and int(
                        nvtx_range_type) < args.profile_end_step - 1:
                step_count += 1
                step_time = self._to_float(infos[1])
                total_step_time += step_time

            if nvtx_range_type in _nvtx_meta_data_dict:
                avg_time = self._calculate_avg_time_per_op(infos)
                _nvtx_meta_data_dict[nvtx_range_type] = round(avg_time, 2)
                # print(nvtx_range_type + ' time: ', avg_time)

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
        # print(scheduling_time_dict)
        return parse_status, op_type_list, scheduling_time_dict

    def _print_scheduling_time(self, op_type_list, time_dict):
        print('\n')
        print('{:*^80}'.format('Dygraph Scheduling Profiling Report'))
        print('Time unit: ns\n')
        print('{:^70}  {:^10}'.format('dygraph scheduling process', 'time'))

        for op_type in op_type_list:
            print('\n')
            print('{:-^80}'.format(op_type))
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
            print('{:70}'.format('overall scheduling time of [Forward OP]'),
                  time_dict[op_type]['imperative_avg_time'])
            print('{:^80}'.format('[Backward]'))
            print('{:70}'.format(
                'scheduling time of [TraceOP] other than Run Grad OP'),
                  time_dict[op_type]['bwd_trace_op_call_time'])
            print('{:70}'.format(
                'scheduling time of [OP Compute method] when Grad OP is executed'
            ), time_dict[op_type]['bwd_op_compute_avg_time'])
            print('{:70}'.format('overall scheduling time of [Backward OP]'),
                  time_dict[op_type]['bwd_trace_op_avg_time'])

        print('\n')
        print('{:-^80}'.format(''))
        print('{:70}'.format(
            'overall scheduling time of [Forward and Backward OP]'),
              time_dict['op_call_time_per_step'])
        print('{:70}'.format(
            '[Python API Call Time] and Overhead of [Pybind Binding C++ OP] etc.'
        ), time_dict['python_call_time'])
        print('{:70}'.format('average time for a [Step]'),
              time_dict['step_time'])


def _analyse_model(args):
    cmd = "{} {} {}".format(sys.executable, args.running_script,
                            " ".join(args.running_script_args))
    if args.tool == 'nsys':
        runner = NsightRunnerForModelAnalyse()
        runner.run(cmd, args)


def _analyse_op(args):
    cmd = "{} {} {}".format(sys.executable, args.running_script,
                            " ".join(args.running_script_args))
    if args.tool == 'nsys':
        runner = NsightRunnerForOpAnalyse()
        runner.run(cmd, args)


def dygraph_analyse():
    """
    Used to analyze the scheduling overhead during dygraph execution from the
     report generated by the profile tool.
    """
    args = _parse_args()
    if args.mode == 'model':
        _analyse_model(args)
    elif args.mode == 'op':
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
        _analyse_op(args)


if __name__ == "__main__":
    dygraph_analyse()
