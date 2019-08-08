# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
'''
Fluid model analysis tools 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import subprocess
import sys
from collections import OrderedDict
from operator import mul

# Simple logging config
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import paddle.fluid as fluid
from paddle.fluid import debugger
from paddle.fluid import core

# Command arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir", type=str, required=True, help="Model dir path")
parser.add_argument(
    "--input_file", default="", type=str, help="Input datas file path")
parser.add_argument(
    "--topo_file",
    type=str,
    required=True,
    help="Runtime topology order output file path")
parser.add_argument(
    "--tensor_file",
    default="",
    type=str,
    required=True,
    help="Tensor file path")
parser.add_argument(
    "--tensor_names",
    default="",
    type=str,
    help="If tensor_names is not empty, then only this tensors will be compare")
parser.add_argument(
    "--separator",
    default=",",
    type=str,
    help="Deafult separator, use in string split")
parser.add_argument(
    "--output_tensor",
    default=0,
    type=int,
    help="dump fluid runntime tensors or not")
parser.add_argument(
    "--tensor_output_file",
    default="./tensor_output_py",
    type=str,
    help="dump fluid runntime tensors filepath")
parser.add_argument(
    "--tensor_output_length",
    default=-1,
    type=int,
    help="Output tensor data length, dims size will be used if tensor_output_length < 0"
)
parser.add_argument(
    "--only_first",
    default=1,
    type=int,
    help="If only output the first mismatch vars info or not")
parser.add_argument(
    "--output_file",
    default="./diff.txt",
    type=str,
    help="dump diff info filepath")
parser.add_argument(
    "--threshold", default=1e-5, type=float, help="float value diff threshold")


# Help functions
def load_file(filename, delim=None):
    """
    Load file help function
    """
    with open(filename) as fd:
        for line in fd:
            line = line.strip()
            assert len(line) != ""
            if delim:
                line = line.split(delim)
            yield line


class FluidModelExecutor(object):
    """
    A fluid inference model executeor
    """

    def __init__(self, model_dir, input_file):
        self.model_dir = model_dir
        self.place = fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)
        self.scope = fluid.core.Scope()
        self.input_data = self._load_input_file(input_file)

        self.program, self.feed_target_names, self.fetch_targets = self._load_inference_model(
        )

    def infer_var_list(self,
                       arg_names=None,
                       out_data_len=-1,
                       dump_tensor=False,
                       dump_tensor_file=''):
        """
        Get variables' tensor in var_list
        """
        with fluid.scope_guard(self.scope):
            global_block = self.program.global_block()
            feed_list = self._prepare_feed_data(global_block,
                                                self.feed_target_names)
            fetch_targets = self._fetch_tmp_vars(global_block, arg_names)
            results = self.exe.run(program=self.program,
                                   feed=feed_list,
                                   fetch_list=fetch_targets,
                                   return_numpy=False)
            return self._get_results(
                results,
                fetch_targets,
                arg_names=arg_names,
                need_save=dump_tensor,
                save_path=dump_tensor_file,
                out_data_len=out_data_len)

    def draw_graph(self, output_path='./', filename='debug'):
        """
        Draw graph with graphviz
        """
        dot_path = os.path.join([output_path, filename + '.dot'])
        pdf_path = os.path.join([output_path, filename + '.pdf'])
        debugger.draw_block_graphviz(self.program.global_block(), path=dot_path)
        cmd = ["dot", "-Tpdf", dot_path, "-o", pdf_path]
        subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

    def _prepare_feed_data(self, block, feed_target_names):
        feed_dict = dict()

        def fill_data(np_dtype, col, shape):
            if self.input_data:
                input_size = reduce(mul, shape)
                assert len(self.input_data[0]) > col
                data = self.input_data[0][col].split(' ')
                assert len(data) == input_size
                return np.array(
                    map(np_dtype, data), dtype=np_dtype).reshape(shape)
            else:
                return np.ones(shape, dtype=np_dtype)

        # TODO(sangoly): support multiple feed fields 
        assert len(feed_target_names) == 1
        for idx, name in enumerate(feed_target_names):
            var = block.var(name)
            np_shape = list(var.shape)
            # TODO(sangoly): support batch
            if np_shape[0] == -1:
                np_shape[0] = 1
            if var.dtype == core.VarDesc.VarType.INT32:
                feed_dict[name] = fill_data(np.int32, idx, np_shape)
            elif var.dtype == core.VarDesc.VarType.INT64:
                feed_dict[name] = fill_data(np.int64, idx, np_shape)
            elif var.dtype == core.VarDesc.VarType.FP16:
                feed_dict[name] = fill_data(np.float16, idx, np_shape)
            elif var.dtype == core.VarDesc.VarType.FP32:
                feed_dict[name] = fill_data(np.float32, idx, np_shape)
            elif var.dtype == core.VarDesc.VarType.FP64:
                feed_dict[name] = fill_data(np.float64, idx, np_shape)
            else:
                raise TypeError("Data type is not supported")
        return feed_dict

    def _load_input_file(self, input_file=None):
        input_data = []
        if not input_file:
            return input_data
        logger.info("Loading input file %s ..." % input_file)
        for line in load_file(input_file, "\t"):
            input_data.append(line)
        return input_data

    def _load_inference_model(self):
        with fluid.scope_guard(self.scope):
            model_abs_path = os.path.join(self.model_dir, 'model')
            param_abs_path = os.path.join(self.model_dir, 'params')
            if os.path.exists(model_abs_path) and os.path.exists(
                    param_abs_path):
                return fluid.io.load_inference_model(self.model_dir, exe,
                                                     'model', 'params')
            else:
                return fluid.io.load_inference_model(self.model_dir, self.exe)

    def _fetch_tmp_vars(self, block, var_names_list=None):
        fetch_var = block.var('fetch')
        old_fetch_names = set([var.name for var in self.fetch_targets])
        new_fetch_vars = [block.var(name) for name in old_fetch_names]
        i = len(new_fetch_vars)
        if var_names_list is None:
            var_names_list = block.vars.keys()
        for var_name in var_names_list:
            if var_name in old_fetch_names: continue
            new_fetch_vars.append(block.var(var_name))
            block.append_op(
                type='fetch',
                inputs={'X': [var_name]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i})
            i = i + 1
        return new_fetch_vars

    def _get_results(self,
                     results,
                     new_fetch_targets,
                     need_save=False,
                     arg_names=None,
                     save_path='',
                     out_data_len=10):
        res = OrderedDict()
        old_fetch_names = set([var.name for var in self.fetch_targets])
        if need_save:
            out_fd = open(save_path, 'w')
        for result in results:
            idx = results.index(result)
            name = new_fetch_targets[idx].name
            dim = [v if v >= 0 else 1 for v in new_fetch_targets[idx].shape]
            size = min(reduce(mul, dim),
                       out_data_len) if out_data_len > 0 else reduce(mul, dim)
            values = list(np.array(result).flatten())[:size]
            res[name] = {"dim": dim, "values": values}
            if need_save:
                if arg_names and name not in arg_names: continue
                dim_str = '{' + ','.join(map(str, dim)) + '}'
                out_fd.write('\t'.join(
                    [name, dim_str, ' '.join(map(str, values))]) + '\n')
        if need_save:
            out_fd.close()
        return res


class Analyser(object):
    """
    A FLuid model analysis tool
    """

    def __init__(self, args):
        self.args = args
        self.tensors = OrderedDict()
        self.topo = {}
        self.input = []
        logger.info("Loading fluid inference model %s ..." % args.model_dir)
        self.predictor = FluidModelExecutor(args.model_dir, args.input_file)

    def analysis(self):
        """
        Analyser work function
        """
        self._load_topo_file()
        self._load_tensor_file()
        arg_names = self.args.tensor_names.split(',') if self.args.tensor_names != "" \
                                           else self.tensors.keys()
        infer_results = self.predictor.infer_var_list(
            out_data_len=self.args.tensor_output_length,
            arg_names=arg_names,
            dump_tensor=self.args.output_tensor,
            dump_tensor_file=self.args.tensor_output_file)
        if self.args.tensor_names == "":
            self._check_diff_nodes(infer_results)

    def _parse_topo_field(self, field):
        params = [item.split(':')[1].strip() for item in field[1:-1].split(' ')]
        params = [item.split('#') for item in params if item != ""]
        return [item for lst in params for item in lst]

    def _load_topo_file(self):
        if self.args.topo_file == "":
            raise ValueError("Topo file path in empty")
        logger.info("Loading topo file %s ..." % self.args.topo_file)
        for line in load_file(self.args.topo_file, '\t'):
            op_type, inputs, outputs = line
            for name in self._parse_topo_field(outputs):
                if name not in self.topo:
                    self.topo[name] = []
                self.topo[name].append(line)

    def _load_tensor_file(self):
        if self.args.tensor_file == "":
            raise ValueError("Tensor file path in empty")
        logger.info("Loading tensor file %s ..." % args.tensor_file)
        for line in load_file(args.tensor_file, "\t"):
            name, dim, values = line
            dim = map(int, dim[1:-1].split(','))
            values = map(float, values.split(' '))

            dim_size = reduce(mul, dim)
            value_size = len(values)
            assert dim_size == value_size, \
                        "Dim size mismatch with data: %d vs %d" % (dim_size, value_size)

            self.tensors[name] = {"dim": dim, "values": values}

    def _check_diff_nodes(self, results):
        """
        NOTE: The tensor output by c++ debug tool is according to runtime topology order,
              so we can find the first ops (may be one of them) with error results
        """
        assert len(self.tensors) == len(results), \
                "FLuid output tensor'size mismatch with `tensor_file`"
        diff_vars = []
        flag = False
        for k in self.tensors:
            if k not in results:
                raise KeyError("Have not found infer result for `%s`" % k)
            if len(self.tensors[k]['values']) != len(results[k]['values']):
                raise ValueError(
                    "Argname: %s size mismatch with `tensor_file`: %d vs %d" %
                    (k, len(self.tensors[k]['values']),
                     len(results[k]['values'])))
            for i in range(len(self.tensors[k]['values'])):
                if abs(self.tensors[k]['values'][i] - results[k]['values'][
                        i]) > args.threshold:
                    diff_vars.append(k)
                    if args.only_first:
                        flag = True
                    break
            if flag: break
        self._output_diff_nodes(results, diff_vars)

    def _output_diff_nodes(self, results, diff_vars):
        def output_param_info(inputs, outputs, infos, fd):
            def tensor_repr(name):
                return '\t'.join([
                    name, '{' + ','.join(map(str, infos[name]['dim'])) + '}',
                    ' '.join(map(str, infos[name]['values']))
                ])

            for name in self._parse_topo_field(inputs):
                if name not in infos: continue
                fd.write(tensor_repr(name) + '\n')
            for name in self._parse_topo_field(outputs):
                if name not in infos: continue
                fd.write(tensor_repr(name) + '\n')

        if len(diff_vars) == 0:
            logger.info("No diff found. Congratulation!")
            return
        logger.info("Total diff vars: %d" % len(diff_vars))
        with open(self.args.output_file, 'w') as fd:
            for var in diff_vars:
                if var not in self.topo:
                    raise KeyError("%s not in any op's output params, " % var +
                                   "please check your model and input")
                fd.write(
                    '>>>>>>>>>>>>>>>>>>DIFF VARIABLE: %s<<<<<<<<<<<<<<<<<<<\n' %
                    var)
                for idx, (op_type, inputs,
                          outputs) in enumerate(self.topo[var]):
                    op_repr = '\t'.join([op_type, inputs, outputs])
                    logger.info("dump diff info: ------------ %s" % op_repr)
                    fd.write(op_repr + '\n')
                    fd.write(
                        "--------------- Tensor File info ---------------\n")
                    output_param_info(inputs, outputs, self.tensors, fd)
                    fd.write(
                        "--------------- Fluid Tensor info ---------------\n")
                    output_param_info(inputs, outputs, results, fd)
                    fd.write("\n\n")


if __name__ == "__main__":
    args = parser.parse_args()
    analyser = Analyser(args)
    analyser.analysis()
