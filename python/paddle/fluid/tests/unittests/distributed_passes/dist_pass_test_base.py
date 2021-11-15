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

import unittest
import paddle
import os
import random
import sys
import pickle
import shlex
import shutil
import inspect
import numpy as np
from collections import OrderedDict
from paddle.distributed.fleet.launch_utils import run_with_coverage


def prepare_python_path_and_return_module(path):
    dirname, filename = os.path.split(path)
    py_suffix = ".py"
    assert filename.endswith(py_suffix), filename

    env_name = 'PYTHONPATH'
    python_path = env_name
    if python_path:
        paths = [p for p in python_path.split(":") if p]
        if dirname not in paths:
            paths.append(dirname)
        python_path = ":".join(paths)
    else:
        python_path = path
    os.environ[env_name] = python_path
    return filename[:-len(py_suffix)]


def remove_path_if_exists(path):
    if not os.path.exists(path):
        return

    if os.path.isfile(path):
        os.remove(path)
    else:
        shutil.rmtree(path)


# NOTE: only support GPU now
class DistPassTestBase(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        seed = int(os.environ.get('SEED', -1))
        if seed <= 0:
            seed = np.random.randint(low=1, high=1000000, size=[1])[0]
            os.environ['SEED'] = str(seed)
        self.seed = seed
        paddle.seed(self.seed)

        self.rtol = 1e-5
        self.atol = 1e-8
        self.equal_nan = False

        self.init()

    def init(self):
        pass

    def get_model(self, place, **kwargs):
        raise NotImplementedError()

    def apply_passes(self, main_prog, startup_prog):
        raise NotImplementedError()

    def check_main(self, gpus=None, **kwargs):
        no_pass_rets = self._distributed_launch(
            apply_pass=False, gpus=gpus, **kwargs)
        pass_rets = self._distributed_launch(
            apply_pass=True, gpus=gpus, **kwargs)
        self.check_results(no_pass_rets, pass_rets)

    def check_results(self, no_pass_rets, pass_rets):
        self.assertEqual(len(no_pass_rets), len(pass_rets))
        for no_pass_ret, pass_ret in zip(no_pass_rets, pass_rets):
            self.assertEqual(len(no_pass_ret), len(pass_ret))
            for i, (out_var_no_pass,
                    out_var_pass) in enumerate(zip(no_pass_ret, pass_ret)):
                if out_var_no_pass is None:
                    self.assertTrue(out_var_pass is None)
                else:
                    self.assertTrue(
                        np.allclose(
                            out_var_no_pass,
                            out_var_pass,
                            rtol=self.rtol,
                            atol=self.atol,
                            equal_nan=self.equal_nan))

    @classmethod
    def _to_var_names(cls, program, names_or_vars):
        if not isinstance(names_or_vars, (list, tuple)):
            names_or_vars = [names_or_vars]
        ret_var_names = []
        for name_or_var in names_or_vars:
            if isinstance(name_or_var, str):
                ret_var_names.append(name_or_var)
            else:
                ret_var_names.append(name_or_var.name)
        return ret_var_names

    def _run_gpu_main(self, apply_pass, dump_file, **kwargs):
        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        place = paddle.CUDAPlace(gpu_id)
        scope = paddle.static.Scope()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            with paddle.static.scope_guard(scope):
                with paddle.fluid.unique_name.guard():
                    main_prog, startup_prog, inputs, outputs, reader = self.get_model(
                        place, **kwargs)
                    inputs = self._to_var_names(main_prog, inputs)
                    outputs = self._to_var_names(main_prog, outputs)
                    if apply_pass:
                        self.apply_passes(main_prog, startup_prog)

        all_fetch_values = []
        exe = paddle.static.Executor(place)
        with paddle.static.scope_guard(scope):
            exe.run(startup_prog)
            for batch_id, input_data in enumerate(reader()):
                assert len(input_data) == len(inputs), "{} vs {}".format(
                    len(input_data), len(inputs))
                feed = dict(zip(inputs, input_data))
                fetch_values = exe.run(main_prog, feed=feed, fetch_list=outputs)
                if paddle.distributed.get_rank() == 0:
                    output_dict = OrderedDict(zip(outputs, fetch_values))
                    print('batch {}, outputs {}'.format(batch_id, output_dict))
                all_fetch_values.append(fetch_values)
        with open(dump_file, "wb") as f:
            pickle.dump(all_fetch_values, f)

    def _distributed_launch(self, apply_pass, gpus=None, **kwargs):
        if gpus is None:
            num_gpus = paddle.device.cuda.device_count()
            gpus = list(range(num_gpus))
        else:
            num_gpus = len(gpus)

        gpus = ','.join([str(gpu_id) for gpu_id in gpus])

        pid = os.getpid()
        if apply_pass:
            output_dir = "test_with_pass_{}".format(pid)
        else:
            output_dir = "test_without_pass_{}".format(pid)
        remove_path_if_exists(output_dir)
        os.makedirs(output_dir, mode=777)

        input_dump_file = os.path.join(output_dir, 'inputs')
        if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
            run_with_coverage(True)
            coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
        else:
            coverage_args = []

        file_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            with open(input_dump_file, 'wb') as f:
                pickle.dump(kwargs, f)

            cmd = [
                sys.executable,
                "-u",
            ] + coverage_args + [
                "-m",
                "launch",
                "--log_dir",
                output_dir,
                "--gpus",
                gpus,
                os.path.join(file_dir, "pass_run_main.py"),
                "--file_path",
                inspect.getfile(type(self)),
                "--class_name",
                type(self).__name__,
                "--input_file",
                input_dump_file,
                "--output_dir",
                output_dir,
            ] + (["--apply_pass"] if apply_pass else [])
            cmd = [shlex.quote(c) for c in cmd]
            prepare_python_path_and_return_module(__file__)
            exitcode = os.system(' '.join(cmd))
            self.assertEqual(
                exitcode, 0,
                "Pass failed with apply_pass = {}".format(apply_pass))

            results = []
            for i in range(num_gpus):
                dump_file = '{0}/{1}.bin'.format(output_dir, i)
                self.assertTrue(
                    os.path.exists(dump_file),
                    "Pass failed with apply_pass = {}".format(apply_pass))
                with open(dump_file, "rb") as f:
                    results.append(pickle.load(f))
            return results
        finally:
            if int(os.environ.get("DEBUG", 0)) == 0:
                remove_path_if_exists(output_dir)
