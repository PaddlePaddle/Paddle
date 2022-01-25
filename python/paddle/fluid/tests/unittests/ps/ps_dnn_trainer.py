# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed.fleet.base.role_maker as role_maker
from paddle.distributed.ps.utils.ps_program_builder import *
import paddle.distributed.fleet as fleet
from paddle.fluid.tests.unittests.ps.ps_dnn_model import StaticModel
import argparse
import time
import sys
import paddle
import os
import warnings
import logging
import ast
import numpy as np
import struct

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def is_distributed_env():
    node_role = os.getenv("TRAINING_ROLE")
    logger.info("-- Role: {} --".format(node_role))
    if node_role is None:
        return False
    else:
        return True


class YamlHelper(object):
    def load_yaml(self, yaml_file, other_part=None):
        part_list = ["runner", "hyper_parameters"]
        if other_part:
            part_list += other_part
        running_config = self.get_all_inters_from_yaml(yaml_file, part_list)
        running_config = self.workspace_adapter(running_config)
        return running_config

    def print_yaml(self, config):
        print(self.pretty_print_envs(config))

    def parse_yaml(self, config):
        vs = [int(i) for i in yaml.__version__.split(".")]
        if vs[0] < 5:
            use_full_loader = False
        elif vs[0] > 5:
            use_full_loader = True
        else:
            if vs[1] >= 1:
                use_full_loader = True
            else:
                use_full_loader = False

        if os.path.isfile(config):
            if six.PY2:
                with open(config, 'r') as rb:
                    if use_full_loader:
                        _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
                    else:
                        _config = yaml.load(rb.read())
                    return _config
            else:
                with open(config, 'r', encoding="utf-8") as rb:
                    if use_full_loader:
                        _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
                    else:
                        _config = yaml.load(rb.read())
                    return _config
        else:
            raise ValueError("config {} can not be supported".format(config))

    def get_all_inters_from_yaml(self, file, filters):
        _envs = self.parse_yaml(file)
        all_flattens = {}

        def fatten_env_namespace(namespace_nests, local_envs):
            for k, v in local_envs.items():
                if isinstance(v, dict):
                    nests = copy.deepcopy(namespace_nests)
                    nests.append(k)
                    fatten_env_namespace(nests, v)
                else:
                    global_k = ".".join(namespace_nests + [k])
                    all_flattens[global_k] = v

        fatten_env_namespace([], _envs)
        ret = {}
        for k, v in all_flattens.items():
            for f in filters:
                if k.startswith(f):
                    ret[k] = v
        return ret

    def workspace_adapter(self, config):
        workspace = config.get("workspace")
        for k, v in config.items():
            if isinstance(v, str) and "{workspace}" in v:
                config[k] = v.replace("{workspace}", workspace)
        return config

    def pretty_print_envs(self, envs, header=None):
        spacing = 2
        max_k = 40
        max_v = 45

        for k, v in envs.items():
            max_k = max(max_k, len(k))

        h_format = "    " + "|{{:>{}s}}{}{{:^{}s}}|\n".format(max_k, " " *
                                                              spacing, max_v)
        l_format = "    " + "|{{:>{}s}}{{}}{{:^{}s}}|\n".format(max_k, max_v)
        length = max_k + max_v + spacing

        border = "    +" + "".join(["="] * length) + "+"
        line = "    +" + "".join(["-"] * length) + "+"

        draws = ""
        draws += border + "\n"

        if header:
            draws += h_format.format(header[0], header[1])
        else:
            draws += h_format.format("PaddleRec Benchmark Envs", "Value")

        draws += line + "\n"

        for k, v in sorted(envs.items()):
            if isinstance(v, str) and len(v) >= max_v:
                str_v = "... " + v[-41:]
            else:
                str_v = v

            draws += l_format.format(k, " " * spacing, str(str_v))

        draws += border

        _str = "\n{}\n".format(draws)
        return _str


def get_strategy(config):
    if not is_distributed_env():
        logger.warn(
            "Not Find Distributed env, Change To local train mode. If you want train with fleet, please use [fleetrun] command."
        )
        return None
    sync_mode = config.get("runner.sync_mode")
    assert sync_mode in ["async", "sync", "geo", "heter", "gpubox"]
    if sync_mode == "sync":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = False
    elif sync_mode == "async":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
    elif sync_mode == "geo":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"k_steps": config.get("runner.geo_step")}
    elif sync_mode == "heter":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"heter_worker_device_guard": "gpu"}
    elif sync_mode == "gpubox":
        print("sync_mode = {}".format(sync_mode))
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"use_ps_gpu": 1}

    strategy.trainer_desc_configs = {
        "dump_fields_path": config.get("runner.dump_fields_path", ""),
        "dump_fields": config.get("runner.dump_fields", []),
        "dump_param": config.get("runner.dump_param", []),
        "stat_var_names": config.get("stat_var_names", [])
    }
    print("strategy:", strategy.trainer_desc_configs)

    if config.get("runner.fs_client.uri") is not None:
        strategy.fs_client_param = {
            "uri": config.get("runner.fs_client.uri", ""),
            "user": config.get("runner.fs_client.user", ""),
            "passwd": config.get("runner.fs_client.passwd", ""),
            "hadoop_bin": config.get("runner.fs_client.hadoop_bin", "hadoop")
        }
    print("strategy:", strategy.fs_client_param)

    strategy.adam_d2sum = config.get("hyper_parameters.adam_d2sum", True)
    table_config = {}
    for x in config:
        if x.startswith("table_parameters"):
            table_name = x.split('.')[1]
            if table_name not in table_config:
                table_config[table_name] = {}
            table_config[table_name][x] = config[x]
    print("table_config:", table_config)
    strategy.sparse_table_configs = table_config
    print("strategy table config:", strategy.sparse_table_configs)

    return strategy


def get_model(config):
    abs_dir = config['config_abs_dir']
    sys.path.append(abs_dir)
    static_model = StaticModel(config)
    return static_model


def parse_args():
    parser = argparse.ArgumentParser("PsTest train script")
    parser.add_argument(
        '-m', '--config_yaml', type=str, required=True, help='config file path')
    parser.add_argument(
        '-bf16',
        '--pure_bf16',
        type=ast.literal_eval,
        default=False,
        help="whether use bf16")

    parser.add_argument(
        '--run_minimize', type=int, default=0, help="test single pass")
    parser.add_argument(
        '--run_single_pass', type=int, default=0, help="test single pass")
    parser.add_argument(
        '--debug_new_minimize', type=int, default=0, help="test single pass")
    parser.add_argument(
        '--debug_new_pass', type=int, default=0, help="test single pass")
    parser.add_argument(
        '--applied_pass_name', type=str, default="", help="test single pass")

    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    yaml_helper = YamlHelper()
    config = yaml_helper.load_yaml(args.config_yaml)
    config["yaml_path"] = args.config_yaml
    config["config_abs_dir"] = args.abs_dir
    config["pure_bf16"] = args.pure_bf16
    config['run_minimize'] = args.run_minimize
    config['run_single_pass'] = args.run_single_pass
    config['debug_new_minimize'] = args.debug_new_minimize
    config['debug_new_pass'] = args.debug_new_pass
    config['applied_pass_name'] = args.applied_pass_name
    yaml_helper.print_yaml(config)
    return config


def bf16_to_fp32(val):
    return np.float32(struct.unpack('<f', struct.pack('<I', val << 16))[0])


class DnnTrainer(object):
    def __init__(self, config):
        self.metrics = {}
        self.config = config
        self.input_data = None
        self.reader = None
        self.exe = None
        self.train_result_dict = {}
        self.train_result_dict["speed"] = []
        self.model = None
        self.pure_bf16 = self.config['pure_bf16']

    def init_fleet_with_gloo(self, use_gloo=True):
        if use_gloo:
            os.environ["PADDLE_WITH_GLOO"] = "1"
            role = role_maker.PaddleCloudRoleMaker()
            fleet.init(role)
        else:
            fleet.init()

    def run_minimize(self):
        self.init_fleet_with_gloo()
        self.model = get_model(self.config)
        logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
        self.input_data = self.model.create_feeds()
        self.metrics = self.model.net(self.input_data)
        strategy = get_strategy(self.config)
        learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        inner_optimizer = paddle.optimizer.Adam(learning_rate, lazy_mode=True)

        if self.config['debug_new_minimize'] == 1:
            from paddle.distributed.fleet.meta_optimizers.ps_optimizer import ParameterServerOptimizer
            ps_optimizer = ParameterServerOptimizer(inner_optimizer)
            ps_optimizer._set_basic_info(self.model._cost, None, optimizer,
                                         strategy)
            ps_optimizer.minimize_impl(self.model._cost)
        else:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(inner_optimizer, strategy)
            optimizer.minimize(self.model._cost)

        if fleet.is_server():
            _main_file = "run_minimize" + "_debug_new_minimize: " + str(
                self.config['debug_new_minimize']) + "_server_main.prototxt"
        elif fleet.is_worker():
            _main_file = "run_minimize" + "_debug_new_minimize: " + str(
                self.config['debug_new_minimize']) + "_worker_main.prototxt"
        debug_program(_main_file, self.model._cost.block.program)

    def run_single_pass(self):
        self.model = get_model(config)
        input_data = self.model.create_feeds()
        metrics = self.model.net(input_data)
        loss = self.model._cost
        strategy = get_strategy(config)
        learning_rate = config.get("hyper_parameters.optimizer.learning_rate")
        inner_optimizer = paddle.optimizer.Adam(learning_rate, lazy_mode=True)
        if self.config['debug_new_pass'] == 1:
            from paddle.distributed.fleet.meta_optimizers.ps_optimizer import ParameterServerOptimizer
            ps_optimizer = ParameterServerOptimizer(inner_optimizer)
            ps_optimizer._set_basic_info(loss, None, inner_optimizer, strategy)
            inner_opt.minimize(loss)
            ps_optimizer.init_ps_pass_context(loss, None)
            _main = ps_optimizer.context['cloned_main']

            append_send_ops_pass = new_pass(config["applied_pass_name"],
                                            ps_optimizer.context)
            append_send_ops_pass.apply([_main], [], ps_optimizer.context)

        else:
            from paddle.fluid.incubate.fleet.parameter_server.ir import public as public
            compiled_config = public.CompileTimeStrategy(
                loss.block.program, None, strategy, role_maker)
            compiled_config.strategy = strategy
            _main = compiled_config.origin_main_program.clone()
            _startup = compiled_config.origin_startup_program.clone()
            from paddle.fluid.incubate.fleet.parameter_server.ir import trainer_pass as worker
            _main = worker.append_send_ops_pass(_main, compiled_config)

        if fleet.is_server():
            _main_file = str(config[
                "applied_pass_name"]) + "debug_new_pass: " + str(self.config[
                    'debug_new_pass']) + "_server_main.prototxt"
        elif fleet.is_worker():
            _main_file = str(config[
                "applied_pass_name"]) + "debug_new_pass: " + str(self.config[
                    'debug_new_pass']) + "_worker_main.prototxt"
        debug_program(_main_file, _main)


if __name__ == "__main__":
    if fleet.is_server():
        logger.info("server: {} started".format(fleet.server_index()))
    else:
        logger.info("worker: {} started".format(fleet.worker_index()))
    paddle.enable_static()
    config = parse_args()
    os.environ["CPU_NUM"] = str(config.get("runner.thread_num"))
    benchmark_main = DnnTrainer(config)
    if config['run_single_pass'] == 1:
        benchmark_main.run_single_pass()
    elif config['run_minimize'] == 1:
        benchmark_main.run_minimize()
