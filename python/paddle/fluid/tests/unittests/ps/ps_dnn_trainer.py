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
import argparse
import time
import sys
import yaml, six, copy
import paddle
import os
import warnings
import ast
import numpy as np
import struct
sys.path.append("..")
from ps_dnn_model import StaticModel

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))


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
            draws += h_format.format("Ps Benchmark Envs", "Value")

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


def get_user_defined_strategy(config):
    if not is_distributed_env():
        logger.warn(
            "Not Find Distributed env, Change To local train mode. If you want train with fleet, please use [fleetrun] command."
        )
        #return None
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
        strategy.pipeline = True
        strategy.pipeline_configs = {
            "accumulate_steps": config.get('runner.micro_num')
        }
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
    a_sync_configs = strategy.a_sync_configs
    a_sync_configs["launch_barrier"] = False
    strategy.a_sync_configs = a_sync_configs
    print("launch_barrier: ", strategy.a_sync_configs["launch_barrier"])

    return strategy


def get_distributed_strategy(user_defined_strategy):
    from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory

    k_steps = user_defined_strategy.a_sync_configs["k_steps"]
    strategy = None

    if not user_defined_strategy.a_sync and k_steps == 0:
        strategy = StrategyFactory.create_sync_strategy()

    if user_defined_strategy.a_sync and k_steps == 0:
        strategy = StrategyFactory.create_async_strategy()

    if user_defined_strategy.a_sync and k_steps > 0:
        strategy = StrategyFactory.create_geo_strategy(k_steps)

    if not strategy:
        raise ValueError("k_steps must be invalid value, please check")

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
        '--run_the_one_ps', type=int, default=0, help="test the_one_ps")
    parser.add_argument(
        '--debug_new_minimize', type=int, default=0, help="test single pass")
    parser.add_argument(
        '--debug_new_pass', type=int, default=0, help="test single pass")
    parser.add_argument(
        '--applied_pass_name', type=str, default="", help="test single pass")
    parser.add_argument(
        '--debug_the_one_ps', type=int, default=0, help="test the_one_ps")

    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    yaml_helper = YamlHelper()
    config = yaml_helper.load_yaml(args.config_yaml)
    config["yaml_path"] = args.config_yaml
    config["config_abs_dir"] = args.abs_dir
    config["pure_bf16"] = args.pure_bf16
    config['run_minimize'] = args.run_minimize
    config['run_single_pass'] = args.run_single_pass
    config['run_the_one_ps'] = args.run_the_one_ps
    config['debug_new_minimize'] = args.debug_new_minimize
    config['debug_new_pass'] = args.debug_new_pass
    config['applied_pass_name'] = args.applied_pass_name
    config['debug_the_one_ps'] = args.debug_the_one_ps
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
        self.role_maker = role_maker.PaddleCloudRoleMaker()

    def init_fleet_with_gloo(self, use_gloo=False):
        if use_gloo:
            os.environ["PADDLE_WITH_GLOO"] = "1"
            fleet.init(self.role_maker)
        else:
            fleet.init()

        if fleet.is_server():
            logger.info("server: {} started".format(fleet.server_index()))
        else:
            logger.info("worker: {} started".format(fleet.worker_index()))

    def run_minimize(self):
        self.init_fleet_with_gloo()
        self.model = get_model(self.config)
        logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
        self.input_data = self.model.create_feeds()
        self.metrics = self.model.net(self.input_data)
        loss = self.model._cost
        user_defined_strategy = get_user_defined_strategy(self.config)
        learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        sync_mode = self.config.get("runner.sync_mode")
        inner_optimizer = paddle.optimizer.Adam(learning_rate, lazy_mode=True)

        self.role_maker._generate_role()  # 必要
        if self.config['debug_new_minimize'] == 1:
            logger.info("entering run_minimize -- new")
            from paddle.distributed.fleet.meta_optimizers.ps_optimizer import ParameterServerOptimizer
            ps_optimizer = ParameterServerOptimizer(inner_optimizer)
            ps_optimizer._set_basic_info(loss, self.role_maker, inner_optimizer,
                                         user_defined_strategy)
            ps_optimizer.minimize_impl(loss)
        else:
            logger.info("entering run_minimize -- old")
            fleet_obj = fleet.distributed_optimizer(
                inner_optimizer, user_defined_strategy)  ## Fleet 对象
            fleet_obj.minimize(loss)

        if fleet.is_server():
            _main_file = ps_log_root_dir + sync_mode + '_run_minimize' + '_debug:_' + str(
                self.config['debug_new_minimize']) + '_server_main.prototxt'
            debug_program(_main_file, loss.block.program)
        elif fleet.is_worker():
            _main_file = ps_log_root_dir + sync_mode + '_run_minimize' + '_debug:_' + str(
                self.config['debug_new_minimize']) + '_worker_main.prototxt'
            debug_program(_main_file, loss.block.program)
        elif self.role_maker._is_heter_worker():
            _main_file = ps_log_root_dir + sync_mode + '_run_minimize' + '_debug:_' + str(
                self.config[
                    'debug_new_minimize']) + '_heter_worker_main.prototxt'
            debug_program(_main_file, loss.block.program)

    def run_single_pass(self):
        self.init_fleet_with_gloo()
        self.model = get_model(config)
        input_data = self.model.create_feeds()
        metrics = self.model.net(input_data)
        loss = self.model._cost
        user_defined_strategy = get_user_defined_strategy(config)
        learning_rate = config.get("hyper_parameters.optimizer.learning_rate")
        sync_mode = self.config.get("runner.sync_mode")
        inner_optimizer = paddle.optimizer.Adam(learning_rate, lazy_mode=True)
        startup_program = paddle.static.default_startup_program()
        inner_optimizer.minimize(loss, startup_program)
        if self.config['debug_new_pass'] == 1:
            logger.info("entering run {} - new".format(
                str(config["applied_pass_name"])))
            from paddle.distributed.fleet.meta_optimizers.ps_optimizer import ParameterServerOptimizer
            ps_optimizer = ParameterServerOptimizer(inner_optimizer)
            ps_optimizer._set_basic_info(loss, self.role_maker, inner_optimizer,
                                         user_defined_strategy)
            ps_optimizer._set_origin_programs([loss])
            ps_optimizer._init_ps_pass_context(loss, startup_program)
            _main = ps_optimizer.pass_ctx._attrs['cloned_main']

            append_send_ops_pass = new_pass(config["applied_pass_name"],
                                            ps_optimizer.pass_ctx._attrs)
            append_send_ops_pass.apply([_main], [None], ps_optimizer.pass_ctx)
        else:
            logger.info("entering run {} - old".format(
                str(config["applied_pass_name"])))
            from paddle.fluid.incubate.fleet.parameter_server.ir import public as public
            dist_strategy = get_distributed_strategy(user_defined_strategy)
            compiled_config = public.CompileTimeStrategy(
                loss.block.program, startup_program, dist_strategy,
                self.role_maker)

            _main = compiled_config.origin_main_program.clone()
            _startup = compiled_config.origin_startup_program.clone()
            from paddle.fluid.incubate.fleet.parameter_server.ir import trainer_pass as worker
            _main = worker.append_send_ops_pass(_main, compiled_config)

        if fleet.is_server():
            _main_file = ps_log_root_dir + sync_mode + "_" + str(config[
                "applied_pass_name"]) + '_debug:_' + str(self.config[
                    'debug_new_pass']) + '_server_main.prototxt'
            debug_program(_main_file, _main)
        elif fleet.is_worker():
            _main_file = ps_log_root_dir + sync_mode + "_" + str(config[
                "applied_pass_name"]) + '_debug:_' + str(self.config[
                    'debug_new_pass']) + '_worker_main.prototxt'
            debug_program(_main_file, _main)

    def run_the_one_ps(self):
        self.init_fleet_with_gloo()
        self.model = get_model(self.config)
        self.input_data = self.model.create_feeds()
        self.metrics = self.model.net(self.input_data)
        loss = self.model._cost
        user_defined_strategy = get_user_defined_strategy(self.config)
        learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        sync_mode = self.config.get("runner.sync_mode")
        inner_optimizer = paddle.optimizer.Adam(learning_rate, lazy_mode=True)

        self.role_maker._generate_role()  # 必要
        if self.config['debug_the_one_ps'] == 1:
            logger.info("entering run_the_one_ps -- new")

            from paddle.distributed.fleet.meta_optimizers.ps_optimizer import ParameterServerOptimizer
            ps_optimizer = ParameterServerOptimizer(inner_optimizer)
            ps_optimizer._set_basic_info(loss, self.role_maker, inner_optimizer,
                                         user_defined_strategy)
            ps_optimizer.minimize_impl(loss)

            from paddle.distributed.ps.the_one_ps import TheOnePSRuntime
            _runtime_handle = TheOnePSRuntime()  # ps 目录下重构版的 TheOnePSRuntime
            _runtime_handle._set_basic_info(ps_optimizer.pass_ctx._attrs)
            if fleet.is_worker():
                worker_desc = _runtime_handle.ps_desc_builder.build_worker_desc(
                )
                with open(ps_log_root_dir + sync_mode + '_' +
                          'new_worker_ps_desc', 'w') as f:
                    f.write(worker_desc)
            if fleet.is_server():
                server_desc = _runtime_handle.ps_desc_builder.build_server_desc(
                )
                with open(ps_log_root_dir + sync_mode + '_' +
                          'new_server_ps_desc', 'w') as f:
                    f.write(server_desc)

        else:
            pass
        '''          
            logger.info("entering run_the_one_ps -- old")
            fleet_obj = fleet.distributed_optimizer(
                inner_optimizer, user_defined_strategy)  
            fleet_obj.minimize(loss)  
            if fleet.is_worker():
                worker_desc = fleet_obj._runtime_handle._get_fleet_proto(is_server=False, is_sync=False)
                server_desc = fleet_obj._runtime_handle._get_fleet_proto(is_server=True, is_sync=False)
                with open(ps_log_root_dir + sync_mode + '_' + 'worker_ps_desc', 'w') as f:
                    f.write(str(worker_desc) + str(server_desc))
            if fleet.is_server():
                server_desc = fleet_obj._runtime_handle._get_fleet_proto(is_server=True, is_sync=False)
                with open(ps_log_root_dir + sync_mode + '_' + 'server_ps_desc', 'w') as f:
                    f.write(str(server_desc) + str(fleet_obj._runtime_handle._get_fs_client_desc().to_string()))
        '''
        if fleet.is_server():
            _main_file = ps_log_root_dir + sync_mode + '_run_the_one_ps' + '_debug:_' + str(
                self.config['debug_the_one_ps']) + '_server_main.prototxt'
            debug_program(_main_file, loss.block.program)
        elif fleet.is_worker():
            _main_file = ps_log_root_dir + sync_mode + '_run_the_one_ps' + '_debug:_' + str(
                self.config['debug_the_one_ps']) + '_worker_main.prototxt'
            debug_program(_main_file, loss.block.program)
        elif self.role_maker._is_heter_worker():
            _main_file = ps_log_root_dir + sync_mode + '_run_the_one_ps' + '_debug:_' + str(
                self.config['debug_the_one_ps']) + '_heter_worker_main.prototxt'
            debug_program(_main_file, loss.block.program)


if __name__ == "__main__":
    paddle.enable_static()
    config = parse_args()
    logger.info(">>>>>>>>>> python process started")
    os.environ["CPU_NUM"] = str(config.get("runner.thread_num"))
    benchmark_main = DnnTrainer(config)
    if config['run_single_pass'] == 1:
        benchmark_main.run_single_pass()
    elif config['run_minimize'] == 1:
        benchmark_main.run_minimize()
    elif config['run_the_one_ps'] == 1:
        benchmark_main.run_the_one_ps()
