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

import os
import unittest
import numpy as np
import time
import paddle
from paddle.distributed.ps.utils.public import ps_log_root_dir, debug_program
import paddle.distributed.fleet as fleet
import paddle.fluid as fluid


def get_dataset(inputs, config, pipe_cmd, role="worker"):
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command(pipe_cmd)
    dataset.set_batch_size(config.get('runner.batch_size'))
    reader_thread_num = int(config.get('runner.reader_thread_num'))
    dataset.set_thread(reader_thread_num)
    train_files_path = config.get('runner.train_files_path')
    print('train_data_files:{}'.format(train_files_path))
    file_list = [
        os.path.join(train_files_path, x) for x in os.listdir(train_files_path)
    ]
    if role == "worker":
        file_list = fleet.util.get_file_shard(file_list)
        print("worker file list: {}".format(file_list))
    elif role == "heter_worker":
        file_list = fleet.util.get_heter_file_shard(file_list)
        print("heter worker file list: {}".format(file_list))

    return dataset, file_list


def fl_ps_train():
    # 0. get role
    import paddle.distributed.fleet.base.role_maker as role_maker
    role_maker = role_maker.PaddleCloudRoleMaker()
    role_maker._generate_role()
    fleet.util._set_role_maker(role_maker)

    # 1. load yaml-config to dict-config
    from ps_dnn_trainer import YamlHelper, StaticModel, get_user_defined_strategy
    yaml_helper = YamlHelper()
    config_yaml_path = '../ps/fl_async_ps_config.yaml'
    config = yaml_helper.load_yaml(config_yaml_path)
    #yaml_helper.print_yaml(config)

    # 2. get static model
    paddle.enable_static()
    model = StaticModel(config)
    feeds_list = model.create_feeds()
    metrics = model.fl_net(feeds_list)
    loss = model._cost

    # 3. compile time - build program_desc
    user_defined_strategy = get_user_defined_strategy(config)
    a_sync_configs = user_defined_strategy.a_sync_configs
    a_sync_configs["launch_barrier"] = True
    user_defined_strategy.a_sync_configs = a_sync_configs
    print("launch_barrier: ",
          user_defined_strategy.a_sync_configs["launch_barrier"])
    learning_rate = config.get("hyper_parameters.optimizer.learning_rate")
    inner_optimizer = paddle.optimizer.Adam(learning_rate, lazy_mode=True)
    from paddle.distributed.fleet.meta_optimizers.ps_optimizer import ParameterServerOptimizer
    ps_optimizer = ParameterServerOptimizer(inner_optimizer)
    ps_optimizer._set_basic_info(loss, role_maker, inner_optimizer,
                                 user_defined_strategy)
    ps_optimizer.minimize_impl(loss)

    # 4. runtime
    from paddle.distributed.ps.the_one_ps import TheOnePSRuntime
    _runtime_handle = TheOnePSRuntime()  # ps 目录下重构版的 TheOnePSRuntime
    _runtime_handle._set_basic_info(ps_optimizer.pass_ctx._attrs)
    epoch_num = int(config.get('runner.epoch_num'))
    # 4.1 run server - build fleet_desc
    if role_maker._is_server():
        _runtime_handle._init_server()
        _runtime_handle._run_server()
    # 4.2 run worker
    elif role_maker._is_worker():
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        _runtime_handle._init_worker()
        print('trainer get dataset')
        inputs = feeds_list[1:-1]
        dataset, file_list = get_dataset(inputs, config,
                                         "python dataset_generator_A.py")
        print("fluid.default_main_program: {}".format(
            fluid.default_main_program()._heter_pipeline_opt))
        for epoch in range(epoch_num):
            # A 方和 B 方如果要以文件粒度 shuffle 时，则需要固定同一个种子
            dataset.set_filelist(file_list)
            start_time = time.time()
            exe.train_from_dataset(program=fluid.default_main_program(),
                                   dataset=dataset,
                                   print_period=2,
                                   debug=False)
            end_time = time.time()
            print("trainer epoch %d finished, use time=%d\n" %
                  ((epoch), end_time - start_time))
        exe.close()
        _runtime_handle._stop_worker()
        print("Fl partyA Trainer Success!")
    else:
        exe = fluid.Executor()
        exe.run(fluid.default_startup_program())
        _runtime_handle._init_worker()
        inputs = [feeds_list[0],
                  feeds_list[-1]]  # 顺序务必要和 dataset_generator_B.py 中保持一致
        dataset, file_list = get_dataset(inputs, config,
                                         "python dataset_generator_B.py",
                                         "heter_worker")
        print("fluid.default_main_program: {}".format(
            fluid.default_main_program()._heter_pipeline_opt))
        for epoch in range(epoch_num):
            dataset.set_filelist(file_list)
            exe.train_from_dataset(program=fluid.default_main_program(),
                                   dataset=dataset,
                                   print_period=2,
                                   debug=False)
        exe.close()
        _runtime_handle._stop_worker()
        print("Fl partB Trainer Success!")


if __name__ == '__main__':
    fl_ps_train()
