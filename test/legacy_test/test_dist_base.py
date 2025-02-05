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

import argparse
import ast
import os

os.environ['FLAGS_enable_pir_api'] = '0'

import pickle
import random
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing

import numpy as np

import paddle
from paddle import base
from paddle.base import compiler
from paddle.distributed.fleet.meta_optimizers import (
    RawProgramOptimizer as RawProgram,
)
from paddle.incubate.distributed.fleet import role_maker
from paddle.incubate.distributed.fleet.collective import (
    DistributedStrategy,
    fleet,
)

RUN_STEP = 5
DEFAULT_BATCH_SIZE = 2
DIST_UT_PORT = 0


def remove_glog_envs(envs):
    if not envs:
        return envs

    glog_envs = ['GLOG_v', 'GLOG_logtostderr', 'GLOG_vmodule']
    envs = dict(envs)
    for env in glog_envs:
        if env in envs:
            del envs[env]
    return envs


def get_dump_file(rank):
    return f"./out_dump_{os.getpid()}_{rank}.pickled"


def modify_envs(envs, rank=0):
    if not envs:
        envs = {}
    envs = remove_glog_envs(envs)
    dump_file = get_dump_file(rank)
    envs['DUMP_FILE'] = dump_file
    if os.path.exists(dump_file):
        os.remove(dump_file)
    return envs


def dump_output(x):
    path = os.environ['DUMP_FILE']
    with open(path, 'wb') as f:
        pickle.dump(x, f)


def load_and_remove_dump_file(rank=0):
    path = get_dump_file(rank)
    with open(path, 'rb') as f:
        out = pickle.load(f)
    os.remove(path)
    return out


def print_to_err(class_name, log_str):
    localtime = time.asctime(time.localtime(time.time()))
    print_str = localtime + "\t" + class_name + "\t" + log_str
    sys.stderr.buffer.write(pickle.dumps(print_str))


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _insert_comm_op(opt, loss, build_strategy=None):
    opt = RawProgram(opt)
    role = paddle.distributed.fleet.base.role_maker.PaddleCloudRoleMaker(
        is_collective=True
    )
    strategy = paddle.distributed.fleet.DistributedStrategy()
    if build_strategy is not None:
        strategy.build_strategy = build_strategy
    opt._set_basic_info(loss, role, opt, strategy)

    # following code is a copy of RawProgramOptimizer.minimize except init_comm_group
    opt.endpoints = opt.role_maker._get_trainer_endpoints()
    opt.current_endpoint = opt.endpoints[opt.role_maker._worker_index()]
    opt.rank = opt.role_maker._worker_index()
    opt.nranks = opt.role_maker._worker_num()
    startup_program = paddle.static.default_startup_program()
    opt.startup_program = startup_program

    block = loss.block
    program = block.program
    opt.main_program = program

    optimize_ops, params_grads = opt.inner_opt.minimize(loss, startup_program)

    opt.main_program = program
    if opt.nranks > 1:
        opt._transpile_main_program(loss)


class TestDistRunnerBase:
    def get_model(
        self,
        batch_size=DEFAULT_BATCH_SIZE,
        lr=0.1,
        single_device=False,
        use_dgc=False,
        dist_strategy=None,
    ):
        raise NotImplementedError(
            "get_model should be implemented by child classes."
        )

    @staticmethod
    def get_transpiler(
        trainer_id,
        main_program,
        pserver_endpoints,
        trainers,
        sync_mode,
        dc_asgd=False,
        current_endpoint=None,
        nccl_comm_num=1,
        hogwild_mode=False,
    ):
        # NOTE: import base until runtime, or else forking processes will cause error.
        config = paddle.distributed.transpiler.DistributeTranspilerConfig()
        config.enable_dc_asgd = dc_asgd
        config.sync_mode = sync_mode
        config.runtime_split_send_recv = hogwild_mode

        if nccl_comm_num > 1:
            config.nccl_comm_num = nccl_comm_num
        # config.runtime_split_send_recv = True
        t = paddle.distributed.transpiler.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id=trainer_id,
            program=main_program,
            pservers=pserver_endpoints,
            trainers=trainers,
            sync_mode=sync_mode,
            current_endpoint=current_endpoint,
        )
        return t

    @staticmethod
    def get_lr_scheduler(program):
        lr_scheduler = None
        if hasattr(program, 'lr_scheduler'):
            from paddle.optimizer.lr import LRScheduler

            lr_scheduler = program.lr_scheduler
            assert isinstance(lr_scheduler, LRScheduler), "must be LRScheduler"
        return lr_scheduler

    def run_pserver(self, args):
        self.lr = args.lr
        self.get_model(batch_size=args.batch_size)
        # NOTE: pserver should not call memory optimize

        t = self.get_transpiler(
            trainer_id=args.trainer_id,
            main_program=base.default_main_program(),
            pserver_endpoints=args.endpoints,
            trainers=args.trainers,
            sync_mode=args.sync_mode,
            dc_asgd=args.dc_asgd,
            hogwild_mode=args.hogwild,
        )
        pserver_prog = t.get_pserver_program(args.current_endpoint)
        startup_prog = t.get_startup_program(
            args.current_endpoint, pserver_prog
        )

        place = base.CPUPlace()
        exe = base.Executor(place)
        exe.run(startup_prog)
        print_to_err(type(self).__name__, "run pserver startup program done.")
        exe.run(pserver_prog)
        print_to_err(type(self).__name__, "run pserver main program done.")

    def run_pipeline_trainer(self, args):
        self.lr = args.lr

        dist_strategy = DistributedStrategy()
        (
            test_program,
            avg_cost,
            train_reader,
            test_reader,
            batch_acc,
            predict,
            data_loader,
        ) = self.get_model(
            batch_size=args.batch_size, dist_strategy=dist_strategy
        )

        device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        eprint(type(self).__name__, f"device_id: {device_id}.")
        place = base.CUDAPlace(device_id)

        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        eprint(type(self).__name__, "run worker startup program done.")

        data_loader.set_sample_list_generator(train_reader, place)
        data_loader.start()
        print_to_err(type(self).__name__, "begin to train on trainer")
        out_losses = []

        main_program = base.default_main_program()
        lr_scheduler = self.get_lr_scheduler(main_program)
        for i in range(RUN_STEP):
            loss = exe.run(main_program, fetch_list=[avg_cost])
            loss = loss[0] if loss else None
            out_losses.append(loss)
            print_to_err(type(self).__name__, f"run step {i} finished")
            if lr_scheduler is not None:
                lr_scheduler.step()

        data_loader.reset()
        print_to_err(type(self).__name__, "trainer run finished")

        dump_output(out_losses)

    def run_use_fleet_api_20_trainer(self, args):
        """
        1. remove codes for DistributedStrategy and leave the DistributedStrategy part to get_model()
        2. to run with fleet 2.0 api, set flags _use_fleet_api and _use_fleet_api_20 to True
        3. for now, not support test for model save
        """
        assert args.update_method == "nccl2" or "bkcl"

        self.lr = args.lr
        print_to_err("use_fleet 2.0", "fleet.node_num:")

        (
            test_program,
            avg_cost,
            train_reader,
            test_reader,
            batch_acc,
            predict,
        ) = self.get_model(batch_size=args.batch_size)

        if base.core.is_compiled_with_cuda():
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = base.CUDAPlace(device_id)
        elif base.core.is_compiled_with_xpu():
            device_id = int(os.getenv("FLAGS_selected_xpus", "0"))
            place = base.XPUPlace(device_id)
        else:
            raise ValueError(
                "fleet dygraph api must in paddlepaddle-xpu or paddlepaddle-gpu."
            )

        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        eprint(type(self).__name__, "run worker startup program done.")

        feed_var_list = [
            var
            for var in base.default_main_program().global_block().vars.values()
            if var.is_data
        ]

        eprint("feed_var_list:", feed_var_list)

        if feed_var_list[0].name == 'label':
            feed_var_list.reverse()

        feeder = base.DataFeeder(feed_var_list, place)
        reader_generator = train_reader()

        def get_data():
            origin_batch = next(reader_generator)
            if (
                paddle.distributed.get_world_size() == 1
                and args.update_method == 'gloo'
            ):  # Gloo single mode
                return origin_batch

            elif args.update_method != "local" and args.use_reader_alloc:
                new_batch = []
                for offset, item in enumerate(origin_batch):
                    if offset % 2 == args.trainer_id:
                        new_batch.append(item)
                return new_batch
            else:
                return origin_batch

        print_to_err(type(self).__name__, "begin to train on trainer")
        out_losses = []
        for i in range(RUN_STEP):
            (loss,) = exe.run(
                base.default_main_program(),
                fetch_list=[avg_cost.name],
                feed=feeder.feed(get_data()),
            )
            out_losses.append(float(loss))
            print_to_err(type(self).__name__, f"run step {i} finished")
        print_to_err(type(self).__name__, "trainer run finished")
        print_to_err(type(self).__name__, f"dist losses: {out_losses}")

        dump_output(out_losses)

    def run_use_fleet_api_trainer(self, args):
        assert args.update_method == "nccl2" or "bkcl"
        backend = "bkcl" if args.update_method == "bkcl" else "nccl"
        paddle.distributed.collective._init_parallel_env(backend)

        self.lr = args.lr

        dist_strategy = DistributedStrategy()
        dist_strategy.fuse_memory_size = 1  # MB
        dist_strategy.fuse_laryer_size = 1
        if args.use_local_sgd:
            dist_strategy.use_local_sgd = True
        if args.ut4grad_allreduce:
            dist_strategy._ut4grad_allreduce = True
        if args.sync_batch_norm:
            dist_strategy.sync_batch_norm = True

        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        print_to_err("use_fleet", "fleet.node_num:")
        # "fleet.node_id:", fleet.node_id(),
        # "fleet.trainer_num:", fleet.worker_num())

        (
            test_program,
            avg_cost,
            train_reader,
            test_reader,
            batch_acc,
            predict,
        ) = self.get_model(
            batch_size=args.batch_size, dist_strategy=dist_strategy
        )

        trainer_prog = fleet._origin_program
        dist_prog = fleet.main_program

        if base.core.is_compiled_with_cuda():
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = base.CUDAPlace(device_id)
        elif base.core.is_compiled_with_xpu():
            device_id = int(os.getenv("FLAGS_selected_xpus", "0"))
            place = base.XPUPlace(device_id)
        else:
            raise ValueError(
                "fleet dygraph api must in paddlepaddle-xpu or paddlepaddle-gpu."
            )

        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        eprint(type(self).__name__, "run worker startup program done.")

        feed_var_list = [
            var
            for var in trainer_prog.global_block().vars.values()
            if var.is_data
        ]

        eprint("feed_var_list:", feed_var_list)

        # tmp add this code to pass python35 gcc8 CI
        # Fixme(gongweibao, wangxi), need fix fleet api program order
        if feed_var_list[0].name == 'label':
            feed_var_list.reverse()

        feeder = base.DataFeeder(feed_var_list, place)
        reader_generator = train_reader()

        def get_data():
            origin_batch = next(reader_generator)
            if args.update_method != "local" and args.use_reader_alloc:
                new_batch = []
                for offset, item in enumerate(origin_batch):
                    if offset % 2 == args.trainer_id:
                        new_batch.append(item)
                return new_batch
            else:
                return origin_batch

        print_to_err(type(self).__name__, "begin to train on trainer")
        out_losses = []
        for i in range(RUN_STEP):
            (loss,) = exe.run(
                dist_prog,
                fetch_list=[avg_cost.name],
                feed=feeder.feed(get_data()),
            )
            out_losses.append(float(loss))
            print_to_err(type(self).__name__, f"run step {i} finished")
        print_to_err(type(self).__name__, "trainer run finished")

        dump_output(out_losses)

        if args.save_model:
            model_save_dir = "/tmp"
            if fleet.worker_index() == 0:
                model_save_dir_base = os.path.join(
                    model_save_dir, "base_persistables"
                )
                model_save_dir_fleet = os.path.join(
                    model_save_dir, "fleet_persistables"
                )
                infer_save_dir_base = os.path.join(
                    model_save_dir, "base_infer/infer"
                )
                infer_save_dir_fleet = os.path.join(
                    model_save_dir, "fleet_infer/infer"
                )
            else:
                model_save_dir_base = os.path.join(
                    model_save_dir, "base_persistables_2"
                )
                model_save_dir_fleet = os.path.join(
                    model_save_dir, "fleet_persistables_2"
                )
                infer_save_dir_base = os.path.join(
                    model_save_dir, "base_infer_2/infer_2"
                )
                infer_save_dir_fleet = os.path.join(
                    model_save_dir, "fleet_infer_2/infer_2"
                )
            paddle.distributed.io.save_persistables(
                exe, model_save_dir_base, fleet._origin_program
            )
            fleet.save_persistables(executor=exe, dirname=model_save_dir_fleet)
            paddle.static.io.save_inference_model(
                path_prefix=infer_save_dir_base,
                feed_vars=feed_var_list,
                fetch_vars=[avg_cost],
                executor=exe,
                program=fleet._origin_program,
            )
            fleet.save_inference_model(
                exe, infer_save_dir_fleet, feed_var_list, [avg_cost]
            )

    def run_trainer(self, args):
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        build_stra = base.BuildStrategy()
        # FIXME force disable enable_inplace and memory_optimize
        build_stra.enable_inplace = False
        build_stra.memory_optimize = False

        if args.fuse_all_reduce is not None:
            sys.stderr.write(f'fuse_all_reduce={args.fuse_all_reduce}')
            build_stra.fuse_all_reduce_ops = args.fuse_all_reduce

        if args.hogwild:
            build_stra.async_mode = True

        if args.enable_backward_deps:
            build_stra.enable_backward_optimizer_op_deps = True

        if args.use_reduce:
            build_stra.reduce_strategy = (
                base.BuildStrategy.ReduceStrategy.Reduce
            )
        else:
            build_stra.reduce_strategy = (
                base.BuildStrategy.ReduceStrategy.AllReduce
            )
        pass_builder = None
        if args.batch_merge_repeat > 1:
            pass_builder = build_stra._finalize_strategy_and_create_passes()
            mypass = pass_builder.insert_pass(0, "multi_batch_merge_pass")
            mypass.set("num_repeats", args.batch_merge_repeat)

        if (
            args.update_method == "nccl2"
            or args.update_method == "nccl2_reduce_layer"
        ):
            build_stra.num_trainers = len(args.endpoints.split(","))
            build_stra.trainer_id = args.trainer_id
        else:
            # case args.update_method == "nccl2_reduce_layer":
            build_stra.num_trainers = 1
            build_stra.trainer_id = 0

        self.lr = args.lr
        if args.nccl2_reduce_layer_local_run:
            (
                test_program,
                avg_cost,
                train_reader,
                test_reader,
                batch_acc,
                predict,
            ) = self.get_model(batch_size=args.batch_size, single_device=True)
        elif args.use_dgc:
            (
                test_program,
                avg_cost,
                train_reader,
                test_reader,
                batch_acc,
                predict,
            ) = self.get_model(
                batch_size=args.batch_size,
                use_dgc=args.use_dgc,
                build_strategy=build_stra,
            )
        else:
            (
                test_program,
                avg_cost,
                train_reader,
                test_reader,
                batch_acc,
                predict,
            ) = self.get_model(batch_size=args.batch_size)

        if args.update_method == "pserver":
            print_to_err(
                type(self).__name__,
                "begin to run transpile on trainer with pserver mode",
            )
            t = self.get_transpiler(
                trainer_id=args.trainer_id,
                main_program=base.default_main_program(),
                pserver_endpoints=args.endpoints,
                trainers=args.trainers,
                sync_mode=args.sync_mode,
                dc_asgd=args.dc_asgd,
                hogwild_mode=args.hogwild,
            )

            trainer_prog = t.get_trainer_program()
            print_to_err(
                type(self).__name__,
                "get trainer program done with pserver mode.",
            )
        elif (
            args.update_method == "nccl2"
            or args.update_method == "nccl2_reduce_layer"
        ):
            # transpile for nccl2
            config = paddle.distributed.transpiler.DistributeTranspilerConfig()
            config.mode = "nccl2"
            config.nccl_comm_num = args.nccl_comm_num
            if args.use_hallreduce:
                config.use_hierarchical_allreduce = True
                config.hierarchical_allreduce_inter_nranks = (
                    args.hallreduce_inter_nranks
                )
            print_to_err(
                type(self).__name__,
                "begin to run transpile on trainer with nccl2 mode",
            )
            nccl2_t = paddle.distributed.transpiler.DistributeTranspiler(
                config=config
            )
            nccl2_t.transpile(
                args.trainer_id,
                program=base.default_main_program(),
                startup_program=base.default_startup_program(),
                trainers=args.endpoints,
                current_endpoint=args.current_endpoint,
            )
            print_to_err(
                type(self).__name__, "get trainer program done. with nccl2 mode"
            )
            trainer_prog = base.default_main_program()
        else:
            print_to_err(
                type(self).__name__,
                "do nothing about main program, just use it",
            )
            trainer_prog = base.default_main_program()
            print_to_err(type(self).__name__, "use main program done.")

        # FIXME(gongwb):wait pserver initialization.
        time.sleep(1)

        if args.use_cuda:
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = base.CUDAPlace(device_id)
        else:
            place = base.CPUPlace()

        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        print_to_err(type(self).__name__, "run worker startup program done.")

        print_to_err(type(self).__name__, "begin to compile with data parallel")
        binary = compiler.CompiledProgram(
            trainer_prog, build_strategy=build_stra
        )
        print_to_err(type(self).__name__, "program compiled with data parallel")

        feed_var_list = [
            var
            for var in trainer_prog.global_block().vars.values()
            if var.is_data
        ]

        feeder = base.DataFeeder(feed_var_list, place)
        reader_generator = train_reader()

        def get_data():
            origin_batch = next(reader_generator)
            if args.update_method != "local" and args.use_reader_alloc:
                new_batch = []
                for offset, item in enumerate(origin_batch):
                    if offset % 2 == args.trainer_id:
                        new_batch.append(item)
                return new_batch
            else:
                return origin_batch

        lr_scheduler = self.get_lr_scheduler(trainer_prog)
        print_to_err(type(self).__name__, "begin to train on trainer")
        out_losses = []
        for i in range(RUN_STEP):
            (loss,) = exe.run(
                binary, fetch_list=[avg_cost.name], feed=feeder.feed(get_data())
            )
            out_losses.append(float(loss))
            print_to_err(type(self).__name__, f"run step {i} finished")
            if lr_scheduler is not None:
                lr_scheduler.step()

        print_to_err(type(self).__name__, "trainer run finished\n")
        # print_to_err(type(self).__name__, "out_losses")

        sys.stdout = old_stdout
        dump_output(out_losses)


class TestParallelDyGraphRunnerBase:
    def get_model(self):
        raise NotImplementedError(
            "get_model should be implemented by child classes."
        )

    def run_one_loop(self, model, opt, data):
        raise NotImplementedError(
            "train_one_loop should be implemented by the child classes."
        )

    def _get_data(self, batch, args):
        if (
            paddle.distributed.get_world_size() == 1
            and args.update_method == 'gloo'
        ):  # Gloo single mode
            return batch
        elif args.update_method != "local":
            new_batch = []

            # NOTE(@xiongkun03) args.diff_batch means batch length is different:
            # such as : batch = [2,3,4,5], then the first rank will get [2]  and
            # the second rank will get [3,4,5].
            # this function is for test sparse_embedding_differ_length
            if hasattr(args, "diff_batch") and args.diff_batch:
                assert (
                    len(batch) > 2
                ), "in differ_batch mode, len(batch) must > 2."
                if paddle.distributed.get_rank() == 0:
                    new_batch.append(batch[0])
                elif paddle.distributed.get_rank() == 1:
                    new_batch.extend(list(batch[1:]))
                else:
                    raise NotImplementedError(
                        "Current TestParallelDyGraphRunnerBase don't support world_size > 2"
                    )
                return new_batch
            else:
                for offset, item in enumerate(batch):
                    if offset % 2 == args.trainer_id:
                        new_batch.append(item)
                return new_batch
        else:
            return batch

    def run_trainer(self, args):
        seed = 90
        if args.update_method == 'gloo':
            place = base.CPUPlace()
        elif base.core.is_compiled_with_cuda():
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = base.CUDAPlace(device_id)
        elif base.core.is_compiled_with_xpu():
            device_id = int(os.getenv("FLAGS_selected_xpus", "0"))
            place = base.XPUPlace(device_id)
        else:
            assert "Only support CUDAPlace or XPUPlace or CPU(Gloo) for now."

        with base.dygraph.guard(place):
            paddle.seed(seed)
            np.random.seed(seed)
            import random

            random.seed(seed)
            model, train_reader, opt = self.get_model()
            nranks = len(args.endpoints.split(",")) if args.endpoints else 1

            # if args.update_method == "nccl2":
            if args.update_method == "nccl2" or args.update_method == "bkcl":
                strategy = paddle.distributed.parallel.ParallelStrategy()
                strategy.nranks = nranks
                strategy.local_rank = args.trainer_id
                strategy.trainer_endpoints = args.endpoints.split(",")
                strategy.current_endpoint = args.current_endpoint
                paddle.distributed.init_parallel_env()
                print_to_err(
                    type(self).__name__,
                    "begin to prepare context in dygraph with nccl2",
                )
                if not args.find_unused_parameters:
                    model = paddle.DataParallel(
                        model, strategy, find_unused_parameters=False
                    )
                else:
                    model = paddle.DataParallel(
                        model, strategy, find_unused_parameters=True
                    )
                print_to_err(type(self).__name__, "model built in dygraph")

            elif args.update_method == "gloo":
                paddle.distributed.init_parallel_env()
                if not args.find_unused_parameters:
                    model = paddle.DataParallel(
                        model, find_unused_parameters=False
                    )
                else:
                    model = paddle.DataParallel(
                        model, find_unused_parameters=True
                    )

            out_losses = []
            print_to_err(type(self).__name__, "begin to run dygraph training")
            for step_id, data in enumerate(train_reader()):
                data = self._get_data(data, args)
                if step_id == RUN_STEP:
                    break
                loss = self.run_one_loop(model, opt, data)
                if step_id % 10 == 0:
                    print_to_err(
                        type(self).__name__,
                        f"loss at step {step_id}: {loss.numpy().item():f}",
                    )
                out_losses.append(loss.numpy())

                loss.backward()

                opt.minimize(loss)
                if not args.accumulate_gradient:
                    model.clear_gradients()
        dump_output(out_losses)

    def run_trainer_with_spawn(self, args):
        # 1. enable dygraph
        paddle.disable_static()

        # 2. init seed
        seed = 90
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # get trainer id
        paddle.distributed.parallel._get_global_parallel_env()
        args.trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))

        # 3. init parallel env
        if args.update_method in ["nccl2", "gloo"]:
            paddle.distributed.init_parallel_env()

        # 4. train model
        model, train_reader, opt = self.get_model()
        if args.update_method in ["nccl2", "gloo"]:
            model = paddle.DataParallel(
                model, find_unused_parameters=args.find_unused_parameters
            )

        out_losses = []
        for step_id, data in enumerate(train_reader()):
            data = self._get_data(data, args)
            if step_id == RUN_STEP:
                break
            loss = self.run_one_loop(model, opt, data)
            out_losses.append(loss.numpy())

            loss.backward()

            opt.minimize(loss)
            model.clear_gradients()
        return out_losses

    def run_use_fleet_api_trainer(self, args):
        from paddle.distributed import fleet

        # 1. enable dygraph
        paddle.disable_static()

        # 2. init seed
        seed = 90
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # get trainer id
        paddle.distributed.parallel._get_global_parallel_env()
        args.trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))

        # set strategy
        strategy = fleet.DistributedStrategy()
        if args.find_unused_parameters:
            strategy.find_unused_parameters = True

        # 3. init parallel env
        if args.update_method == "nccl2" or "bkcl":
            fleet.init(is_collective=True, strategy=strategy)

        # 4. train model
        model, train_reader, opt = self.get_model()
        if args.update_method == "nccl2" or "bkcl":
            opt = fleet.distributed_optimizer(opt)
            model = fleet.distributed_model(model)

        out_losses = []
        for step_id, data in enumerate(train_reader()):
            data = self._get_data(data, args)
            if step_id == RUN_STEP:
                break
            loss = self.run_one_loop(model, opt, data)
            out_losses.append(loss.numpy())

            loss.backward()

            opt.step()
            if not args.accumulate_gradient:
                opt.clear_grad()
        dump_output(out_losses)


def runtime_main(test_class):
    parser = argparse.ArgumentParser(description='Run dist test.')
    parser.add_argument(
        '--role', type=str, required=True, choices=['pserver', 'trainer']
    )
    parser.add_argument('--endpoints', type=str, required=False, default="")
    parser.add_argument(
        '--update_method',
        type=str,
        default="local",
        choices=[
            "pserver",
            "nccl2",
            "bkcl",
            "local",
            "nccl2_reduce_layer",
            "gloo",
        ],
    )
    parser.add_argument('--trainer_id', type=int, required=False, default=0)
    parser.add_argument('--trainers', type=int, required=False, default=1)
    parser.add_argument('--nccl_comm_num', type=int, required=False, default=1)
    parser.add_argument('--enable_backward_deps', action='store_true')
    parser.add_argument('--use_hallreduce', action='store_true')
    parser.add_argument('--use_pipeline', action='store_true')
    parser.add_argument('--use_fleet_api', action='store_true')
    parser.add_argument('--use_fleet_api_20', action='store_true')
    parser.add_argument('--use_local_sgd', action='store_true')
    parser.add_argument('--diff_batch', action='store_true')
    parser.add_argument('--ut4grad_allreduce', action='store_true')
    parser.add_argument(
        '--hallreduce_inter_nranks', type=int, required=False, default=2
    )
    parser.add_argument(
        '--current_endpoint', type=str, required=False, default=""
    )
    parser.add_argument('--sync_mode', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--use_xpu', action='store_true')
    parser.add_argument('--use_dgc', action='store_true')
    parser.add_argument('--accumulate_gradient', action='store_true')
    parser.add_argument('--find_unused_parameters', action='store_true')
    parser.add_argument('--use_reduce', action='store_true')
    parser.add_argument('--dc_asgd', action='store_true')
    parser.add_argument('--hogwild', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument(
        '--use_reader_alloc', action='store_true', required=False
    )
    parser.add_argument('--batch_size', required=False, type=int, default=2)
    parser.add_argument('--lr', required=False, type=float, default=0.001)
    parser.add_argument(
        '--batch_merge_repeat', required=False, type=int, default=1
    )
    parser.add_argument(
        '--nccl2_reduce_layer_local_run',
        required=False,
        type=bool,
        default=False,
    )
    parser.add_argument('--sync_batch_norm', action='store_true')
    parser.add_argument(
        '--fuse_all_reduce', required=False, type=ast.literal_eval, default=None
    )

    args = parser.parse_args()

    if args.update_method == 'gloo':
        paddle.set_device("cpu")

    model = test_class()
    if args.role == "pserver" and args.update_method == "pserver":
        model.run_pserver(args)
    elif args.use_fleet_api:
        model.run_use_fleet_api_trainer(args)
    elif args.use_fleet_api_20:
        model.run_use_fleet_api_20_trainer(args)
    elif args.use_pipeline:
        model.run_pipeline_trainer(args)
    else:
        model.run_trainer(args)


class TestDistBase(unittest.TestCase):
    def _setup_config(self):
        raise NotImplementedError("tests should have _setup_config implemented")

    def _after_setup_config(self):
        if self._enforce_place == "CPU":
            self.__use_cuda = False
            self.__use_xpu = False
            self._use_dgc = False
        elif self._enforce_place == "GPU":
            self.__use_cuda = True
            self.__use_xpu = False
        elif self._enforce_place == "XPU":
            self.__use_cuda = False
            self.__use_xpu = True
            self._use_dgc = False
        else:
            if base.core.is_compiled_with_cuda():
                self.__use_cuda = True
            else:
                self.__use_cuda = False
                self._use_dgc = False

        if self._use_reduce:
            assert not self._use_dgc

    def setUp(self):
        self._trainers = 2
        self._pservers = 2
        self._port_set = set()
        self._python_interp = sys.executable
        self._sync_mode = True
        self._hogwild_mode = False
        self._enforce_place = None
        self._use_reduce = False
        self._dc_asgd = False  # must use with async mode
        self._use_reader_alloc = True
        self._nccl2_mode = False
        self._bkcl_mode = False
        self._gloo_mode = False  # now, support gloo backend
        self._pipeline_mode = False
        self._mp_mode = False
        self._diff_batch = False
        # FIXME(typhoonzero): I added this stupid argument to enable
        # testing allreduce layers, which users can call layers.allreduce
        # to accumulate tensors at anywhere. Find a better way to do this
        # test, reduce check this argument everywhere.
        self._nccl2_reduce_layer = False
        self._lr = 0.001
        self._use_dgc = False
        self._dygraph = False
        self._nccl_comm_num = 1
        self._enable_backward_deps = False
        self._use_fleet_api = False
        self._use_fleet_api_20 = False
        self._use_local_sgd = False
        self._ut4grad_allreduce = False
        self._use_hallreduce = False
        self._save_model = False
        self._fuse_all_reduce = None
        self._accumulate_gradient = False
        self._find_unused_parameters = False
        self._setup_config()

        global DIST_UT_PORT
        if DIST_UT_PORT == 0 and os.getenv("PADDLE_DIST_UT_PORT"):
            DIST_UT_PORT = int(os.getenv("PADDLE_DIST_UT_PORT"))

        if DIST_UT_PORT == 0:
            self._ps_endpoints = f"127.0.0.1:{self._find_free_port()},127.0.0.1:{self._find_free_port()}"
        else:
            self._ps_endpoints = (
                f"127.0.0.1:{DIST_UT_PORT},127.0.0.1:{DIST_UT_PORT + 1}"
            )
            DIST_UT_PORT += 2
            self._dist_port = DIST_UT_PORT

        self._after_setup_config()

        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _find_free_port(self):
        def __free_port():
            with closing(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ) as s:
                s.bind(('', 0))
                print_to_err(
                    type(self).__name__, f"socket name: {s.getsockname()[1]}"
                )
                return s.getsockname()[1]

        while True:
            port = __free_port()
            if port not in self._port_set:
                self._port_set.add(port)
                return port

    def start_pserver(
        self, model_file, check_error_log, required_envs, log_name=""
    ):
        ps0_ep, ps1_ep = self._ps_endpoints.split(",")
        ps_cmd = "%s"

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            required_envs['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            ps_cmd += " -m coverage run --branch -p"

        ps_cmd += " %s --role pserver --endpoints %s --trainer_id 0 --current_endpoint %s --trainers %d --update_method pserver"

        ps0_cmd = ps_cmd % (
            self._python_interp,
            model_file,
            self._ps_endpoints,
            ps0_ep,
            self._trainers,
        )
        ps1_cmd = ps_cmd % (
            self._python_interp,
            model_file,
            self._ps_endpoints,
            ps1_ep,
            self._trainers,
        )

        if self._sync_mode:
            ps0_cmd += " --sync_mode"
            ps1_cmd += " --sync_mode"

        path0 = os.path.join(self.temp_dir.name, log_name + "_ps0_err.log")
        path1 = os.path.join(self.temp_dir.name, log_name + "_ps1_err.log")
        ps0_pipe = open(path0, "wb")
        ps1_pipe = open(path1, "wb")

        ps0_proc = subprocess.Popen(
            ps0_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=ps0_pipe,
            env=modify_envs(required_envs),
        )
        ps1_proc = subprocess.Popen(
            ps1_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=ps1_pipe,
            env=modify_envs(required_envs),
        )

        return ps0_proc, ps1_proc, ps0_pipe, ps1_pipe

    def _run_local(
        self,
        model,
        envs,
        check_error_log=False,
        batch_size=DEFAULT_BATCH_SIZE,
        batch_merge_repeat=1,
        log_name="",
        devices="1",
    ):
        cmd = self._python_interp
        envs['PADDLE_TRAINER_ENDPOINTS'] = self._ps_endpoints

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            envs['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            cmd += " -m coverage run --branch -p"

        cmd += (
            f" {model} --role trainer --update_method local --lr {self._lr:f}"
        )

        if batch_size != DEFAULT_BATCH_SIZE:
            cmd += f" --batch_size {batch_size}"
        if batch_merge_repeat > 1:
            cmd += f" --batch_merge_repeat {batch_merge_repeat}"
        if self._nccl2_reduce_layer:
            cmd += " --nccl2_reduce_layer_local_run 1"

        if self.__use_cuda:
            cmd += " --use_cuda"
            env_local = {
                "CUDA_VISIBLE_DEVICES": devices,
                "PADDLE_TRAINERS_NUM": "1",
                "PADDLE_TRAINER_ID": "0",
            }
        elif self.__use_xpu:
            cmd += " --use_xpu"
            env_local = {
                "FLAGS_selected_xpus": devices,
                "PADDLE_TRAINERS_NUM": "1",
                "PADDLE_TRAINER_ID": "0",
            }
        else:
            env_local = {'CPU_NUM': '1'}

        # not use dgc in single card
        if len(devices) > 1 and self._use_dgc:
            cmd += " --use_dgc"

        if self._accumulate_gradient:
            cmd += " --accumulate_gradient"

        if self._find_unused_parameters:
            cmd += " --find_unused_parameters"

        env_local.update(envs)
        # print(f"local_cmd: {cmd}, env: {env_local}")

        if check_error_log:
            path = os.path.join(self.temp_dir.name, log_name + "_local.log")
            err_log = open(path, "wb")
            local_proc = subprocess.Popen(
                cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=err_log,
                env=modify_envs(env_local),
            )
        else:
            local_proc = subprocess.Popen(
                cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=modify_envs(env_local),
            )

        local_out, local_err = local_proc.communicate()

        if check_error_log:
            err_log.close()

        # sys.stderr.write('local_stderr: %s\n' % local_err)

        return load_and_remove_dump_file()

    def _run_local_gloo(
        self,
        model,
        envs,
        check_error_log=False,
        batch_size=DEFAULT_BATCH_SIZE,
        batch_merge_repeat=1,
        log_name="",
        devices="0",
    ):
        saved_endpoints = self._ps_endpoints
        self._ps_endpoints = self._ps_endpoints.split(',')[0]
        result = self._run_cluster_gloo(model, envs, 'gloo', False, log_name)
        self._ps_endpoints = saved_endpoints
        return result

    def _run_cluster(self, model, envs, check_error_log, log_name):
        # Run dist train to compare with local results
        ps0, ps1, ps0_pipe, ps1_pipe = self.start_pserver(
            model, check_error_log, envs, log_name=log_name
        )

        ps0_ep, ps1_ep = self._ps_endpoints.split(",")

        tr_cmd = "%s"

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            envs['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            tr_cmd += " -m coverage run --branch -p"

        tr_cmd += " %s --role trainer --endpoints %s --trainer_id %d --current_endpoint %s --trainers %d --update_method pserver --lr %f"

        tr0_cmd = tr_cmd % (
            self._python_interp,
            model,
            self._ps_endpoints,
            0,
            ps0_ep,
            self._trainers,
            self._lr,
        )
        tr1_cmd = tr_cmd % (
            self._python_interp,
            model,
            self._ps_endpoints,
            1,
            ps1_ep,
            self._trainers,
            self._lr,
        )

        if self._sync_mode:
            tr0_cmd += " --sync_mode"
            tr1_cmd += " --sync_mode"
        if self._hogwild_mode:
            tr0_cmd += " --hogwild"
            tr1_cmd += " --hogwild"
        if self._use_reduce:
            tr0_cmd += " --use_reduce"
            tr1_cmd += " --use_reduce"
        if self._use_reader_alloc:
            tr0_cmd += " --use_reader_alloc"
            tr1_cmd += " --use_reader_alloc"
        if self.__use_cuda:
            tr0_cmd += " --use_cuda"
            tr1_cmd += " --use_cuda"
            env0 = {"CUDA_VISIBLE_DEVICES": "0"}
            env1 = {"CUDA_VISIBLE_DEVICES": "1"}
        else:
            env0 = {'CPU_NUM': '1'}
            env1 = {'CPU_NUM': '1'}

        env0.update(envs)
        env1.update(envs)

        # print(f"tr0_cmd: {tr0_cmd}, env: {env0}")
        # print(f"tr1_cmd: {tr1_cmd}, env: {env1}")

        path0 = os.path.join(self.temp_dir.name, log_name + "_tr0_err.log")
        path1 = os.path.join(self.temp_dir.name, log_name + "_tr1_err.log")
        tr0_pipe = open(path0, "wb")
        tr1_pipe = open(path1, "wb")

        print_to_err(type(self).__name__, "going to start trainer process 0")
        tr0_proc = subprocess.Popen(
            tr0_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=tr0_pipe,
            env=modify_envs(env0, 0),
        )
        print_to_err(type(self).__name__, "going to start trainer process 1")
        tr1_proc = subprocess.Popen(
            tr1_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=tr1_pipe,
            env=modify_envs(env1, 1),
        )

        # Wait until trainer process terminate
        while True:
            stat0 = tr0_proc.poll()
            time.sleep(0.1)
            if stat0 is not None:
                break
        while True:
            stat1 = tr1_proc.poll()
            time.sleep(0.1)
            if stat1 is not None:
                break

        tr0_out, tr0_err = tr0_proc.communicate()
        tr1_out, tr1_err = tr1_proc.communicate()

        # close trainer file
        tr0_pipe.close()
        tr1_pipe.close()
        ps0_pipe.close()
        ps1_pipe.close()

        ps0.terminate()
        ps1.terminate()

        return load_and_remove_dump_file(0), load_and_remove_dump_file(1)

    def _get_gloo_trainer_cmd(
        self, model, ep, update_method, trainer_id, trainer_num
    ):
        env = {}
        tr_cmd = "%s -u"

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            tr_cmd += " -m coverage run --branch -p"

        tr_cmd += " %s --role trainer --endpoints %s --trainer_id %d --current_endpoint %s --update_method %s --lr %f"

        tr_cmd = tr_cmd % (
            self._python_interp,
            model,
            self._ps_endpoints,
            trainer_id,
            ep,
            update_method,
            self._lr,
        )

        if self._use_reduce:
            tr_cmd += " --use_reduce"
        if self._use_reader_alloc:
            tr_cmd += " --use_reader_alloc"
        # assert self._use_reduce == False, "gloo not support _use_reduce"
        # assert self._use_reader_alloc == False, "gloo not support _use_reduce"
        if self._save_model:
            tr_cmd += " --save_model"
        if self._diff_batch:
            tr_cmd += " --diff_batch"
        self.__use_cuda = False
        self.__use_xpu = False
        assert not self.__use_cuda, "gloo not support use cuda"
        assert not self.__use_xpu, "gloo not support use xpu"
        tr_cmd += " --use_cpu"
        env.update(
            {
                "PADDLE_TRAINERS_NUM": f"{trainer_num}",
                "PADDLE_TRAINER_ID": f"{trainer_id}",
                "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
                "PADDLE_CURRENT_ENDPOINT": ep,
                "PADDLE_DISTRI_BACKEND": "gloo",
                "GLOG_v": "2",
            }
        )

        assert not self._use_dgc, "gloo not support use dgc"

        if self._accumulate_gradient:
            tr_cmd += " --accumulate_gradient"

        if self._find_unused_parameters:
            tr_cmd += " --find_unused_parameters"

        assert not self._pipeline_mode, "gloo not support use pipeline"

        if self._enable_backward_deps:  # build strategy, save it
            tr_cmd += " --enable_backward_deps"

        if self._fuse_all_reduce is not None:
            tr_cmd += f" --fuse_all_reduce {self._fuse_all_reduce}"

        assert not self._use_fleet_api, "gloo not support use fleet api"
        assert not self._use_fleet_api_20, "gloo not support use fleet api"
        return tr_cmd, env

    def _get_nccl2_trainer_cmd(
        self, model, ep, update_method, trainer_id, trainer_num
    ):
        env = {}
        tr_cmd = "%s -u"

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            tr_cmd += " -m coverage run --branch -p"

        tr_cmd += " %s --role trainer --endpoints %s --trainer_id %d --current_endpoint %s --update_method %s --lr %f"

        tr_cmd = tr_cmd % (
            self._python_interp,
            model,
            self._ps_endpoints,
            trainer_id,
            ep,
            update_method,
            self._lr,
        )

        if self._use_reduce:
            tr_cmd += " --use_reduce"
        if self._use_reader_alloc:
            tr_cmd += " --use_reader_alloc"
        if self._save_model:
            tr_cmd += " --save_model"
        if self.__use_cuda:
            tr_cmd += " --use_cuda"
            env.update(
                {
                    "FLAGS_selected_gpus": f"{0}",
                    "CUDA_VISIBLE_DEVICES": f"{trainer_id}",
                    "PADDLE_TRAINERS_NUM": f"{trainer_num}",
                    "PADDLE_TRAINER_ID": f"{trainer_id}",
                    "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
                    "PADDLE_CURRENT_ENDPOINT": ep,
                }
            )
        # TODO(liuyuhui):XPU_VISIBLE_DEVICES is not working right now,
        # will update it after Badiu Kunlun partners' support.
        elif self.__use_xpu:
            tr_cmd += " --use_xpu"
            env.update(
                {
                    "FLAGS_selected_xpus": f"{trainer_id}",
                    # "XPU_VISIBLE_DEVICES": "{}".format(trainer_id + 1),
                    "PADDLE_TRAINERS_NUM": f"{trainer_num}",
                    "PADDLE_TRAINER_ID": f"{trainer_id}",
                    "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
                    "PADDLE_CURRENT_ENDPOINT": ep,
                    "GLOG_v": "2",
                }
            )
        else:
            env.update({'CPU_NUM': '1'})

        if self._use_dgc:
            tr_cmd += " --use_dgc"

        if self._accumulate_gradient:
            tr_cmd += " --accumulate_gradient"

        if self._find_unused_parameters:
            tr_cmd += " --find_unused_parameters"

        if self._pipeline_mode:
            tr_cmd += " --use_pipeline"
        if self._mp_mode:
            env = {"FLAGS_selected_gpus": f"{trainer_id}"}

        if self._nccl_comm_num > 1:
            tr_cmd += f" --nccl_comm_num {self._nccl_comm_num}"

        if self._use_hallreduce:
            tr_cmd += " --use_hallreduce --hallreduce_inter_nranks 2"

        if self._enable_backward_deps:
            tr_cmd += " --enable_backward_deps"

        if self._fuse_all_reduce is not None:
            tr_cmd += f" --fuse_all_reduce {self._fuse_all_reduce}"

        if self._use_fleet_api:
            tr_cmd += (
                " --use_fleet_api_20"
                if self._use_fleet_api_20
                else " --use_fleet_api"
            )
            if self._use_local_sgd:
                tr_cmd += " --use_local_sgd"
            if self._ut4grad_allreduce:
                tr_cmd += " --ut4grad_allreduce"
            if hasattr(self, '_sync_batch_norm') and self._sync_batch_norm:
                tr_cmd += " --sync_batch_norm"

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            env['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')

        return tr_cmd, env

    def _run_cluster_gloo(
        self, model, envs, update_method, check_error_log, log_name
    ):
        assert (
            update_method == "gloo"
        ), f"_run_cluster_gloo must have update_method: gloo, but get {update_method}"
        assert (
            not self._use_hallreduce
        ), "_run_cluster_gloo must have _use_hallreduce = false"

        worker_endpoints = self._ps_endpoints.split(",")

        trainer_num = len(worker_endpoints)

        procs = []
        pipes = []
        for i in range(0, trainer_num):
            tr_cmd, tr_env = self._get_gloo_trainer_cmd(
                model, worker_endpoints[i], update_method, i, trainer_num
            )
            tr_env.update(envs)
            tr_env["GLOG_vmodule"] = 'gloo_context=4'
            tr_env["GLOG_v"] = '3'
            # print(
            # f"use_hallreduce:{self._use_hallreduce} tr_cmd:{tr_cmd}, env: {tr_env}"
            # )

            path = os.path.join(
                self.temp_dir.name, log_name + f"_tr{i}_err.log"
            )
            tr_pipe = open(path, "wb")

            print_to_err(
                type(self).__name__,
                f"going to start process {i} with nccl2",
            )
            tr_proc = subprocess.Popen(
                tr_cmd.strip().split(" "),
                stdout=subprocess.PIPE,
                stderr=tr_pipe,
                env=modify_envs(tr_env, i),
            )

            procs.append(tr_proc)
            pipes.append(tr_pipe)

        outs = []
        for i in range(0, trainer_num):
            tr_out, tr_err = procs[i].communicate()
            outs.append(tr_out)
            pipes[i].close()
            sys.stderr.write(f'trainer {i} stderr: {tr_err}\n')

        if trainer_num == 1:
            if check_error_log:
                print("outs[0]:", outs[0])
            return load_and_remove_dump_file(0)

        else:
            if check_error_log:
                print("outs[0]:", outs[0])
                print("outs[1]:", outs[1])
            return load_and_remove_dump_file(0), load_and_remove_dump_file(1)

    def _run_cluster_nccl2(
        self, model, envs, update_method, check_error_log, log_name
    ):
        if self._use_hallreduce:
            self._ps_endpoints = ""

            global DIST_UT_PORT
            if DIST_UT_PORT == 0:
                # NOTE(wangxi). hallreduce test must use 4cards after nccl>=2.7
                for i in range(0, 4):
                    self._ps_endpoints += f"127.0.0.1:{self._find_free_port()},"
            else:
                for i in range(0, 4):
                    self._ps_endpoints += "127.0.0.1:%s," % (DIST_UT_PORT + i)
                DIST_UT_PORT += 4
            self._ps_endpoints = self._ps_endpoints[:-1]

        # NOTE: we reuse ps_endpoints as nccl2 worker endpoints
        worker_endpoints = self._ps_endpoints.split(",")

        trainer_num = len(worker_endpoints)

        procs = []
        pipes = []
        for i in range(0, trainer_num):
            tr_cmd, tr_env = self._get_nccl2_trainer_cmd(
                model, worker_endpoints[i], update_method, i, trainer_num
            )
            tr_env.update(envs)
            print(
                f"use_hallreduce:{self._use_hallreduce} tr_cmd:{tr_cmd}, env: {tr_env}"
            )

            path = os.path.join(
                self.temp_dir.name, log_name + f"_tr{i}_err.log"
            )
            tr_pipe = open(path, "wb")

            print_to_err(
                type(self).__name__,
                f"going to start process {i} with nccl2",
            )
            tr_proc = subprocess.Popen(
                tr_cmd.strip().split(" "),
                stdout=subprocess.PIPE,
                stderr=tr_pipe,
                env=modify_envs(tr_env, i),
            )

            procs.append(tr_proc)
            pipes.append(tr_pipe)

        outs = []
        for i in range(0, trainer_num):
            tr_out, tr_err = procs[i].communicate()
            outs.append(tr_out)
            pipes[i].close()
            sys.stderr.write(f'trainer {i} stderr: {tr_err}\n')

        if check_error_log:
            print("outs[0]:", outs[0])
            print("outs[1]:", outs[1])

        return load_and_remove_dump_file(0), load_and_remove_dump_file(1)

    def _run_pipeline(self, model, envs, check_error_log, log_name):
        # NOTE: we reuse ps_endpoints as nccl2 worker endpoints
        worker_endpoints = self._ps_endpoints.split(",")
        update_method = "nccl2"

        trainer_num = len(worker_endpoints)

        procs = []
        pipes = []
        for i in range(0, trainer_num):
            tr_cmd, tr_env = self._get_nccl2_trainer_cmd(
                model, worker_endpoints[i], update_method, i, trainer_num
            )
            tr_env.update(envs)
            tr_env['CUDA_VISIBLE_DEVICES'] = "0,1"
            tr_env['NCCL_SHM_DISABLE'] = '1'
            tr_env['FLAGS_selected_gpus'] = str(i)
            tr_env['FLAGS_cudnn_deterministic'] = '0'
            print(f"tr_cmd:{tr_cmd}, env: {tr_env}")

            path = os.path.join(self.temp_dir.name + f"tr{i}_err.log")
            tr_pipe = open(path, "wb")

            print_to_err(
                type(self).__name__,
                f"going to start process {i} with nccl2",
            )
            tr_proc = subprocess.Popen(
                tr_cmd.strip().split(" "),
                stdout=subprocess.PIPE,
                stderr=tr_pipe,
                env=modify_envs(tr_env, i),
            )

            procs.append(tr_proc)
            pipes.append(tr_pipe)

        outs = []
        for i in range(0, trainer_num):
            tr_out, tr_err = procs[i].communicate()
            outs.append(tr_out)
            pipes[i].close()
            sys.stderr.write(f'trainer {i} stderr: {tr_err}\n')

        if check_error_log:
            print("outs[0]:", outs[0])
            print("outs[1]:", outs[1])
        return load_and_remove_dump_file(0), load_and_remove_dump_file(1)

    def _get_required_envs(self, check_error_log=False, need_envs={}):
        # TODO(typhoonzero): should auto adapt GPU count on the machine.
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_rpc_deadline": "30000",  # 5sec to fail fast
            "FLAGS_rpc_retry_bind_port": "50",
            "FLAGS_cudnn_deterministic": "1",
            "FLAGS_rpc_disable_reuse_port": "1",
            "http_proxy": "",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_SHM_DISABLE": "1",
            "FLAGS_new_executor_static_build": "1",
        }

        if check_error_log:
            required_envs["GLOG_vmodule"] = (
                "alloc_continuous_space_op=10,"
                "alloc_continuous_space_for_grad_pass=10,fast_threaded_ssa_graph_executor=10,executor=10,operator=10,"
                "gen_nccl_id_op=10,gen_nccl_id_op_help=10,nccl_helper=10,grpc_client=10,"
                "grpc_server=10,request_handler_impl=10,section_worker=10"
            )
            required_envs["GLOG_logtostderr"] = "1"

        if os.getenv('NVIDIA_TF32_OVERRIDE', '') is not None:
            required_envs['NVIDIA_TF32_OVERRIDE'] = os.getenv(
                'NVIDIA_TF32_OVERRIDE', ''
            )

        required_envs.update(need_envs)
        return required_envs

    def check_with_place(
        self,
        model_file,
        delta=1e-3,
        check_error_log=False,
        need_envs={},
        log_name="",
    ):
        self.check_with_place_func(
            model_file=model_file,
            delta=delta,
            check_error_log=check_error_log,
            need_envs=need_envs,
            log_name=log_name,
        )

    def check_with_place_func(
        self,
        model_file,
        delta=1e-3,
        check_error_log=False,
        need_envs={},
        log_name="",
    ):
        required_envs = self._get_required_envs(check_error_log, need_envs)

        if self._gloo_mode:
            local_losses = self._run_local_gloo(
                model_file, required_envs, check_error_log, log_name=log_name
            )
        else:
            local_losses = self._run_local(
                model_file, required_envs, check_error_log, log_name=log_name
            )

        if self._nccl2_mode:
            if self._nccl2_reduce_layer:
                tr0_losses, tr1_losses = self._run_cluster_nccl2(
                    model_file,
                    required_envs,
                    update_method="nccl2_reduce_layer",
                    check_error_log=check_error_log,
                    log_name=log_name,
                )
            else:
                tr0_losses, tr1_losses = self._run_cluster_nccl2(
                    model_file,
                    required_envs,
                    update_method='nccl2',
                    check_error_log=check_error_log,
                    log_name=log_name,
                )
        elif self._bkcl_mode:
            tr0_losses, tr1_losses = self._run_cluster_nccl2(
                model_file,
                required_envs,
                update_method='bkcl',
                check_error_log=check_error_log,
                log_name=log_name,
            )
        elif self._gloo_mode:
            # gloo mode, cpu only parallel train @xiongkun03
            tr0_losses, tr1_losses = self._run_cluster_gloo(
                model_file,
                required_envs,
                update_method='gloo',
                check_error_log=check_error_log,
                log_name=log_name,
            )
        elif self._pipeline_mode:
            tr0_losses, tr1_losses = self._run_pipeline(
                model_file, required_envs, check_error_log, log_name=log_name
            )
        else:
            tr0_losses, tr1_losses = self._run_cluster(
                model_file, required_envs, check_error_log, log_name=log_name
            )

        for step_id in range(RUN_STEP):
            local_loss = local_losses[step_id]
            tr0_loss = tr0_losses[step_id]
            tr1_loss = tr1_losses[step_id]
            if self._pipeline_mode:
                dist_loss = np.array([tr1_loss])
            else:
                dist_loss = (np.array([tr0_loss]) + np.array([tr1_loss])) / 2
            print("=======", local_loss, ":", dist_loss[0], "=======")
            self.assertAlmostEqual(local_loss, dist_loss[0], delta=delta)

    def check_with_place_multi_cards(
        self,
        model_file,
        delta=1e-3,
        check_error_log=False,
        need_envs={},
        log_name="",
    ):
        # need open p2p or shm otherwise multi cards mode will hang
        need_envs.update({"NCCL_P2P_DISABLE": "0", "NCCL_SHM_DISABLE": "0"})

        required_envs = self._get_required_envs(check_error_log, need_envs)

        if self._use_dgc:
            multi_cards_losses = self._run_local(
                model_file,
                required_envs,
                check_error_log,
                log_name=log_name + "_dgc_2cards",
                devices="0,1",
            )

            self._use_dgc = False
            base_losses = self._run_local(
                model_file,
                required_envs,
                check_error_log,
                log_name=log_name + "_base_2cards",
                devices="0,1",
            )

            self._use_dgc = True

            for step_id in range(RUN_STEP):
                base_loss = base_losses[step_id]
                multi_cards_loss = multi_cards_losses[step_id]
                print("=======", base_loss, ":", multi_cards_loss, "=======")
                self.assertAlmostEqual(base_loss, multi_cards_loss, delta=delta)
