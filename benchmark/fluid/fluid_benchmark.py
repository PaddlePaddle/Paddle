# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import cProfile
import time
import os

import numpy as np

import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler
import paddle.fluid.transpiler.distribute_transpiler as distribute_transpiler

from args import *


def append_nccl2_prepare(trainer_id):
    if trainer_id >= 0:
        # append gen_nccl_id at the end of startup program
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        port = os.getenv("PADDLE_PSERVER_PORT")
        worker_ips = os.getenv("PADDLE_TRAINER_IPS")
        worker_endpoints = []
        for ip in worker_ips.split(","):
            worker_endpoints.append(':'.join([ip, port]))
        num_trainers = len(worker_endpoints)
        current_endpoint = os.getenv("PADDLE_CURRENT_IP") + ":" + port
        worker_endpoints.remove(current_endpoint)

        nccl_id_var = fluid.default_startup_program().global_block().create_var(
            name="NCCLID",
            persistable=True,
            type=fluid.core.VarDesc.VarType.RAW)
        fluid.default_startup_program().global_block().append_op(
            type="gen_nccl_id",
            inputs={},
            outputs={"NCCLID": nccl_id_var},
            attrs={
                "endpoint": current_endpoint,
                "endpoint_list": worker_endpoints,
                "trainer_id": trainer_id
            })
        return nccl_id_var, num_trainers, trainer_id
    else:
        raise Exception("must set positive PADDLE_TRAINER_ID env variables for "
                        "nccl-based dist train.")


def dist_transpile(trainer_id, args):
    if trainer_id < 0:
        return None, None

    # the port of all pservers, needed by both trainer and pserver
    port = os.getenv("PADDLE_PSERVER_PORT", "6174")
    # comma separated ips of all pservers, needed by trainer and
    # pserver
    pserver_ips = os.getenv("PADDLE_PSERVER_IPS", "")
    eplist = []
    for ip in pserver_ips.split(","):
        eplist.append(':'.join([ip, port]))
    pserver_endpoints = ",".join(eplist)
    # total number of workers/trainers in the job, needed by
    # trainer and pserver
    trainers = int(os.getenv("PADDLE_TRAINERS"))
    # the IP of the local machine, needed by pserver only
    current_endpoint = os.getenv("PADDLE_CURRENT_IP", "") + ":" + port
    # the role, should be either PSERVER or TRAINER
    training_role = os.getenv("PADDLE_TRAINING_ROLE")

    t = distribute_transpiler.DistributeTranspiler()
    t.transpile(
        trainer_id,
        pservers=pserver_endpoints,
        trainers=trainers,
        sync_mode=not args.async_mode,
        slice_var_up=not args.no_split_var)
    if training_role == "PSERVER":
        pserver_program = t.get_pserver_program(current_endpoint)
        pserver_startup_program = t.get_startup_program(current_endpoint,
                                                        pserver_program)
        return pserver_program, pserver_startup_program
    elif training_role == "TRAINER":
        train_program = t.get_trainer_program()
        return train_program, fluid.default_startup_program()
    else:
        raise ValueError(
            'PADDLE_TRAINING_ROLE environment variable must be either TRAINER or PSERVER'
        )


def test(exe, inference_program, test_reader, feeder, batch_acc):
    accuracy_evaluator = fluid.metrics.Accuracy()
    for batch_id, data in enumerate(test_reader()):
        acc = exe.run(inference_program,
                      feed=feeder.feed(data),
                      fetch_list=[batch_acc])
        accuracy_evaluator.update(value=np.array(acc), weight=len(data))

    return accuracy_evaluator.eval()


# TODO(wuyi): replace train, train_parallel, test functions with new trainer
# API once it is ready.
def train(avg_loss, infer_prog, optimizer, train_reader, test_reader, batch_acc,
          args, train_prog, startup_prog):
    if os.getenv("PADDLE_TRAINING_ROLE") == "PSERVER":
        place = core.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        exe.run(train_prog)
        return

    if args.use_fake_data:
        raise Exception(
            "fake data is not supported in single GPU test for now.")

    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if not args.use_reader_op:
        feed_var_list = [
            var for var in train_prog.global_block().vars.itervalues()
            if var.is_data
        ]
        feeder = fluid.DataFeeder(feed_var_list, place)

    iters, num_samples, start_time = 0, 0, time.time()
    for pass_id in range(args.pass_num):
        train_losses = []
        if not args.use_reader_op:
            reader_generator = train_reader()
        batch_id = 0
        data = None
        while True:
            if not args.use_reader_op:
                data = next(reader_generator, None)
                if data == None:
                    break
            if iters == args.iterations:
                break
            if iters == args.skip_batch_num:
                start_time = time.time()
                num_samples = 0

            if args.use_reader_op:
                try:
                    loss = exe.run(train_prog, fetch_list=[avg_loss])
                except fluid.core.EnforceNotMet as ex:
                    break
            else:
                loss = exe.run(train_prog,
                               feed=feeder.feed(data),
                               fetch_list=[avg_loss])
            iters += 1
            batch_id += 1
            # FIXME(wuyi): For use_reader_op, if the current
            # pass is not the last, the last batch of this pass
            # is also equal to args.batch_size.
            if args.use_reader_op:
                num_samples += args.batch_size * args.gpus
            else:
                num_samples += len(data)
            train_losses.append(loss)
            print("Pass: %d, Iter: %d, Loss: %f\n" %
                  (pass_id, iters, np.mean(train_losses)))
        print_train_time(start_time, time.time(), num_samples)
        print("Pass: %d, Loss: %f" % (pass_id, np.mean(train_losses))),
        # evaluation
        if not args.no_test and batch_acc and not args.use_reader_op:
            pass_test_acc = test(exe, infer_prog, test_reader, feeder,
                                 batch_acc)
            print(", Test Accuracy: %f" % pass_test_acc)
        print("\n")
        # TODO(wuyi): add warmup passes to get better perf data.
        exit(0)


# TODO(wuyi): replace train, train_parallel, test functions with new trainer
# API once it is ready.
def train_parallel(avg_loss, infer_prog, optimizer, train_reader, test_reader,
                   batch_acc, args, train_prog, startup_prog, nccl_id_var,
                   num_trainers, trainer_id):
    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    if not args.use_reader_op:
        feed_var_list = [
            var for var in train_prog.global_block().vars.itervalues()
            if var.is_data
        ]
        feeder = fluid.DataFeeder(feed_var_list, place)

    # generate fake:
    if args.use_fake_data:
        for var in feed_var_list:
            v = startup_prog.global_block().clone_variable(var)
            var.persistable = True
            v.persistable = True

            real_shape = list(var.shape)
            real_shape[0] = args.batch_size / args.gpus
            startup_prog.global_block().append_op(
                outputs={"Out": v},
                type="fill_constant",
                attrs={"shape": real_shape,
                       "value": 1.0,
                       "dtype": var.dtype})

    if nccl_id_var and trainer_id == 0:
        #FIXME(wuyi): wait other trainer to start listening
        time.sleep(30)

    startup_exe = fluid.Executor(place)
    startup_exe.run(startup_prog)
    strategy = fluid.ExecutionStrategy()
    strategy.num_threads = 1
    strategy.allow_op_delay = False
    exe = fluid.ParallelExecutor(
        True,
        avg_loss.name,
        exec_strategy=strategy,
        num_trainers=num_trainers,
        trainer_id=trainer_id)

    for pass_id in range(args.pass_num):
        num_samples = 0
        iters = 0
        start_time = time.time()
        if not args.use_reader_op:
            reader_generator = train_reader()
        batch_id = 0
        data = None
        while True:
            if not args.use_reader_op:
                data = next(reader_generator, None)
                if data == None:
                    break
            if iters == args.iterations:
                break
            if args.profile and pass_id == 0 and batch_id == 5:
                profiler.start_profiler("All")
            elif args.profile and pass_id == 0 and batch_id == 10:
                profiler.stop_profiler("total", "/tmp/profile_%d" % trainer_id)

            if iters == args.skip_batch_num:
                start_time = time.time()
                num_samples = 0
            if args.use_fake_data or args.use_reader_op:
                try:
                    loss, = exe.run([avg_loss.name])
                except fluid.core.EnforceNotMet as ex:
                    break
            else:
                loss, = exe.run([avg_loss.name], feed=feeder.feed(data))
            if args.use_reader_op:
                num_samples += args.batch_size * args.gpus
            else:
                num_samples += len(data)
            iters += 1
            if batch_id % 1 == 0:
                print("Pass %d, batch %d, loss %s" %
                      (pass_id, batch_id, np.array(loss)))
            batch_id += 1

        print_train_time(start_time, time.time(), num_samples)
        if not args.no_test and batch_acc and not args.use_reader_op:
            # we have not implement record io for test
            # skip test when use args.use_reader_op
            test_acc = test(startup_exe, infer_prog, test_reader, feeder,
                            batch_acc)
            print("Pass: %d, Test Accuracy: %f\n" % (pass_id, test_acc))


def print_arguments(args):
    vars(args)['use_nvprof'] = (vars(args)['use_nvprof'] and
                                vars(args)['device'] == 'GPU')
    print('----------- Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def print_train_time(start_time, end_time, num_samples):
    train_elapsed = end_time - start_time
    examples_per_sec = num_samples / train_elapsed
    print('\nTotal examples: %d, total time: %.5f, %.5f examples/sed\n' %
          (num_samples, train_elapsed, examples_per_sec))


def print_paddle_envs():
    print('----------- Configuration envs -----------')
    for k in os.environ:
        if "PADDLE_" in k:
            print "ENV %s:%s" % (k, os.environ[k])
    print('------------------------------------------------')


def main():
    args = parse_args()
    print_arguments(args)
    print_paddle_envs()

    # the unique trainer id, starting from 0, needed by trainer
    # only
    nccl_id_var, num_trainers, trainer_id = (
        None, 1, int(os.getenv("PADDLE_TRAINER_ID", "0")))

    if args.use_cprof:
        pr = cProfile.Profile()
        pr.enable()
    model_def = __import__("models.%s" % args.model, fromlist=["models"])
    train_args = list(model_def.get_model(args))
    train_args.append(args)
    # Run optimizer.minimize(avg_loss)
    train_args[2].minimize(train_args[0])
    if args.memory_optimize:
        fluid.memory_optimize(fluid.default_main_program())

    if args.update_method == "pserver":
        train_prog, startup_prog = dist_transpile(trainer_id, args)
        if not train_prog:
            raise Exception(
                "Must configure correct environments to run dist train.")
        train_args.extend([train_prog, startup_prog])
        if args.gpus > 1 and os.getenv("PADDLE_TRAINING_ROLE") == "TRAINER":
            train_args.extend([nccl_id_var, num_trainers, trainer_id])
            train_parallel(*train_args)
        train(*train_args)
        exit(0)

    # for other update methods, use default programs
    train_args.append(fluid.default_main_program())
    train_args.append(fluid.default_startup_program())

    if args.update_method == "nccl2":
        nccl_id_var, num_trainers, trainer_id = append_nccl2_prepare(trainer_id)
    if args.gpus == 1:
        # NOTE: parallel executor use profiler interanlly
        if args.use_nvprof and args.device == 'GPU':
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                train(*train_args)
        else:
            train(*train_args)
    else:
        if args.device == "CPU":
            raise Exception("Only support GPU perf with parallel exe")
        train_args.extend([nccl_id_var, num_trainers, trainer_id])
        train_parallel(*train_args)


if __name__ == "__main__":
    main()
