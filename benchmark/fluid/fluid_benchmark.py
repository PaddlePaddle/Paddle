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

BENCHMARK_MODELS = [
    "machine_translation", "resnet", "vgg", "mnist", "stacked_dynamic_lstm"
]


def parse_args():
    parser = argparse.ArgumentParser('Fluid model benchmarks.')
    parser.add_argument(
        '--model',
        type=str,
        choices=BENCHMARK_MODELS,
        default='resnet',
        help='The model to run benchmark with.')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='The minibatch size.')
    # TODO(wuyi): add this option back.
    # parser.add_argument(
    #     '--use_fake_data',
    #     action='store_true',
    #     help='use real data or fake data')
    parser.add_argument(
        '--skip_batch_num',
        type=int,
        default=5,
        help='The first num of minibatch num to skip, for better performance test'
    )
    parser.add_argument(
        '--iterations', type=int, default=80, help='The number of minibatches.')
    parser.add_argument(
        '--pass_num', type=int, default=100, help='The number of passes.')
    parser.add_argument(
        '--data_format',
        type=str,
        default='NCHW',
        choices=['NCHW', 'NHWC'],
        help='The data data_format, now only support NCHW.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Use ParallelExecutor to run multi GPU tests.')
    parser.add_argument(
        '--data_set',
        type=str,
        default='flowers',
        choices=['cifar10', 'flowers'],
        help='Optional dataset for benchmark.')
    parser.add_argument(
        '--infer_only', action='store_true', help='If set, run forward only.')
    parser.add_argument(
        '--use_cprof', action='store_true', help='If set, use cProfile.')
    parser.add_argument(
        '--use_nvprof',
        action='store_true',
        help='If set, use nvprof for CUDA.')
    parser.add_argument(
        '--with_test',
        action='store_true',
        help='If set, test the testset during training.')
    parser.add_argument(
        '--memory_optimize',
        action='store_true',
        help='If set, optimize runtime memory before start.')
    parser.add_argument(
        '--update_method',
        type=str,
        default='local',
        choices=['local', 'pserver', 'nccl2'],
        help='Choose parameter update method, can be local, pserver, nccl2.')
    args = parser.parse_args()
    return args


def append_nccl2_prepare():
    if os.getenv("PADDLE_TRAINER_ID", None) != None:
        # append gen_nccl_id at the end of startup program
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        port = os.getenv("PADDLE_PORT")
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
        raise Exception(
            "must set PADDLE_TRAINER_ID env variables for dist train.")


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
          args):
    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    feed_var_list = [
        var
        for var in fluid.default_main_program().global_block().vars.itervalues()
        if var.is_data
    ]
    feeder = fluid.DataFeeder(feed_var_list, place)

    iters, num_samples, start_time = 0, 0, time.time()
    for pass_id in range(args.pass_num):
        train_losses = []
        for batch_id, data in enumerate(train_reader()):
            if iters == args.skip_batch_num:
                start_time = time.time()
                num_samples = 0
            if iters == args.iterations:
                break
            loss = exe.run(fluid.default_main_program(),
                           feed=feeder.feed(data),
                           fetch_list=[avg_loss])
            iters += 1
            num_samples += len(data)
            train_losses.append(loss)
            print("Pass: %d, Iter: %d, Loss: %f\n" %
                  (pass_id, iters, np.mean(train_losses)))
        train_elapsed = time.time() - start_time
        examples_per_sec = num_samples / train_elapsed
        print('\nTotal examples: %d, total time: %.5f, %.5f examples/sec\n' %
              (num_samples, train_elapsed, examples_per_sec))
        print("Pass: %d, Loss: %f" % (pass_id, np.mean(train_losses)))
        # evaluation
        if args.with_test:
            pass_test_acc = test(exe, infer_prog, test_reader, feeder,
                                 batch_acc)
            print(", Test Accuracy: %f" % pass_test_acc)
        print("\n")
        # TODO(wuyi): add warmup passes to get better perf data.
        exit(0)


def train_parallel(avg_loss, infer_prog, optimizer, train_reader, test_reader,
                   batch_acc, args, nccl_id_var, num_trainers, trainer_id):
    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    startup_exe = fluid.Executor(place)
    startup_exe.run(fluid.default_startup_program())
    exe = fluid.ParallelExecutor(
        True,
        avg_loss.name,
        num_threads=1,
        allow_op_delay=False,
        num_trainers=num_trainers,
        trainer_id=trainer_id)
    feed_var_list = [
        var
        for var in fluid.default_main_program().global_block().vars.itervalues()
        if var.is_data
    ]
    feeder = fluid.DataFeeder(feed_var_list, place)
    for pass_id in range(args.pass_num):
        num_samples = 0
        iters = 0
        start_time = time.time()
        for batch_id, data in enumerate(train_reader()):
            if iters == args.skip_batch_num:
                start_time = time.time()
                num_samples = 0
            if iters == args.iterations:
                break
            loss, = exe.run([avg_loss.name], feed=feeder.feed(data))
            num_samples += len(data)
            iters += 1
            if batch_id % 1 == 0:
                print("Pass %d, batch %d, loss %s" %
                      (pass_id, batch_id, np.array(loss)))
        train_elapsed = time.time() - start_time
        examples_per_sec = num_samples / train_elapsed
        print('\nTotal examples: %d, total time: %.5f, %.5f examples/sed\n' %
              (num_samples, train_elapsed, examples_per_sec))
        if args.with_test:
            test_acc = test(startup_exe, infer_prog, test_reader, feeder,
                            batch_acc)
            print("Pass: %d, Test Accuracy: %f\n" % (pass_id, test_acc))
        exit(0)


def print_arguments(args):
    vars(args)['use_nvprof'] = (vars(args)['use_nvprof'] and
                                vars(args)['device'] == 'GPU')
    print('----------- resnet Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def main():
    args = parse_args()
    print_arguments(args)
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
        # TODO(wuyi): do transpile here
        exit(0)

    nccl_id_var, num_trainers, trainer_id = None, 1, 0
    if args.update_method == "nccl2":
        nccl_id_var, num_trainers, trainer_id = append_nccl2_prepare()
    if args.parallel == 0:
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
