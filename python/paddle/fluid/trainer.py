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

import os
import core
import framework
import executor
import data_feeder
import contextlib

# optimizer is same as the parameter of Trainer.__init__. Rename it to opt_module
import optimizer as opt_module
import distribute_transpiler

__all__ = [
    'Trainer',
    'BeginEpochEvent',
    'EndEpochEvent',
    'BeginStepEvent',
    'EndStepEvent',
]


class BeginEpochEvent(object):
    def __init__(self, epoch_id):
        self.epoch = epoch_id


class EndEpochEvent(object):
    def __init__(self, epoch_id):
        self.epoch = epoch_id


class BeginStepEvent(object):
    def __init__(self, epoch_id, step_id):
        self.epoch = epoch_id
        self.step = step_id


class EndStepEvent(object):
    def __init__(self, epoch_id, step_id):
        self.epoch = epoch_id
        self.step = step_id


class Trainer(object):
    """

    Args:
        program_func(callable): A function which will return loss. The loss must be a scaler.
        optimizer(optimizer.Optimizer): The optimizer should be an instance of Optimizer
        place: The device place of this trainer.
    """

    def __init__(self, program_func, optimizer, param_path=None, place=None):
        # 1. we need to generate a framework.Program by calling
        # program_func. Reference: fluid.program_guard in
        # test_word2vec.py
        self.scope = core.Scope()

        self.startup_program = framework.Program()
        self.train_program = framework.Program()

        with framework.program_guard(self.train_program, self.startup_program):
            loss = program_func()
            if not isinstance(optimizer, opt_module.Optimizer):
                raise TypeError(
                    "The optimizer should be an instance of Optimizer")

            optimize_ops, params_grads = optimizer.minimize(loss)

        self.place = Trainer._check_and_get_place(place)

        self.dist_transpile_if_necessary(optimize_ops, params_grads)

        # 2. move the default_main_program to self.program and run the
        # default_startup program on an empty core.Scope()
        # Run startup program
        with self._prog_and_scope_guard():
            exe = executor.Executor(place)
            exe.run(self.startup_program)

        if param_path:
            # load params from param_path into scope
            # TODO(yuyang): This depends on parameters implementation.
            pass

    def dist_transpile_if_necessary(self, optimize_ops, params_grads):
        if "PADDLE_TRAINING_ROLE" not in os.environ:
            return

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
        # the unique trainer id, starting from 0, needed by trainer
        # only
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        # the role, should be either PSERVER or TRAINER
        training_role = os.getenv("PADDLE_TRAINING_ROLE")
        with self._prog_and_scope_guard():
            t = distribute_transpiler.DistributeTranspiler()
            t.transpile(
                trainer_id, pservers=pserver_endpoints, trainers=trainers)
            if training_role == "PSERVER":
                self.train_program = t.get_pserver_program(current_endpoint)
                self.startup_program = t.get_startup_program(current_endpoint,
                                                             self.train_program)
            elif training_role == "TRAINER":
                self.train_program = t.get_trainer_program()
            else:
                raise ValueError(
                    'TRAINING_ROLE environment variable must be either TRAINER or PSERVER'
                )

    def train(self,
              num_epochs,
              event_handler,
              reader=None,
              parallel=False,
              feed_order=None):
        """
        Train the model.

        Args:
            num_epochs: The number of epoch. An epoch will process all data in reader
            event_handler: The event handler. A function with type (ev:Event)->void
            reader:
            parallel: True if use multi-CPUs or multi-GPUs
            feed_order: Feeding order of reader. None will following the defining
                order in program

        Returns:

        """
        if parallel:
            raise NotImplementedError(
                "Parallel Executor version of trainer is not implemented")

        training_role = os.getenv("PADDLE_TRAINING_ROLE", "")
        if training_role == "PSERVER":
            with self._prog_and_scope_guard():
                exe = executor.Executor(self.place)
                exe.run()
                return

        self._train_by_executor(num_epochs, event_handler, reader, feed_order)

    def test(self, reader):
        pass

    def save_params(self, param_path):
        # reference: save_persistables in io.py
        pass

    @staticmethod
    def _check_and_get_place(place):
        """
        Check the type of place or get the default place
        Args:
            place(None|core.CUDAPlace|core.CPUPlace): the place that trainer will be executed on.

        Raises:
            TypeError if the type mismatched.

        Returns:
            the original place if it is not None.
            if fluid is compiled with CUDA, returns CUDAPlace(0) by default.
            Otherwise returns CPUPlace by default.
        """
        if place is None:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            else:
                return core.CPUPlace()
        else:
            if not isinstance(place, core.CUDAPlace) and not isinstance(
                    place, core.CPUPlace):
                raise TypeError("Place should be either CUDAPlace or CPUPlace")
            return place

    @contextlib.contextmanager
    def _prog_and_scope_guard(self):
        with framework.program_guard(
                main_program=self.train_program,
                startup_program=self.startup_program):
            with executor.scope_guard(self.scope):
                yield

    def _train_by_executor(self, num_epochs, event_handler, reader, feed_order):
        """
        Train by Executor and single device.

        Args:
            num_epochs:
            event_handler:
            reader:
            feed_order:

        Returns:

        """
        with self._prog_and_scope_guard():
            exe = executor.Executor(self.place)
            if feed_order is None:
                feed_var_list = [
                    var
                    for var in self.train_program.global_block(
                    ).vars.itervalues()
                    if hasattr(var, 'is_data') and var.is_data
                ]
            else:
                feed_var_list = [
                    self.train_program.global_block().var(var_name)
                    for var_name in feed_order
                ]

            feeder = data_feeder.DataFeeder(
                feed_list=feed_var_list, place=self.place)
            for epoch_id in range(num_epochs):
                event_handler(BeginEpochEvent(epoch_id))
                for step_id, data in enumerate(reader()):
                    event_handler(BeginStepEvent(epoch_id, step_id))
                    exe.run(feed=feeder.feed(data), fetch_list=[])
                    event_handler(EndStepEvent(epoch_id, step_id))
                event_handler(EndEpochEvent(epoch_id))
