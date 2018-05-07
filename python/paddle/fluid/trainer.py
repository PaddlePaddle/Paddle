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

import core
import framework
import executor
import data_feeder
import contextlib
import pdb

# optimizer is same as the parameter of Trainer.__init__. Rename it to opt_module
import optimizer as opt_module

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
            self.test_output = loss
            self.test_program = self.train_program.clone()
            if not isinstance(optimizer, opt_module.Optimizer):
                raise TypeError(
                    "The optimizer should be an instance of Optimizer")

            optimizer.minimize(loss)

        self.place = Trainer._check_and_get_place(place)

        # 2. move the default_main_program to self.program and run the
        # default_startup program on an empty core.Scope()
        # Run startup program
        exe = executor.Executor(place)
        exe.run(self.startup_program, scope=self.scope)

        if param_path:
            # load params from param_path into scope
            # TODO(yuyang): This depends on parameters implementation.
            pass

        # TODO(helin): support distributed training

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

        self._train_by_executor(num_epochs, event_handler, reader, feed_order)

    def test(self, reader, feed_order=None):
        """
        Test the model on given test data

        Args:
            reader: The reader that yields test data.
            feed_order: Feeding order of reader. None will following the defining
                order in program
        """

        return self._test_by_executor(reader, feed_order, [self.test_output])

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
                    for var in self.train_program.global_block()
                    .vars.itervalues() if var.is_data
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

    def _test_by_executor(self, reader, feed_order, fetch_list):
        with executor.scope_guard(self.scope):
            if feed_order is None:
                feed_var_list = [
                    var
                    for var in self.train_program.global_block()
                    .vars.itervalues() if var.is_data
                ]
            else:
                feed_var_list = [
                    self.train_program.global_block().var(var_name)
                    for var_name in feed_order
                ]

            exe = executor.Executor(self.place)

            feeder = data_feeder.DataFeeder(
                feed_list=feed_var_list, place=self.place)
            accumulated_loss = 0
            count = 0
            for data in reader():
                loss, = exe.run(program=self.test_program,
                                feed=feeder.feed(data),
                                fetch_list=fetch_list)
                accumulated_loss += loss[0]
                count += 1

            return accumulated_loss / count
