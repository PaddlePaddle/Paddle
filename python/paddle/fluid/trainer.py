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
        network_func(callable): A function which will return loss. The loss must be a scaler.
        optimizer(optimizer.Optimizer): The optimizer should be an instance of Optimizer
        params:
        place: The device place of this trainer.
    """

    def __init__(self, network_func, optimizer, params=None, place=None):
        # 1. we need to generate a framework.Program by calling
        # network_func. Reference: fluid.program_guard in
        # test_word2vec.py
        self.scope = self._get_scope_from_params(params)

        self.startup_program = framework.Program()
        self.train_program = framework.Program()

        with framework.program_guard(self.train_program, self.startup_program):
            loss = network_func()
            if not isinstance(optimizer, opt_module.Optimizer):
                raise TypeError(
                    "The optimizer should be an instance of Optimizer")

            optimizer.minimize(loss)

        self.place = Trainer._check_and_get_place(place)

        # 2. move the default_main_program to self.program and run the
        # default_startup program on an empty core.Scope()
        # Run startup program
        if params is None:
            exe = executor.Executor(place)
            exe.run(self.startup_program, scope=self.scope)

        # 3. call self.params.add_vars with the initialized scope, it
        # will add the new vars of the initialized scope into
        # self.params.
        # TODO(yuyang): This depends on parameters implementation.

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

    def test(self, reader):
        pass

    def _get_scope_from_params(self, params):
        """
        Get Scope from parameter object.
        Args:
            params(Parameter|None): The parameter object instance. Could be None.

        Returns: New scope if params is None. Or params.scope()
        NOTE: This method is WIP. Not fully implemented.
        """
        if params is None:
            return core.Scope()  # new scope when params is None
        else:
            raise NotImplementedError("Not implemented right now.")

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
