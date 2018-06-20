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

import contextlib
import os

import core

import data_feeder
import executor
import framework
import io
# optimizer is same as the parameter of Trainer.__init__. Rename it to opt_module
import optimizer as opt_module
import parallel_executor
from transpiler import distribute_transpiler

__all__ = [
    'Trainer', 'BeginEpochEvent', 'EndEpochEvent', 'BeginStepEvent',
    'EndStepEvent', 'CheckpointConfig'
]


class BeginEpochEvent(object):
    """
    The begin of a training epoch.

    Args:
        epoch_id(int): The current epoch ID.
    """

    def __init__(self, epoch_id):
        self.epoch = epoch_id


class EndEpochEvent(object):
    """
    The end of a training epoch.

    Args:
        epoch_id(int): The current epoch ID.
    """

    def __init__(self, epoch_id):
        self.epoch = epoch_id


class BeginStepEvent(object):
    """
    The begin of a training epoch.

    Args:
        epoch_id(int): The current epoch ID.
        step_id(int): The current step ID.
    """

    def __init__(self, epoch_id, step_id):
        self.epoch = epoch_id
        self.step = step_id
        self.fetch_metrics = True
        """
        If fetch_metrics is true, the metrics will be fetched at the 
        EndStepEvent. Default is True.
        """


class EndStepEvent(object):
    """
    The end of a training step.

    Args:
        epoch_id(int): The current epoch ID.
        step_id(int): The current step ID.
        metrics(list): A list of fetched tensor. The order of this list is same
            as the :code:`train_func` returns.
    """

    def __init__(self, epoch_id, step_id, metrics):
        self.epoch = epoch_id
        self.step = step_id
        self.metrics = metrics


class CheckpointConfig(object):
    """
    Parameter object for :code:`fluid.io.save_checkpoint` and
    :code:`fluid.Trainer`. Used to configuration how to save checkpoint.

    Args:
        checkpoint_dir(str): Directory path to save check point. Default is the
            current directory.

        max_num_checkpoints(int): The max number of local check points.
        epoch_interval(int): Every number of epoch to save check point.
        step_interval(int): Every number of step to save check point.

    Examples:
        >>> config = fluid.CheckpointConfig("./checkpoints")
        >>> trainer = fluid.Trainer(train_func=train_program,
        >>>                         place=place,
        >>>                         optimizer_func=optimizer_func,
        >>>                         checkpoint_config=config)
        >>> trainer.train(...)
    """

    def __init__(self,
                 checkpoint_dir=None,
                 max_num_checkpoints=3,
                 epoch_interval=1,
                 step_interval=10):
        if checkpoint_dir is None:
            self.checkpoint_dir = os.getcwd()
        else:
            self.checkpoint_dir = checkpoint_dir

        self.max_num_checkpoints = max_num_checkpoints

        if epoch_interval < 1:
            self.epoch_interval = 1
        else:
            self.epoch_interval = epoch_interval

        if step_interval < 1:
            self.step_interval = 10
        else:
            self.step_interval = step_interval

        self.epoch_id = 0
        self.step_id = 0
        self.load_serial = None
        self.is_pserver = False


def check_and_get_place(place):
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


class Trainer(object):
    """
    A trainer wraps MultiGPU/MultiNode training loops and can be used to train a
    simple neural network easily.

    This API takes a :code:`train_func`. A :code:`train_func` is a function that
    return loss as it first return value. The reset value can be fetched by
    EndStepEvent.metrics

    This API also takes a :code:`optimizer_func` that will return an optimizer
    instance.

    For example, to train a MLP for MNIST dataset, the sample program is

    >>> import paddle.fluid as fluid
    >>>
    >>> def mlp(image, layer_sizes=[200, 100], activation="relu", num_classes=10):
    >>>     hidden = image
    >>>     for layer_size in layer_sizes:
    >>>         hidden = fluid.layers.fc(input=hidden, size=layer_size, act=activation)
    >>>     return fluid.layers.fc(input=hidden, size=num_classes, act="softmax")
    >>>
    >>> def train_mnist_mlp():
    >>>     img = fluid.layers.data(name='image', shape=[784])
    >>>     label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    >>>     prediction = mlp(img)
    >>>     return fluid.layers.mean(fluid.layers.cross_entropy(prediction, label))
    >>>
    >>> def optimizer():
    >>>     return fluid.optimizer.Adam()
    >>>
    >>> trainer = Trainer(train_func=train_mnist_mlp,
    >>>                   optimizer_func=optimizer,
    >>>                   place=fluid.CUDAPlace(0),
    >>>                   parallel=True)
    >>>
    >>> def train_callback(event):
    >>>     if isinstance(event, fluid.EndStepEvent):
    >>>         print "Epoch ID", event.epoch, "Step ID",\
    >>>             event.step, "AvgLoss", event.metrics[0]
    >>>     elif isinstance(event, fluid.EndEpochEvent):
    >>>         trainer.save_params("./model_{0}".format(event.epoch))
    >>>
    >>> trainer.train(num_epochs=100, event_handler=train_callback)

    For more example, please see :ref:`api_guide_high_level_api`.


    Args:
        train_func(callable): A function which will return loss. The loss must be
            a scalar tensor.
        optimizer_func(callable): A function that returns an Optimizer object.
        place(CUDAPlace|CPUPlace): The device place of this trainer. If
            :code:`parallel=True,` all CUDA Places will be used if :code:`place`
            is a :code:`CUDAPlace`.
        parallel(bool): True if use multiple devices.
        checkpoint_config(CheckpointConfig): Configuration about how to save
            checkpoints.
    """

    def __init__(self,
                 train_func,
                 optimizer_func,
                 param_path=None,
                 place=None,
                 parallel=False,
                 checkpoint_config=None):
        self.__stop = False
        self.parallel = parallel

        # config for checkpoint
        # only chief worker will save variables
        self.trainer_id = 0
        self.checkpoint_cfg = checkpoint_config
        if self.checkpoint_cfg:
            assert isinstance(self.checkpoint_cfg, CheckpointConfig)
            serial = io.get_latest_checkpoint_serial(
                self.checkpoint_cfg.checkpoint_dir)
            self.checkpoint_cfg.load_serial = serial if serial >= 0 else None

        self.scope = core.Scope()

        # 1. we need to generate a framework.Program by calling
        # program_func. Reference: fluid.program_guard in
        # test_word2vec.py

        self.startup_program = framework.Program()
        self.train_program = framework.Program()

        with framework.program_guard(self.train_program, self.startup_program):
            program_func_outs = train_func()
            self.train_func_outputs = program_func_outs if isinstance(
                program_func_outs, list) else [program_func_outs]
            self.test_program = self.train_program.clone(for_test=True)

            # The first element of program_func_outs is loss.
            loss = self.train_func_outputs[0]

            optimizer = optimizer_func()
            if not isinstance(optimizer, opt_module.Optimizer):
                raise TypeError(
                    "The optimizer should be an instance of Optimizer")
            optimize_ops, params_grads = optimizer.minimize(loss)

        self.place = check_and_get_place(place)

        self._dist_transpile_if_necessary(optimize_ops, params_grads)

        # 2. move the default_main_program to self.program and run the
        # default_startup program on an empty core.Scope()
        # Run startup program
        with self._prog_and_scope_guard():
            exe = executor.Executor(place)
            exe.run(self.startup_program)

        if self.checkpoint_cfg and self.checkpoint_cfg.load_serial:
            with self._prog_and_scope_guard():
                exe = executor.Executor(place)
                io.load_checkpoint(exe, self.checkpoint_cfg.checkpoint_dir,
                                   self.checkpoint_cfg.load_serial,
                                   self.startup_program)

            if not self.checkpoint_cfg.is_pserver:
                epoch_id, step_id = io.load_trainer_args(
                    self.checkpoint_cfg.checkpoint_dir,
                    self.checkpoint_cfg.load_serial, self.trainer_id,
                    self._get_checkpoint_load_args())
                self.checkpoint_cfg.epoch_id = int(epoch_id)
                self.checkpoint_cfg.step_id = int(step_id)

        if param_path and os.path.isdir(param_path):
            # load params from param_path into scope
            io.load_persist_vars_without_grad(
                exe, dirname=param_path, program=self.startup_program)

    def _transpile_nccl2_dist(self):
        # PADDLE_TRAINER_IPS
        if "PADDLE_TRAINER_IPS" not in os.environ:
            self.nccl_id_var = None
        else:
            self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
            port = os.getenv("PADDLE_PSERVER_PORT")
            worker_ips = os.getenv("PADDLE_TRAINER_IPS")
            worker_endpoints = []
            for ip in worker_ips.split(","):
                worker_endpoints.append(':'.join([ip, port]))
            self.num_trainers = len(worker_endpoints)
            current_endpoint = os.getenv("POD_IP") + ":" + port
            worker_endpoints.remove(current_endpoint)
            # TODO(wuyi): use self.nccl_id_var, self.num_trainers and self.trainer_id
            # in ParallelExecutor to start
            # distributed training using NCCL2
            self.nccl_id_var = self.startup_program.global_block().create_var(
                name="NCCLID", persistable=True, type=core.VarDesc.VarType.RAW)
            self.startup_program.global_block().append_op(
                type="gen_nccl_id",
                inputs={},
                outputs={"NCCLID": self.nccl_id_var},
                attrs={
                    "endpoint": current_endpoint,
                    "endpoint_list": worker_endpoints,
                    "trainer_id": self.trainer_id
                })

    def _dist_transpile_if_necessary(self, optimize_ops, params_grads):
        self._transpile_nccl2_dist()
        if self.nccl_id_var != None:
            return

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
        self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))

        # the role, should be either PSERVER or TRAINER
        training_role = os.getenv("PADDLE_TRAINING_ROLE")
        with self._prog_and_scope_guard():
            t = distribute_transpiler.DistributeTranspiler()
            t.transpile(
                self.trainer_id, pservers=pserver_endpoints, trainers=trainers)
            if training_role == "PSERVER":
                if self.checkpoint_cfg:
                    self.is_pserver = True

                self.train_program = t.get_pserver_program(current_endpoint)
                self.startup_program = t.get_startup_program(current_endpoint,
                                                             self.train_program)
            elif training_role == "TRAINER":
                self.train_program = t.get_trainer_program()
            else:
                raise ValueError(
                    'TRAINING_ROLE environment variable must be either TRAINER or PSERVER'
                )

    def stop(self):
        """
        stop training
        """
        self.__stop = True

    def train(self, num_epochs, event_handler, reader=None, feed_order=None):
        """
        Start the train loop to train the model.

        Args:
            num_epochs(int): The number of epoch. An epoch will process all data in reader
            event_handler(callable): The event handler. A function with type (ev:Event)->void
            reader(callable): A reader creator object. See also
                :ref:`api_guide_python_reader` .
            feed_order(list): Feeding order of reader. None will following the defining
                order in program

        Returns:
            None
        """
        training_role = os.getenv("PADDLE_TRAINING_ROLE", "")
        if training_role == "PSERVER":
            with self._prog_and_scope_guard():
                exe = executor.Executor(self.place)
                exe.run()
                return
        if self.parallel:
            self._train_by_parallel_executor(num_epochs, event_handler, reader,
                                             feed_order)
        else:
            self._train_by_executor(num_epochs, event_handler, reader,
                                    feed_order)

    def test(self, reader, feed_order):
        """
        Test the model on given test data

        Args:
            reader(callable): The reader that yields test data.
            feed_order(list): Feeding order of reader. None will following the
                defining order in program
        """

        return self._test_by_executor(reader, feed_order,
                                      self.train_func_outputs)

    def save_params(self, param_path):
        """
        Save all parameters into :code:`param_path`.

        Args:
            param_path(str): The path to save parameters.

        Returns:
            None
        """
        with self._prog_and_scope_guard():
            exe = executor.Executor(self.place)
            io.save_persistables(exe, dirname=param_path)

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
            feed_var_list = build_feed_var_list(self.train_program, feed_order)
            feeder = data_feeder.DataFeeder(
                feed_list=feed_var_list, place=self.place)
            exe = executor.Executor(self.place)
            reader = feeder.decorate_reader(reader, multi_devices=False)
            self._train_by_any_executor(event_handler, exe, num_epochs, reader)

    def _train_by_any_executor(self, event_handler, exe, num_epochs, reader):
        if self.checkpoint_cfg:
            epochs = [
                epoch_id for epoch_id in range(num_epochs)
                if epoch_id >= self.checkpoint_cfg.epoch_id
            ]
        else:
            epochs = [epoch_id for epoch_id in range(num_epochs)]

        for epoch_id in epochs:
            event_handler(BeginEpochEvent(epoch_id))
            for step_id, data in enumerate(reader()):
                if self.__stop:
                    if self.checkpoint_cfg:
                        self._clean_checkpoint()
                    return

                if self.checkpoint_cfg and self.checkpoint_cfg.load_serial \
                    and self.checkpoint_cfg.step_id >= step_id and self.checkpoint_cfg.epoch_id == epoch_id:
                    continue

                begin_event = BeginStepEvent(epoch_id, step_id)
                event_handler(begin_event)
                if begin_event.fetch_metrics:
                    metrics = exe.run(feed=data,
                                      fetch_list=[
                                          var.name
                                          for var in self.train_func_outputs
                                      ])
                else:
                    metrics = exe.run(feed=data, fetch_list=[])

                if self.checkpoint_cfg:
                    self._save_checkpoint(epoch_id, step_id)
                event_handler(EndStepEvent(epoch_id, step_id, metrics))
            event_handler(EndEpochEvent(epoch_id))
        if self.checkpoint_cfg:
            self._clean_checkpoint()

    def _test_by_executor(self, reader, feed_order, fetch_list):
        with executor.scope_guard(self.scope):
            feed_var_list = build_feed_var_list(self.test_program, feed_order)
            feeder = data_feeder.DataFeeder(
                feed_list=feed_var_list, place=self.place)
            exe = executor.Executor(self.place)
            accumulated = len(fetch_list) * [0]
            count = 0
            for data in reader():
                outs = exe.run(program=self.test_program,
                               feed=feeder.feed(data),
                               fetch_list=fetch_list)
                accumulated = [x[0] + x[1][0] for x in zip(accumulated, outs)]
                count += 1

            return [x / count for x in accumulated]

    def _train_by_parallel_executor(self, num_epochs, event_handler, reader,
                                    feed_order):
        with self._prog_and_scope_guard():
            pe = self._get_or_create_parallel_executor()
            feed_var_list = build_feed_var_list(self.train_program, feed_order)
            feeder = data_feeder.DataFeeder(
                feed_list=feed_var_list, place=self.place)
            reader = feeder.decorate_reader(reader, multi_devices=True)
            self._train_by_any_executor(event_handler, pe, num_epochs, reader)

    def _get_parallel_executor(self):
        return getattr(self, 'parallel_executor', None)

    def _get_or_create_parallel_executor(self):
        if self._get_parallel_executor() is None:
            self.parallel_executor = parallel_executor.ParallelExecutor(
                use_cuda=isinstance(self.place, core.CUDAPlace),
                loss_name=self.train_func_outputs[0].name)
        return self._get_parallel_executor()

    def _clean_checkpoint(self):
        assert self.checkpoint_cfg
        io.clean_checkpoint(checkpoint_dir=self.checkpoint_cfg.checkpoint_dir)

    def _get_checkpoint_load_args(self):
        """
        epoch_id and step_id are runtime arguments, they are not variables, will load them independently.
        """
        return ["epoch_id", "step_id"]

    def _get_checkpoint_save_args(self, epoch_id, step_id):
        """
        epoch_id and step_id are runtime arguments, they are not variables, will save them independently.
        """
        trainer_args = {}
        trainer_args["epoch_id"] = epoch_id
        trainer_args["step_id"] = step_id
        return trainer_args

    def _save_checkpoint(self, epoch_id, step_id):
        assert self.checkpoint_cfg

        if epoch_id % self.checkpoint_cfg.epoch_interval == 0 and step_id % self.checkpoint_cfg.step_interval == 0:
            exe = executor.Executor(self.place)
            io.save_checkpoint(
                executor=exe,
                checkpoint_dir=self.checkpoint_cfg.checkpoint_dir,
                trainer_id=self.trainer_id,
                trainer_args=self._get_checkpoint_save_args(epoch_id, step_id),
                main_program=self.train_program,
                max_num_checkpoints=self.checkpoint_cfg.max_num_checkpoints)


def build_feed_var_list(program, feed_order):
    if not isinstance(program, framework.Program):
        raise TypeError("The 'program' should be an object of Program")

    if isinstance(feed_order, list):
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]
    else:
        if not isinstance(feed_order, dict):
            raise TypeError(
                "The 'feed_order' should be either None, list or dict.")
        if not sorted(feed_order.values()) == range(len(feed_order)):
            raise ValueError(
                "The values of 'feed_order' should be a permutation of [0, len(feed_order))"
            )
        sorted_pair_list = sorted(feed_order.items(), key=lambda item: item[1])
        feed_var_list = [
            program.global_block().var(pair[0]) for pair in sorted_pair_list
        ]
    return feed_var_list
