# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import numbers
import os
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np

import paddle
from paddle.utils import try_import

from .progressbar import ProgressBar

if TYPE_CHECKING:
    from typing import Any, Literal, TypedDict

    from collection.abc import Iterator, Sequence
    from typing_extensions import TypeAlias

    from .model import Model

    _CallbackMode: TypeAlias = Literal["train", "eval", "predict"]

    class _CallbackParams(TypedDict):
        batch_size: int
        epochs: int
        steps: int
        verbose: int
        metrics: list[str]

    class _CallbackLogs(TypedDict):
        loss: float
        metrics: list[str]
        batch_size: int


__all__ = []


def config_callbacks(
    callbacks: Sequence[Callback] | Callback | None = None,
    model: Model | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    steps: int | None = None,
    log_freq: int = 2,
    verbose: int = 2,
    save_freq: int = 1,
    save_dir: str | None = None,
    metrics: list[str] | None = None,
    mode: Literal["train", "test"] = 'train',
) -> CallbackList:
    _cbks = callbacks or []
    cbks: list[Callback] = list(
        _cbks if isinstance(_cbks, (list, tuple)) else [_cbks]
    )
    if not any(isinstance(k, ProgBarLogger) for k in cbks) and verbose:
        cbks = [ProgBarLogger(log_freq, verbose=verbose), *cbks]

    if not any(isinstance(k, ModelCheckpoint) for k in cbks):
        cbks = [*cbks, ModelCheckpoint(save_freq, save_dir)]

    for k in cbks:
        if isinstance(k, EarlyStopping):
            k.save_dir = save_dir
    if not any(isinstance(k, LRScheduler) for k in cbks):
        cbks = [*cbks, LRScheduler()]

    cbk_list = CallbackList(cbks)
    cbk_list.set_model(model)
    metrics = metrics or [] if mode != 'test' else []
    params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps,
        'verbose': verbose,
        'metrics': metrics,
    }
    cbk_list.set_params(params)
    return cbk_list


class CallbackList:
    def __init__(self, callbacks: Sequence[Callback] | None = None) -> None:
        # copy
        assert callbacks is not None
        self.callbacks = list(callbacks)
        self.params = {}
        self.model = None

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def __iter__(self) -> Iterator[Callback]:
        return iter(self.callbacks)

    def set_params(self, params: _CallbackParams) -> None:
        for c in self.callbacks:
            c.set_params(params)

    def set_model(self, model: Model) -> None:
        for c in self.callbacks:
            c.set_model(model)

    def _call(self, name: str, *args: Any) -> None:
        for c in self.callbacks:
            func = getattr(c, name)
            func(*args)

    def _check_mode(self, mode: _CallbackMode) -> None:
        assert mode in [
            'train',
            'eval',
            'predict',
        ], 'mode should be train, eval or predict'

    def on_begin(
        self, mode: _CallbackMode, logs: _CallbackLogs | None = None
    ) -> None:
        self._check_mode(mode)
        name = f'on_{mode}_begin'
        self._call(name, logs)

    def on_end(
        self, mode: _CallbackMode, logs: _CallbackLogs | None = None
    ) -> None:
        self._check_mode(mode)
        name = f'on_{mode}_end'
        self._call(name, logs)

    def on_epoch_begin(
        self, epoch: int | None = None, logs: _CallbackLogs | None = None
    ) -> None:
        self._call('on_epoch_begin', epoch, logs)

    def on_epoch_end(
        self, epoch: int | None = None, logs: _CallbackLogs | None = None
    ) -> None:
        self._call('on_epoch_end', epoch, logs)

    def on_batch_begin(
        self,
        mode: _CallbackMode,
        step: int | None = None,
        logs: _CallbackLogs | None = None,
    ) -> None:
        self._check_mode(mode)
        name = f'on_{mode}_batch_begin'
        self._call(name, step, logs)

    def on_batch_end(
        self,
        mode: _CallbackMode,
        step: int | None = None,
        logs: _CallbackLogs | None = None,
    ) -> None:
        self._check_mode(mode)
        name = f'on_{mode}_batch_end'
        self._call(name, step, logs)


class Callback:
    """
    Base class used to build new callbacks. And new callbacks could also
    terminate training by setting `model.stop_training=True`.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # build a simple model checkpoint callback
            >>> class ModelCheckpoint(paddle.callbacks.Callback):
            ...     def __init__(self, save_freq=1, save_dir=None):
            ...         self.save_freq = save_freq
            ...         self.save_dir = save_dir
            ...
            ...     def on_epoch_end(self, epoch, logs=None):
            ...         if self.model is not None and epoch % self.save_freq == 0:
            ...             path = '{}/{}'.format(self.save_dir, epoch)
            ...             print('save checkpoint at {}'.format(path))
            ...             self.model.save(path)

    """

    model: Model | None
    params: _CallbackParams

    def __init__(self) -> None:
        self.model = None
        self.params = {}  # type: ignore

    def set_params(self, params: _CallbackParams) -> None:
        """
        Set parameters, which is dict. The keys contain:

          - 'batch_size': an integer. Number of samples per batch.
          - 'epochs': an integer. Number of epochs.
          - 'steps': an integer. Number of steps of one epoch.
          - 'verbose': an integer. Verbose mode is 0, 1 or 2. 0 = silent, 1 = progress bar, 2 = one line per epoch.
          - 'metrics': a list of str. Names of metrics, including 'loss' and the names of paddle.metric.Metric.
        """
        self.params = params

    def set_model(self, model: Model) -> None:
        """model is instance of paddle.Model."""
        self.model = model

    def on_train_begin(self, logs: _CallbackLogs | None = None) -> None:
        """Called at the start of training.

        Args:
            logs (dict): The logs is a dict or None.
        """

    def on_train_end(self, logs: _CallbackLogs | None = None) -> None:
        """Called at the end of training.

        Args:
            logs (dict): The logs is a dict or None. The keys of logs
                passed by paddle.Model contains 'loss', metric names and
                `batch_size`.
        """

    def on_eval_begin(self, logs: _CallbackLogs | None = None) -> None:
        """Called at the start of evaluation.

        Args:
            logs (dict): The logs is a dict or None. The keys of logs
                passed by paddle.Model contains 'steps' and 'metrics',
                The `steps` is number of total steps of validation dataset.
                The `metrics` is a list of str including 'loss' and the names
                of paddle.metric.Metric.
        """

    def on_eval_end(self, logs: _CallbackLogs | None = None) -> None:
        """Called at the end of evaluation.

        Args:
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict contains 'loss', metrics and 'batch_size'
                of last batch of validation dataset.
        """

    def on_predict_begin(self, logs: _CallbackLogs | None = None) -> None:
        """Called at the beginning of predict.

        Args:
            logs (dict): The logs is a dict or None.
        """

    def on_predict_end(self, logs: _CallbackLogs | None = None) -> None:
        """Called at the end of predict.

        Args:
            logs (dict): The logs is a dict or None.
        """

    def on_epoch_begin(
        self, epoch: int, logs: _CallbackLogs | None = None
    ) -> None:
        """Called at the beginning of each epoch.

        Args:
            epoch (int): The index of epoch.
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is None.
        """

    def on_epoch_end(
        self, epoch: int, logs: _CallbackLogs | None = None
    ) -> None:
        """Called at the end of each epoch.

        Args:
            epoch (int): The index of epoch.
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of last batch.
        """

    def on_train_batch_begin(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        """Called at the beginning of each batch in training.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is empty.
        """

    def on_train_batch_end(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        """Called at the end of each batch in training.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of current batch.
        """

    def on_eval_batch_begin(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        """Called at the beginning of each batch in evaluation.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is empty.
        """

    def on_eval_batch_end(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        """Called at the end of each batch in evaluation.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of current batch.
        """

    def on_predict_batch_begin(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        """Called at the beginning of each batch in predict.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None.
        """

    def on_predict_batch_end(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        """Called at the end of each batch in predict.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None.
        """


class ProgBarLogger(Callback):
    """
    Logger callback function to print loss and metrics to stdout. It supports
    silent mode (not print), progress bar or one line per each printing,
    see arguments for more detailed.

    Args:
        log_freq (int): The frequency, in number of steps,
            the logs such as loss, metrics are printed. Default: 1.
        verbose (int): The verbosity mode, should be 0, 1, or 2.
            0 = silent, 1 = progress bar, 2 = one line each printing, 3 = 2 +
            time counter, such as average reader cost, samples per second.
            Default: 2.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.vision.transforms as T
            >>> from paddle.vision.datasets import MNIST
            >>> from paddle.static import InputSpec

            >>> inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
            >>> labels = [InputSpec([None, 1], 'int64', 'label')]

            >>> transform = T.Compose([
            ...     T.Transpose(),
            ...     T.Normalize([127.5], [127.5])
            ... ])
            >>> train_dataset = MNIST(mode='train', transform=transform)

            >>> lenet = paddle.vision.models.LeNet()
            >>> model = paddle.Model(lenet,
            ...     inputs, labels)

            >>> optim = paddle.optimizer.Adam(0.001, parameters=lenet.parameters())
            >>> model.prepare(optimizer=optim,
            ...             loss=paddle.nn.CrossEntropyLoss(),
            ...             metrics=paddle.metric.Accuracy())

            >>> callback = paddle.callbacks.ProgBarLogger(log_freq=10)
            >>> model.fit(train_dataset, batch_size=64, callbacks=callback)
    """

    epochs: int | None
    steps: int | None
    progbar: ProgressBar | None
    verbose: int
    log_freq: int

    def __init__(self, log_freq: int = 1, verbose: int = 2) -> None:
        self.epochs = None
        self.steps = None
        self.progbar = None
        self.verbose = verbose
        self.log_freq = log_freq

    def _is_print(self):
        return self.verbose and paddle.distributed.ParallelEnv().local_rank == 0

    def on_train_begin(self, logs: _CallbackLogs | None = None) -> None:
        self.epochs = self.params['epochs']
        assert self.epochs
        self.train_metrics = self.params['metrics']
        assert self.train_metrics

        self._train_timer = {
            'data_time': 0,
            'batch_time': 0,
            'count': 0,
            'samples': 0,
        }
        if self._is_print():
            print(
                "The loss value printed in the log is the current step, and the metric is the average value of previous steps."
            )

    def on_epoch_begin(
        self, epoch: int | None = None, logs: _CallbackLogs | None = None
    ) -> None:
        self.steps = self.params['steps']
        self.epoch = epoch
        self.train_step = 0
        if self.epochs and self._is_print():
            print(f'Epoch {epoch + 1}/{self.epochs}')
        self.train_progbar = ProgressBar(num=self.steps, verbose=self.verbose)

        self._train_timer['batch_start_time'] = time.time()

    def _updates(self, logs: _CallbackLogs | None, mode: _CallbackMode) -> None:
        values = []
        metrics = getattr(self, f'{mode}_metrics')
        progbar = getattr(self, f'{mode}_progbar')
        steps = getattr(self, f'{mode}_step')

        for k in metrics:
            if k in logs:
                values.append((k, logs[k]))

        if self.verbose == 3 and hasattr(self, f'_{mode}_timer'):
            timer = getattr(self, f'_{mode}_timer')
            cnt = timer['count'] if timer['count'] > 0 else 1.0
            samples = timer['samples'] if timer['samples'] > 0 else 1.0
            values.append(
                ('avg_reader_cost', "%.5f sec" % (timer['data_time'] / cnt))
            )
            values.append(
                ('avg_batch_cost', "%.5f sec" % (timer['batch_time'] / cnt))
            )
            values.append(
                (
                    'ips',
                    "%.5f samples/sec"
                    % (samples / (timer['data_time'] + timer['batch_time'])),
                )
            )
            timer['count'] = 0
            timer['samples'] = 0
            timer['data_time'] = 0.0
            timer['batch_time'] = 0.0

        progbar.update(steps, values)

    def on_train_batch_begin(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        self._train_timer['batch_data_end_time'] = time.time()
        self._train_timer['data_time'] += (
            self._train_timer['batch_data_end_time']
            - self._train_timer['batch_start_time']
        )

    def on_train_batch_end(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        logs = logs or {}
        self.train_step += 1

        self._train_timer['batch_time'] += (
            time.time() - self._train_timer['batch_data_end_time']
        )
        self._train_timer['count'] += 1
        samples = logs.get('batch_size', 1)
        self._train_timer['samples'] += samples
        if self._is_print() and self.train_step % self.log_freq == 0:
            if self.steps is None or self.train_step < self.steps:
                self._updates(logs, 'train')
        self._train_timer['batch_start_time'] = time.time()

    def on_epoch_end(
        self, epoch: int, logs: _CallbackLogs | None = None
    ) -> None:
        logs = logs or {}
        if self._is_print() and (self.steps is not None):
            self._updates(logs, 'train')

    def on_eval_begin(self, logs: _CallbackLogs | None = None) -> None:
        self.eval_steps = logs.get('steps', None)
        self.eval_metrics = logs.get('metrics', [])
        self.eval_step = 0
        self.evaled_samples = 0

        self._eval_timer = {
            'data_time': 0,
            'batch_time': 0,
            'count': 0,
            'samples': 0,
        }

        self.eval_progbar = ProgressBar(
            num=self.eval_steps, verbose=self.verbose
        )
        if self._is_print():
            print('Eval begin...')

        self._eval_timer['batch_start_time'] = time.time()

    def on_eval_batch_begin(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        self._eval_timer['batch_data_end_time'] = time.time()
        self._eval_timer['data_time'] += (
            self._eval_timer['batch_data_end_time']
            - self._eval_timer['batch_start_time']
        )

    def on_eval_batch_end(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        logs = logs or {}
        self.eval_step += 1
        samples = logs.get('batch_size', 1)
        self.evaled_samples += samples

        self._eval_timer['batch_time'] += (
            time.time() - self._eval_timer['batch_data_end_time']
        )
        self._eval_timer['count'] += 1
        samples = logs.get('batch_size', 1)
        self._eval_timer['samples'] += samples

        if self._is_print() and self.eval_step % self.log_freq == 0:
            if self.eval_steps is None or self.eval_step < self.eval_steps:
                self._updates(logs, 'eval')

        self._eval_timer['batch_start_time'] = time.time()

    def on_predict_begin(self, logs: _CallbackLogs | None = None) -> None:
        self.test_steps = logs.get('steps', None)
        self.test_metrics = logs.get('metrics', [])
        self.test_step = 0
        self.tested_samples = 0

        self._test_timer = {
            'data_time': 0,
            'batch_time': 0,
            'count': 0,
            'samples': 0,
        }

        self.test_progbar = ProgressBar(
            num=self.test_steps, verbose=self.verbose
        )
        if self._is_print():
            print('Predict begin...')

        self._test_timer['batch_start_time'] = time.time()

    def on_predict_batch_begin(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        self._test_timer['batch_data_end_time'] = time.time()
        self._test_timer['data_time'] += (
            self._test_timer['batch_data_end_time']
            - self._test_timer['batch_start_time']
        )

    def on_predict_batch_end(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        logs = logs or {}
        self.test_step += 1
        samples = logs.get('batch_size', 1)
        self.tested_samples += samples

        self._test_timer['batch_time'] += (
            time.time() - self._test_timer['batch_data_end_time']
        )
        self._test_timer['count'] += 1
        samples = logs.get('batch_size', 1)
        self._test_timer['samples'] += samples

        if self.test_step % self.log_freq == 0 and self._is_print():
            if self.test_steps is None or self.test_step < self.test_steps:
                self._updates(logs, 'test')

        self._test_timer['batch_start_time'] = time.time()

    def on_eval_end(self, logs: _CallbackLogs | None = None) -> None:
        logs = logs or {}
        if self._is_print() and (self.eval_steps is not None):
            self._updates(logs, 'eval')
            print(f'Eval samples: {self.evaled_samples}')

    def on_predict_end(self, logs: _CallbackLogs | None = None) -> None:
        logs = logs or {}
        if self._is_print():
            if self.test_step % self.log_freq != 0 or self.verbose == 1:
                self._updates(logs, 'test')
            print(f'Predict samples: {self.tested_samples}')


class ModelCheckpoint(Callback):
    """
    Model checkpoint callback function to save model weights and optimizer
    state during training in conjunction with model.fit(). Currently,
    ModelCheckpoint only supports saving after a fixed number of epochs.

    Args:
        save_freq(int): The frequency, in number of epochs, the model checkpoint
            are saved. Default: 1.
        save_dir(str|None): The directory to save checkpoint during training.
            If None, will not save checkpoint. Default: None.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.vision.transforms as T
            >>> from paddle.vision.datasets import MNIST
            >>> from paddle.static import InputSpec

            >>> inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
            >>> labels = [InputSpec([None, 1], 'int64', 'label')]

            >>> transform = T.Compose([
            ...     T.Transpose(),
            ...     T.Normalize([127.5], [127.5])
            ... ])
            >>> train_dataset = MNIST(mode='train', transform=transform)

            >>> lenet = paddle.vision.models.LeNet()
            >>> model = paddle.Model(lenet,
            ...     inputs, labels)

            >>> optim = paddle.optimizer.Adam(0.001, parameters=lenet.parameters())
            >>> model.prepare(optimizer=optim,
            ...             loss=paddle.nn.CrossEntropyLoss(),
            ...             metrics=paddle.metric.Accuracy())

            >>> callback = paddle.callbacks.ModelCheckpoint(save_dir='./temp')
            >>> model.fit(train_dataset, batch_size=64, callbacks=callback)
    """

    def __init__(self, save_freq: int = 1, save_dir: str | None = None) -> None:
        self.save_freq = save_freq
        self.save_dir = save_dir

    def on_epoch_begin(
        self, epoch: int | None = None, logs: _CallbackLogs | None = None
    ) -> None:
        self.epoch = epoch

    def _is_save(self):
        return (
            self.model
            and self.save_dir
            and paddle.distributed.ParallelEnv().local_rank == 0
        )

    def on_epoch_end(
        self, epoch: int, logs: _CallbackLogs | None = None
    ) -> None:
        if self._is_save() and self.epoch % self.save_freq == 0:
            path = f'{self.save_dir}/{epoch}'
            print(f'save checkpoint at {os.path.abspath(path)}')
            self.model.save(path)

    def on_train_end(self, logs: _CallbackLogs | None = None) -> None:
        if self._is_save():
            path = f'{self.save_dir}/final'
            print(f'save checkpoint at {os.path.abspath(path)}')
            self.model.save(path)


class LRScheduler(Callback):
    """Lr scheduler callback function

    Args:
        by_step(bool, optional): whether to update learning rate scheduler
            by step. Default: True.
        by_epoch(bool, optional): whether to update learning rate scheduler
            by epoch. Default: False.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.vision.transforms as T
            >>> from paddle.static import InputSpec

            >>> inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
            >>> labels = [InputSpec([None, 1], 'int64', 'label')]

            >>> transform = T.Compose([
            ...     T.Transpose(),
            ...     T.Normalize([127.5], [127.5])
            ... ])
            >>> train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)

            >>> lenet = paddle.vision.models.LeNet()
            >>> model = paddle.Model(lenet,
            ...     inputs, labels)

            >>> base_lr = 1e-3
            >>> boundaries = [5, 8]
            >>> wamup_steps = 4

            >>> def make_optimizer(parameters=None):
            ...     momentum = 0.9
            ...     weight_decay = 5e-4
            ...     values = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
            ...     learning_rate = paddle.optimizer.lr.PiecewiseDecay(
            ...         boundaries=boundaries, values=values)
            ...     learning_rate = paddle.optimizer.lr.LinearWarmup(
            ...         learning_rate=learning_rate,
            ...         warmup_steps=wamup_steps,
            ...         start_lr=base_lr / 5.,
            ...         end_lr=base_lr,
            ...         verbose=True)
            ...     optimizer = paddle.optimizer.Momentum(
            ...         learning_rate=learning_rate,
            ...         weight_decay=weight_decay,
            ...         momentum=momentum,
            ...         parameters=parameters)
            ...     return optimizer

            >>> optim = make_optimizer(parameters=lenet.parameters())
            >>> model.prepare(optimizer=optim,
            ...             loss=paddle.nn.CrossEntropyLoss(),
            ...             metrics=paddle.metric.Accuracy())

            >>> # if LRScheduler callback not set, an instance LRScheduler update by step
            >>> # will be created auto.
            >>> model.fit(train_dataset, batch_size=64)

            >>> # create a learning rate scheduler update by epoch
            >>> callback = paddle.callbacks.LRScheduler(by_step=False, by_epoch=True)
            >>> model.fit(train_dataset, batch_size=64, callbacks=callback)
    """

    def __init__(self, by_step: bool = True, by_epoch: bool = False) -> None:
        if by_step and by_epoch:
            raise ValueError(
                "by_step option is mutually exclusive with by_epoch"
            )

        self.by_step = by_step
        self.by_epoch = by_epoch

    def on_epoch_end(
        self, epoch: int, logs: _CallbackLogs | None = None
    ) -> None:
        if self.by_epoch:
            if (
                self.model._optimizer
                and hasattr(self.model._optimizer, '_learning_rate')
                and isinstance(
                    self.model._optimizer._learning_rate,
                    paddle.optimizer.lr.LRScheduler,
                )
            ):
                self.model._optimizer._learning_rate.step()

    def on_train_batch_end(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        if self.by_step:
            if (
                self.model._optimizer
                and hasattr(self.model._optimizer, '_learning_rate')
                and isinstance(
                    self.model._optimizer._learning_rate,
                    paddle.optimizer.lr.LRScheduler,
                )
            ):
                self.model._optimizer._learning_rate.step()


class EarlyStopping(Callback):
    """Stop training when the given monitor stopped improving during evaluation
    by setting `model.stop_training=True`.

    Args:
        monitor(str): Quantity to be monitored. Default: 'loss'.
        mode(str|None): Mode should be one of 'auto', 'min' or 'max'. In 'min'
            mode, training will stop until monitored quantity stops decreasing.
            In 'max' mode, training will stop until monitored quantity stops
            increasing. In 'auto' mode, exact mode can be inferred by the name
            of monitor. If 'acc' in monitor, the mode will be considered as
            'max', otherwise the mode will be set to 'min'. Default: 'auto'.
        patience(int): Number of epochs with no improvement after which
            training will be stopped. Default: 0.
        verbose(int): The verbosity mode, should be 0 or 1. When verbose=0,
            logs will not be printed. When verbose=1, logs will be printed.
            Default: 1.
        min_delta(int|float): The minimum change of monitored quantity. If
            the change is less than min_delta, model could be considered as no
            improvement. Default: 0.
        baseline(int|float|None): Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline. Default: None.
        save_best_model(bool): Whether to save best model. Default: True.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle import Model
            >>> from paddle.static import InputSpec
            >>> from paddle.vision.models import LeNet
            >>> from paddle.vision.datasets import MNIST
            >>> from paddle.metric import Accuracy
            >>> from paddle.nn import CrossEntropyLoss
            >>> import paddle.vision.transforms as T

            >>> device = paddle.set_device('cpu')
            >>> sample_num = 200
            >>> save_dir = './best_model_checkpoint'
            >>> transform = T.Compose(
            ...     [T.Transpose(), T.Normalize([127.5], [127.5])])
            >>> train_dataset = MNIST(mode='train', transform=transform)
            >>> val_dataset = MNIST(mode='test', transform=transform)
            >>> net = LeNet()
            >>> optim = paddle.optimizer.Adam(
            ...     learning_rate=0.001, parameters=net.parameters())

            >>> inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
            >>> labels = [InputSpec([None, 1], 'int64', 'label')]

            >>> model = Model(net, inputs=inputs, labels=labels)
            >>> model.prepare(
            ...     optim,
            ...     loss=CrossEntropyLoss(reduction="sum"),
            ...     metrics=[Accuracy()])
            >>> callbacks = paddle.callbacks.EarlyStopping(
            ...     'loss',
            ...     mode='min',
            ...     patience=1,
            ...     verbose=1,
            ...     min_delta=0,
            ...     baseline=None,
            ...     save_best_model=True)
            >>> model.fit(train_dataset,
            ...           val_dataset,
            ...           batch_size=64,
            ...           log_freq=200,
            ...           save_freq=10,
            ...           save_dir=save_dir,
            ...           epochs=20,
            ...           callbacks=[callbacks])
    """

    def __init__(
        self,
        monitor: str = 'loss',
        mode: Literal['auto', 'min', 'max'] = 'auto',
        patience: int = 0,
        verbose: int = 1,
        min_delta: float = 0,
        baseline: float | None = None,
        save_best_model: bool = True,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait_epoch = 0
        self.best_weights = None
        self.stopped_epoch = 0
        self.save_best_model = save_best_model
        # The value of `save_dir` is set in function `config_callbacks`
        self.save_dir: str | None = None
        if mode not in ['auto', 'min', 'max']:
            warnings.warn(
                f'EarlyStopping mode {mode} is unknown, '
                'fallback to auto mode.'
            )
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        # When mode == 'auto', the mode should be inferred by `self.monitor`
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs: _CallbackLogs | None = None) -> None:
        self.wait_epoch = 0
        if self.baseline is not None:
            self.best_value = self.baseline
        else:
            self.best_value = np.inf if self.monitor_op == np.less else -np.inf
            self.best_weights = None

    def on_eval_end(self, logs: _CallbackLogs | None = None) -> None:
        if logs is None or self.monitor not in logs:
            warnings.warn(
                'Monitor of EarlyStopping should be loss or metric name.'
            )
            return
        current = logs[self.monitor]
        if isinstance(current, (list, tuple)):
            current = current[0]
        elif isinstance(current, numbers.Number):
            current = current
        else:
            return

        if self.monitor_op(current - self.min_delta, self.best_value):
            self.best_value = current
            self.wait_epoch = 0
            if self.save_best_model and self.save_dir is not None:
                path = os.path.join(self.save_dir, 'best_model')
                self.model.save(path)
        else:
            self.wait_epoch += 1
        if self.wait_epoch >= self.patience:
            self.model.stop_training = True
            if self.verbose > 0:
                print(f'Epoch {self.stopped_epoch + 1}: Early stopping.')
                if self.save_best_model and self.save_dir is not None:
                    print(
                        'Best checkpoint has been saved at {}'.format(
                            os.path.abspath(
                                os.path.join(self.save_dir, 'best_model')
                            )
                        )
                    )
        self.stopped_epoch += 1


class VisualDL(Callback):
    """
    VisualDL callback class. After storing the loss values and evaluation metrics in a log file during the training time , the panel is launched to view the visual results.

    Args:
        log_dir (str): The directory to save visualdl log file.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.vision.transforms as T
            >>> from paddle.static import InputSpec

            >>> inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
            >>> labels = [InputSpec([None, 1], 'int64', 'label')]

            >>> transform = T.Compose([
            ...     T.Transpose(),
            ...     T.Normalize([127.5], [127.5])
            ... ])
            >>> train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
            >>> eval_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

            >>> net = paddle.vision.models.LeNet()
            >>> model = paddle.Model(net, inputs, labels)

            >>> optim = paddle.optimizer.Adam(0.001, parameters=net.parameters())
            >>> model.prepare(optimizer=optim,
            ...             loss=paddle.nn.CrossEntropyLoss(),
            ...             metrics=paddle.metric.Accuracy())

            >>> ## uncomment following lines to fit model with visualdl callback function
            >>> # callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir')
            >>> # model.fit(train_dataset, eval_dataset, batch_size=64, callbacks=callback)

    """

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.epochs = None
        self.steps = None
        self.epoch = 0

    def _is_write(self) -> bool:
        return paddle.distributed.ParallelEnv().local_rank == 0

    def on_train_begin(self, logs: _CallbackLogs | None = None) -> None:
        self.epochs = self.params['epochs']
        assert self.epochs
        self.train_metrics = self.params['metrics']
        assert self.train_metrics
        self._is_fit = True
        self.train_step = 0

    def on_epoch_begin(
        self, epoch: int | None = None, logs: _CallbackLogs | None = None
    ) -> None:
        self.steps = self.params['steps']
        self.epoch = epoch

    def _updates(self, logs: int, mode: _CallbackMode) -> None:
        if not self._is_write():
            return
        if not hasattr(self, 'writer'):
            visualdl = try_import('visualdl')
            self.writer = visualdl.LogWriter(self.log_dir)

        metrics = getattr(self, f'{mode}_metrics')
        current_step = getattr(self, f'{mode}_step')

        if mode == 'train':
            total_step = current_step
        else:
            total_step = self.epoch

        for k in metrics:
            if k in logs:
                temp_tag = mode + '/' + k

                if isinstance(logs[k], (list, tuple)):
                    temp_value = logs[k][0]
                elif isinstance(logs[k], numbers.Number):
                    temp_value = logs[k]
                else:
                    continue

                self.writer.add_scalar(
                    tag=temp_tag, step=total_step, value=temp_value
                )

    def on_train_batch_end(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        logs = logs or {}
        self.train_step += 1

        if self._is_write():
            self._updates(logs, 'train')

    def on_eval_begin(self, logs: _CallbackLogs | None = None) -> None:
        self.eval_steps = logs.get('steps', None)
        self.eval_metrics = logs.get('metrics', [])
        self.eval_step = 0
        self.evaled_samples = 0

    def on_train_end(self, logs: _CallbackLogs | None = None) -> None:
        if hasattr(self, 'writer'):
            self.writer.close()
            delattr(self, 'writer')

    def on_eval_end(self, logs: _CallbackLogs | None = None) -> None:
        if self._is_write():
            self._updates(logs, 'eval')

            if (not hasattr(self, '_is_fit')) and hasattr(self, 'writer'):
                self.writer.close()
                delattr(self, 'writer')


class WandbCallback(Callback):
    """Track your training and system metrics using `Weights and Biases <https://docs.wandb.ai>`_.

    **Installation and set-up**

    Install with pip and log in to your W&B account:

    .. code-block:: bash

        pip install wandb
        wandb login

    Args:
        project(str|None, optional): Name of the project. Default: uncategorized
        entity(str|None, optional): Name of the team/user creating the run. Default: Logged in user
        name(str|None, optional): Name of the run. Default: randomly generated by wandb
        dir(str|None, optional): Directory in which all the metadata is stored. Default: `wandb`
        mode(str|None, optional): Can be "online", "offline" or "disabled". Default: "online".
        job_type(str|None, optional): the type of run, for grouping runs together. Default: None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.vision.transforms as T
            >>> from paddle.static import InputSpec

            >>> inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
            >>> labels = [InputSpec([None, 1], 'int64', 'label')]

            >>> transform = T.Compose([
            ...     T.Transpose(),
            ...     T.Normalize([127.5], [127.5])
            ... ])
            >>> train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
            >>> eval_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

            >>> net = paddle.vision.models.LeNet()
            >>> model = paddle.Model(net, inputs, labels)

            >>> optim = paddle.optimizer.Adam(0.001, parameters=net.parameters())
            >>> model.prepare(optimizer=optim,
            ...             loss=paddle.nn.CrossEntropyLoss(),
            ...             metrics=paddle.metric.Accuracy())

            >>> ## uncomment following lines to fit model with wandb callback function
            >>> # callback = paddle.callbacks.WandbCallback(project='paddle_mnist')
            >>> # model.fit(train_dataset, eval_dataset, batch_size=64, callbacks=callback)

    """

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        name: str | None = None,
        dir: str | None = None,
        mode: Literal["online", "offline", "disabled"] | None = None,
        job_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.wandb = try_import(
            "wandb",
            "You want to use `wandb` which is not installed yet install it with `pip install wandb`",
        )

        self.wandb_args = {
            'project': project,
            'name': name,
            'entity': entity,
            'dir': dir,
            'mode': mode,
            'job_type': job_type,
        }

        self._run = None
        self.wandb_args.update(**kwargs)

        _ = self.run

    def _is_write(self):
        return paddle.distributed.ParallelEnv().local_rank == 0

    @property
    def run(self):
        if self._is_write():
            if self._run is None:
                if self.wandb.run is not None:
                    warnings.warn(
                        "There is a wandb run already in progress and newly created instances"
                        " of `WandbCallback` will reuse this run. If this is not desired"
                        " , call `wandb.finish()` before instantiating `WandbCallback`."
                    )
                    self._run = self.wandb.run
                else:
                    self._run = self.wandb.init(**self.wandb_args)

        return self._run

    def on_train_begin(self, logs: _CallbackLogs | None = None) -> None:
        self.epochs = self.params['epochs']
        assert self.epochs
        self.train_metrics = self.params['metrics']
        assert self.train_metrics
        self._is_fit = True
        self.train_step = 0

        if self._is_write():
            self.run.define_metric("train/step")
            self.run.define_metric("train/*", step_metric="train/step")

            self.run.define_metric("epoch")
            self.run.define_metric("eval/*", step_metric="epoch")

    def on_epoch_begin(
        self, epoch: int, logs: _CallbackLogs | None = None
    ) -> None:
        self.steps = self.params['steps']
        self.epoch = epoch

    def _updates(self, logs: _CallbackLogs | None, mode: _CallbackMode) -> None:
        if not self._is_write():
            return

        metrics = getattr(self, f'{mode}_metrics')
        current_step = getattr(self, f'{mode}_step')

        _metrics = {}

        if mode == 'train':
            total_step = current_step
            _metrics.update({'train/step': total_step})
        else:
            total_step = self.epoch
            _metrics.update({'epoch': total_step})

        for k in metrics:
            if k in logs:
                temp_tag = mode + '/' + k

                if isinstance(logs[k], (list, tuple)):
                    _metrics.update({temp_tag: logs[k][0]})
                elif isinstance(logs[k], numbers.Number):
                    _metrics.update({temp_tag: logs[k]})
                else:
                    continue

        self.run.log(_metrics)

    def on_train_batch_end(
        self, step: int, logs: _CallbackLogs | None = None
    ) -> None:
        logs = logs or {}
        self.train_step += 1

        if self._is_write():
            self._updates(logs, 'train')

    def on_eval_begin(self, logs: _CallbackLogs | None = None) -> None:
        self.eval_steps = logs.get('steps', None)
        self.eval_metrics = logs.get('metrics', [])
        self.eval_step = 0
        self.evaled_samples = 0

    def on_train_end(self, logs: _CallbackLogs | None = None) -> None:
        if self._is_write():
            self.run.finish()

    def on_eval_end(self, logs: _CallbackLogs | None = None) -> None:
        if self._is_write():
            self._updates(logs, 'eval')

            if (not hasattr(self, '_is_fit')) and hasattr(self, 'run'):
                self.run.finish()
                delattr(self, 'run')


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric of evaluation has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        monitor(str, optional): Quantity to be monitored. Default: 'loss'.
        factor(float, optional): factor by which the learning rate will be reduced.
            `new_lr = lr * factor`. Default: 0.1.
        patience(int, optional): Number of epochs with no improvement after which
            learning rate will be reduced. Default: 10.
        verbose(int, optional): The verbosity mode. 0: quiet, 1: update messages.
            Default: 1.
        mode(str, optional): one of `{'auto', 'min', 'max'}`. In `'min'` mode,
            the learning rate will be reduced when the quantity monitored has
            stopped decreasing. In 'max' mode, learning rate will reduce until
            monitored quantity stops increasing. In 'auto' mode, exact mode
            can be inferred by the name of monitor. If 'acc' in monitor, the
            mode will be considered as 'max', otherwise the mode will be set
            to 'min'. Default: 'auto'.
        min_delta(int|float, optional): threshold for measuring the new optimum,
            to only focus on significant changes. Default: 0.
        cooldown(int, optional): number of epochs to wait before resuming normal operation after
            lr has been reduced. Default: 0.
        min_lr(float, optional): lower bound on the learning rate. Default: 0.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle import Model
            >>> from paddle.static import InputSpec
            >>> from paddle.vision.models import LeNet
            >>> from paddle.vision.datasets import MNIST
            >>> from paddle.metric import Accuracy
            >>> from paddle.nn.layer.loss import CrossEntropyLoss
            >>> import paddle.vision.transforms as T
            >>> sample_num = 200
            >>> transform = T.Compose(
            ...      [T.Transpose(), T.Normalize([127.5], [127.5])])
            >>> train_dataset = MNIST(mode='train', transform=transform)
            >>> val_dataset = MNIST(mode='test', transform=transform)
            >>> net = LeNet()
            >>> optim = paddle.optimizer.Adam(
            ...     learning_rate=0.001, parameters=net.parameters())
            >>> inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
            >>> labels = [InputSpec([None, 1], 'int64', 'label')]
            >>> model = Model(net, inputs=inputs, labels=labels)
            >>> model.prepare(
            ...     optim,
            ...     loss=CrossEntropyLoss(),
            ...     metrics=[Accuracy()])
            >>> callbacks = paddle.callbacks.ReduceLROnPlateau(patience=3, verbose=1)
            >>> model.fit(train_dataset,
            ...             val_dataset,
            ...             batch_size=64,
            ...             log_freq=200,
            ...             save_freq=10,
            ...             epochs=20,
            ...             callbacks=[callbacks])

    """

    monitor: str
    factor: float
    patience: int
    verbose: int
    mode: Literal['auto', 'min', 'max']
    min_delta: float
    cooldown: int
    min_lr: float

    def __init__(
        self,
        monitor: str = 'loss',
        factor: float = 0.1,
        patience: int = 10,
        verbose: int = 1,
        mode: Literal['auto', 'min', 'max'] = 'auto',
        min_delta: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0,
    ) -> None:
        super().__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError(
                'ReduceLROnPlateau ' 'does not support a factor >= 1.0.'
            )

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.epoch = 0
        self._reset()

    def _reset(self) -> None:
        """Resets wait counter and cooldown counter."""
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn(
                f'Learning rate reduction mode {self.mode} is unknown, '
                'fallback to auto mode.'
            )
            self.mode = 'auto'
        if self.mode == 'min' or (
            self.mode == 'auto' and 'acc' not in self.monitor
        ):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs: _CallbackLogs | None = None) -> None:
        self._reset()

    def on_eval_end(self, logs: _CallbackLogs | None = None) -> None:
        if logs is None or self.monitor not in logs:
            warnings.warn(
                'Monitor of ReduceLROnPlateau should be loss or metric name.'
            )
            return
        else:
            try:
                lr = self.model._optimizer._learning_rate
                if not isinstance(lr, float):
                    warnings.warn(
                        f'Expected learning_rate be float, bug got {type(lr)}.'
                    )
                    return
            except Exception as e:
                warnings.warn(
                    f'There are something wrong when get learning_rate from optimizer: {e}.'
                )
                return

        current = logs[self.monitor]
        if isinstance(current, (list, tuple)):
            current = current[0]
        elif isinstance(current, numbers.Number):
            current = current
        else:
            return

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = self.model._optimizer.get_lr()
                if old_lr > np.float32(self.min_lr):
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    self.model._optimizer._learning_rate = new_lr
                    if (
                        self.verbose > 0
                        and paddle.distributed.ParallelEnv().local_rank == 0
                    ):
                        print(
                            f'\nEpoch {self.epoch + 1}: ReduceLROnPlateau reducing learning '
                            f'rate to {new_lr}.'
                        )
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
        self.epoch += 1

    def in_cooldown(self) -> bool:
        return self.cooldown_counter > 0
