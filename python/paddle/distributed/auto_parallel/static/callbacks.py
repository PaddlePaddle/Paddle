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

import json
import os
import time

import paddle
import paddle.distributed.auto_parallel.utils as auto_utils
from paddle.hapi.callbacks import (
    Callback,
    CallbackList,
    LRScheduler,
    ModelCheckpoint,
    ProgBarLogger,
)

from ..interface import CollectionNames, get_collection


def config_callbacks(
    callbacks=None,
    engine=None,
    batch_size=None,
    epochs=None,
    steps=None,
    log_freq=2,
    verbose=2,
    save_freq=1,
    latest_checkpoint_meta=None,
    save_dir=None,
    metrics=None,
    acc_step=1,
    mode='train',
):
    cbks = callbacks or []
    cbks = cbks if isinstance(cbks, (list, tuple)) else [cbks]

    if not any(isinstance(k, ProgBarLogger) for k in cbks) and verbose:
        cbks = [ProgBarLoggerAuto(log_freq, verbose=verbose)] + cbks

    if not any(isinstance(k, LRScheduler) for k in cbks):
        cbks = [LRSchedulerAuto()] + cbks

    print(f"debug config_callbacks save_dir: {save_dir}")
    if not any(isinstance(k, ModelCheckpoint) for k in cbks):
        cbks = cbks + [
            # ModelCheckpointAuto( **{ "save_freq": save_freq, "save_dir": save_dir, "latest_checkpoint_meta": latest_checkpoint_meta }
            ModelCheckpointAuto(
                save_freq=save_freq,
                save_dir=save_dir,
                latest_checkpoint_meta=latest_checkpoint_meta,
            )
        ]

    if not any(isinstance(k, Profiler) for k in cbks) and verbose == 3:
        cbks = cbks + [Profiler(timer_only=True)]

    if not any(isinstance(k, History) for k in cbks):
        cbks = cbks + [History()]

    for i, k in enumerate(cbks):
        if isinstance(k, ProgBarLogger):
            cbks[i] = ProgBarLoggerAuto(k.log_freq, k.verbose)
        if isinstance(k, LRScheduler):
            cbks[i] = LRSchedulerAuto(k.by_step, k.by_epoch)
        if isinstance(k, ModelCheckpoint):
            # cbks[i] = ModelCheckpointAuto( **{ "save_freq": save_freq, "save_dir": save_dir, "latest_checkpoint_meta": latest_checkpoint_meta })
            cbks[i] = ModelCheckpointAuto(
                save_freq=save_freq,
                save_dir=save_dir,
                latest_checkpoint_meta=latest_checkpoint_meta,
            )

    cbk_list = CallbackList(cbks)
    cbk_list.set_model(engine)
    metrics = metrics or [] if mode != 'test' else []
    params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps,
        'verbose': verbose,
        'metrics': metrics,
        'acc_step': acc_step,
    }
    cbk_list.set_params(params)
    return cbk_list


class ProgBarLoggerAuto(ProgBarLogger):
    def __init__(self, log_freq=1, verbose=2):
        super().__init__(log_freq, verbose)

    def _is_print(self):
        return True

    def _updates(self, logs, mode):
        values = []
        metrics = getattr(self, '%s_metrics' % (mode))
        progbar = getattr(self, '%s_progbar' % (mode))
        steps = getattr(self, '%s_step' % (mode))

        for k in metrics:
            if k in logs:
                values.append((k, logs[k]))

        if 'lr' in logs:
            values.append(('lr', logs['lr']))

        fetches_logs = logs.get('fetches', {})
        collect_logging = get_collection(CollectionNames.LOGGING)
        for name, var in collect_logging:
            k = name or var.name
            if k in fetches_logs:
                values.append((k, fetches_logs[k]))

        out_logs = logs.get('outputs', {})
        for k in out_logs:
            values.append((k, out_logs[k]))

        if self.verbose == 3 and hasattr(self, '_%s_timer' % (mode)):
            timer = getattr(self, '_%s_timer' % (mode))
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

    def on_eval_batch_end(self, step, logs=None):
        logs = logs or {}
        self.eval_step += 1
        samples = self.params['batch_size']
        self.evaled_samples += samples

        self._eval_timer['batch_time'] += (
            time.time() - self._eval_timer['batch_data_end_time']
        )
        self._eval_timer['count'] += 1
        samples = self.params['batch_size']
        self._eval_timer['samples'] += samples

        if self._is_print() and self.eval_step % self.log_freq == 0:
            if self.eval_steps is None or self.eval_step < self.eval_steps:
                self._updates(logs, 'eval')

        self._eval_timer['batch_start_time'] = time.time()


class LRSchedulerAuto(LRScheduler):
    def __init__(self, by_step=True, by_epoch=False):
        super().__init__(by_step, by_epoch)

    def on_epoch_begin(self, epoch=None, logs=None):
        self.acc_step = self.params["acc_step"]
        self.epoch = epoch
        self.train_step = 0

    def on_train_batch_end(self, step, logs=None):
        self.train_step += 1

        if self.by_step and self.train_step % self.acc_step == 0:
            if (
                self.model._optimizer
                and hasattr(self.model._optimizer, '_learning_rate')
                and isinstance(
                    self.model._optimizer._learning_rate,
                    paddle.optimizer.lr.LRScheduler,
                )
            ):
                self.model._optimizer._learning_rate.step()


class History(Callback):
    def __init__(self):
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.model.history = self


class Profiler(Callback):
    def __init__(self, *args, **kwargs):
        self.prof = paddle.profiler.Profiler(*args, **kwargs)

    def on_epoch_begin(self, epoch=None, logs=None):
        self.epoch = epoch
        self.train_step = 0
        self.batch_size = self.params["batch_size"]
        self.steps = self.params['steps']

    def on_train_begin(self, logs=None):
        self.prof.start()

    def on_train_batch_end(self, step, logs=None):
        self.train_step += 1
        self.prof.step(num_samples=self.batch_size)
        print(
            "step {}:{}".format(
                self.train_step, self.prof.step_info(unit='samples')
            )
        )

    def on_train_end(self, logs=None):
        self.prof.stop()
        self.prof.summary()


class ModelCheckpointAuto(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        self._checkpoint_meta = kwargs.get("latest_checkpoint_meta", None)
        print(f"debug callbacks args: {args}, kwargs: {kwargs}")
        save_freq = kwargs.get("save_freq", 1)
        save_dir = kwargs.get("save_dir", None)
        if self._checkpoint_meta:
            save_freq = self._checkpoint_meta.get("save_freq", 1)
            save_dir = self._checkpoint_meta.get("save_dir", None)
        super().__init__(save_freq=save_freq, save_dir=save_dir)
        # model params for restore check between checkpoint and args
        self._rank_id = paddle.distributed.get_rank()
        self._rank_size = paddle.distributed.get_world_size()
        self._latest_checkpoint_dir = self.save_dir

    @property
    def epochs(self):
        return self._checkpoint_meta.get("epochs", 0)

    @epochs.setter
    def epochs(self, value):
        self._checkpoint_meta["epochs"] = value

    @property
    def steps(self):
        return self._checkpoint_meta.get("steps", 0)

    @steps.setter
    def steps(self, value):
        self._checkpoint_meta["steps"] = value

    @property
    def keep_checkpoint_max_num(self):
        return self._checkpoint_meta.get("keep_checkpoint_max_num", None)

    @property
    def save_checkpoint_every_n_step(self):
        return self._checkpoint_meta.get("save_checkpoint_every_n_step", None)

    @property
    def save_checkpoint_every_n_epoch(self):
        return self._checkpoint_meta.get("save_checkpoint_every_n_epoch", None)

    @property
    def rng_state(self):
        return self._checkpoint_meta.get("rng_state", None)

    @property
    def checkpoint_meta_path(self):
        latest_ckpt_path = None
        if self.save_dir:
            latest_ckpt_path = os.path.join(
                self.save_dir, "latest_checkpoint.txt"
            )
        return latest_ckpt_path

    def _save_checkpoint_meta(self, latest_checkpoint_path):
        # 保存ckpt、dataloader状态数据
        print(
            f"save checkpoint meta: {self._checkpoint_meta} path: {latest_checkpoint_path}"
        )
        self._checkpoint_meta["latest_checkpoint_path"] = latest_checkpoint_path
        with open(self.checkpoint_meta_path, "w") as wobj:
            print(f'debug checkpoint_meta: {self._checkpoint_meta}')
            wobj.write(f"{json.dumps(self._checkpoint_meta)}")
        return True

    def _load_checkpoint_meta(self):
        # 恢复ckpt、dataloader状态
        with open(self.checkpoint_meta_path, "rb") as robj:
            self._checkpoint_meta = json.loads(robj.read())
        return self.get_checkpoint_meta

    def _is_save(self):
        return self.model and self.save_dir

    def _save_checkpoint(self, path):
        self.model.save(path)
        if self._rank_id == 0:
            self._save_checkpoint_meta(path)
        auto_utils.update_checkpoint_filelist(
            self.save_dir, self.keep_checkpoint_max_num
        )

    def _get_timestamp(self):
        now = int(time.time())
        return time.strftime("%Y%m%d%H%M%S", time.localtime(now))

    def on_train_batch_end(self, step, logs=None):
        self.steps = step + 1
        if (
            self._is_save()
            and self.save_checkpoint_every_n_step is not None
            and self.steps % self.save_checkpoint_every_n_step == 0
        ):
            path = (
                f"{self.save_dir}/epoch{self.epochs}_step{self.steps}/default"
            )
            print(f'save checkpoint at {os.path.abspath(path)}')
            self._save_checkpoint(path)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs = epoch + 1
        """
        if self._is_save() and self.epochs % self.save_freq == 0:
            path = f'{self.save_dir}/epoch{epoch}'
            print(f'save checkpoint at {os.path.abspath(path)}')
            self._save_checkpoint(path)
        """

    def on_train_end(self, logs=None):
        print(
            f"debug callbacks ModelCheckpoint on_train_end begin, is_save: {self._is_save()}"
        )
        if self._is_save():
            print(
                f"debug callbacks ModelCheckpoint on_train_end save: {self.save_dir}"
            )
            path = (
                f"{self.save_dir}/epoch{self.epochs-1}_step{self.steps}/default"
            )
            print(f'save checkpoint at {os.path.abspath(path)}')
            self._save_checkpoint(path)
