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

from .dist_saver import DistributedSaver
from ...hapi.callbacks import Callback


class ModelCheckpoint(Callback):
    def __init__(self, save_freq=1, save_dir=None, prefix='mckp'):
        self.prefix = prefix
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.saver = DistributedSaver()

    def on_epoch_begin(self, epoch=None, logs=None):
        self.epoch = epoch

    def on_train_batch_begin(self, step=None, logs=None):
        self.step = step

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch % self.save_freq == 0:
            path = '{}/{}'.format(self.save_dir, epoch)
            print('save checkpoint at {}'.format(os.path.abspath(path)))

    def on_train_end(self, logs=None):
        path = '{}/final'.format(self.save_dir)
        print('save checkpoint at {}'.format(os.path.abspath(path)))
