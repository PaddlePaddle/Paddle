#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

__all__ = ['Strategy']


class Strategy(object):
    """
    Base class for all strategies.
    """

    def __init__(self, start_epoch=0, end_epoch=0):
        """
        Args:
            start_epoch: The first epoch to apply the strategy.
            end_epoch: The last epoch to apply the strategy.
        """
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def __getstate__(self):
        d = {}
        for key in self.__dict__:
            if key not in ["start_epoch", "end_epoch"]:
                d[key] = self.__dict__[key]
        return d

    def on_compression_begin(self, context):
        pass

    def on_epoch_begin(self, context):
        pass

    def on_epoch_end(self, context):
        pass

    def on_batch_begin(self, context):
        pass

    def on_batch_end(self, context):
        pass

    def on_compression_end(self, context):
        pass

    def restore_from_checkpoint(self, context):
        pass
