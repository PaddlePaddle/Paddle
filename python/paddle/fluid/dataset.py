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

from paddle.fluid.proto import data_feed_pb2
from google.protobuf import text_format
from . import core
__all__ = ['DatasetFactory']


class DatasetFactory(object):
    def __init__(self):
        pass

    def create_dataset(self, datafeed_class):
        try:
            dataset = globals()[datafeed_class]()
            return dataset
        except:
            raise ValueError("datafeed class %s does not exist" %
                             datafeed_class)


class DatasetBase(object):
    def __init__(self):
        # define class name here
        # to decide whether we need create in memory instance
        self.proto_desc = data_feed_pb2.DataFeedDesc()
        self.proto_desc.pipe_command = "cat"
        self.dataset = core.Dataset()

    def set_pipe_command(self, pipe_command):
        """
        Set pipe command of current dataset
        A pipe command is a UNIX pipeline command that can be used only

        """
        self.proto_desc.pipe_command = pipe_command

    def set_batch_size(self, batch_size):
        """
        Set batch size. Will be effective during training

        Example:
            >>> data_feed = fluid.DataFeedDesc('data.proto')
            >>> data_feed.set_batch_size(128)

        Args:
            batch_size: batch size

        """
        self.proto_desc.batch_size = batch_size

    def set_thread(self, thread_num):
        self.dataset.set_thread_num(thread_num)

    def set_filelist(self, filelist):
        self.dataset.set_filelist(filelist)

    def set_use_var(self, var_list):
        multi_slot = self.proto_desc.multi_slot_desc
        for var in var_list:
            slot_var = multi_slot.slots.add()
            slot_var.is_used = True
            slot_var.name = var.name
            if var.lod_level == 0:
                slot_var.is_dense = True
            if var.dtype == core.VarDesc.VarType.FP32:
                slot_var.type = "float32"
            elif var.dtype == core.VarDesc.VarType.INT64:
                slot_var.type = "uint64"
            else:
                raise ValueError(
                    "Currently, fluid.dataset only supports dtype=float32 and dtype=int64"
                )

    def desc(self):
        """
        Returns a protobuf message for this DataFeedDesc

        Example:
            >>> data_feed = fluid.DataFeedDesc('data.proto')
            >>> print(data_feed.desc())

        Returns:
            A string message
        """
        return text_format.MessageToString(self.proto_desc)


class InMemoryDataset(DatasetBase):
    def __init__(self):
        super(InMemoryDataset, self).__init__()
        self.proto_desc.name = "MultiSlotInMemoryDataFeed"

    def load_into_memory(self):
        self.dataset.set_data_feed_desc(self.desc())
        self.dataset.load_into_memory()

    def local_shuffle(self):
        self.dataset.local_shuffle()

    def global_shuffle(self):
        from .distributed import ps_instance
        instance = ps_instance.PaddlePSInstance(1, 2)
        self.dataset.set_trainer_num(instance.get_worker_num())
        self.global_shuffle()


class QueueDataset(DatasetBase):
    def __init__(self):
        super(QueueDataset, self).__init__()
        self.proto_desc.name = "MultiSlotDataFeed"
