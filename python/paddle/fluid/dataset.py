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
    """
    DatasetFactory is a factory which create dataset by its name,
    you can create "QueueDataset" or "InMemoryDataset",
    the default is "QueueDataset".

    Example:
        dataset = paddle.fluid.DatasetFactory.create_dataset("InMemoryDataset")
    """

    def __init__(self):
        """
        Init
        """
        pass

    def create_dataset(self, datafeed_class="QueueDataset"):
        """
        Create "QueueDataset" or "InMemoryDataset",
        the default is "QueueDataset".
        """
        try:
            dataset = globals()[datafeed_class]()
            return dataset
        except:
            raise ValueError("datafeed class %s does not exist" %
                             datafeed_class)


class DatasetBase(object):
    """
    Base dataset class
    """

    def __init__(self):
        """
        Init
        """
        # define class name here
        # to decide whether we need create in memory instance
        self.proto_desc = data_feed_pb2.DataFeedDesc()
        self.proto_desc.pipe_command = "cat"
        self.dataset = core.Dataset("MultiSlotDataset")
        self.thread_num = 0

    def set_pipe_command(self, pipe_command):
        """
        Set pipe command of current dataset
        A pipe command is a UNIX pipeline command that can be used only

        Example:
            >>> dataset.set_pipe_command("python my_script.py")

        Args:
            pipe_command: pipe command

        """
        self.proto_desc.pipe_command = pipe_command

    def set_batch_size(self, batch_size):
        """
        Set batch size. Will be effective during training

        Example:
            >>> dataset.set_batch_size(128)

        Args:
            batch_size: batch size

        """
        self.proto_desc.batch_size = batch_size

    def set_thread(self, thread_num):
        """
        Set thread num, it is the num of readers.

        Example:
            >>> dataset.set_thread(12)

        Args:
            thread_num: thread num
        """
        self.dataset.set_thread_num(thread_num)
        self.thread_num = thread_num

    def set_filelist(self, filelist):
        """
        Set file list in current worker.

        Example:
            >>> dataset.set_filelist(['a.txt', 'b.txt'])

        Args:
            filelist: file list
        """
        self.dataset.set_filelist(filelist)

    def set_use_var(self, var_list):
        """
        Set Variables which you will use.

        Example:
            >>> dataset.set_use_var([data, label])

        Args:
            var_list: variable list
        """
        multi_slot = self.proto_desc.multi_slot_desc
        for var in var_list:
            slot_var = multi_slot.slots.add()
            slot_var.is_used = True
            slot_var.name = var.name
            if var.lod_level == 0:
                slot_var.is_dense = True
            if var.dtype == core.VarDesc.VarType.FP32:
                slot_var.type = "float"
            elif var.dtype == core.VarDesc.VarType.INT64:
                slot_var.type = "uint64"
            else:
                raise ValueError(
                    "Currently, fluid.dataset only supports dtype=float32 and dtype=int64"
                )

    def set_hdfs_config(self, fs_name, fs_ugi):
        """
        Set hdfs config: fs name ad ugi

        Example:
            >>> dataset.set_hdfs_config("my_fs_name", "my_fs_ugi")

        Args:
            fs_name: fs name
            fs_ugi: fs ugi
        """
        self.dataset.set_hdfs_config(fs_name, fs_ugi)

    def _prepare_to_run(self):
        """
        Set data_feed_desc before load or shuffle,
        user no need to call this function.
        """
        self.dataset.set_data_feed_desc(self.desc())

    def desc(self):
        """
        Returns a protobuf message for this DataFeedDesc

        Example:
            >>> print(dataset.desc())

        Returns:
            A string message
        """
        return text_format.MessageToString(self.proto_desc)


class InMemoryDataset(DatasetBase):
    """
    InMemoryDataset, it will load data into memory
    and shuffle data before training

    Example:
        dataset = paddle.fluid.DatasetFactory.create_dataset("InMemoryDataset")
    """

    def __init__(self):
        """
        Init
        """
        super(InMemoryDataset, self).__init__()
        self.proto_desc.name = "MultiSlotInMemoryDataFeed"

    def load_into_memory(self):
        """
        Load data into memory

        Example:
            >>> import paddle.fluid as fluid
            >>> dataset = fluid.DatasetFactory.create_dataset("InMemoryDataset")
            >>> filelist = ["a.txt", "b.txt"]
            >>> dataset.set_filelist(filelist)
            >>> dataset.load_into_memory()
        """
        self._prepare_to_run()
        self.dataset.load_into_memory()

    def local_shuffle(self):
        """
        Local shuffle

        Example:
            >>> import paddle.fluid as fluid
            >>> dataset = fluid.DatasetFactory.create_dataset("InMemoryDataset")
            >>> filelist = ["a.txt", "b.txt"]
            >>> dataset.set_filelist(filelist)
            >>> dataset.local_shuffle()
        """
        self.dataset.local_shuffle()

    def global_shuffle(self, fleet=None):
        """
        Global shuffle.
        If you run distributed, you should pass fleet instead of None.

        Example:
            >>> import paddle.fluid as fluid
            >>> import paddle.fluid.incubate.fleet.parameter_server as fleet
            >>> dataset = fluid.DatasetFactory.create_dataset("InMemoryDataset")
            >>> filelist = ["a.txt", "b.txt"]
            >>> dataset.set_filelist(filelist)
            >>> dataset.global_shuffle(fleet)

        Args:
            fleet: fleet singleton. Default None.
        """
        trainer_num = 1
        if fleet is not None:
            fleet.fleet_instance.role_maker_._barrier_worker()
            trainer_num = fleet.worker_num()
        self.dataset.register_client2client_msg_handler()
        self.dataset.set_trainer_num(trainer_num)
        if fleet is not None:
            fleet.fleet_instance.role_maker_._barrier_worker()
        self.dataset.global_shuffle()
        if fleet is not None:
            fleet.fleet_instance.role_maker_._barrier_worker()


class QueueDataset(DatasetBase):
    """
    QueueDataset, it will process data streamly.

    Example:
        import paddle.fluid as fluid
        dataset = fluid.DatasetFactory.create_dataset("QueueDataset")
    """

    def __init__(self):
        """
        Init
        """
        super(QueueDataset, self).__init__()
        self.proto_desc.name = "MultiSlotDataFeed"

    def local_shuffle(self):
        """
        Local shuffle

        QueueDataset does not support local shuffle
        """
        raise NotImplementedError(
            "QueueDataset does not support local shuffle, "
            "please use InMemoryDataset for local_shuffle")

    def global_shuffle(self, fleet=None):
        """
        Global shuffle
        """
        raise NotImplementedError(
            "QueueDataset does not support global shuffle, "
            "please use InMemoryDataset for global_shuffle")
