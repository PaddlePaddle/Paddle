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

__all__ = ['DataFeedDesc']


class DataFeedDesc(object):
    """
    Datafeed descriptor, describing input training data format. This class is
    currently only used for AsyncExecutor (See comments for class AsyncExecutor
    for a brief introduction)

    DataFeedDesc shall be initialized from a valid protobuf message from disk:
    >>> data_feed = fluid.DataFeedDesc('data.proto')

    See :code:`paddle/fluid/framework/data_feed.proto` for message definition.
    A typical message might look like:

    >>> name: "MultiSlotDataFeed"
    >>> batch_size: 2
    >>> multi_slot_desc {
    >>>     slots {
    >>>         name: "words"
    >>>         type: "uint64"
    >>>         is_dense: false
    >>>         is_used: true
    >>>     }
    >>>     slots {
    >>>         name: "label"
    >>>         type: "uint64"
    >>>         is_dense: false
    >>>         is_used: true
    >>>     }
    >>> }

    However, users usually shouldn't care about the message format; instead,
    they are encouragd to use :code:`Data Generator` as a tool to generate a
    valid data description, in the process of converting their raw log files to
    training files acceptable to AsyncExecutor.

    DataFeedDesc can also be changed during runtime. Once you got familiar with
    what each field mean, you can modify it to better suit your need. E.g.:
    >>> data_feed.set_batch_size(128)
    >>> data_feed.set_dense_slots('wd')  # The slot named 'wd' will be dense
    >>> data_feed.set_use_slots('wd')    # The slot named 'wd' will be used

    Finally, the content can be dumped out for debugging purpose:
    >>> print(data_feed.desc())

    Args:
        proto_file(string): Disk file containing a data feed description.
    
    """

    def __init__(self, proto_file):
        self.proto_desc = data_feed_pb2.DataFeedDesc()
        with open(proto_file, 'r') as f:
            text_format.Parse(f.read(), self.proto_desc)
        if self.proto_desc.name == "MultiSlotDataFeed":
            self.__name_to_index = {
                slot.name: i
                for i, slot in enumerate(self.proto_desc.multi_slot_desc.slots)
            }

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

    def set_dense_slots(self, dense_slots_name):
        """
        Set if a specific slot will be dense. Will be effective during training.
        features for a dense slot will be fed into a Tensor, while those for a
        sparse slot will be fed into a LoDTensor

        Example:
            >>> data_feed = fluid.DataFeedDesc('data.proto')
            >>> data_feed.set_dense_slots(['words'])

        Args:
            dense_slots_name: a list of slot names which will be set dense

        Note:
            Default is sparse for all slots
        """
        if self.proto_desc.name != "MultiSlotDataFeed":
            raise ValueError(
                "Only MultiSlotDataFeed need set_dense_slots, pls check your datafeed.proto"
            )
        for name in dense_slots_name:
            self.proto_desc.multi_slot_desc.slots[self.__name_to_index[
                name]].is_dense = True

    def set_pipe_command(self, pipe_command):
        """
        Pipeline command will be set with this function. In IO runtime, 
        pipeline command will be executed given user provided input raw
        files.

        Example:
            >>> data_feed = fluid.DataFeedDesc('data.proto')
            >>> data_feed.set_pipe_command('awk -F '\t' '{print $2}'')

        Args:
            pipe_command: a command string of shell command
        
        Note:
            Default is cat, i.e., cat user's input file list to data feed
        """
        self.proto_desc.pipe_command = pipe_command

    def set_use_slots(self, use_slots_name):
        """
        Set if a specific slot will be used for training. A dataset shall
        contain a lot of features, through this function one can select which
        ones will be used for a specific model.

        Example:
            >>> data_feed = fluid.DataFeedDesc('data.proto')
            >>> data_feed.set_use_slots(['words'])

        Args:
            use_slots_name: a list of slot names which will be used in training

        Note:
            Default is not used for all slots
        """
        if self.proto_desc.name != "MultiSlotDataFeed":
            raise ValueError(
                "Only MultiSlotDataFeed need set_use_slots, pls check your datafeed.proto"
            )
        for name in use_slots_name:
            self.proto_desc.multi_slot_desc.slots[self.__name_to_index[
                name]].is_used = True

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
