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

from google.protobuf import text_format

from paddle.base.proto import data_feed_pb2

__all__ = []


class DataFeedDesc:
    r"""
    :api_attr: Static Graph

    Datafeed descriptor, describing input training data format.

    DataFeedDesc shall be initialized from a valid protobuf message from disk.

    See :code:`paddle/base/framework/data_feed.proto` for message definition.
    A typical message might look like:

    Examples:
        .. code-block:: python

            >>> import paddle.base as base
            >>> with open("data.proto", "w") as f:
            ...     f.write('name: "MultiSlotDataFeed"\n')
            ...     f.write('batch_size: 2\n')
            ...     f.write('multi_slot_desc {\n')
            ...     f.write('    slots {\n')
            ...     f.write('        name: "words"\n')
            ...     f.write('        type: "uint64"\n')
            ...     f.write('        is_dense: false\n')
            ...     f.write('        is_used: true\n')
            ...     f.write('    }\n')
            ...     f.write('    slots {\n')
            ...     f.write('        name: "label"\n')
            ...     f.write('        type: "uint64"\n')
            ...     f.write('        is_dense: false\n')
            ...     f.write('        is_used: true\n')
            ...     f.write('    }\n')
            ...     f.write('}')
            >>> data_feed = base.DataFeedDesc('data.proto')

        However, users usually shouldn't care about the message format; instead,
        they are encouraged to use :code:`Data Generator` as a tool to generate a
        valid data description, in the process of converting their raw log files to
        training files acceptable to Executor.

        DataFeedDesc can also be changed during runtime. Once you got familiar with
        what each field mean, you can modify it to better suit your need. E.g.:

        .. code-block:: python

            >>> import paddle.base as base
            >>> data_feed = base.DataFeedDesc('data.proto')
            >>> data_feed.set_batch_size(128)
            >>> data_feed.set_dense_slots(['words'])  # The slot named 'words' will be dense
            >>> data_feed.set_use_slots(['words'])    # The slot named 'words' will be used

            >>> # Finally, the content can be dumped out for debugging purpose:

            >>> print(data_feed.desc())

    Args:
        proto_file(string): Disk file containing a data feed description.

    """

    def __init__(self, proto_file):
        self.proto_desc = data_feed_pb2.DataFeedDesc()
        self.proto_desc.pipe_command = "cat"
        with open(proto_file, 'r') as f:
            text_format.Parse(f.read(), self.proto_desc)
        if self.proto_desc.name == "MultiSlotDataFeed":
            self.__name_to_index = {
                slot.name: i
                for i, slot in enumerate(self.proto_desc.multi_slot_desc.slots)
            }

    def set_batch_size(self, batch_size):
        r"""
        Set :attr:`batch_size` in ``paddle.base.DataFeedDesc`` . :attr:`batch_size` can be changed during training.

        Examples:
            .. code-block:: python

                >>> import paddle.base as base
                >>> with open("data.proto", "w") as f:
                ...     f.write('name: "MultiSlotDataFeed"\n')
                ...     f.write('batch_size: 2\n')
                ...     f.write('multi_slot_desc {\n')
                ...     f.write('    slots {\n')
                ...     f.write('        name: "words"\n')
                ...     f.write('        type: "uint64"\n')
                ...     f.write('        is_dense: false\n')
                ...     f.write('        is_used: true\n')
                ...     f.write('    }\n')
                ...     f.write('    slots {\n')
                ...     f.write('        name: "label"\n')
                ...     f.write('        type: "uint64"\n')
                ...     f.write('        is_dense: false\n')
                ...     f.write('        is_used: true\n')
                ...     f.write('    }\n')
                ...     f.write('}')
                >>> data_feed = base.DataFeedDesc('data.proto')
                >>> data_feed.set_batch_size(128)

        Args:
            batch_size (int): The number of batch size.

        Returns:
            None.

        """
        self.proto_desc.batch_size = batch_size

    def set_dense_slots(self, dense_slots_name):
        r"""
        Set slots in :attr:`dense_slots_name` as dense slots. **Note: In default, all slots are sparse slots.**

        Features for a dense slot will be fed into a Tensor, while those for a
        sparse slot will be fed into a LoDTensor.

        Examples:
            .. code-block:: python

                >>> import paddle.base as base
                >>> with open("data.proto", "w") as f:
                ...     f.write('name: "MultiSlotDataFeed"\n')
                ...     f.write('batch_size: 2\n')
                ...     f.write('multi_slot_desc {\n')
                ...     f.write('    slots {\n')
                ...     f.write('        name: "words"\n')
                ...     f.write('        type: "uint64"\n')
                ...     f.write('        is_dense: false\n')
                ...     f.write('        is_used: true\n')
                ...     f.write('    }\n')
                ...     f.write('    slots {\n')
                ...     f.write('        name: "label"\n')
                ...     f.write('        type: "uint64"\n')
                ...     f.write('        is_dense: false\n')
                ...     f.write('        is_used: true\n')
                ...     f.write('    }\n')
                ...     f.write('}')
                >>> data_feed = base.DataFeedDesc('data.proto')
                >>> data_feed.set_dense_slots(['words'])

        Args:
            dense_slots_name (list(str)): a list of slot names which will be set dense.

        Returns:
            None.

        """
        if self.proto_desc.name != "MultiSlotDataFeed":
            raise ValueError(
                "Only MultiSlotDataFeed needs set_dense_slots, please check your datafeed.proto"
            )
        for name in dense_slots_name:
            self.proto_desc.multi_slot_desc.slots[
                self.__name_to_index[name]
            ].is_dense = True

    def set_use_slots(self, use_slots_name):
        r"""
        Set if a specific slot will be used for training. A dataset shall
        contain a lot of features, through this function one can select which
        ones will be used for a specific model.

        Examples:
            .. code-block:: python

                >>> import paddle.base as base
                >>> with open("data.proto", "w") as f:
                ...     f.write('name: "MultiSlotDataFeed"\n')
                ...     f.write('batch_size: 2\n')
                ...     f.write('multi_slot_desc {\n')
                ...     f.write('    slots {\n')
                ...     f.write('        name: "words"\n')
                ...     f.write('        type: "uint64"\n')
                ...     f.write('        is_dense: false\n')
                ...     f.write('        is_used: true\n')
                ...     f.write('    }\n')
                ...     f.write('    slots {\n')
                ...     f.write('        name: "label"\n')
                ...     f.write('        type: "uint64"\n')
                ...     f.write('        is_dense: false\n')
                ...     f.write('        is_used: true\n')
                ...     f.write('    }\n')
                ...     f.write('}')
                >>> data_feed = base.DataFeedDesc('data.proto')
                >>> data_feed.set_use_slots(['words'])

        Args:
            use_slots_name: a list of slot names which will be used in training

        Note:
            Default is not used for all slots
        """
        if self.proto_desc.name != "MultiSlotDataFeed":
            raise ValueError(
                "Only MultiSlotDataFeed needs set_use_slots, please check your datafeed.proto"
            )
        for name in use_slots_name:
            self.proto_desc.multi_slot_desc.slots[
                self.__name_to_index[name]
            ].is_used = True

    def desc(self):
        r"""
        Returns a protobuf message for this DataFeedDesc

        Examples:
            .. code-block:: python

                >>> import paddle.base as base
                >>> with open("data.proto", "w") as f:
                ...     f.write('name: "MultiSlotDataFeed"\n')
                ...     f.write('batch_size: 2\n')
                ...     f.write('multi_slot_desc {\n')
                ...     f.write('    slots {\n')
                ...     f.write('        name: "words"\n')
                ...     f.write('        type: "uint64"\n')
                ...     f.write('        is_dense: false\n')
                ...     f.write('        is_used: true\n')
                ...     f.write('    }\n')
                ...     f.write('    slots {\n')
                ...     f.write('        name: "label"\n')
                ...     f.write('        type: "uint64"\n')
                ...     f.write('        is_dense: false\n')
                ...     f.write('        is_used: true\n')
                ...     f.write('    }\n')
                ...     f.write('}')
                >>> data_feed = base.DataFeedDesc('data.proto')
                >>> print(data_feed.desc())

        Returns:
            A string message
        """
        return text_format.MessageToString(self.proto_desc)
