# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
This module provide a wrapper(decorator) to wrap a data process method into a
PyDataProvider. Some examples are shown `here <data_provider/python_case.html>`_.
"""

import struct
import array
import random
import gc
import logging
import pstats
import sys
import numpy
import functools

__all__ = [
    'DenseSlot', 'SlotType', 'SparseNonValueSlot', 'StringSlot',
    'SparseValueSlot', 'IndexSlot', 'PoolSize', 'GeneralPyDataProvider',
    'provider', 'init_hook_wrapper'
]

try:  # Just for profile mode, will try to import cProfile first.
    # Most python will contains cProfile, cProfile/profile are basically same.
    # ref: https://docs.python.org/2/library/profile.html#introduction-to-the-profilers
    import cProfile as profile
except ImportError:
    import profile

try:
    import cPickle as pickle
except ImportError:
    import pickle

import io


class SlotType(object):  # Just a hint for user.
    pass


class DenseSlot(SlotType):
    """
    Dense Slot Type: Each item is the value of a Dense Vector.

    Its yield format for :code:`provider` is:

    - **NonSeq**: [float, float, ... ]
    - **Seq**: [[float, float, ...], [float, float ....], ... ]
    - **SubSeq**: [[[float, float, ...], [float ....], ...] ,  \
                   [[float, float, ...], [float ....], ...] , ...]
    """

    def __init__(self, dim):
        """
        :param dim: slot dimension
        :type dim: int
        """
        self.dim = dim
        self.type = 0


class SparseNonValueSlot(SlotType):
    """
    Sparse NonValue Slot Type: Each item is the id of a Sparse Vector.

    Its yield format for :code:`provider` is:

    - **NonSeq**: [int, int, ...]
    - **Seq**: [[int, int, ...], [int, int, ...], ... ]
    - **SubSeq**: [[[int, int, ...], [int, ....], ...] ,  \
                   [[int, int, ...], [int, ....], ...] , ...]
    """

    def __init__(self, dim):
        """
        :param dim: slot dimension
        :type dim: int
        """
        self.dim = dim
        self.type = 1


class SparseValueSlot(SlotType):
    """
    Sparse Value Slot Type: Each item is the id and value of a Sparse Vector.

    Its yield format for :code:`provider` is:

    - **NonSeq**: [(int, float), (int, float), ... ]
    - **Seq**: [[(int,float), (int, float), ... ], \
                [(int, float), (int, float), ...], ... ]
    - **SubSeq**: [[[(int,float), ...], [(int, float), ....], ...] ,  \
                   [[(int,float), ...], [(int, float), ....], ...] , ...]
    """

    def __init__(self, dim):
        """
        :param dim: slot dimension.
        :type dim: int
        """
        self.dim = dim
        self.type = 2


class IndexSlot(SlotType):
    """
    Index Value Slot Type: Each item is the id of Label.

    Its yield format for :code:`provider` is:

    - **NonSeq**: int
    - **Seq**:  [int, int, ....]
    - **SubSeq**: [[int, int, ...], [int, int, ...], ... ]
    """

    def __init__(self, dim):
        """
        :param dim: slot dimension
        :type dim: int
        """
        self.dim = dim
        self.type = 3


class StringSlot(SlotType):
    """
    String Value Slot Type: Each item is a string for printout, \
                            can be used in DataLayer too.

    Its yield format for :code:`provider` is:

    - **NonSeq**: string
    - **Seq**: [string, string, ....]
    - **SubSeq**:  [[string, string, ...], [string, string, ...], ... ]
    """

    def __init__(self, dim):
        """
        :param dim: slot dimension
        :type dim: string
        """
        self.dim = dim
        self.type = 6


class SparseNonValueHandler(object):
    """
    Private Class, Use for converting python object to paddle string.
    """

    def __init__(self):
        self.offsets = []
        self.value = []
        self.offset_count = 0

    def __call__(self, ele):
        """
        It will be invoked when scan each sparse data.

        :param ele: list of sparse data, maybe non-value [ idx, ... ] or value.
                    [ (idx, val), ... ]
        :type ele: list
        """
        self.offsets.append(self.offset_count)
        self.offset_count += len(ele)
        self.processElement(ele)

    def processElement(self, ele):
        """
        Process for element list. See __call__ for more document.
        """
        self.value += ele

    def done(self, data_stream, int_packer):
        """
        Dump data to stream.
        :param data_stream: Output Stream.
        :param int_packer:  A struct.Struct("i") object
        """
        data_stream.write(array.array("i", self.offsets).tostring())
        data_stream.write(int_packer.pack(self.offset_count))
        data_stream.write(array.array("i", self.value).tostring())


class SparseValueHandler(SparseNonValueHandler):
    """
    Private class, use for converting python obj to paddle string.
    """

    def __init__(self):
        SparseNonValueHandler.__init__(self)
        self.weight = []

    def processElement(self, ele):
        for idx, w in ele:
            self.value.append(idx)
            self.weight.append(w)

    def done(self, data_stream, int_packer):
        SparseNonValueHandler.done(self, data_stream, int_packer)
        data_stream.write(int_packer.pack(self.offset_count))
        data_stream.write(array.array("f", self.weight).tostring())


class StringHandler(object):
    """
    Private Class, Use for converting python object to paddle string.
    """

    def __init__(self, data_stream, int_packer):
        self.data_stream = data_stream
        self.int_packer = int_packer

    def __call__(self, ele):
        """
        It will be invoked when scan each string data.
        :param ele: string data
        :type ele: str
        """
        self.data_stream.write(self.int_packer.pack(len(ele)))
        self.data_stream.write(array.array("c", ele).tostring())


class GeneralPyDataProvider:
    def __init__(self, *file_list, **kwargs):
        """
        :param file_list: input file_list
        """
        del kwargs  # unused
        gc.disable()
        assert isinstance(self.logger, logging.Logger)
        self.use_seq_flag = hasattr(self, "use_seq_flag") and self.use_seq_flag
        self.slots_num = len(self.getSlots())
        self.file_list = list(file_list)
        self.generators = map(self.generateData, self.file_list)
        self.int_packer = struct.Struct("i")
        self.head_packer = struct.Struct("ii")
        self.float_packer = struct.Struct("f")
        self.shuffler = lambda *args, **kwargs: None
        self.data_pool = []
        self.has_subseq = []
        self.has_checked = False

        self.debug = hasattr(self, "debug") and self.debug

        if hasattr(self, "profile_filename") and isinstance(
                self.profile_filename, str):
            self.profile_count = 0
            self.is_profile = True
        else:
            self.is_profile = False

        if not hasattr(self, "file_count") or not isinstance(self.file_count,
                                                             int):
            self.file_count = sys.maxint

        if not hasattr(self, "can_over_batch_size"):
            self.can_over_batch_size = True
        elif not self.can_over_batch_size:
            self.logger.warn(
                "User should ensure every data size is not larger than batch"
                " size when can_over_batch_size = False")

        self.data_pool_idx = 0

    def reset(self):
        """Reset all data in provider."""

        self.logger.debug("reset dataprovider.")
        self.generators = map(self.generateData, self.file_list)
        self.shuffler = lambda *args, **kwargs: None
        self.data_pool = []
        self.data_pool_idx = 0
        if self.file_count != 0:
            self.max_pool_size = 0

        # When use Profile, each pass will print a profile result.
        if self.is_profile:
            if hasattr(self, "profiler") and isinstance(self.profiler,
                                                        profile.Profile):
                self.profiler.disable()
                fn = "%s_%d" % (self.profile_filename, self.profile_count)
                sortby = "cumulative"
                with open(fn, "w") as f:
                    pstats.Stats(
                        self.profiler,
                        stream=f).sort_stats(sortby).print_stats()
                self.logger.info("saving profile to file %s" % fn)
                self.profile_count += 1
            self.logger.info("resetting profile")
            self.profiler = profile.Profile()
            self.profiler.enable()

    def shuffle(self):
        """ shuffle data"""
        if not self.should_shuffle:
            return
        else:
            self.logger.debug("shuffling data.")
            random.shuffle(self.generators)
            self.shuffler = random.shuffle

    def getSlots(self):
        """
        :return : return a list of SlotType
        :rtype: list
        """
        return []

    def generateData(self, fn):
        """
        :param fn: file name
        :return: a generator to yield data one by one.
        """
        raise NotImplementedError

    def calculateDataBatchSize(self, data):
        """
        :param data: One sample which yield by generateData
        :type data: list
        :return: The batch size that the data contribute.
        :rtype: int
        """
        return 1

    def getHeader(self):
        """return paddle header format"""
        ret = self.head_packer.pack(self.slots_num, self.use_seq_flag)
        for obj in self.getSlots():
            ret += self.head_packer.pack(obj.type, obj.dim)
        return ret

    def getHeaderNative(self):
        return self.use_seq_flag, self.getSlots()

    def getNextBatchNative(self, batch_size):
        ret_list = []
        self.__prepareData(batch_size, ret_list)
        return ret_list

    def getNextBatch(self, batch_size):
        """
        :param batch_size: the batch_size approximately return.
        :return: return paddle pyDataProvider format, just see documents.
        :rtype: str

        NOTE: If can_over_batch_size is True, the return batch_size >= input batch_size.
              Otherwise, the return batch_size < input batch_size, BUT USER MUST ENSURE THAT each data's batch size
              is less than input batch_size.
        """
        ret_list = []
        current_batch_size = self.__prepareData(batch_size, ret_list)
        # create unified format for ret_list with differnt slots_num
        if self.slots_num == 1:
            ret_list = [ret_list]

        if current_batch_size == 0:
            return self.int_packer.pack(current_batch_size)
        data_bytes = io.BytesIO()
        seq_bytes = io.BytesIO()
        subseq_bytes = io.BytesIO()
        data_stream = io.BufferedWriter(data_bytes)
        seq_stream = io.BufferedWriter(seq_bytes)
        subseq_stream = io.BufferedWriter(subseq_bytes)

        def convertDataImpl(idx, data_callback):
            """
            This method will handle sequence in return data. invoke data_callback one by one.
            :param idx: the slot index.
            :param data_callback: a callback, which type is (each sample) => None.
            """
            indices = 0
            slot_sample_num = len(ret_list)
            if self.use_seq_flag:
                slot_sample_num = 0
                if self.has_subseq[idx]:  # has sub-sequence
                    slot_subseq_num = 0
                    for dat in ret_list:
                        dat = dat[idx]
                        slot_subseq_num += len(dat)
                        for sub_dat in dat:
                            slot_sample_num += len(sub_dat)
                    subseq_stream.write(self.int_packer.pack(slot_subseq_num))
                else:
                    for dat in ret_list:
                        dat = dat[idx]
                        slot_sample_num += len(dat)
                seq_stream.write(self.int_packer.pack(len(ret_list)))
            data_stream.write(self.int_packer.pack(slot_sample_num))

            for dat in ret_list:
                dat = dat[idx]
                if self.use_seq_flag:
                    seq_stream.write(self.int_packer.pack(indices))
                    if self.has_subseq[idx]:  # has sub-sequence
                        for sub_dat in dat:
                            writeDataStream(sub_dat, data_callback)
                            subseq_stream.write(self.int_packer.pack(indices))
                            indices += len(sub_dat)
                    else:
                        writeDataStream(dat, data_callback)
                        indices += len(dat)
                else:
                    writeDataStream(dat, data_callback)

        def writeDataStream(dat, data_callback):
            if self.use_seq_flag > 0:
                if data_callback is None:  # Special for index slot
                    data_stream.write(array.array("i", dat).tostring())
                else:
                    for ele in dat:
                        data_callback(ele)
            else:
                if data_callback is None:  # Special for index slot
                    data_stream.write(self.int_packer.pack(dat))
                else:
                    data_callback(dat)

        try:
            for i in range(self.slots_num):
                slot = self.getSlots()[i]
                # According to the data_type, each slot data will be converted to binary
                if isinstance(slot, DenseSlot):
                    convertDataImpl(i, lambda e: data_stream.write(
                        array.array("f", e).tostring()))
                elif isinstance(slot, SparseNonValueSlot):
                    handler = SparseNonValueHandler()
                    convertDataImpl(i, handler)
                    handler.done(data_stream, self.int_packer)
                elif isinstance(slot, SparseValueSlot):
                    handler = SparseValueHandler()
                    convertDataImpl(i, handler)
                    handler.done(data_stream, self.int_packer)
                elif isinstance(slot, IndexSlot):
                    convertDataImpl(i, None)
                elif isinstance(slot, StringSlot):
                    handler = StringHandler(data_stream, self.int_packer)
                    convertDataImpl(i, handler)
                else:
                    raise RuntimeError("The data_type must be 0/1/2/3/6")
            data_stream.flush()
            seq_stream.flush()
            subseq_stream.flush()

            return "".join([
                self.int_packer.pack(current_batch_size), data_bytes.getvalue(),
                seq_bytes.getvalue(), subseq_bytes.getvalue()
            ])

        finally:
            data_stream.close()
            seq_stream.close()
            subseq_stream.close()
            data_bytes.close()
            seq_bytes.close()
            subseq_bytes.close()

    def hasSubseq(self, ret_list):
        # create unified format for ret_list with differnt slots_num
        if self.slots_num == 1:
            ret_list = [ret_list]
        # decide whether slot has sub-sequence using its first sample
        for i in range(self.slots_num):
            slot = self.getSlots()[i]
            dat = ret_list[0][i][0]
            if isinstance(slot, IndexSlot) or isinstance(slot, StringSlot):
                if isinstance(dat, list) or isinstance(dat, numpy.ndarray):
                    self.has_subseq.append(1)  # has_subseq = True
                    continue
            elif isinstance(dat[0], list) or isinstance(dat[0], numpy.ndarray):
                self.has_subseq.append(1)  # has_subseq = True
                continue
            self.has_subseq.append(0)  # has_subseq = False

    def checkOrder(self):
        first_noSubseq_slot = self.slots_num
        last_subseq_slot = -1
        for i in range(self.slots_num):
            if not self.has_subseq[i]:
                first_noSubseq_slot = i
                break
        for i in range(self.slots_num):
            if self.has_subseq[i]:
                last_subseq_slot = i
        if first_noSubseq_slot < last_subseq_slot:
            raise RuntimeError(
                "slot hasSubseq must put before than slot without subseq")
        self.has_checked = True

    def __prepareData(self, batch_size, ret_list):
        current_batch_size = 0
        could_exit = False
        while not could_exit:
            if len(self.data_pool) == 0:
                self.data_pool_idx = 0
                self.fillPool()
            if len(self.data_pool) != 0:
                for idx in xrange(self.data_pool_idx, len(self.data_pool)):
                    current_batch_size += self.calculateDataBatchSize(
                        self.data_pool[idx])
                    if current_batch_size >= batch_size:
                        could_exit = True
                        break
                if current_batch_size > batch_size and not self.can_over_batch_size:  # if cannot over batch size
                    current_batch_size -= self.calculateDataBatchSize(
                        self.data_pool[idx])
                    idx -= 1

                ret_list += self.data_pool[self.data_pool_idx:idx + 1]

                # for speed reason, just shift left index, not delete data actually.
                self.data_pool_idx = idx + 1

                if self.data_pool_idx == len(self.data_pool):
                    self.data_pool = []
            else:
                break
        if self.use_seq_flag and not self.has_checked:  # compute self.has_subseq and checkOrder only at first time
            self.hasSubseq(ret_list)
            self.checkOrder()
        return current_batch_size

    def fillPool(self):
        """
        Fill the pool to max_pool_size. If max_pool_size is None, then read file_count to pool.
        """
        if self.max_pool_size == 0:
            for i in xrange(min(self.file_count, len(self.generators))):
                self.data_pool += list(self.generators[i])
            self.generators = self.generators[min(self.file_count,
                                                  len(self.generators)):]
            self.max_pool_size = len(self.data_pool)
        else:
            while len(self.data_pool) < self.max_pool_size and len(
                    self.generators) != 0:
                try:
                    self.data_pool.append(self.generators[0].next())
                except StopIteration:
                    self.generators.pop(0)
        self.shuffler(self.data_pool)


class PoolSize(object):
    """Max number of sample which contains in provider."""

    def __init__(self, pool_size):
        self.size = pool_size


def default_init_hook(cls, *args, **kwargs):
    """ default hook, do nothing """
    del cls, args, kwargs


def provider(slots=None,
             use_seq=False,
             should_shuffle=True,
             pool_size=1,
             can_over_batch_size=True,
             calc_batch_size=lambda data: 1,
             debug=False,
             init_hook=default_init_hook,
             profile_filename=None):
    """
    The decorator for PyDataProvider. User should use this to create Provider class.
    User should only concern how to read sample from file.

    So the basic usage is:

    ..  code-block:: python

        @provider(some data provider config here...)
        def process(obj, file_name):
            while not at end of file_name:
                sample = readOneSampleFromFile(file_name)
                yield sample.

    The configuration of data provider should be setup by:

    :param init_hook: A callback will be invoked when PyDataProvider instance \
                      created. The parameter is (obj, \*args, \*\*kwargs).

                      - **obj**: actually data provider instance, which \
                                 contains some global objects in obj.xxxxx, \
                                 and is used by process function.

                        1. **obj.slots**: a list of SlotType Object. Can be \
                                          set in init. For example, obj.slots = \
                                          [DenseSlot(9), IndexSlot(2)].
                        2. **obj.logger**: a logger object. User can invoke \
                                          obj.logger.info(), obj.logger.fatal(), etc.

                      - **args** and **kwargs**: the data provider __init__ \
                                                 parameters. For example, load_data_args \
                                                 will be found in \*\*kwargs, \
                                                 and if you want to recieve \
                                                 it from trainer_config, \
                                                 recommand to use init_hook_wrapper
    :type init_hook: callable

    :param pool_size:
                      - **int**: it will read at most pool_size files to memory.
                      - **PoolSize**: it will read at most PoolSize.size samples to memory.
                      - If not set, it will read all the files to memory.
    :type pool_size: int | PoolSize

    :param slots: Specify the SlotTypes, can also be set in init_hook. It has two formats:

                  - A list of SlotType objects. For example, slots = \
                    [DenseSlot(9), IndexSlot(2)].
                  - A method return a list of SlotTypes, and the parameter of \
                    method is (obj, \*file_list, \*\*kwargs).
    :type slots: list | callable

    :param use_seq:  False if use no sequence (Default). True if use sequence:

                     - If sequence has **no sub-sequence**: Each slot will \
                       return a list of data. This list is one sequence. \
                       So the return format likes \
                       [[a0, a1, a2], [b1, b2, b3, b4], [c1]].
                     - If sequence has **sub-sequence**: Each slot will return \
                       a nested-list of data. This list contains several \
                       sub-lists, each sub-list is one sub-sequence. \
                       So the return format likes \
                       [[[a0, a1, a2], [a4, a5]], [[b1, b2, b3, b4], [b5, b6]], [[c1], [c2]]].
    :type use_seq: bool

    :param should_shuffle: True if data should shuffle.
    :type should_shuffle: bool

    :param calc_batch_size: The method calculate each data's batch size.

                            - Default is the batch size of one sample.
                            - User can customize by **lamda** funtion. For example, \
                              :code:`calc_batch_size = lambda data : len(data)` \
                              means calculating the token number of a sequence data.
    :type calc_batch_size: callable

    :param can_over_batch_size: Whether :code:`actual batch size >= input batch size`

                                - **True** (>=): getNextBatch method can return more data (Default).
                                - **False** (<): user must ensure that each data's batch size < input batch size.
    :type can_over_batch_size: bool

    :param debug: True if enable debug logger and some debug check. Default is False.
    :type debug: bool

    :param profile_filename: None if disable profile (Default). Otherwise, \
                             the data provider will dump profile result when \
                             reset. And the dump filename is \
                             **<profile_filename>_<reset_count>**.
    :type profile_filename: None | Str
    """

    def _wrapper(handler):
        class Cls(GeneralPyDataProvider):
            """ Real PyDataProvider Class. """

            def __init__(self, *file_list, **kwargs):
                logging.basicConfig(
                    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s]"
                    " %(message)s")

                self.logger = logging.getLogger("")
                if debug:
                    self.logger.setLevel(logging.DEBUG)
                    self.logger.debug("Running pydataprovider in debug mode.")
                else:
                    self.logger.setLevel(logging.INFO)

                init_hook(self, *file_list, **kwargs)
                if callable(slots):
                    self.slots = slots(self, *file_list, **kwargs)
                elif slots is not None:
                    self.slots = slots

                if isinstance(pool_size, int):
                    self.max_pool_size = 0
                    self.file_count = pool_size
                elif isinstance(pool_size, PoolSize):
                    self.max_pool_size = pool_size.size
                    self.file_count = 0
                else:
                    raise RuntimeError
                self.can_over_batch_size = can_over_batch_size
                self.debug = debug
                self.profile_filename = profile_filename
                self.use_seq_flag = use_seq
                self.should_shuffle = should_shuffle
                GeneralPyDataProvider.__init__(self, *file_list, **kwargs)

            def getSlots(self):
                return self.slots

            def generateData(self, f):
                return handler(self, f)

            def calculateDataBatchSize(self, data):
                return calc_batch_size(data)

        return Cls

    return _wrapper


def init_hook_wrapper(func):
    """
    Wrap a method for PyDataProviderWrapper's init_hook. This method can
    receive parameter from trainer_config's load_data_args. The load_data_args
    must pass a pickle.dumps() value, and dump a map as keyword args. The
    wrapped method :code:`func` will receive them as keyword args.

    So an example usage is:

    ..  code-block:: python

        @init_hook_wrapper
        def hook(obj, dictionary, file_list, **kwargs):
            obj.dictionary = dictionary
            obj.slots = [IndexSlot(len(obj.dictionary)),
                         IndexSlot(len(open(file_list[0], "r").readlines()))]

    :param func: init_hook function
    :type func: callable
    :return: wrapped method, can be passed into @provider.
    """

    @functools.wraps(func)
    def wrapper(obj, *file_list, **kwargs):
        args = kwargs.get("load_data_args", dict())
        if isinstance(args, basestring):
            args = pickle.loads(args)
        args['file_list'] = file_list
        func(obj=obj, **args)

    return wrapper
