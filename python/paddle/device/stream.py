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

import ctypes

import paddle


class Event(paddle.fluid.core.eager.EventBase):
    '''
    A event rapper around EventBase.

    Parameters:
        enable_timing (bool, optional): indicates if the event should measure time, default is False
        blocking (bool, optional): if True, ``wait`` will be blocking, default is False
        interprocess (bool): if True, the event can be shared between processes, default is False

    Returns:
        Event: The event.
    '''

    def __new__(cls, enable_timing=False, blocking=False, interprocess=False):
        return super(Event, cls).__new__(
            cls,
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess,
        )

    def record(self, stream=None):
        '''
        Records the event in a given stream.

        Parameters:
            stream(Stream, optional): The given stream. By default, stream is None,
            event will be recorded in current_stream.

        Returns:
            None.
        '''
        if stream is None:
            stream = current_stream()
        super(Event, self).record(stream)

    def wait(self, stream=None):
        '''
        Makes all future work submitted to the given stream wait for this
        event.

        Parameters:
            stream(Stream, optional): The given stream. By default, stream is None,
            current_stream will wait for this event.

        Returns:
            None.
        '''
        if stream is None:
            stream = current_stream()
        super(Event, self).wait(stream)

    def query(self):
        '''
        Checks if all work currently captured by event has completed.

        Returns:
            bool: Whether all work currently captured by event has completed.
        '''
        return super(Event, self).query()

    def elapsed_time(self, end_event):
        '''
        Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.

        Returns:
            int: The time.
        '''
        return super(Event, self).elapsed_time(end_event)

    def synchronize(self):
        '''
        Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

        Returns:
            None.
        '''
        super(Event, self).synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.event)

    def __repr__(self):
        if self.event:
            return '<paddle.device.Event {0:#x}>'.format(
                self._as_parameter_.value
            )
        else:
            return '<paddle.device.Event uninitialized>'


class Stream(paddle.fluid.core.eager.StreamBase):
    '''
    A device stream wrapper around StreamBase.

    Parameters:
        device(str|paddle.CUDAPlace(n)|paddle.CustomPlace(n)): Which device the stream runn on. If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``xpu``, ``npu``, ``mlu``, ``gpu:x``, ``xpu:x``, ``npu:x``, ``mlu:x`` and ``ipu``,
            where ``x`` is the index of the GPUs, XPUs, NPUs or MLUs. And it can be paddle.CUDAPlace(n) or paddle.CustomPlace(n).
        priority(int, optional): priority of the CUDA stream. Can be either
            -1 (high priority) or 0 (low priority). By default, streams have
            priority 0. The priority only supporded on GPU.

    Returns:
        Stream: The stream.

    Examples:

        .. code-block:: python

            import paddle

            s = paddle.device.Stream()
    '''

    def __new__(cls, device=None, priority=0):
        if device is None:
            place = paddle.framework._current_expected_place()
        else:
            place = paddle.device._convert_to_place(device)
        super(Stream, cls).__new__(place, priority)

    def wait_event(self, event):
        '''
        Makes all future work submitted to the stream wait for an event.

        Parameters:
            event (Event): an event to wait for.

        Returns:
            None.
        '''
        event.wait(self)

    def wait_stream(self, stream):
        '''
        Synchronizes with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Parameters:
            stream (Stream): a stream to synchronize.

        Returns:
            None.
        '''
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        '''
        Records an event.

        Parameters:
            event (Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Event: Recorded event.
        '''
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self):
        '''
        Checks if all the work submitted has been completed.

        Returns:
            bool: Whether all kernels in this stream are completed.
        '''
        return super(Stream, self).query()

    def synchronize(self):
        '''
        Wait for all the kernels in this stream to complete.

        Returns:
            None.
        '''
        super(Stream, self).synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.stream)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super(Stream, self).__eq__(o)
        return False

    def __hash__(self):
        return hash((self.stream, self.device))

    def __repr__(self):
        return '<paddle.device.Stream device={0} stream={1:#x}>'.format(
            self.device, self.stream
        )


def current_stream(device=None):
    '''
    Return the current CUDA stream by the device.

    Parameters:
        device(str|paddle.CUDAPlace(n)|paddle.CustomPlace(n)): The device which want to get stream from.  If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``xpu``, ``npu``, ``mlu``, ``gpu:x``, ``xpu:x``, ``npu:x``, ``mlu:x`` and ``ipu``,
            where ``x`` is the index of the GPUs, XPUs, NPUs or MLUs. And it can be paddle.CUDAPlace(n) or paddle.CustomPlace(n).

    Returns:
        Stream: The stream to the device.

    Examples:
        .. code-block:: python

            import paddle

            s1 = paddle.device.current_stream()

            s2 = paddle.device.current_stream("gpu:0")

            place = paddle.CustomPlace('custom_cpu', 0)
            s3 = paddle.device.current_stream(place)

    '''
    if device is None:
        place = paddle.framework._current_expected_place()
    else:
        place = paddle.device._convert_to_place(device)
    return paddle.fluid.core.eager.current_stream(place)


def default_stream(device=None):
    '''
    Returns the default Stream for a given device.

    Parameters:
        device(str|paddle.CUDAPlace(n)|paddle.CustomPlace(n)): The device which want to get default stream from.  If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``xpu``, ``npu``, ``mlu``, ``gpu:x``, ``xpu:x``, ``npu:x``, ``mlu:x`` and ``ipu``,
            where ``x`` is the index of the GPUs, XPUs, NPUs or MLUs. And it can be paddle.CUDAPlace(n) or paddle.CustomPlace(n).

    Returns:
        Stream: The default stream to the device.
    '''
    if device is None:
        place = paddle.framework._current_expected_place()
    else:
        place = paddle.device._convert_to_place(device)
    return paddle.fluid.core.eager.default_stream(place)


def set_stream(stream):
    '''
    Set the current stream.

    Parameters:
        stream(Stream): The selected stream.

    Returns:
        Stream: The previous stream.

    '''
    return paddle.fluid.core.eager.set_stream(stream)


def synchronize(device=None):
    '''
    Wait for the compute on the given device to finish.

    Parameters:
        device(str|paddle.CUDAPlace(n)|paddle.CustomPlace(n)): The device which want to wait for.  If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``xpu``, ``npu``, ``mlu``, ``gpu:x``, ``xpu:x``, ``npu:x``, ``mlu:x`` and ``ipu``,
            where ``x`` is the index of the GPUs, XPUs, NPUs or MLUs. And it can be paddle.CUDAPlace(n) or paddle.CustomPlace(n).

    Examples:
        .. code-block:: python

            import paddle

            paddle.device.synchronize()
            paddle.device.synchronize("gpu:0")
            place = paddle.CustomPlace('custom_cpu', 0)
            paddle.device.synchronize(place)

    '''

    if device is None:
        place = paddle.framework._current_expected_place()
    else:
        place = paddle.device._convert_to_place(device)

    paddle.fluid.core.eager.synchronize(place)


class stream(object):
    '''
    Notes:
        This API only supports dynamic graph mode currently.

    A context manager that specifies the current stream context by the given stream.

    Parameters:
        stream(Stream, optional): the selected stream. If stream is None, just yield.

    Returns:
        None.

    Examples:

        .. code-block:: python

            import paddle

            s = paddle.device.Stream()
            data1 = paddle.ones(shape=[20])
            data2 = paddle.ones(shape=[20])
            data3 = data1 + data2
            with paddle.device.stream(s):
                s.wait_stream(paddle.device.default_stream())
                data4 = data1 + data3

    '''

    def __init__(self, stream=None):
        self.stream = stream

    def __enter__(self):
        cur_stream = self.stream
        if cur_stream is None:
            return

        self.src_prev_stream = current_stream()
        if self.src_prev_stream.device != cur_stream.device:
            self.tmp_place = paddle.fluid.framework._current_expected_place()
            paddle.fluid.framework._set_expected_place(cur_stream.device)
            self.dst_prev_stream = current_stream(cur_stream.device)
            set_stream(cur_stream)
        else:
            set_stream(cur_stream)

    def __exit__(self, *args):
        cur_stream = self.stream
        if cur_stream is None:
            return

        if self.src_prev_stream.device != cur_stream.device:
            set_stream(self.dst_prev_stream)
            paddle.fluid.framework._set_expected_place(self.tmp_place)
            set_stream(self.src_prev_stream)
        else:
            set_stream(self.src_prev_stream)
