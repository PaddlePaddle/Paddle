# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define the functions to manipulate devices
import re
import os
import ctypes
import paddle
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.framework import is_compiled_with_cinn  # noqa: F401
from paddle.fluid.framework import is_compiled_with_cuda  # noqa: F401
from paddle.fluid.framework import is_compiled_with_rocm  # noqa: F401
from . import cuda
from . import xpu

__all__ = [  # noqa
    'get_cudnn_version',
    'set_device',
    'get_device',
    'XPUPlace',
    'IPUPlace',
    'is_compiled_with_xpu',
    'is_compiled_with_ipu',
    'is_compiled_with_cinn',
    'is_compiled_with_cuda',
    'is_compiled_with_rocm',
    'is_compiled_with_custom_device',
    'get_all_device_type',
    'get_all_custom_device_type',
    'get_available_device',
    'get_available_custom_device',
    'Stream',
    'Event',
    'current_stream',
    'set_stream',
    'stream_guard',
    'synchronize',
]

_cudnn_version = None


def is_compiled_with_custom_device(device_type):
    """
    Whether paddle was built with Paddle_CUSTOM_DEVICE .

    Args:
        std::string, the registered device type, like "npu".
    Return:
        bool, ``True`` if CustomDevice is supported, otherwise ``False``.

    Examples:
        .. code-block:: python

            import paddle
            support_npu = paddle.device.is_compiled_with_custom_device("npu")
    """
    return core.is_compiled_with_custom_device(device_type)


def is_compiled_with_ipu():
    """
    Whether paddle was built with WITH_IPU=ON to support Graphcore IPU.

    Returns (bool): `True` if IPU is supported, otherwise `False`.

    Examples:
        .. code-block:: python

            import paddle
            support_ipu = paddle.is_compiled_with_ipu()
    """
    return core.is_compiled_with_ipu()


def IPUPlace():
    """
    Return a Graphcore IPU Place

    Examples:
        .. code-block:: python

            # required: ipu

            import paddle
            place = paddle.device.IPUPlace()
    """
    return core.IPUPlace()


def is_compiled_with_xpu():
    """
    Whether paddle was built with WITH_XPU=ON to support Baidu Kunlun

    Returns (bool): whether paddle was built with WITH_XPU=ON

    Examples:
        .. code-block:: python

            import paddle
            support_xpu = paddle.device.is_compiled_with_xpu()
    """
    return core.is_compiled_with_xpu()


def XPUPlace(dev_id):
    """
    Return a Baidu Kunlun Place

    Parameters:
        dev_id(int): Baidu Kunlun device id

    Examples:
        .. code-block:: python

            # required: xpu

            import paddle
            place = paddle.device.XPUPlace(0)
    """
    return core.XPUPlace(dev_id)


def get_cudnn_version():
    """
    This function return the version of cudnn. the retuen value is int which represents the
    cudnn version. For example, if it return 7600, it represents the version of cudnn is 7.6.

    Returns:
        int: A int value which represents the cudnn version. If cudnn version is not installed, it return None.

    Examples:
        .. code-block:: python

            import paddle

            cudnn_version = paddle.device.get_cudnn_version()



    """
    global _cudnn_version
    if not core.is_compiled_with_cuda():
        return None
    if _cudnn_version is None:
        cudnn_version = int(core.cudnn_version())
        _cudnn_version = cudnn_version
        if _cudnn_version < 0:
            return None
        else:
            return cudnn_version
    else:
        return _cudnn_version


def _convert_to_place(device):
    lower_device = device.lower()
    if device in core.get_all_custom_device_type():
        selected_devices = os.getenv(f"FLAGS_selected_{device}s", "0").split(
            ","
        )
        device_id = int(selected_devices[0])
        place = core.CustomPlace(device, device_id)
    elif lower_device == 'cpu':
        place = core.CPUPlace()
    elif lower_device == 'gpu':
        if not core.is_compiled_with_cuda():
            raise ValueError(
                "The device should not be 'gpu', "
                "since PaddlePaddle is not compiled with CUDA"
            )
        place = core.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
    elif lower_device == 'xpu':
        if not core.is_compiled_with_xpu():
            raise ValueError(
                "The device should not be 'xpu', "
                "since PaddlePaddle is not compiled with XPU"
            )
        selected_xpus = os.getenv("FLAGS_selected_xpus", "0").split(",")
        device_id = int(selected_xpus[0])
        place = core.XPUPlace(device_id)
    elif lower_device == 'ipu':
        if not core.is_compiled_with_ipu():
            raise ValueError(
                "The device should not be 'ipu', "
                "since PaddlePaddle is not compiled with IPU"
            )
        place = core.IPUPlace()
    else:
        avaliable_gpu_device = re.match(r'gpu:\d+', lower_device)
        avaliable_xpu_device = re.match(r'xpu:\d+', lower_device)
        if avaliable_gpu_device:
            if not core.is_compiled_with_cuda():
                raise ValueError(
                    "The device should not be {}, since PaddlePaddle is "
                    "not compiled with CUDA".format(avaliable_gpu_device)
                )
            device_info_list = device.split(':', 1)
            device_id = device_info_list[1]
            device_id = int(device_id)
            place = core.CUDAPlace(device_id)
        if avaliable_xpu_device:
            if not core.is_compiled_with_xpu():
                raise ValueError(
                    "The device should not be {}, since PaddlePaddle is "
                    "not compiled with XPU".format(avaliable_xpu_device)
                )
            device_info_list = device.split(':', 1)
            device_id = device_info_list[1]
            device_id = int(device_id)
            place = core.XPUPlace(device_id)
        if not avaliable_gpu_device and not avaliable_xpu_device:
            device_info_list = device.split(':', 1)
            device_type = device_info_list[0]
            if device_type in core.get_all_custom_device_type():
                device_id = device_info_list[1]
                device_id = int(device_id)
                place = core.CustomPlace(device_type, device_id)
            else:
                raise ValueError(
                    "The device must be a string which is like 'cpu', {}".format(
                        ', '.join(
                            f"'{x}', '{x}:x'"
                            for x in ['gpu', 'xpu', 'npu']
                            + core.get_all_custom_device_type()
                        )
                    )
                )
    return place


def set_device(device):
    """
    Paddle supports running calculations on various types of devices, including CPU, GPU, XPU, NPU and IPU.
    They are represented by string identifiers. This function can specify the global device
    which the OP will run.

    Parameters:
        device(str): This parameter determines the specific running device.
            It can be ``cpu``, ``gpu``, ``xpu``, ``npu``, ``gpu:x``, ``xpu:x``, ``npu:x`` and ``ipu``,
            where ``x`` is the index of the GPUs, XPUs or NPUs.

    Examples:

     .. code-block:: python

        import paddle

        paddle.device.set_device("cpu")
        x1 = paddle.ones(name='x1', shape=[1, 2], dtype='int32')
        x2 = paddle.zeros(name='x2', shape=[1, 2], dtype='int32')
        data = paddle.stack([x1,x2], axis=1)
    """
    place = _convert_to_place(device)
    framework._set_expected_place(place)
    return place


def get_device():
    """
    This function can get the current global device of the program is running.
    It's a string which is like 'cpu', 'gpu:x', 'xpu:x' and 'npu:x'. if the global device is not
    set, it will return a string which is 'gpu:x' when cuda is avaliable or it
    will return a string which is 'cpu' when cuda is not avaliable.

    Examples:

     .. code-block:: python

        import paddle
        device = paddle.device.get_device()

    """
    device = ''
    place = framework._current_expected_place()
    if isinstance(place, core.CPUPlace):
        device = 'cpu'
    elif isinstance(place, core.CUDAPlace):
        device_id = place.get_device_id()
        device = 'gpu:' + str(device_id)
    elif isinstance(place, core.XPUPlace):
        device_id = place.get_device_id()
        device = 'xpu:' + str(device_id)
    elif isinstance(place, core.IPUPlace):
        num_devices = core.get_ipu_device_count()
        device = f"ipus:{{0-{num_devices - 1}}}"
        device = f"ipus:{{0-{num_devices - 1}}}"
    elif isinstance(place, core.CustomPlace):
        device_id = place.get_device_id()
        device_type = place.get_device_type()
        device = device_type + ':' + str(device_id)
    else:
        raise ValueError(f"The device specification {place} is invalid")

    return device


def get_all_device_type():
    """
    Get all available device types.

    Returns:
        A list of all available device types.

    Examples:
        .. code-block:: python

            import paddle
            paddle.device.get_all_device_type()

            # Case 1: paddlepaddle-cpu package installed, and no custom device registerd.
            # Output: ['cpu']

            # Case 2: paddlepaddle-gpu package installed, and no custom device registerd.
            # Output: ['cpu', 'gpu']

            # Case 3: paddlepaddle-cpu package installed, and custom deivce 'CustomCPU' is registerd.
            # Output: ['cpu', 'CustomCPU']

            # Case 4: paddlepaddle-gpu package installed, and custom deivce 'CustomCPU' and 'CustomGPU' is registerd.
            # Output: ['cpu', 'gpu', 'CustomCPU', 'CustomGPU']
    """
    return core.get_all_device_type()


def get_all_custom_device_type():
    """
    Get all available custom device types.

    Returns:
        A list of all available custom device types.

    Examples:
        .. code-block:: python

            import paddle
            paddle.device.get_all_custom_device_type()

            # Case 1: paddlepaddle-gpu package installed, and no custom device registerd.
            # Output: None

            # Case 2: paddlepaddle-gpu package installed, and custom deivce 'CustomCPU' and 'CustomGPU' is registerd.
            # Output: ['CustomCPU', 'CustomGPU']
    """
    return core.get_all_custom_device_type()


def get_available_device():
    """
    Get all available devices.

    Returns:
        A list of all available devices.

    Examples:
        .. code-block:: python

            import paddle
            paddle.device.get_available_device()

            # Case 1: paddlepaddle-cpu package installed, and no custom device registerd.
            # Output: ['cpu']

            # Case 2: paddlepaddle-gpu package installed, and no custom device registerd.
            # Output: ['cpu', 'gpu:0', 'gpu:1']

            # Case 3: paddlepaddle-cpu package installed, and custom deivce 'CustomCPU' is registerd.
            # Output: ['cpu', 'CustomCPU']

            # Case 4: paddlepaddle-gpu package installed, and custom deivce 'CustomCPU' and 'CustomGPU' is registerd.
            # Output: ['cpu', 'gpu:0', 'gpu:1', 'CustomCPU', 'CustomGPU:0', 'CustomGPU:1']
    """
    return core.get_available_device()


def get_available_custom_device():
    """
    Get all available custom devices.

    Returns:
       A list of all available custom devices.

    Examples:
        .. code-block:: python

            import paddle
            paddle.device.get_available_custom_device()

            # Case 1: paddlepaddle-gpu package installed, and no custom device registerd.
            # Output: None

            # Case 2: paddlepaddle-gpu package installed, and custom deivce 'CustomCPU' and 'CustomGPU' is registerd.
            # Output: ['CustomCPU', 'CustomGPU:0', 'CustomGPU:1']
    """
    return core.get_available_custom_device()


class Event:
    '''
    A device event wrapper around StreamBase.
    Parameters:
        device(str|paddle.CUDAPlace(n)|paddle.CustomPlace(n)): Which device the stream runn on. If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``gpu:x``,``custom_device``, ``custom_device:x``, where ``custom_device`` is the name of CustomDevicec,
            where ``x`` is the index of the GPUs, XPUs. And it can be paddle.CUDAPlace(n) or paddle.CustomPlace(n).
        enable_timing (bool, optional): indicates if the event should measure time, default is False
        blocking (bool, optional): if True, ``wait`` will be blocking, default is False
        interprocess (bool): if True, the event can be shared between processes, default is False
    Returns:
        Event: The event.
    Examples:
        .. code-block:: python
            # required: custom_device
            import paddle

            paddle.set_device('custom_cpu')
            e1 = paddle.device.Event()
            e2 = paddle.device.Event('custom_cpu')
            e3 = paddle.device.Event('custom_cpu:0')
            e4 = paddle.device.Event(paddle.CustomPlace('custom_cpu', 0))
    '''

    def __init__(
        self,
        device=None,
        enable_timing=False,
        blocking=False,
        interprocess=False,
    ):
        if device is None:
            self.device = paddle.framework._current_expected_place()
        elif isinstance(device, str):
            self.device = paddle.device._convert_to_place(device)
        else:
            self.device = device

        if paddle.is_compiled_with_cuda() and isinstance(
            self.device, paddle.CUDAPlace
        ):
            self.event_base = core.CUDAEvent(
                enable_timing, blocking, interprocess
            )
        elif isinstance(self.device, paddle.CustomPlace):
            self.event_base = core.CustomDeviceEvent(
                self.device.get_device_type(),
                self.device.get_device_id(),
                enable_timing,
                blocking,
                interprocess,
            )
        else:
            raise TypeError(
                "device should be gpu, xpu, {}".format(
                    ",".join(paddle.device.get_all_custom_device_type())
                )
            )

    def record(self, stream=None):
        '''
        Records the event in a given stream.
        Parameters:
            stream(Stream, optional): The given stream. By default, stream is None,
            event will be recorded in current_stream.
        Returns:
            None.
        Examples:
            .. code-block:: python
                # required: custom_device
                import paddle

                paddle.set_device('custom_cpu')
                e = paddle.device.Event()
                e.record()

                s = paddle.device.Stream()
                e.record(s)
        '''
        if stream is None:
            stream = current_stream(self.device)

        self.event_base.record(stream.stream_base)

    def query(self):
        '''
        Checks if all work currently captured by event has completed.
        Returns:
            bool: Whether all work currently captured by event has completed.
        Examples:
            .. code-block:: python
                # required: custom_device
                import paddle

                paddle.set_device('custom_cpu')
                e = paddle.device.Event()
                e.record()
                e.query()
        '''
        return self.event_base.query()

    def elapsed_time(self, end_event):
        '''
        Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        Returns:
            int: The time.
        Examples:
            .. code-block:: python
                # required: custom_device
                import paddle

                paddle.set_device('custom_cpu')
                e1 = paddle.device.Event()
                e1.record()

                e2 = paddle.device.Event()
                e2.record()
                e1.elapsed_time(e2)
        '''
        return 0

    def synchronize(self):
        '''
        Waits for the event to complete.
        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        Returns:
            None.
        Examples:
            .. code-block:: python
                # required: custom_device
                import paddle

                paddle.set_device('custom_cpu')
                e = paddle.device.Event()
                e.record()
                e.synchronize()
        '''
        self.event_base.synchronize()

    def __repr__(self):
        return self.event_base


class Stream:
    '''
    A device stream wrapper around StreamBase.
    Parameters:
        device(str|paddle.CUDAPlace(n)|paddle.CustomPlace(n)): Which device the stream runn on. If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``gpu:x``,``custom_device``, ``custom_device:x``, where ``custom_device`` is the name of CustomDevicec,
            where ``x`` is the index of the GPUs, XPUs. And it can be paddle.CUDAPlace(n) or paddle.CustomPlace(n).
        priority(int, optional): priority of the CUDA stream. Can be either
            1 (high priority) or 2 (low priority). By default, streams have
            priority 2.
    Returns:
        Stream: The stream.
    Examples:
        .. code-block:: python
            # required: custom_device
            import paddle

            paddle.set_device('custom_cpu')
            s1 = paddle.device.Stream()
            s2 = paddle.device.Stream('custom_cpu')
            s3 = paddle.device.Stream('custom_cpu:0')
            s4 = paddle.device.Stream(paddle.CustomPlace('custom_cpu', 0))
    '''

    def __init__(self, device=None, priority=2, stream_base=None):
        if stream_base is not None:
            if isinstance(
                stream_base, (core.CUDAStream, core.CustomDeviceStream)
            ):
                self.stream_base = stream_base
                self.device = stream_base.place
            else:
                raise TypeError(
                    "stream_base should be CUDAStream, CustomDeviceStream"
                )
            return

        if device is None:
            self.device = paddle.framework._current_expected_place()
        elif isinstance(device, str):
            self.device = paddle.device._convert_to_place(device)
        else:
            self.device = device

        if paddle.is_compiled_with_cuda() and isinstance(
            self.device, paddle.CUDAPlace
        ):
            self.stream_base = core.CUDAStream(
                self.device.get_device_id(), priority
            )
        elif isinstance(self.device, paddle.CustomPlace):
            self.stream_base = core.CustomDeviceStream(
                self.device.get_device_type(),
                self.device.get_device_id(),
                priority,
                blocking=False,
            )
        else:
            raise TypeError(
                "device should be gpu, xpu, {}".format(
                    ",".join(paddle.device.get_all_custom_device_type())
                )
            )

    def wait_event(self, event):
        '''
        Makes all future work submitted to the stream wait for an event.
        Parameters:
            event (Event): an event to wait for.
        Returns:
            None.
        Examples:
            .. code-block:: python
                # required: custom_device
                import paddle

                paddle.set_device('custom_cpu')
                s1 = paddle.device.Stream()
                s2 = paddle.device.Stream()
                e = paddle.device.Event()
                e.record(s1)
                s2.wait_event(e)
        '''
        self.stream_base.wait_event(event.event_base)

    def wait_stream(self, stream):
        '''
        Synchronizes with another stream.
        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.
        Parameters:
            stream (Stream): a stream to synchronize.
        Returns:
            None.
        Examples:
            .. code-block:: python
                # required: custom_device
                import paddle

                paddle.set_device('custom_cpu')
                s1 = paddle.device.Stream()
                s2 = paddle.device.Stream()
                s1.wait_stream(s2)
        '''
        self.stream_base.wait_stream(stream.stream_base)

    def record_event(self, event=None):
        '''
        Records an event.
        Parameters:
            event (Event, optional): event to record. If not given, a new one
                will be allocated.
        Returns:
            Event: Recorded event.
        Examples:
            .. code-block:: python
                # required: custom_device
                import paddle

                paddle.set_device('custom_cpu')
                s = paddle.device.Stream()
                e1 = s.record_event()

                e2 = paddle.device.Event()
                s.record_event(e2)
        '''
        if event is None:
            event = Event(self.device)
        event.record(self)
        return event

    def query(self):
        '''
        Checks if all the work submitted has been completed.
        Returns:
            bool: Whether all kernels in this stream are completed.
        Examples:
            .. code-block:: python
                # required: custom_device
                import paddle

                paddle.set_device('custom_cpu')
                s = paddle.device.Stream()
                s.query()
        '''
        return self.stream_base.query()

    def synchronize(self):
        '''
        Wait for all the kernels in this stream to complete.
        Returns:
            None.
        Examples:
            .. code-block:: python
                # required: custom_device
                import paddle

                paddle.set_device('custom_cpu')
                s = paddle.device.Stream()
                s.synchronize()
        '''
        self.stream_base.synchronize()

    @property
    def _as_parameter_(self):
        if isinstance(self.stream_base, core.CUDAStream):
            return ctypes.c_void_p(self.stream_base.cuda_stream)
        else:
            return ctypes.c_void_p(self.stream_base.raw_stream)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self):
        return hash((self.stream_base, self.device))

    def __repr__(self):
        return '<paddle.device.Stream device={} stream={:#x}>'.format(
            self.device, self._as_parameter_.value
        )


def current_stream(device=None):
    '''
    Return the current stream by the device.
    Parameters:
        device(str|paddle.CUDAPlace(n)|paddle.CustomPlace(n)): The device which want to get stream from.  If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``gpu:x``,``custom_device``, ``custom_device:x``, where ``custom_device`` is the name of CustomDevicec,
            where ``x`` is the index of the GPUs, CustomDevicecs. And it can be paddle.CUDAPlace(n) or paddle.CustomPlace(n).
    Returns:
        Stream: The stream to the device.
    Examples:
        .. code-block:: python
            # required: custom_device
            import paddle

            paddle.set_device('custom_cpu')
            s1 = paddle.device.current_stream()
            s2 = paddle.device.current_stream("custom_cpu:0")
            place = paddle.CustomPlace('custom_cpu', 0)
            s3 = paddle.device.current_stream(place)
    '''
    if device is None:
        place = paddle.framework._current_expected_place()
    elif isinstance(device, str):
        place = paddle.device._convert_to_place(device)
    else:
        place = device

    if paddle.is_compiled_with_cuda() and isinstance(place, paddle.CUDAPlace):
        return Stream(
            stream_base=core._get_current_stream(place.get_device_id())
        )
    elif isinstance(place, paddle.CustomPlace):
        return Stream(
            stream_base=core._get_current_custom_device_stream(
                place.get_device_type(), place.get_device_id()
            )
        )
    else:
        raise TypeError(
            "device should be gpu, xpu, {}".format(
                ",".join(paddle.device.get_all_custom_device_type())
            )
        )


def set_stream(stream):
    '''
    Set the current stream.
    Parameters:
        stream(Stream): The selected stream.
    Returns:
        Stream: The previous stream.
    Examples:
        .. code-block:: python
            # required: custom_device
            import paddle

            paddle.set_device('custom_cpu')
            s = paddle.device.Stream()
            paddle.device.set_stream(s)
    '''

    prev_stream = current_stream(stream.stream_base.place)

    if paddle.is_compiled_with_cuda() and isinstance(
        stream.stream_base.place, paddle.CUDAPlace
    ):
        core._set_current_stream(stream.stream_base)
    elif isinstance(stream.stream_base.place, paddle.CustomPlace):
        core._set_current_custom_device_stream(
            stream.stream_base.place.get_device_type(),
            stream.stream_base.place.get_device_id(),
            stream.stream_base,
        )
    else:
        raise TypeError(
            "device should be gpu, xpu, {}".format(
                ",".join(paddle.device.get_all_custom_device_type())
            )
        )

    return prev_stream


class stream_guard:
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
            # required: custom_device
            import paddle

            paddle.set_device('custom_cpu')
            s = paddle.device.Stream()
            data1 = paddle.ones(shape=[20])
            data2 = paddle.ones(shape=[20])
            data3 = data1 + data2
            with paddle.device.stream_guard(s):
                s.wait_stream(paddle.device.default_stream())
                data4 = data1 + data3
    '''

    def __init__(self, stream=None):
        self.stream = stream

    def __enter__(self):
        cur_stream = self.stream
        if cur_stream is None:
            return

        self.src_prev_stream = current_stream(cur_stream.device)
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


def synchronize(device=None):
    '''
    Wait for the compute on the given device to finish.
    Parameters:
        device(str|paddle.CUDAPlace(n)|paddle.XPUPlace(n)|paddle.CustomPlace(n)): The device which want to wait for.  If device is None, the device is the current device. Default: None.
            It can be ``gpu``, ``gpu:x``, ``xpu``, ``xpu:x``, ``custom_device``, ``custom_device:x``, where ``custom_device`` is the name of CustomDevicec,
            where ``x`` is the index of the GPUs, XPUs. And it can be paddle.CUDAPlace(n) or paddle.XPUPlace(n) or paddle.CustomPlace(n).
    Examples:
        .. code-block:: python
            # required: custom_device
            import paddle

            paddle.set_device('custom_cpu')
            paddle.device.synchronize()
            paddle.device.synchronize("custom_cpu:0")
            place = paddle.CustomPlace('custom_cpu', 0)
            paddle.device.synchronize(place)
    '''

    if device is None:
        place = paddle.framework._current_expected_place()
    elif isinstance(device, str):
        place = paddle.device._convert_to_place(device)
    else:
        place = device

    if paddle.is_compiled_with_cuda() and isinstance(place, paddle.CUDAPlace):
        core._device_synchronize(place.get_device_id())
    elif paddle.is_compiled_with_xpu() and isinstance(place, paddle.XPUPlace):
        core._xpu_device_synchronize(place.get_device_id())
    elif isinstance(place, paddle.CustomPlace):
        core._synchronize_custom_device(
            place.get_device_type(), place.get_device_id()
        )
    else:
        raise TypeError(
            "device should be gpu, xpu, {}".format(
                ",".join(paddle.device.get_all_custom_device_type())
            )
        )
