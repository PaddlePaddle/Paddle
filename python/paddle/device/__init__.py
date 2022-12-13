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
import paddle
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.framework import is_compiled_with_cinn  # noqa: F401
from paddle.fluid.framework import is_compiled_with_cuda  # noqa: F401
from paddle.fluid.framework import is_compiled_with_rocm  # noqa: F401
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager
from . import cuda
from . import xpu

__all__ = [  # noqa
    'get_cudnn_version',
    'set_device',
    'get_device',
    'XPUPlace',
    'IPUPlace',
    'MLUPlace',
    'is_compiled_with_xpu',
    'is_compiled_with_ipu',
    'is_compiled_with_cinn',
    'is_compiled_with_cuda',
    'is_compiled_with_rocm',
    'is_compiled_with_npu',
    'is_compiled_with_mlu',
    'is_compiled_with_custom_device',
    'get_all_device_type',
    'get_all_custom_device_type',
    'get_available_device',
    'get_available_custom_device',
    'Stream',
    'Event',
    'current_stream',
    'stream_guard',
    'synchronize',
]

_cudnn_version = None


# TODO: WITH_ASCEND_CL may changed to WITH_NPU or others in the future
# for consistent.
def is_compiled_with_npu():
    """
    Whether paddle was built with WITH_ASCEND_CL=ON to support Ascend NPU.

    Return:
        bool, ``True`` if NPU is supported, otherwise ``False``.

    Examples:
        .. code-block:: python

            import paddle
            support_npu = paddle.device.is_compiled_with_npu()
    """
    return core.is_compiled_with_npu()


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


def is_compiled_with_mlu():
    """
    Whether paddle was built with WITH_MLU=ON to support Cambricon MLU

    Returns (bool): whether paddle was built with WITH_MLU=ON

    Examples:
        .. code-block:: python

            # required: mlu

            import paddle
            support_mlu = paddle.device.is_compiled_with_mlu()
    """
    return core.is_compiled_with_mlu()


def MLUPlace(dev_id):
    """
    Return a Cambricon MLU Place

    Parameters:
        dev_id(int): MLU device id

    Examples:
        .. code-block:: python

            # required: mlu

            import paddle
            place = paddle.device.MLUPlace(0)
    """
    return core.MLUPlace(dev_id)


def get_cudnn_version():
    """
    This funciton return the version of cudnn. the retuen value is int which represents the
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
        selected_devices = os.getenv(
            "FLAGS_selected_{}s".format(device), "0"
        ).split(",")
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
        place = core.CUDAPlace(ParallelEnv().dev_id)
    elif lower_device == 'xpu':
        if not core.is_compiled_with_xpu():
            raise ValueError(
                "The device should not be 'xpu', "
                "since PaddlePaddle is not compiled with XPU"
            )
        selected_xpus = os.getenv("FLAGS_selected_xpus", "0").split(",")
        device_id = int(selected_xpus[0])
        place = core.XPUPlace(device_id)
    elif lower_device == 'npu':
        if not core.is_compiled_with_npu():
            raise ValueError(
                "The device should not be 'npu', "
                "since PaddlePaddle is not compiled with NPU"
            )
        selected_npus = os.getenv("FLAGS_selected_npus", "0").split(",")
        device_id = int(selected_npus[0])
        place = core.NPUPlace(device_id)
    elif lower_device == 'ipu':
        if not core.is_compiled_with_ipu():
            raise ValueError(
                "The device should not be 'ipu', "
                "since PaddlePaddle is not compiled with IPU"
            )
        place = core.IPUPlace()
    elif lower_device == 'mlu':
        if not core.is_compiled_with_mlu():
            raise ValueError(
                "The device should not be 'mlu', "
                "since PaddlePaddle is not compiled with MLU"
            )
        selected_mlus = os.getenv("FLAGS_selected_mlus", "0").split(",")
        device_id = int(selected_mlus[0])
        place = core.MLUPlace(device_id)
    else:
        avaliable_gpu_device = re.match(r'gpu:\d+', lower_device)
        avaliable_xpu_device = re.match(r'xpu:\d+', lower_device)
        avaliable_npu_device = re.match(r'npu:\d+', lower_device)
        avaliable_mlu_device = re.match(r'mlu:\d+', lower_device)
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
        if avaliable_npu_device:
            if not core.is_compiled_with_npu():
                device_info_list = device.split(':', 1)
                device_type = device_info_list[0]
                if device_type in core.get_all_custom_device_type():
                    device_id = device_info_list[1]
                    device_id = int(device_id)
                    place = core.CustomPlace(device_type, device_id)
                    return place
                else:
                    raise ValueError(
                        "The device should not be {}, since PaddlePaddle is "
                        "not compiled with NPU or compiled with custom device".format(
                            avaliable_npu_device
                        )
                    )
            device_info_list = device.split(':', 1)
            device_id = device_info_list[1]
            device_id = int(device_id)
            place = core.NPUPlace(device_id)
        if avaliable_mlu_device:
            if not core.is_compiled_with_mlu():
                raise ValueError(
                    "The device should not be {}, since PaddlePaddle is "
                    "not compiled with mlu".format(avaliable_mlu_device)
                )
            device_info_list = device.split(':', 1)
            device_id = device_info_list[1]
            device_id = int(device_id)
            place = core.MLUPlace(device_id)
        if (
            not avaliable_gpu_device
            and not avaliable_xpu_device
            and not avaliable_npu_device
            and not avaliable_mlu_device
        ):
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
                            "'{}', '{}:x'".format(x, x)
                            for x in ['gpu', 'xpu', 'npu', 'mlu']
                            + core.get_all_custom_device_type()
                        )
                    )
                )
    return place


def set_device(device):
    """
    Paddle supports running calculations on various types of devices, including CPU, GPU, XPU, NPU, MLU and IPU.
    They are represented by string identifiers. This function can specify the global device
    which the OP will run.

    Parameters:
        device(str): This parameter determines the specific running device.
            It can be ``cpu``, ``gpu``, ``xpu``, ``npu``, ``mlu``, ``gpu:x``, ``xpu:x``, ``npu:x``, ``mlu:x`` and ``ipu``,
            where ``x`` is the index of the GPUs, XPUs, NPUs or MLUs.

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
    This funciton can get the current global device of the program is running.
    It's a string which is like 'cpu', 'gpu:x', 'xpu:x', 'mlu:x' and 'npu:x'. if the global device is not
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
    elif isinstance(place, core.NPUPlace):
        device_id = place.get_device_id()
        device = 'npu:' + str(device_id)
    elif isinstance(place, core.IPUPlace):
        num_devices = core.get_ipu_device_count()
        device = "ipus:{{0-{}}}".format(num_devices - 1)
    elif isinstance(place, core.MLUPlace):
        device_id = place.get_device_id()
        device = 'mlu:' + str(device_id)
    elif isinstance(place, core.CustomPlace):
        device_id = place.get_device_id()
        device_type = place.get_device_type()
        device = device_type + ':' + str(device_id)
    else:
        raise ValueError("The device specification {} is invalid".format(place))

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


def _convert_place_to_device_type(place):
    if isinstance(place, str):
        return place
    elif isinstance(place, paddle.CUDAPlace):
        return 'gpu'
    elif isinstance(place, paddle.CustomPlace):
        return place.get_device_type()
    raise TypeError(
        "place should be paddle.CUDAPlace or paddle.CustomPlace or str"
    )


def _set_current_stream(stream):
    device = _convert_place_to_device_type(stream.stream.place)

    if device == "gpu":
        core._set_current_stream(stream.stream)
    elif device in paddle.device.get_all_custom_device_type():
        core._set_current_custom_device_stream(
            device, stream.stream.place.get_device_id(), stream.stream
        )
    else:
        raise TypeError(
            "device should be gpu, xpu, {}".format(
                ",".join(paddle.device.get_all_custom_device_type())
            )
        )


class Stream(object):
    def __init__(
        self,
        device=None,
        device_id=None,
        priority=2,
        blocking=False,
        stream=None,
    ):
        if stream is not None:
            if isinstance(stream, (core.CUDAStream, core.CustomDeviceStream)):
                self.stream = stream
            else:
                raise TypeError(
                    "device should be CUDAStream or CustomDeviceStream"
                )
        else:
            if device is None:
                device = framework._current_expected_place()

            if isinstance(device, (paddle.CUDAPlace, paddle.CustomPlace, str)):
                device = _convert_place_to_device_type(device)
                if not isinstance(device, str):
                    device_id = device.get_device_id()
            else:
                raise TypeError(
                    "device should be paddle.CUDAPlace or paddle.CustomPlace or str"
                )

            if device_id is None:
                device_id = -1

            if device == "gpu":
                self.stream = core.CUDAStream(device_id, priority)
            elif device in paddle.device.get_all_custom_device_type():
                self.stream = core.CustomDeviceStream(
                    device, device_id, priority, blocking
                )
            else:
                raise TypeError(
                    "device should be gpu, {}".format(
                        ",".join(paddle.device.get_all_custom_device_type())
                    )
                )

    def wait_event(self, event):
        """
        Synchronize with the given event.

        Examples:
            .. code-block:: python
                # required: custom_device

                import paddle

                paddle.set_device('custom_cpu')

                s = paddle.device.Stream()

                e = paddle.device.Event()

                s.wait_event(e)
        """
        self.stream.wait_event(event.event)

    def record_event(self, event):
        """
        Record an event on the stream.

        Examples:
            .. code-block:: python
                # required: custom_device

                import paddle

                paddle.set_device('custom_cpu')

                s = paddle.device.Stream()

                e = paddle.device.Event()

                s.record_event(e)
        """
        self.stream.record_event(event.event)

    def wait_stream(self, stream):
        """
        Synchronize with the given stream.

        Examples:
            .. code-block:: python
                # required: custom_device

                import paddle

                paddle.set_device('custom_cpu')

                s1 = paddle.device.Stream()

                s2 = paddle.device.Stream()

                s1.wait_stream(s2)
        """
        self.stream.wait_stream(stream.stream)

    def synchronize(self):
        """
        Wait for the stream to finish.

        Examples:
            .. code-block:: python
                # required: custom_device

                import paddle

                paddle.set_device('custom_cpu')

                s = paddle.device.Stream()

                s.synchronize()
        """
        self.stream.synchronize()

    def query(self):
        """
        Query the status of stream.

        Returns:
            bool: the status of stream.

        Examples:
            .. code-block:: python
                # required: custom_device

                import paddle

                paddle.set_device('custom_cpu')

                s = paddle.device.Stream()

                s.query()
        """
        return self.stream.query()

    @property
    def raw_stream(self):
        """
        Get the raw stream of paddle.device.Stream object.

        Returns:
            int: the raw stream.

        Examples:
            .. code-block:: python
                # required: custom_device

                import paddle

                paddle.set_device('custom_cpu')

                s = paddle.device.Stream()

                print(s.raw_stream)
        """
        if isinstance(self.stream, core.CUDAStream):
            return self.stream.cuda_stream
        else:
            return self.stream.raw_stream


class Event(object):
    def __init__(
        self,
        device=None,
        device_id=None,
        enable_timing=False,
        blocking=False,
        interprocess=False,
    ):
        if device is None:
            device = framework._current_expected_place()

        if isinstance(device, (paddle.CUDAPlace, paddle.CustomPlace, str)):
            device = _convert_place_to_device_type(device)
            if not isinstance(device, str):
                device_id = device.get_device_id()
        else:
            raise TypeError(
                "device should be paddle.CUDAPlace or paddle.CustomPlace or str"
            )

        if device_id is None:
            device_id = -1

        if device == "gpu":
            self.event = core.CUDAEvent(enable_timing, blocking, interprocess)
        elif device in paddle.device.get_all_custom_device_type():
            self.event = core.CustomDeviceEvent(
                device, device_id, enable_timing, blocking, interprocess
            )
        else:
            raise TypeError(
                "device should be gpu, {}".format(
                    ",".join(paddle.device.get_all_custom_device_type())
                )
            )

    def record(self, stream):
        """
        Record the event on the given stream.

        Examples:
            .. code-block:: python
                # required: custom_device

                import paddle

                paddle.set_device('custom_cpu')

                s = paddle.device.Stream()

                e = paddle.device.Event()

                e.record(s)
        """
        self.event.record(stream.stream)

    def synchronize(self):
        """
        Wait for the event to finish.

        Examples:
            .. code-block:: python
                # required: custom_device

                import paddle

                paddle.set_device('custom_cpu')

                e = paddle.device.Event()

                e.synchronize()
        """
        self.event.synchronize()

    def query(self):
        """
        Query the status of event.

        Returns:
            bool: the status of event.

        Examples:
            .. code-block:: python
                # required: custom_device

                import paddle

                paddle.set_device('custom_cpu')

                e = paddle.device.Event()

                e.query()
        """
        return self.event.query()


def current_stream(device=None, device_id=None):
    '''
    Return the current stream by the device.

    Parameters:
        device(paddle.CUDAPlace|paddle.CustomPlace|str): The device or the type of the device which want to get stream from, If device is None, the device is the current expected place. Default: None.
        device_id(int, Optional): The id of the device which want to get stream from, If device_id is None, the device_id is the id of the current device. Default: None.

    Returns:
        paddle.device.Stream: the stream to the device.

    Examples:
        .. code-block:: python
            # required: custom_device

            import paddle

            s1 = paddle.device.current_stream()

            s2 = paddle.device.current_stream('custom_cpu')

            s2 = paddle.device.current_stream('custom_cpu', 0)

            s3 = paddle.device.current_stream(paddle.CustomPlace('custom_cpu', 0))

    '''

    if device is None:
        device = framework._current_expected_place()

    if isinstance(device, (paddle.CUDAPlace, paddle.CustomPlace, str)):
        device = _convert_place_to_device_type(device)
        if not isinstance(device, str):
            device_id = device.get_device_id()
    else:
        raise TypeError(
            "device should be paddle.CUDAPlace or paddle.CustomPlace or str"
        )

    if device_id is None:
        device_id = -1

    if device == "gpu":
        return Stream(stream=core._get_current_stream(device_id))
    elif device in paddle.device.get_all_custom_device_type():
        return Stream(
            stream=core._get_current_custom_device_stream(device, device_id)
        )
    else:
        raise TypeError(
            "device should be gpu, {}".format(
                ",".join(paddle.device.get_all_custom_device_type())
            )
        )


@signature_safe_contextmanager
def stream_guard(stream):
    '''
    Note:
        This API only supports dygraph mode currently.

    A context manager that specifies the current stream context by the given stream.

    Parameters:
        stream(paddle.device.Stream): the selected stream. If stream is None, just yield.

    Examples:
        .. code-block:: python
            # required: custom_device

            import paddle

            s = paddle.device.Stream('custom_cpu')

            data1 = paddle.ones(shape=[20])

            data2 = paddle.ones(shape=[20])

            with paddle.device.stream_guard(s):
                data3 = data1 + data2

    '''

    if stream is not None and not isinstance(stream, paddle.device.Stream):
        raise TypeError("stream type should be paddle.device.Stream")

    if stream is None:
        yield

    place = stream.stream.place
    device = _convert_place_to_device_type(place)
    device_id = place.get_device_id()
    cur_stream = current_stream(device, device_id)

    if cur_stream.raw_stream == stream.raw_stream:
        yield
    else:
        _set_current_stream(stream)
        pre_stream = cur_stream
        try:
            yield
        finally:
            _set_current_stream(pre_stream)


def synchronize(device=None, device_id=None):
    '''
    Wait for the compute on the given device to finish.

    Parameters:
        device(paddle.CUDAPlace|paddle.CustomPlace|str): The device or the type of the device, If device is None, the device is the current expected place. Default: None.
        device_id(int, Optional): The id of the device, If device_id is None, the device_id is the id of the current device. Default: None.

    Examples:
        .. code-block:: python
            # required: custom_device

            import paddle

            paddle.device.synchronize()

            paddle.device.synchronize('custom_cpu')

            paddle.device.synchronize('custom_cpu', 0)

            paddle.device.synchronize(paddle.CustomPlace('custom_cpu', 0))

    '''
    if device is None:
        device = framework._current_expected_place()

    if isinstance(device, (paddle.CUDAPlace, paddle.CustomPlace, str)):
        device = _convert_place_to_device_type(device)
        if not isinstance(device, str):
            device_id = device.get_device_id()
    else:
        raise TypeError(
            "device should be paddle.CUDAPlace or paddle.CustomPlace or str"
        )

    if device_id is None:
        device_id = -1

    if device == "gpu":
        core._device_synchronize(device_id)
    elif device in paddle.device.get_all_custom_device_type():
        core._synchronize_custom_device(device, device_id)
    else:
        raise TypeError(
            "device should be gpu, {}".format(
                ",".join(paddle.device.get_all_custom_device_type())
            )
        )
