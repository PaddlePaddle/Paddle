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

import paddle
from paddle.fluid import core
from paddle.fluid.core import CustomDeviceEvent as Event  # noqa: F401
from paddle.fluid.core import CustomDeviceStream as Stream  # noqa: F401
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager

__all__ = [
    'current_device',
    'synchronize_device',
    'device_count',
    'set_current_device',
    'stream_guard',
    'current_stream',
    'Event',
    'Stream',
]


def current_device(device_type):
    '''
    Return the device used for computation.

    Parameters:
        device(paddle.CustomPlace()|str): The device or the type of the device.
        device_id(int, Optional): The id of the device. Default: 0.

    Returns:
        int: the id of custom device used for computation.

    Examples:
        .. code-block:: python

            # required: custom_device
            import paddle

            paddle.device.custom.current_device('custom_cpu')
    '''
    return core._get_current_custom_device(device_type)


def synchronize_device(device, device_id=None):
    '''
    Wait for the compute on the given custom device to finish.

    Parameters:
        device(paddle.CustomPlace()|str): The device or the type of the device.
        device_id(int, Optional): The id of the device, If device_id is None, the device_id is the id of the current device. Default: None.

    Examples:
        .. code-block:: python

            # required: custom_device
            import paddle

            paddle.device.cuda.synchronize('custom_cpu')

            paddle.device.cuda.synchronize('custom_cpu', 0)

            paddle.device.cuda.synchronize(paddle.CustomPlace('custom_cpu', 0))

    '''
    if isinstance(device, str):
        if device_id is None:
            device_id = current_device()
        core._synchronize_custom_device(paddle.CustomPlace(device, device_id))
    elif isinstance(device, paddle.CustomPlace):
        core._synchronize_custom_device(device)
    else:
        raise TypeError("device should be paddle.CustomPlace or str")


def device_count(device_type):
    '''
    Return the number of custom devices available.

    Parameters:
        device_type(str): The type of the device.

    Returns:
        int: the number of custom devices available.

    Examples:
        .. code-block:: python

            # required: custom_device
            import paddle

            paddle.device.custom.device_count('custom_cpu')

    '''
    return core._custom_device_count(device_type)


def set_current_device(device, device_id=0):
    '''
    Set the device used for computation.

    Parameters:
        device(paddle.CustomPlace()|str): The device or the type of the device.
        device_id(int, Optional): The id of the device. Default: 0.

    Examples:
        .. code-block:: python

            # required: custom_device
            import paddle

            paddle.device.custom.set_current_device('custom_cpu')

            paddle.device.custom.set_current_device('custom_cpu', 0)

            paddle.device.custom.set_current_device(paddle.CustomPlace('custom_cpu', 0))

    '''
    if isinstance(device, str):
        core._set_current_custom_device(paddle.CustomPlace(device, device_id))
    elif isinstance(device, paddle.CustomPlace):
        core._set_current_custom_device(device)
    else:
        raise TypeError("device should be paddle.CustomPlace or str")


def current_stream(device, device_id=None):
    '''
    Return the current custom stream by the device.

    Parameters:
        device(paddle.CustomPlace()|str): The device or the type of the device which want to get stream from.
        device_id(int, Optional): The id of the device which want to get stream from, If device_id is None, the device_id is the id of the current device. Default: None.

    Returns:
        CustomDeviceStream: the stream to the device.

    Examples:
        .. code-block:: python

            # required: custom_device
            import paddle

            s1 = paddle.device.custom.current_stream('custom_cpu')

            s2 = paddle.device.custom.current_stream('custom_cpu', 0)

            s3 = paddle.device.custom.current_stream(paddle.CustomPlace('custom_cpu', 0))

    '''

    if isinstance(device, str):
        if device_id is None:
            device_id = current_device()
        return core._get_current_custom_device_stream(
            paddle.CustomPlace(device, device_id)
        )
    elif isinstance(device, paddle.CustomPlace):
        return core._get_current_custom_device_stream(device)
    else:
        raise TypeError("device should be paddle.CustomPlace or str")


@signature_safe_contextmanager
def stream_guard(stream):
    '''
    **Notes**:
        **This API only supports dygraph mode currently.**

    A context manager that specifies the current stream context by the given stream.

    Parameters:
        stream(paddle.device.custom.Stream): the selected stream. If stream is None, just yield.

    Examples:
        .. code-block:: python

            # required: custom_device
            import paddle

            s = paddle.device.custom.Stream('custom_cpu')

            data1 = paddle.ones(shape=[20])

            data2 = paddle.ones(shape=[20])

            with paddle.device.custom.stream_guard(s):
                data3 = data1 + data2

    '''

    if stream is not None and not isinstance(
        stream, paddle.device.custom.Stream
    ):
        raise TypeError("stream type should be paddle.device.custom.Stream")

    cur_stream = core._get_current_custom_device_stream(stream.place)
    if stream is None or cur_stream.raw_stream == stream.raw_stream:
        yield
    else:
        core._set_current_custom_device_stream(stream.place, stream)
        pre_stream = cur_stream
        try:
            yield
        finally:
            stream = core._set_current_custom_device_stream(
                stream.place, pre_stream
            )
