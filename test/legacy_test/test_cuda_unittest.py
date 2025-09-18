# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
# test_cuda_unittest.py
import ctypes
import types
import unittest

import numpy as np
from op_test import get_device, is_custom_device

import paddle
from paddle.cuda import (
    Stream,
    StreamContext,
    _device_to_paddle,
    check_error,
    current_stream,
    get_device_capability,
    get_device_name,
    get_device_properties,
    is_available,
    mem_get_info,
    stream,
    synchronize,
)


class TestCudaCompat(unittest.TestCase):
    # ---------------------
    # _device_to_paddle test
    # ---------------------
    def test_device_to_paddle_none(self):
        self.assertIsNone(_device_to_paddle(None))

    def test_device_to_paddle_int(self):
        self.assertEqual(_device_to_paddle(0), 'gpu:0')
        self.assertEqual(_device_to_paddle(2), 'gpu:2')

    def test_device_to_paddle_str(self):
        self.assertEqual(_device_to_paddle('cuda:0'), 'gpu:0')
        self.assertEqual(_device_to_paddle('gpu:1'), 'gpu:1')

    def test_device_to_paddle_invalid(self):
        with self.assertRaises(TypeError):
            _device_to_paddle(1.5)

    # ---------------------
    # is_available test
    # ---------------------
    def test_is_available(self):
        if paddle.is_compiled_with_cuda() or is_custom_device():
            self.assertIsInstance(is_available(), bool)

    # ---------------------
    # synchronize test
    # ---------------------
    def test_synchronize(self):
        if paddle.is_compiled_with_cuda() or is_custom_device():
            try:
                synchronize(None)
                synchronize(0)
                synchronize('cuda:0')
                synchronize('gpu:0')
            except Exception as e:
                self.fail(f"synchronize raised Exception {e}")

    # ---------------------
    # current_stream test
    # ---------------------
    def test_current_stream(self):
        if paddle.is_compiled_with_cuda() or is_custom_device():
            stream = current_stream(None)
            self.assertIsNotNone(stream)
            stream = current_stream(0)
            self.assertIsNotNone(stream)

    # ---------------------
    # get_device_properties test
    # ---------------------
    def test_get_device_properties(self):
        if paddle.is_compiled_with_cuda() or is_custom_device():
            props = get_device_properties(0)
            self.assertTrue(hasattr(props, 'name'))
            self.assertTrue(hasattr(props, 'total_memory'))

    # ---------------------
    # get_device_name / get_device_capability test
    # ---------------------
    def test_device_name_and_capability(self):
        if paddle.is_compiled_with_cuda() or is_custom_device():
            name = get_device_name(0)
            self.assertIsInstance(name, str)

            cap = get_device_capability(0)
            self.assertIsInstance(cap, tuple)
            self.assertEqual(len(cap), 2)

    def test_stream_creation(self):
        if paddle.is_compiled_with_cuda() or is_custom_device():
            s = Stream()
            s1 = Stream()
            self.assertIsInstance(s, paddle.device.Stream)
            self.assertIsInstance(s1, paddle.device.Stream)

    def test_stream_context(self):
        if paddle.is_compiled_with_cuda() or is_custom_device():
            s = Stream(device=get_device(), priority=2)
            with stream(s):
                ctx = stream(s)
                self.assertIsInstance(ctx, StreamContext)
                current = current_stream()
                self.assertEqual(current.stream_base, s.stream_base)

    def test_nested_streams(self):
        if paddle.is_compiled_with_cuda() or is_custom_device():
            s1 = Stream()
            s2 = Stream()
            with stream(s1):
                with stream(s2):
                    current = paddle.cuda.current_stream()
                    self.assertEqual(current.stream_base, s2.stream_base)
                current = paddle.cuda.current_stream()
                self.assertEqual(current.stream_base, s1.stream_base)

    @unittest.skipIf(
        (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_rocm()
        ),
        reason="Skip if not in CUDA env",
    )
    def test_cudart_integrity(self):
        cuda_rt_module = paddle.cuda.cudart()
        self.assertIsNotNone(cuda_rt_module)
        self.assertIsInstance(cuda_rt_module, types.ModuleType)

        cuda_version = paddle.version.cuda()
        if int(cuda_version.split(".")[0]) < 12:
            self.assertTrue(hasattr(cuda_rt_module, "cudaOutputMode_"))
            self.assertTrue(hasattr(cuda_rt_module, "cudaProfilerInitialize"))

            self.assertTrue(
                hasattr(cuda_rt_module.cudaOutputMode_, "KeyValuePair")
            )
            self.assertEqual(cuda_rt_module.cudaOutputMode_.KeyValuePair, 0)

            self.assertTrue(hasattr(cuda_rt_module.cudaOutputMode_, "CSV"))
            self.assertEqual(cuda_rt_module.cudaOutputMode_.CSV, 1)

        self.assertTrue(hasattr(cuda_rt_module, "cudaError_"))
        self.assertTrue(hasattr(cuda_rt_module.cudaError_, "success"))
        self.assertEqual(cuda_rt_module.cudaError_.success, 0)

        func_list = [
            "cudaGetErrorString",
            "cudaProfilerStart",
            "cudaProfilerStop",
            "cudaHostRegister",
            "cudaHostUnregister",
            "cudaStreamCreate",
            "cudaStreamDestroy",
            "cudaMemGetInfo",
        ]
        for f in func_list:
            self.assertTrue(hasattr(cuda_rt_module, f))

    @unittest.skipIf(
        (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_rocm()
        ),
        reason="Skip if not in CUDA env",
    )
    def test_cudart_function(self):
        cuda_rt_module = paddle.cuda.cudart()

        # cudaGetErrorString
        err_str = cuda_rt_module.cudaGetErrorString(
            cuda_rt_module.cudaError_.success
        )
        self.assertIsInstance(err_str, str)

        # cudaMemGetInfo
        free_mem, total_mem = cuda_rt_module.cudaMemGetInfo(0)
        self.assertIsInstance(free_mem, int)
        self.assertIsInstance(total_mem, int)
        self.assertGreaterEqual(total_mem, free_mem)
        self.assertGreater(free_mem, 0)

        # cudaHostRegister / cudaHostUnregister
        buf = np.zeros(1024, dtype=np.float32)
        ptr = buf.ctypes.data
        err = cuda_rt_module.cudaHostRegister(ptr, buf.nbytes, 0)
        self.assertEqual(err, cuda_rt_module.cudaError_.success)
        err = cuda_rt_module.cudaHostUnregister(ptr)
        self.assertEqual(err, cuda_rt_module.cudaError_.success)

        # cudaStreamCreate / cudaStreamDestroy
        stream = ctypes.c_size_t(0)
        err = cuda_rt_module.cudaStreamCreate(ctypes.addressof(stream))
        assert err == cuda_rt_module.cudaError_.success

        err = cuda_rt_module.cudaStreamDestroy(stream.value)
        assert err == cuda_rt_module.cudaError_.success

        err = cuda_rt_module.cudaProfilerStart()
        self.assertEqual(err, cuda_rt_module.cudaError_.success)
        err = cuda_rt_module.cudaProfilerStop()
        self.assertEqual(err, cuda_rt_module.cudaError_.success)

    @unittest.skipIf(
        (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_rocm()
        ),
        reason="Skip if not in CUDA env",
    )
    def test_mem_get_info(self):
        a, b = mem_get_info(paddle.device.get_device())
        self.assertGreaterEqual(a, 0)
        self.assertGreaterEqual(b, 0)

        a, b = mem_get_info('cuda:0')
        self.assertGreaterEqual(a, 0)
        self.assertGreaterEqual(b, 0)

        a, b = mem_get_info()
        self.assertGreaterEqual(a, 0)
        self.assertGreaterEqual(b, 0)

        with self.assertRaises(ValueError):
            a, b = mem_get_info(0)

    @unittest.skipIf(
        (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_rocm()
        ),
        reason="Skip if not in CUDA env",
    )
    def test_check_error(self):
        check_error(0)

        with self.assertRaisesRegex(RuntimeError, "invalid argument"):
            check_error(1)

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            check_error(2)


class TestExternalStream(unittest.TestCase):
    def test_get_stream_from_external(self):
        # Only run test if CUDA is available
        if not paddle.cuda.is_available():
            return

        # Test case 1: Device specified by integer ID
        device_id = 0
        original_stream = paddle.cuda.Stream(device_id)
        original_raw_ptr = original_stream.stream_base.raw_stream

        external_stream = paddle.cuda.get_stream_from_external(
            original_raw_ptr, device_id
        )
        self.assertEqual(
            original_raw_ptr, external_stream.stream_base.raw_stream
        )

        # Test case 2: Device specified by CUDAPlace
        device_place = paddle.CUDAPlace(0)
        original_stream = paddle.device.Stream(device_place)
        original_raw_ptr = original_stream.stream_base.raw_stream

        external_stream = paddle.device.get_stream_from_external(
            original_raw_ptr, device_place
        )
        self.assertEqual(
            original_raw_ptr, external_stream.stream_base.raw_stream
        )

        # Test case 3: Device not specified (None)
        device_none = None
        original_stream = paddle.cuda.Stream(device_none)
        original_raw_ptr = original_stream.stream_base.raw_stream

        external_stream = paddle.cuda.get_stream_from_external(
            original_raw_ptr, device_none
        )
        self.assertEqual(
            original_raw_ptr, external_stream.stream_base.raw_stream
        )

        # Test case 4: Verify original stream remains valid after external stream deletion
        del external_stream
        with paddle.cuda.stream(original_stream):
            current_stream = paddle.cuda.current_stream(device_none)

        self.assertEqual(
            current_stream.stream_base.raw_stream, original_raw_ptr
        )


if __name__ == '__main__':
    unittest.main()
