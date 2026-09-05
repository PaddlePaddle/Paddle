# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import threading
import unittest

import paddle
from paddle.base import core, framework


class TestDeviceThreadSafety(unittest.TestCase):
    """
    Test the thread safety of device management in Paddle.

    This test suite verifies that:
    1. Main thread device settings remain unchanged
    2. Child threads can set their own devices independently
    3. Unset child threads inherit the main thread's device
    4. Concurrent device access is thread-safe
    """

    def setUp(self):
        """Set up test environment"""
        paddle.disable_static()
        self.results = {}
        self.lock = threading.Lock()

    def tearDown(self):
        """Clean up after tests"""
        paddle.enable_static()

    def _get_current_device(self):
        """Helper to get current device string"""
        place = framework._current_expected_place()
        if isinstance(place, core.CPUPlace):
            return "cpu"
        elif isinstance(place, core.CUDAPlace):
            return f"gpu:{place.get_device_id()}"
        elif isinstance(place, core.XPUPlace):
            return f"xpu:{place.get_device_id()}"
        elif hasattr(place, "get_device_type"):  # Custom device
            return f"{place.get_device_type()}:{place.get_device_id()}"
        else:
            return str(place)

    def _get_place_type(self):
        """Helper to get place type string"""
        place = framework._current_expected_place()
        if isinstance(place, core.CPUPlace):
            return "cpu"
        elif isinstance(place, core.CUDAPlace):
            return "gpu"
        elif isinstance(place, core.XPUPlace):
            return "xpu"
        else:
            return "unknown"

    def _worker_set_device(self, thread_id, device):
        """Worker function that sets its own device"""
        paddle.set_device(device)
        with self.lock:
            self.results[thread_id] = self._get_current_device()

    def _worker_get_device(self, thread_id):
        """Worker function that only gets device (no set)"""
        with self.lock:
            self.results[thread_id] = self._get_current_device()

    def _worker_set_and_verify(self, thread_id, device, iterations=100):
        """Worker that repeatedly sets and verifies its device"""
        for i in range(iterations):
            paddle.set_device(device)
            current = self._get_current_device()
            if current != device:
                with self.lock:
                    self.results[thread_id] = (
                        f"Failed at iteration {i}: got {current}, expected {device}"
                    )
                return
        with self.lock:
            self.results[thread_id] = "success"

    def _worker_stress_test(self, thread_id, devices):
        """Worker that randomly switches between multiple devices"""
        import random

        for i in range(100):
            device = random.choice(devices)
            paddle.set_device(device)
            current = self._get_current_device()
            if current != device:
                with self.lock:
                    self.results[thread_id] = (
                        f"Failed: expected {device}, got {current}"
                    )
                return
        with self.lock:
            self.results[thread_id] = "success"

    # ==================== Test Cases ====================

    def test_main_thread_behavior_unchanged(self):
        """
        Test Case 1: Verify main thread behavior remains unchanged.

        Expected: Main thread can set and get device normally,
        and the device persists across operations.
        """
        # Set CPU
        paddle.set_device('cpu')
        self.assertEqual(self._get_current_device(), 'cpu')

        # Create tensor - should be on CPU
        x = paddle.ones([2, 3])
        self.assertTrue(x.place.is_cpu_place())

        # Set GPU if available
        if core.is_compiled_with_cuda():
            paddle.set_device('gpu:0')
            self.assertEqual(self._get_current_device(), 'gpu:0')

            # Create tensor - should be on GPU
            y = paddle.ones([2, 3])
            self.assertTrue(y.place.is_gpu_place())
            self.assertTrue(y.place.is_gpu_place())

    def test_child_thread_device_isolation(self):
        """
        Test Case 2: Verify child threads can set their own devices independently.

        Expected: Each child thread's device setting should not affect others.
        """
        main_device = 'cpu'
        thread1_device = 'gpu:0' if core.is_compiled_with_cuda() else 'cpu'
        thread2_device = (
            'gpu:1'
            if core.is_compiled_with_cuda() and core.get_cuda_device_count() > 1
            else 'cpu'
        )

        paddle.set_device(main_device)

        # Create threads with different device settings
        t1 = threading.Thread(
            target=self._worker_set_device, args=(1, thread1_device)
        )
        t2 = threading.Thread(
            target=self._worker_set_device, args=(2, thread2_device)
        )
        t3 = threading.Thread(
            target=self._worker_get_device, args=(3,)
        )  # No set, should inherit

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

        # Verify each thread got expected device
        self.assertEqual(self.results[1], thread1_device)
        self.assertEqual(self.results[2], thread2_device)
        self.assertEqual(self.results[3], main_device)

        # Verify main thread unaffected
        self.assertEqual(self._get_current_device(), main_device)

    def test_device_inheritance_chain(self):
        """
        Test Case 3: Verify device inheritance works correctly.

        Expected: New threads inherit the MAIN THREAD's device at the time they START,
        not at the time they are created.
        """
        paddle.set_device('cpu')

        def worker(thread_id):
            with self.lock:
                self.results[thread_id] = self._get_current_device()

        # 创建两个线程（都还没启动）
        t1 = threading.Thread(target=worker, args=(1,))
        t2 = threading.Thread(target=worker, args=(2,))

        # 启动第一个线程前，主线程是CPU
        t1.start()  # 此时主线程是CPU，t1应继承CPU
        t1.join()

        # 改变主线程设备
        if core.is_compiled_with_cuda():
            paddle.set_device('gpu:0')

        # 启动第二个线程，此时主线程是GPU，t2应继承GPU
        t2.start()
        t2.join()

        # 验证
        self.assertEqual(self.results[1], 'cpu')
        if core.is_compiled_with_cuda():
            self.assertEqual(self.results[2], 'gpu:0')
        else:
            self.assertEqual(self.results[2], 'cpu')

    def test_concurrent_set_device_same_thread(self):
        """
        Test Case 4: Verify a single thread can change its device multiple times.

        Expected: Device changes should only affect the changing thread.
        """
        paddle.set_device('cpu')

        def worker():
            # Change device multiple times
            if core.is_compiled_with_cuda():
                paddle.set_device('gpu:0')
                self.assertEqual(self._get_place_type(), 'gpu')

                if core.get_cuda_device_count() > 1:
                    paddle.set_device('gpu:1')
                    self.assertEqual(self._get_place_type(), 'gpu')
                else:
                    paddle.set_device('gpu:0')
                    self.assertEqual(self._get_place_type(), 'gpu')

            paddle.set_device('cpu')
            self.assertEqual(self._get_place_type(), 'cpu')

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # Main thread should still be on CPU
        self.assertEqual(self._get_place_type(), 'cpu')

    def test_device_persistence_in_thread(self):
        """
        Test Case 5: Verify device setting persists within a thread.

        Expected: Once set, a thread's device should remain consistent.
        """
        paddle.set_device('cpu')

        def worker():
            device = 'gpu:0' if core.is_compiled_with_cuda() else 'cpu'
            paddle.set_device(device)
            device_type = 'gpu' if 'gpu' in device else 'cpu'

            # Multiple operations should all use same device
            for _ in range(10):
                x = paddle.ones([2, 3])
                if device_type == 'gpu':
                    self.assertTrue(x.place.is_gpu_place())
                else:
                    self.assertTrue(x.place.is_cpu_place())
                self.assertEqual(self._get_place_type(), device_type)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

    def test_multiple_threads_same_device(self):
        """
        Test Case 6: Verify multiple threads can use the same device.

        Expected: Threads setting same device should not interfere.
        """
        target_device = 'gpu:0' if core.is_compiled_with_cuda() else 'cpu'
        paddle.set_device('cpu')  # Main thread stays on CPU

        threads = []
        for i in range(5):
            t = threading.Thread(
                target=self._worker_set_and_verify, args=(i, target_device, 50)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All threads should have succeeded
        for i in range(5):
            self.assertEqual(self.results[i], "success")

        # Main thread unchanged
        self.assertEqual(self._get_place_type(), 'cpu')

    def test_device_switch_stress(self):
        """
        Test Case 7: Stress test with rapid device switching.

        Expected: No race conditions or device pollution under heavy load.
        """

        paddle.set_device('cpu')
        devices = ['gpu:0', 'gpu:1', 'cpu']

        if not core.is_compiled_with_cuda() or core.get_cuda_device_count() < 2:
            devices = ['gpu:0', 'cpu']
        if not core.is_compiled_with_cuda():
            self.skipTest(
                "Need at least 1 GPU and GPU version of paddle for this test"
            )

        threads = []
        for i in range(10):
            t = threading.Thread(
                target=self._worker_stress_test, args=(i, devices)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All threads should have succeeded
        for i in range(10):
            self.assertEqual(self.results[i], "success")

        # Main thread unchanged
        self.assertEqual(self._get_place_type(), 'cpu')

    def test_device_get_without_set(self):
        """
        Test Case 8: Verify getting device without setting works.

        Expected: Should return current thread's device (inherited or default).
        """
        paddle.set_device('cpu')

        def worker():
            # Just get device without setting
            device = self._get_current_device()
            with self.lock:
                self.results[threading.current_thread().name] = device

        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, name=f"Worker-{i}")
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All threads should report 'cpu'
        for i in range(3):
            self.assertEqual(self.results[f"Worker-{i}"], 'cpu')

    def test_nested_threads_device_independence(self):
        """
        Test Case 9: Verify nested threads can set devices independently.

        Expected: Each thread (including nested ones) can set its own device.
        Note: Thread-local storage does NOT automatically propagate to nested threads.
        """
        paddle.set_device('cpu')
        results = {}
        lock = threading.Lock()

        def inner_worker(outer_id, inner_id, expected_device):
            # 内层线程需要显式设置自己的设备
            paddle.set_device(expected_device)
            current = self._get_current_device()
            with lock:
                results[f"inner_{outer_id}_{inner_id}"] = current

        def outer_worker(outer_id):
            # 外层线程设置自己的设备
            if outer_id == 1 and core.is_compiled_with_cuda():
                paddle.set_device('gpu:0')
                expected_inner_device = 'gpu:0'
            else:
                paddle.set_device('cpu')
                expected_inner_device = 'cpu'

            # 记录外层线程的设备
            outer_device = self._get_current_device()
            with lock:
                results[f"outer_{outer_id}"] = outer_device

            # 创建并启动内层线程（需要显式传递期望的设备）
            inner_threads = []
            for i in range(2):
                t = threading.Thread(
                    target=inner_worker,
                    args=(outer_id, i, expected_inner_device),
                )
                inner_threads.append(t)
                t.start()

            # 等待内层线程完成
            for t in inner_threads:
                t.join()

        # 创建并启动外层线程
        outer_threads = []
        for i in range(2):
            t = threading.Thread(target=outer_worker, args=(i,))
            outer_threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in outer_threads:
            t.join()

        # 验证结果
        self.assertEqual(results["outer_0"], 'cpu')
        self.assertEqual(results["inner_0_0"], 'cpu')
        self.assertEqual(results["inner_0_1"], 'cpu')

        if core.is_compiled_with_cuda():
            self.assertEqual(results["outer_1"], 'gpu:0')
            self.assertEqual(results["inner_1_0"], 'gpu:0')
            self.assertEqual(results["inner_1_1"], 'gpu:0')


class TestDeviceThreadSafetyStatic(unittest.TestCase):
    """
    Test thread safety in static graph mode.
    """

    def setUp(self):
        paddle.enable_static()
        self.main_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()

    def tearDown(self):
        paddle.disable_static()

    def test_static_main_thread_behavior(self):
        """Test device setting in static mode on main thread"""
        with paddle.static.program_guard(
            self.main_program, self.startup_program
        ):
            # Set device
            paddle.set_device('cpu')

            # Create executor with specific place
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(self.startup_program)

            # Verify executor's place is CPU
            self.assertIsInstance(exe.place, core.CPUPlace)

            # Create some ops to verify they work
            x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
            y = paddle.static.data(name='y', shape=[2, 3], dtype='float32')
            z = paddle.add(x, y)

            # 使用 numpy 数组作为 feed 数据
            import numpy as np

            feed = {
                'x': np.ones([2, 3], dtype='float32'),
                'y': np.ones([2, 3], dtype='float32'),
            }

            # Run a simple inference to verify
            result = exe.run(fetch_list=[z], feed=feed)
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].shape, (2, 3))

            # Test GPU if available
            if core.is_compiled_with_cuda():
                # Set GPU device
                paddle.set_device('gpu:0')

                # Create GPU executor
                gpu_place = paddle.CUDAPlace(0)
                gpu_exe = paddle.static.Executor(gpu_place)
                gpu_exe.run(self.startup_program)

                # Verify executor's place is GPU
                self.assertIsInstance(gpu_exe.place, core.CUDAPlace)

                # Run on GPU
                result = gpu_exe.run(fetch_list=[z], feed=feed)
                self.assertIsNotNone(result)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].shape, (2, 3))

    def test_static_mode_device_persistence(self):
        """Test that device setting persists across multiple executions"""
        with paddle.static.program_guard(
            self.main_program, self.startup_program
        ):
            # Set device
            paddle.set_device('cpu')

            # Create executor
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(self.startup_program)

            # Create ops
            x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
            y = paddle.static.data(name='y', shape=[2, 3], dtype='float32')
            z = paddle.add(x, y)

            # 使用 numpy 数组
            import numpy as np

            feed = {
                'x': np.ones([2, 3], dtype='float32'),
                'y': np.ones([2, 3], dtype='float32'),
            }

            # Run multiple times
            for _ in range(5):
                result = exe.run(fetch_list=[z], feed=feed)
                self.assertIsNotNone(result)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].shape, (2, 3))
                # Device should remain CPU
                self.assertIsInstance(exe.place, core.CPUPlace)


if __name__ == '__main__':
    unittest.main()
