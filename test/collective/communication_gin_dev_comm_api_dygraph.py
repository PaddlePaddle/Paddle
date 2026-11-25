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

import ctypes
import os

import paddle.distributed as dist
from paddle.distributed.collective import (
    restart_process_group,
    shutdown_process_group,
)

# Start of the Unavailable error that ProcessGroupNCCL raises from every GIN
# primitive when Paddle was built without PADDLE_WITH_GIN_OVERLAP, the GPU is
# older than sm_90, or the loaded NCCL has no device API.
GIN_UNAVAILABLE_MESSAGE = (
    "The GIN primitives of ProcessGroupNCCL are not available"
)
# test_communication_gin_dev_comm_api.py passes the path of a file in which
# rank 0 writes that error when it gets it; the test then skips.
SKIP_FILE_ENV = "GIN_DEV_COMM_TEST_SKIP_FILE"

_GIN_CONTEXT_COUNT = 4
_WINDOW_ALIGNMENT = 4096


def _raises(error_type, message, fn, *args, **kwargs):
    """Assert that a call raises `error_type` with `message` in its text."""
    try:
        fn(*args, **kwargs)
    except error_type as error:
        assert message in str(error), (message, str(error))
        return
    raise AssertionError(f"the call should have raised {error_type.__name__}")


class GinDevCommTestCase:
    """Exercises the setup of the GIN primitives of ProcessGroupNCCL on a
    group of two GPUs of one node: the device communicator
    (get_or_create_dev_comm) and the host windows (alloc_host_window),
    including their teardown when the group is shut down and restarted, and
    a new group after destroy_process_group().

    Every rank asserts on its own, so a failure anywhere fails the launcher.
    """

    def __init__(self):
        dist.init_parallel_env()
        self._rank = dist.get_rank()
        self._world = dist.get_world_size()
        assert self._world == 2, self._world
        self._group = dist.new_group(list(range(self._world)))
        self._pg = self._group.process_group

    def unavailable_reason(self):
        """Creates the device communicator (the first, collective call) and
        returns None, or returns the error when the GIN primitives are not
        available in this build or on this GPU."""
        try:
            self._dev_comm_info = self._pg.get_or_create_dev_comm(
                _GIN_CONTEXT_COUNT
            )
        except RuntimeError as error:
            if GIN_UNAVAILABLE_MESSAGE not in str(error):
                raise
            return str(error)
        return None

    def _check_dev_comm_info(self, dev_comm_info):
        assert dev_comm_info["rank"] == self._rank, dev_comm_info
        assert dev_comm_info["nranks"] == self._world, dev_comm_info
        # Two GPUs of one node: one LSA team holding both ranks.
        assert dev_comm_info["lsa_size"] == self._world, dev_comm_info
        assert dev_comm_info["lsa_rank"] == self._rank, dev_comm_info
        assert dev_comm_info["gin_context_count"] >= 1, dev_comm_info

    def _check_host_window(self, pg, requested_bytes):
        window, cpu_ptr, window_bytes = pg.alloc_host_window(requested_bytes)
        assert window != 0 and cpu_ptr != 0, (window, cpu_ptr)
        # Rounded up to the window alignment and the allocation granularity.
        assert window_bytes >= max(requested_bytes, _WINDOW_ALIGNMENT), (
            requested_bytes,
            window_bytes,
        )
        assert window_bytes % _WINDOW_ALIGNMENT == 0, window_bytes
        assert ctypes.string_at(cpu_ptr, window_bytes) == bytes(window_bytes)
        # The CPU can write through the returned pointer.
        ctypes.c_uint64.from_address(cpu_ptr).value = 1 + self._rank
        return window, cpu_ptr, window_bytes

    def test_dev_comm_info(self):
        self._check_dev_comm_info(self._dev_comm_info)
        # Later calls return the cached device communicator.
        cached = self._pg.get_or_create_dev_comm(_GIN_CONTEXT_COUNT)
        assert cached == self._dev_comm_info, cached
        # The default hint is 4.
        assert self._pg.get_or_create_dev_comm() == self._dev_comm_info
        _raises(
            ValueError,
            "hint of 4, but got 2",
            self._pg.get_or_create_dev_comm,
            2,
        )

    def test_alloc_host_window(self):
        windows = set()
        for requested_bytes in (1, 4096, 100000, 3 * 2**20 + 1):
            window, _, window_bytes = self._check_host_window(
                self._pg, requested_bytes
            )
            windows.add(window)
            # Rounding a rounded size changes nothing.
            _, _, again_bytes = self._check_host_window(self._pg, window_bytes)
            assert again_bytes == window_bytes, (window_bytes, again_bytes)
        assert len(windows) == 4, windows
        _raises(ValueError, "must be positive", self._pg.alloc_host_window, 0)

    def test_host_window_before_dev_comm(self):
        # A host window can be allocated before the device communicator of
        # the group is created.
        group = dist.new_group(list(range(self._world)))
        pg = group.process_group
        self._check_host_window(pg, 8192)
        self._check_dev_comm_info(pg.get_or_create_dev_comm(_GIN_CONTEXT_COUNT))
        dist.barrier(group=group)

    def test_shutdown_and_restart(self):
        # Shutting the group down destroys the device communicator and frees
        # the host windows; after a restart another hint is accepted.
        shutdown_process_group(self._group)
        restart_process_group(self._group)
        self._check_dev_comm_info(self._pg.get_or_create_dev_comm(2))
        self._check_host_window(self._pg, 4096)
        dist.barrier(group=self._group)

    def test_new_group_after_destroy(self):
        # After destroy_process_group() a new group gets a device
        # communicator and host windows of its own.
        dist.destroy_process_group(self._group)
        group = dist.new_group(list(range(self._world)))
        pg = group.process_group
        self._check_dev_comm_info(pg.get_or_create_dev_comm(_GIN_CONTEXT_COUNT))
        self._check_host_window(pg, 4096)
        dist.barrier(group=group)

    def run_test_case(self):
        reason = self.unavailable_reason()
        if reason is not None:
            if self._rank == 0:
                with open(os.environ[SKIP_FILE_ENV], "w") as skip_file:
                    skip_file.write(reason)
            return
        # In this order: the last two cases tear down self._group.
        for name in (
            "test_dev_comm_info",
            "test_alloc_host_window",
            "test_host_window_before_dev_comm",
            "test_shutdown_and_restart",
            "test_new_group_after_destroy",
        ):
            getattr(self, name)()
            if self._rank == 0:
                print(f"[gin_dev_comm] {name} passed", flush=True)


if __name__ == "__main__":
    GinDevCommTestCase().run_test_case()
