#!/usr/bin/env python

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

"""
A minimal unit test for testing XPU IPC sharing (_share_xpu and _new_shared_xpu).

This test uses the spawn start method to create two child processes
before the parent creates an XPU tensor. The parent then creates an XPU
tensor, calls _share_xpu to get IPC metadata, and sends that metadata to
the children via a multiprocessing.Queue. Each child sets its XPU device,
reconstructs the shared tensor using _new_shared_xpu, and verifies that its
content matches the expected value.
"""

import multiprocessing
import unittest

# IMPORTANT: Use the spawn method before any CUDA/XPU initialization.
multiprocessing.set_start_method("spawn", force=True)

import paddle
import paddle.incubate.multiprocessing as mp

# We'll use a constant test value.
TEST_VALUE = [1, 2, 3]


def child_reader(queue):
    """
    Child process function:
      - Initializes the XPU device.
      - Reads the IPC metadata from the queue.
      - Reconstructs the shared tensor via _new_shared_xpu.
      - Verifies that its content equals TEST_VALUE.
    """
    try:
        # Set XPU device in child process.
        paddle.set_device("xpu")
        current_device = (
            paddle.get_device() if hasattr(paddle, "get_device") else "xpu"
        )
        # print("[Child] XPU device set to:", current_device)
    except Exception as e:
        # print("[Child] Exception during paddle.set_device:", e)
        raise

    # Get the IPC metadata from the queue.
    ipc_meta = queue.get()
    # print("[Child] Received IPC metadata:", ipc_meta)

    try:
        # Reconstruct the shared tensor.
        # (Note: _new_shared_xpu is a private API; adjust accordingly for your version.)
        shared_tensor = paddle.to_tensor(
            paddle.base.core.LoDTensor._new_shared_xpu(ipc_meta)
        )
        # print(
        #     "[Child] Reconstructed tensor on",
        #     shared_tensor.place,
        #     "with value:",
        #     shared_tensor.numpy(),
        # )
    except Exception as e:
        # print("[Child] Exception during reconstruction:", e)
        raise

    # Verify that the content is as expected.
    expected = paddle.to_tensor(TEST_VALUE, dtype=shared_tensor.dtype)
    # Move to CPU for easy comparison.
    if not (shared_tensor.cpu() == expected).all().item():
        raise ValueError(
            "Child: Reconstructed tensor does not match expected value!"
        )
    # print("[Child] Verification passed.")


class TestXpuIpcSharing(unittest.TestCase):
    def test_ipc_share_read(self):
        """Test that a shared XPU tensor can be reconstructed in a child process."""
        ctx = mp.get_context("spawn")
        # Create a Queue to send the IPC metadata.
        q = ctx.Queue()

        # Spawn two child processes.
        p1 = ctx.Process(target=child_reader, args=(q,))
        p2 = ctx.Process(target=child_reader, args=(q,))
        p1.start()
        p2.start()

        # In the parent process, create an XPU tensor.
        # (This will trigger XPU initialization in the parent—but since we're using spawn,
        # the children will start fresh.)
        paddle.set_device("xpu")
        tensor = paddle.to_tensor(TEST_VALUE, dtype="int32").to("xpu")
        # print(
        #     "[Parent] Created tensor on",
        #     tensor.place,
        #     "with value:",
        #     tensor.cpu().numpy(),
        # )

        # Get the IPC metadata by calling _share_xpu on the tensor.
        ipc_meta = tensor.value().get_tensor()._share_xpu()
        # print("[Parent] IPC metadata:", ipc_meta)

        # Put the same metadata into the queue for each child.
        q.put(ipc_meta)
        q.put(ipc_meta)

        # Wait for children to complete.
        p1.join(10)
        p2.join(10)
        self.assertFalse(p1.is_alive())
        self.assertFalse(p2.is_alive())


if __name__ == "__main__":
    unittest.main()
