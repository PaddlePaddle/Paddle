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

import unittest

from paddle.distributed.fleet.meta_parallel import pipeline_parallel as pp_mod
from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
    PipelineParallelMicroStepCallback,
    PipelineParallelMicroStepLocations,
    register_global_pipeline_parallel_hook,
)
from paddle.distributed.fleet.meta_parallel.pp_utils import (
    p2p_communication as p2p_mod,
)
from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
    P2pHelper,
)

PLACEHOLDER = "placeholder"


class TestMicroStepLocations(unittest.TestCase):
    def test_p2p_issued_enum_member(self):
        self.assertEqual(
            PipelineParallelMicroStepLocations.P2P_ISSUED.value, 'p2p_issued'
        )
        self.assertIs(
            PipelineParallelMicroStepLocations('p2p_issued'),
            PipelineParallelMicroStepLocations.P2P_ISSUED,
        )

    def test_registry_contains_all_locations(self):
        callbacks = PipelineParallelMicroStepCallback()
        self.assertEqual(
            set(callbacks.hooks.keys()),
            set(PipelineParallelMicroStepLocations),
        )
        self.assertEqual(
            callbacks.hooks[PipelineParallelMicroStepLocations.P2P_ISSUED], []
        )

    def test_register_and_fire(self):
        callbacks = PipelineParallelMicroStepCallback()
        calls = []

        callbacks.register_hook(
            PipelineParallelMicroStepLocations.P2P_ISSUED,
            lambda **kw: calls.append(('first', kw)),
        )
        callbacks.register_hook(
            PipelineParallelMicroStepLocations.P2P_ISSUED,
            lambda **kw: calls.append(('second', kw)),
        )
        # a hook on another location must not be triggered by P2P_ISSUED
        callbacks.register_hook(
            PipelineParallelMicroStepLocations.FORWARD_END,
            lambda **kw: calls.append(('forward_end', kw)),
        )

        callbacks.on_location(
            PipelineParallelMicroStepLocations.P2P_ISSUED,
            output_tensor=PLACEHOLDER,
            step_id=7,
        )

        # hooks run in registration order, and only for this location
        self.assertEqual([name for name, _ in calls], ['first', 'second'])
        for _, kwargs in calls:
            self.assertEqual(
                kwargs, {'output_tensor': PLACEHOLDER, 'step_id': 7}
            )

    def test_no_hook_registered_is_noop(self):
        callbacks = PipelineParallelMicroStepCallback()
        callbacks.on_location(
            PipelineParallelMicroStepLocations.P2P_ISSUED,
            output_tensor=None,
            step_id=0,
        )

    def test_invalid_location_message_lists_p2p_issued(self):
        callbacks = PipelineParallelMicroStepCallback()
        with self.assertRaises(AssertionError) as ctx:
            callbacks.register_hook('not_a_location', lambda **kw: None)
        self.assertIn('p2p_issued', str(ctx.exception))

        with self.assertRaises(AssertionError) as ctx:
            callbacks.on_location('not_a_location')
        self.assertIn('p2p_issued', str(ctx.exception))

    def test_global_registration(self):
        location = PipelineParallelMicroStepLocations.P2P_ISSUED
        global_hooks = pp_mod.pipeline_parallel_callbacks_.hooks[location]
        before = len(global_hooks)
        fired = []
        try:
            register_global_pipeline_parallel_hook(
                location, lambda **kw: fired.append(kw)
            )
            self.assertEqual(len(global_hooks), before + 1)
            pp_mod.pipeline_parallel_callbacks_.on_location(
                location, output_tensor=None, step_id=3
            )
            self.assertEqual(fired, [{'output_tensor': None, 'step_id': 3}])
        finally:
            del global_hooks[before:]
        self.assertEqual(len(global_hooks), before)


class _FakeReq:
    def __init__(self, log, name):
        self._log = log
        self._name = name

    def wait(self):
        self._log.append(self._name)


class TestSendForwardOverlapP2pComm(unittest.TestCase):
    """`P2pHelper.send_forward` must be able to defer its wait handles.

    The `P2P_ISSUED` hook is only useful if the send can be issued without
    being waited on inline, so `overlap_p2p_comm=True` has to forward
    `wait_on_reqs=False` to `_p2p_helper` and hand the handles back to the
    caller instead of consuming them.
    """

    def setUp(self):
        self.calls = []
        self.waited = []
        self.handles = [_FakeReq(self.waited, 'req0')]

        def fake_p2p_helper(**kwargs):
            self.calls.append(kwargs)
            reqs = None if kwargs['wait_on_reqs'] else self.handles
            return None, None, reqs

        self._orig = p2p_mod._p2p_helper
        p2p_mod._p2p_helper = fake_p2p_helper

        self.helper = P2pHelper(use_cache=True, dynamic_shape=False)
        # `_send_meta` talks to the pipeline process group, which is out of
        # scope here: this test only pins down the wait-handle plumbing.
        self.meta_sent = []
        self.helper._send_meta = lambda *args, **kwargs: self.meta_sent.append(
            (args, kwargs)
        )

    def tearDown(self):
        p2p_mod._p2p_helper = self._orig

    def test_default_waits_inline_and_returns_none(self):
        ret = self.helper.send_forward(
            PLACEHOLDER, pp_last_stage=False, batch_p2p_comm=False
        )
        self.assertIsNone(ret)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.calls[0]['wait_on_reqs'])
        self.assertIs(self.calls[0]['tensor_send_next'], PLACEHOLDER)
        self.assertFalse(self.calls[0]['batch_p2p_comm'])
        self.assertEqual(len(self.meta_sent), 1)
        self.assertEqual(self.waited, [])

    def test_overlap_returns_handles_without_waiting(self):
        handles = self.helper.send_forward(
            PLACEHOLDER,
            pp_last_stage=False,
            batch_p2p_comm=False,
            overlap_p2p_comm=True,
        )
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.calls[0]['wait_on_reqs'])
        # nothing has been waited on yet: that is the caller's job, after the
        # P2P_ISSUED hook has enqueued its kernels.
        self.assertEqual(self.waited, [])
        self.assertIs(handles, self.handles)
        for handle in handles:
            handle.wait()
        self.assertEqual(self.waited, ['req0'])

    def test_batch_p2p_comm_yields_no_handles(self):
        # `_batched_p2p_ops` runs on the calculation stream, so `_p2p_helper`
        # returns no requests and `send_forward` must degrade gracefully.
        def fake_p2p_helper(**kwargs):
            self.calls.append(kwargs)
            return None, None, None

        p2p_mod._p2p_helper = fake_p2p_helper
        handles = self.helper.send_forward(
            PLACEHOLDER,
            pp_last_stage=False,
            batch_p2p_comm=True,
            overlap_p2p_comm=True,
        )
        self.assertIsNone(handles)
        self.assertTrue(self.calls[0]['batch_p2p_comm'])

    def test_last_stage_sends_nothing(self):
        for overlap in (False, True):
            handles = self.helper.send_forward(
                PLACEHOLDER,
                pp_last_stage=True,
                batch_p2p_comm=False,
                overlap_p2p_comm=overlap,
            )
            self.assertIsNone(handles)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.meta_sent, [])

    def test_skip_check_meta_is_forwarded(self):
        self.helper.send_forward(
            PLACEHOLDER,
            pp_last_stage=False,
            batch_p2p_comm=False,
            skip_check_meta=True,
            overlap_p2p_comm=True,
        )
        self.assertEqual(self.meta_sent[0][1].get('skip_check_meta'), True)


if __name__ == "__main__":
    unittest.main()
