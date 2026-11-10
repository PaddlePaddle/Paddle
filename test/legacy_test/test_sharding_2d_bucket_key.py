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

"""Single-process tests for the keep-whole (2D) bucket key of the Muon and
Mixed sharding optimizers.

A 2D bucket is keyed by ``(color name or None, comm-group ranks)``. One color
name may be used by several comm groups on the same rank (e.g. the experts of
two sub-models with different expert-parallel sizes both use 'moe_expert'), and
each group must get its own bucket so its params are partitioned, reduced and
broadcast in their own group. Groups are stubbed with their ``ranks`` / ``rank``
and collectives with local stand-ins, so no launcher or accelerator is needed.
"""

import types
import unittest
from unittest import mock

import paddle
from paddle.distributed.fleet.meta_optimizers import (
    mixed_sharding_optimizer,
    muon_sharding_optimizer,
)

MODULES = (muon_sharding_optimizer, mixed_sharding_optimizer)


def _optimizer_cls(module):
    if module is muon_sharding_optimizer:
        return module.MuonShardingOptimizer
    return module.MixedShardingOptimizer


def _group(ranks, rank=0):
    return types.SimpleNamespace(ranks=list(ranks), rank=rank)


def _param(name, color=None, numel=16):
    p = types.SimpleNamespace(name=name, shape=[numel], dtype="float32")
    if color is not None:
        p.color = color
    return p


class _StubParam:
    """Records the storage calls of clear/reset_param_storage."""

    def __init__(self, name):
        self.name = name
        self.cleared = False

    def _clear_to_zero_allocation(self):
        self.cleared = True

    def _is_initialized(self):
        return True


def _bare_optimizer(cls, **attrs):
    """An instance without __init__, carrying only ``attrs``."""
    opt = object.__new__(cls)
    # __getattr__ forwards to _inner_opt; set it so missing attributes raise
    # AttributeError instead of recursing.
    opt._inner_opt = None
    for k, v in attrs.items():
        setattr(opt, k, v)
    return opt


class TestParamBucketKey(unittest.TestCase):
    def test_bucket_key(self):
        default = _group([0, 1, 2, 3])
        enc = _group([0, 2])
        dec = _group([0, 4])
        rank = paddle.distributed.get_rank()
        cases = [
            (_param("no_color"), (None, (0, 1, 2, 3))),
            (_param("minus_one", -1), (None, (0, 1, 2, 3))),
            (_param("dict_none", {"color": None}), (None, (0, 1, 2, 3))),
            # The default color always uses the default group.
            (
                _param("dict_none_group", {"color": None, "group": enc}),
                (None, (0, 1, 2, 3)),
            ),
            (
                _param("enc", {"color": "moe_expert", "group": enc}),
                ("moe_expert", (0, 2)),
            ),
            (
                _param("dec", {"color": "moe_expert", "group": dec}),
                ("moe_expert", (0, 4)),
            ),
            (_param("no_group", {"color": "c1"}), ("c1", (0, 1, 2, 3))),
            (_param("plain_name", "c2"), ("c2", (0, 1, 2, 3))),
            (
                _param("solo", {"color": "solo", "group": None}),
                ("solo", (rank,)),
            ),
        ]
        for module in MODULES:
            cls = _optimizer_cls(module)
            for p, expected in cases:
                with self.subTest(cls=cls.__name__, param=p.name):
                    self.assertEqual(
                        cls._param_bucket_key(p, default), expected
                    )

    def test_group_info_has_one_entry_per_group(self):
        default = _group([0, 1, 2, 3], rank=0)
        enc = _group([0, 2], rank=0)
        dec = _group([0, 4], rank=0)
        dec_alias = _group([0, 4], rank=0)  # same members, another object
        params = [
            _param("dense"),
            _param("enc", {"color": "moe_expert", "group": enc}),
            _param("dec", {"color": "moe_expert", "group": dec}),
            _param("dec2", {"color": "moe_expert", "group": dec_alias}),
        ]
        for module in MODULES:
            cls = _optimizer_cls(module)
            with self.subTest(cls=cls.__name__):
                info = cls._build_color_to_group_info_from_params(
                    params, default
                )
                self.assertEqual(
                    list(info),
                    [
                        (None, (0, 1, 2, 3)),
                        ("moe_expert", (0, 2)),
                        ("moe_expert", (0, 4)),
                    ],
                )
                self.assertIs(info[("moe_expert", (0, 2))]["group"], enc)
                self.assertIs(info[("moe_expert", (0, 4))]["group"], dec)
                self.assertEqual(info[("moe_expert", (0, 2))]["world_size"], 2)
                self.assertEqual(info[(None, (0, 1, 2, 3))]["world_size"], 4)

    def test_group_info_without_default_group(self):
        rank = paddle.distributed.get_rank()
        for module in MODULES:
            cls = _optimizer_cls(module)
            with self.subTest(cls=cls.__name__):
                info = cls._build_color_to_group_info_from_params([], None)
                self.assertEqual(
                    info,
                    {
                        (None, (rank,)): {
                            "group": None,
                            "world_size": 1,
                            "rank": 0,
                        }
                    },
                )

    def test_bucket_keys_of_color(self):
        buckets = {
            (None, (0, 1)): [],
            ("moe_expert", (0, 2)): [],
            ("moe_expert", (0, 4)): [],
        }
        for module in MODULES:
            cls = _optimizer_cls(module)
            with self.subTest(cls=cls.__name__):
                opt = _bare_optimizer(cls, _params_2d_by_color=buckets)
                self.assertEqual(
                    opt._bucket_keys_of_color("moe_expert"),
                    [("moe_expert", (0, 2)), ("moe_expert", (0, 4))],
                )
                self.assertEqual(
                    opt._bucket_keys_of_color(None), [(None, (0, 1))]
                )
                # -1 is normalized to None in the keys, so it matches nothing,
                # same as before the key change.
                self.assertEqual(opt._bucket_keys_of_color(-1), [])
                self.assertEqual(opt._bucket_keys_of_color("other"), [])


class TestBucketsOfOneColor(unittest.TestCase):
    """Two buckets share the color 'moe_expert' on different groups."""

    def setUp(self):
        # This rank is local rank 1 of the encoder group and 0 of the decoder.
        self.enc_key = ("moe_expert", (0, 2))
        self.dec_key = ("moe_expert", (0, 4))
        self.default_key = (None, (0, 1, 2, 3))
        self.enc = [_StubParam("enc0"), _StubParam("enc1")]
        self.dec = [_StubParam("dec0"), _StubParam("dec1")]
        self.dense = [_StubParam("dense0")]
        self.master_weights = []

    def _optimizer(self, cls):
        return _bare_optimizer(
            cls,
            _multi_precision=True,
            clear_color=set(),
            _color_to_comm_buffer_list={},
            _params_2d_by_color={
                self.default_key: self.dense,
                self.enc_key: self.enc,
                self.dec_key: self.dec,
            },
            _color_to_group_info={
                self.default_key: {"rank": 0},
                self.enc_key: {"rank": 1},
                self.dec_key: {"rank": 0},
            },
            _rank2params_2d_by_color={
                self.default_key: {0: self.dense},
                self.enc_key: {0: [self.enc[0]], 1: [self.enc[1]]},
                self.dec_key: {0: [self.dec[0]], 1: [self.dec[1]]},
            },
            _create_master_weight=lambda p: self.master_weights.append(p.name),
        )

    def test_clear_and_reset_param_storage(self):
        for module in MODULES:
            cls = _optimizer_cls(module)
            with self.subTest(cls=cls.__name__):
                self.setUp()
                opt = self._optimizer(cls)
                with mock.patch.object(
                    module, "g_shard_bypass_dygraph_optimizer", 0
                ):
                    opt.clear_param_storage("moe_expert")

                # Master weights are made for the params this rank owns in
                # each group, and every param of the color is released.
                self.assertEqual(self.master_weights, ["enc1", "dec0"])
                for p in self.enc + self.dec:
                    self.assertTrue(p.cleared, p.name)
                self.assertFalse(self.dense[0].cleared)
                self.assertEqual(opt.clear_color, {"moe_expert"})

                # reset_param_storage re-allocates every released param of
                # the color, in both groups.
                for p in self.enc + self.dec:
                    p._is_initialized = lambda: False
                shared = []
                fake = types.SimpleNamespace(
                    _share_buffer_to=lambda p: shared.append(p.name)
                )
                with mock.patch.object(
                    paddle, "empty_like", lambda p: fake, create=False
                ):
                    opt.reset_param_storage()
                self.assertEqual(shared, ["enc0", "enc1", "dec0", "dec1"])


class TestMachineBalancedBuckets(unittest.TestCase):
    def test_gathered_keys_match_local_buckets(self):
        """The all-gathered plan is keyed like the local buckets, so each
        bucket of one color is partitioned inside its own group."""
        buckets = {
            (None, tuple(range(8))): [_param("dense", numel=64)],
            ("moe_expert", (0, 2, 4, 6)): [
                _param(f"enc{i}", numel=32) for i in range(4)
            ],
            ("moe_expert", (0, 4)): [
                _param(f"dec{i}", numel=32) for i in range(2)
            ],
        }
        for params in buckets.values():
            for p in params:
                p.dtype = paddle.float32
        for module in MODULES:
            cls = _optimizer_cls(module)
            with self.subTest(cls=cls.__name__):
                opt = _bare_optimizer(
                    cls,
                    comm_buffer_size_MB=0,
                    _global_rank=1,
                    _params_2d_by_color=buckets,
                )
                with mock.patch.object(
                    paddle.distributed,
                    "all_gather_object",
                    lambda out, obj: out.append(obj),
                ):
                    gathered = opt._gather_all_color_group_info()
                self.assertEqual(list(gathered), list(buckets))

                plan = opt._partition_2d_parameters_machine_balanced(
                    gathered, dict.fromkeys(range(8), "host0")
                )
                self.assertEqual(list(plan), list(buckets))
                for key, params in buckets.items():
                    owners = plan[key]
                    self.assertEqual(set(owners), set(range(len(key[1]))))
                    self.assertEqual(
                        sorted(n for names in owners.values() for n in names),
                        sorted(p.name for p in params),
                    )


if __name__ == "__main__":
    unittest.main()
