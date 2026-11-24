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
"""Tests for paddle.distributed.fleet.utils.param_utils.copy_param_attrs and its
use in the sharding optimizers' ``_create_slice_param``. Single process, CPU.
"""

import gc
import types
import unittest
import weakref

import paddle
from paddle.base import unique_name
from paddle.base.framework import EagerParamBase
from paddle.distributed.fleet.meta_optimizers import (
    mixed_sharding_optimizer,
    muon_sharding_optimizer,
)
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    dygraph_sharding_optimizer,
)
from paddle.distributed.fleet.utils.param_utils import (
    DEFAULT_COPY_ATTRS,
    DEFAULT_EXCLUDE_ATTRS,
    copy_param_attrs,
    get_external_attr_names,
)

# The attribute list the sharding optimizers' copy_attr used to copy.
ORIGINAL_COPY_ATTR_LIST = (
    "is_distributed",
    "optimize_attr",
    "do_model_average",
    "need_clip",
    "no_sync",
    "is_firstly_shared",
)

# Set by EagerParamBase.__init__ on every parameter.
ALWAYS_PRESENT_COPY_ATTRS = [
    "is_distributed",
    "optimize_attr",
    "do_model_average",
    "need_clip",
]

# Built-in attributes outside the original list: never copied.
NOT_COPIED_BUILTIN_ATTRS = ("regularizer", "_init_func", "_init_op_creator")


def _new_param(shape=(2, 3)):
    return paddle.create_parameter(list(shape), "float32")


def _new_slice_param(param):
    # Same construction as DygraphShardingOptimizerV2._create_slice_param.
    slice_param = EagerParamBase(shape=[1], dtype=param.dtype)
    slice_param.name = param.name
    return slice_param


class TestGetExternalAttrNames(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def test_fresh_param_has_no_external_attrs(self):
        self.assertEqual(get_external_attr_names(_new_param()), [])

    def test_detects_attached_attrs(self):
        p = _new_param()
        p.no_sync = True
        p.is_firstly_shared = False
        p.color = {"color": "mp"}
        p._private_tag = 1
        self.assertEqual(
            get_external_attr_names(p),
            ["_private_tag", "color", "is_firstly_shared", "no_sync"],
        )

    def test_reassigned_builtin_attrs_are_not_external(self):
        p = _new_param()
        p.optimize_attr = {"learning_rate": 0.5}
        p.need_clip = False
        p.is_distributed = True
        p.regularizer = object()
        self.assertEqual(get_external_attr_names(p), [])

    def test_excludes_grad_and_main_grad_by_default(self):
        p = _new_param()
        p.main_grad = paddle.zeros([2, 3])
        p.tag = 1
        self.assertEqual(get_external_attr_names(p), ["tag"])

    def test_custom_exclude(self):
        p = _new_param()
        p.main_grad = paddle.zeros([2, 3])
        p.my_private_buf = object()
        p.tag = 1
        self.assertEqual(
            get_external_attr_names(
                p, exclude=DEFAULT_EXCLUDE_ATTRS | {"my_private_buf"}
            ),
            ["tag"],
        )

    def test_plain_tensor(self):
        t = paddle.zeros([2])
        self.assertEqual(get_external_attr_names(t), [])
        t.tag = 1
        self.assertEqual(get_external_attr_names(t), ["tag"])

    def test_object_without_dict(self):
        self.assertEqual(get_external_attr_names(1), [])

    def test_probe_does_not_consume_unique_names(self):
        get_external_attr_names(_new_param())  # warm the cache
        from paddle.distributed.fleet.utils.param_utils import (
            _builtin_attr_names,
        )

        _builtin_attr_names.cache_clear()
        with unique_name.guard():
            first = unique_name.generate("_eager_param_base")
            get_external_attr_names(_new_param())
            second = unique_name.generate("_eager_param_base")
        with unique_name.guard():
            expected_first = unique_name.generate("_eager_param_base")
            _new_param()
            expected_second = unique_name.generate("_eager_param_base")
        self.assertEqual(first, expected_first)
        self.assertEqual(second, expected_second)


class TestCopyParamAttrs(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def test_default_list_matches_original_copy_attr(self):
        self.assertEqual(tuple(DEFAULT_COPY_ATTRS), ORIGINAL_COPY_ATTR_LIST)

    def test_copies_every_original_copy_attr_by_reference(self):
        src = _new_param()
        values = {
            "is_distributed": True,
            "optimize_attr": {"learning_rate": 0.5},
            "do_model_average": True,
            "need_clip": False,
            "no_sync": True,
            "is_firstly_shared": True,
        }
        for name, value in values.items():
            setattr(src, name, value)
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertEqual(copied, list(ORIGINAL_COPY_ATTR_LIST))
        for name, value in values.items():
            self.assertIs(getattr(dst, name), value, name)

    def test_copies_original_attrs_with_default_values(self):
        src = _new_param()  # no reassignment, no_sync/is_firstly_shared absent
        dst = _new_slice_param(src)
        dst.optimize_attr = {"learning_rate": 9.0}

        copied = copy_param_attrs(src, dst)

        self.assertEqual(copied, ALWAYS_PRESENT_COPY_ATTRS)
        self.assertIs(dst.optimize_attr, src.optimize_attr)
        self.assertFalse(hasattr(dst, "no_sync"))
        self.assertFalse(hasattr(dst, "is_firstly_shared"))

    def test_copies_class_level_original_attr(self):
        class MyParam(EagerParamBase):
            no_sync = True

        with unique_name.guard():
            src = MyParam(shape=[2, 3], dtype=paddle.float32)
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertIn("no_sync", copied)
        self.assertTrue(dst.no_sync)

    def test_copies_attached_attrs_by_reference(self):
        src = _new_param()
        color = {"color": "mp", "group": object()}
        src.color = color
        src._private_tag = 1
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertEqual(
            copied, [*ALWAYS_PRESENT_COPY_ATTRS, "_private_tag", "color"]
        )
        self.assertIs(dst.color, color)  # shallow copy
        self.assertEqual(dst._private_tag, 1)

    def test_does_not_copy_other_builtin_attrs(self):
        src = _new_param()
        src.regularizer = object()
        src._init_func = lambda *a, **k: None
        src._init_op_creator = lambda *a, **k: None
        dst = _new_slice_param(src)
        before = {n: getattr(dst, n) for n in NOT_COPIED_BUILTIN_ATTRS}

        copied = copy_param_attrs(src, dst)

        self.assertEqual(copied, ALWAYS_PRESENT_COPY_ATTRS)
        for name in NOT_COPIED_BUILTIN_ATTRS:
            self.assertIs(getattr(dst, name), before[name], name)

    def test_does_not_touch_tensor_properties(self):
        src = _new_param()
        src.stop_gradient = True
        src.tag = 1
        dst = _new_slice_param(src)
        dst.stop_gradient = False

        copy_param_attrs(src, dst)

        self.assertFalse(dst.stop_gradient)
        self.assertEqual(dst.name, src.name)
        self.assertEqual(dst.shape, [1])

    def test_does_not_copy_grad_or_main_grad(self):
        src = _new_param()
        src.main_grad = paddle.ones([2, 3])
        loss = (src * 2).sum()
        loss.backward()
        self.assertIsNotNone(src.grad)
        src.tag = 1
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertEqual(copied, [*ALWAYS_PRESENT_COPY_ATTRS, "tag"])
        self.assertFalse(hasattr(dst, "main_grad"))
        self.assertIsNone(dst.grad)

    def test_custom_exclude(self):
        src = _new_param()
        src.my_private_buf = object()
        src.tag = 1
        dst = _new_slice_param(src)

        copied = copy_param_attrs(
            src, dst, exclude=DEFAULT_EXCLUDE_ATTRS | {"my_private_buf"}
        )

        self.assertEqual(copied, [*ALWAYS_PRESENT_COPY_ATTRS, "tag"])
        self.assertFalse(hasattr(dst, "my_private_buf"))

    def test_exclude_wins_over_copy_attrs(self):
        src = _new_param()
        src.no_sync = True
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst, exclude={"no_sync", "main_grad"})

        self.assertNotIn("no_sync", copied)
        self.assertFalse(hasattr(dst, "no_sync"))

    def test_custom_copy_attrs(self):
        src = _new_param()
        src.regularizer = object()
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst, copy_attrs=("regularizer",))

        self.assertEqual(copied, ["regularizer"])
        self.assertIs(dst.regularizer, src.regularizer)

    def test_overwrites_existing_value_on_dst(self):
        src = _new_param()
        src.no_sync = True
        src.tag = "src"
        dst = _new_slice_param(src)
        dst.no_sync = False
        dst.tag = "dst"

        copy_param_attrs(src, dst)

        self.assertTrue(dst.no_sync)
        self.assertEqual(dst.tag, "src")

    def test_src_unchanged(self):
        src = _new_param()
        src.main_grad = paddle.ones([2, 3])
        src.no_sync = True
        src.tag = 1
        before = dict(vars(src))
        dst = _new_slice_param(src)

        copy_param_attrs(src, dst)

        self.assertEqual(vars(src).keys(), before.keys())
        for name, value in before.items():
            self.assertIs(vars(src)[name], value)

    def test_default_exclude_attrs_not_copied(self):
        src = _new_param()
        for name in DEFAULT_EXCLUDE_ATTRS - {"grad"}:  # grad is C-level
            setattr(src, name, object())
        src.tag = 1
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertEqual(copied, [*ALWAYS_PRESENT_COPY_ATTRS, "tag"])
        for name in DEFAULT_EXCLUDE_ATTRS - {"grad"}:
            self.assertNotIn(name, vars(dst), name)

    def test_default_exclude_contains_expected_names(self):
        expected = {
            "grad", "main_grad", "_main_grad_addr", "get_main_grad",
            "_fused_kernel_fp32_grad", "master_weight",
            "comm_buffer_ref", "fw_storage", "bw_storage",
            "_fusion_buffer", "_param_offsets",
            "_fsdp_grad_hooked", "use_count", "status", "unslice",
        }  # fmt: skip
        self.assertEqual(set(DEFAULT_EXCLUDE_ATTRS), expected)

    def test_method_bound_to_src_not_copied(self):
        def get_buf(self):
            return self

        src = _new_param()
        # Same pattern as get_main_grad in auto_parallel/fully_shard_fusion.py,
        # but under a name the exclude list does not know.
        src.get_buf = get_buf.__get__(src)
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertNotIn("get_buf", copied)
        self.assertNotIn("get_buf", vars(dst))

    def test_bound_method_in_copy_attrs_not_copied(self):
        class MyParam(EagerParamBase):
            def no_sync(self):
                return True

        with unique_name.guard():
            src = MyParam(shape=[2, 3], dtype=paddle.float32)
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertNotIn("no_sync", copied)
        self.assertFalse(hasattr(dst, "no_sync"))

    def test_plain_function_and_other_bound_method_copied(self):
        class Owner:
            def hook(self):
                return self

        def init_fn():
            return 1

        owner = Owner()
        src = _new_param()
        src.init_fn = init_fn
        src.owner_hook = owner.hook
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertIn("init_fn", copied)
        self.assertIn("owner_hook", copied)
        self.assertIs(dst.init_fn, init_fn)
        self.assertIs(dst.owner_hook.__self__, owner)

    def test_weak_references_not_copied(self):
        class Buffer:
            def __call__(self):
                return 1

        buf = Buffer()
        src = _new_param()
        src.buf_ref = weakref.ref(buf)
        src.buf_proxy = weakref.proxy(buf)
        src.buf_callable_proxy = weakref.proxy(buf)  # Buffer is callable
        self.assertIs(type(src.buf_callable_proxy), weakref.CallableProxyType)
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertEqual(copied, ALWAYS_PRESENT_COPY_ATTRS)
        for name in ("buf_ref", "buf_proxy", "buf_callable_proxy"):
            self.assertNotIn(name, vars(dst), name)

    def test_dead_weak_proxy_not_copied(self):
        class Buffer:
            pass

        buf = Buffer()
        src = _new_param()
        src.buf_proxy = weakref.proxy(buf)
        del buf
        gc.collect()
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertNotIn("buf_proxy", copied)

    def test_tensor_value_copied(self):
        src = _new_param()
        scale = paddle.ones([1])
        src.scale = scale
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertIn("scale", copied)
        self.assertIs(dst.scale, scale)

    def test_shared_value_mutation_visible_on_both(self):
        src = _new_param()
        src.color = {"color": "mp"}
        dst = _new_slice_param(src)

        copy_param_attrs(src, dst)
        src.color["group"] = "g0"

        self.assertEqual(dst.color, {"color": "mp", "group": "g0"})

    def test_dst_only_attrs_kept(self):
        src = _new_param()
        dst = _new_slice_param(src)
        dst.dst_only = "keep"
        dst.main_grad = paddle.zeros([1])
        main_grad = dst.main_grad
        src.main_grad = paddle.ones([2, 3])

        copy_param_attrs(src, dst)

        self.assertEqual(dst.dst_only, "keep")
        self.assertIs(dst.main_grad, main_grad)

    def test_copy_order_listed_first_then_sorted_external(self):
        src = _new_param()
        src.zeta = 1
        src.alpha = 2
        src.no_sync = True
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst, copy_attrs=("no_sync", "need_clip"))

        self.assertEqual(copied, ["no_sync", "need_clip", "alpha", "zeta"])

    def test_external_name_also_in_copy_attrs_copied_once(self):
        src = _new_param()
        src.tag = 1
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst, copy_attrs=("tag",))

        self.assertEqual(copied, ["tag"])

    def test_accepts_any_iterable_for_copy_attrs_and_exclude(self):
        src = _new_param()
        src.no_sync = True
        src.tag = 1
        src.other = 2
        dst = _new_slice_param(src)

        copied = copy_param_attrs(
            src,
            dst,
            copy_attrs=(n for n in ("no_sync", "need_clip")),
            exclude=["other", "main_grad"],
        )

        self.assertEqual(copied, ["no_sync", "need_clip", "tag"])
        self.assertNotIn("other", vars(dst))

    def test_empty_exclude_still_skips_tied_values(self):
        src = _new_param()
        src.main_grad = paddle.zeros([2, 3])
        src.buf_ref = weakref.ref(src)
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst, exclude=())

        self.assertIn("main_grad", copied)
        self.assertIs(dst.main_grad, src.main_grad)
        self.assertNotIn("buf_ref", copied)

    def test_dead_weakref_not_copied(self):
        class Buffer:
            pass

        buf = Buffer()
        src = _new_param()
        src.buf_ref = weakref.ref(buf)
        del buf
        gc.collect()
        self.assertIsNone(src.buf_ref())
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertNotIn("buf_ref", copied)

    def test_property_in_copy_attrs_copied(self):
        class MyParam(EagerParamBase):
            @property
            def is_firstly_shared(self):
                return True

        with unique_name.guard():
            src = MyParam(shape=[2, 3], dtype=paddle.float32)
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertIn("is_firstly_shared", copied)
        self.assertTrue(dst.is_firstly_shared)

    def test_subclass_init_attrs_counted_as_external(self):
        class MyParam(EagerParamBase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.expert_id = 3

        with unique_name.guard():
            src = MyParam(shape=[2, 3], dtype=paddle.float32)
        dst = _new_slice_param(src)

        copied = copy_param_attrs(src, dst)

        self.assertIn("expert_id", copied)
        self.assertEqual(dst.expert_id, 3)

    def test_layer_parameters(self):
        layer = paddle.nn.Linear(4, 3)
        for param in layer.parameters():
            with self.subTest(name=param.name):
                param.color = {"color": "dp"}
                param.main_grad = paddle.zeros(param.shape)
                dst = _new_slice_param(param)

                copied = copy_param_attrs(param, dst)

                self.assertEqual(copied, [*ALWAYS_PRESENT_COPY_ATTRS, "color"])
                self.assertIs(dst.color, param.color)
                self.assertNotIn("main_grad", vars(dst))

    def test_builtin_names_match_constructor(self):
        from paddle.distributed.fleet.utils.param_utils import (
            _builtin_attr_names,
        )

        with unique_name.guard():
            fresh = EagerParamBase(shape=[4], dtype=paddle.float16, name="x")
        self.assertEqual(_builtin_attr_names(), frozenset(vars(fresh)))
        self.assertTrue(set(ALWAYS_PRESENT_COPY_ATTRS) <= _builtin_attr_names())
        self.assertTrue(set(NOT_COPIED_BUILTIN_ATTRS) <= _builtin_attr_names())


# Each class's _create_slice_param only touches self._slice_params, so it can
# be called on a stub instead of a fully constructed optimizer.
CREATE_SLICE_PARAM_OWNERS = (
    dygraph_sharding_optimizer.DygraphShardingOptimizerV2,
    mixed_sharding_optimizer.MixedShardingOptimizer,
    muon_sharding_optimizer.MuonShardingOptimizer,
)


class TestCreateSliceParam(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def _create(self, cls, param):
        stub = types.SimpleNamespace(_slice_params={})
        slice_param = cls._create_slice_param(stub, param)
        self.assertIs(stub._slice_params[param.name], slice_param)
        return slice_param

    def test_copies_original_and_external_attrs(self):
        for cls in CREATE_SLICE_PARAM_OWNERS:
            with self.subTest(cls=cls.__name__):
                param = _new_param()
                param.optimize_attr = {"learning_rate": 0.5}
                param.need_clip = False
                param.is_distributed = True
                param.no_sync = True
                param.is_firstly_shared = True
                color = {"color": "moe_expert"}
                param.color = color
                param.split_axis = 1

                slice_param = self._create(cls, param)

                self.assertEqual(slice_param.name, param.name)
                self.assertEqual(slice_param.shape, [1])
                for name in ORIGINAL_COPY_ATTR_LIST:
                    self.assertIs(
                        getattr(slice_param, name), getattr(param, name), name
                    )
                self.assertIs(slice_param.color, color)
                self.assertEqual(slice_param.split_axis, 1)

    def test_does_not_copy_grad_storage_or_tied_values(self):
        class Buffer:
            pass

        for cls in CREATE_SLICE_PARAM_OWNERS:
            with self.subTest(cls=cls.__name__):
                buf = Buffer()
                param = _new_param()
                param.main_grad = paddle.zeros([2, 3])
                param.comm_buffer_ref = weakref.ref(buf)
                param.my_buf_ref = weakref.ref(buf)
                param.get_buf = (lambda self: self).__get__(param)
                param.regularizer = object()

                slice_param = self._create(cls, param)

                for name in (
                    "main_grad",
                    "comm_buffer_ref",
                    "my_buf_ref",
                    "get_buf",
                ):
                    self.assertNotIn(name, vars(slice_param), name)
                self.assertIsNot(slice_param.regularizer, param.regularizer)

    def test_does_not_modify_param(self):
        for cls in CREATE_SLICE_PARAM_OWNERS:
            with self.subTest(cls=cls.__name__):
                param = _new_param()
                param.main_grad = paddle.zeros([2, 3])
                param.color = {"color": "mp"}
                before = dict(vars(param))

                self._create(cls, param)

                self.assertEqual(vars(param).keys(), before.keys())
                for name, value in before.items():
                    self.assertIs(vars(param)[name], value, name)

    def test_v2_rejects_duplicate_param_name(self):
        cls = dygraph_sharding_optimizer.DygraphShardingOptimizerV2
        param = _new_param()
        stub = types.SimpleNamespace(_slice_params={})
        cls._create_slice_param(stub, param)

        with self.assertRaises(AssertionError):
            cls._create_slice_param(stub, param)


if __name__ == "__main__":
    unittest.main()
