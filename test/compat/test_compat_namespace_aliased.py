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

import inspect
import pathlib
import sys
import unittest
from contextlib import contextmanager
from functools import wraps
from unittest import mock

import numpy as np

import paddle
from paddle.compat import api_dispatch
from paddle.compat.api_dispatch import _PADDLE_NAMESPACE_SAVED
from paddle.compat.proxy import TORCH_PROXY_FINDER

sys.path.append(str(pathlib.Path(__file__).parent / "fake_modules"))


@contextmanager
def level2_guard():
    paddle.enable_compat(level=2)
    try:
        yield
    finally:
        paddle.disable_compat()


def with_level2(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with level2_guard():
            return func(*args, **kwargs)

    return wrapper


class CompatNamespaceAliasBase(unittest.TestCase):
    """Snapshot the relevant native symbols and the proxy-finder state so every
    test reverts the global namespace/finder no matter how it ends."""

    # (module, attr) pairs that should be perfectly restored after disable.
    EXISTING = [
        (paddle, "sort"),
        (paddle, "split"),
        (paddle, "min"),
        (paddle, "max"),
        (paddle, "unique"),
        (paddle, "median"),
        (paddle, "nanmedian"),
        (paddle, "allclose"),
        (paddle, "equal"),
        (paddle, "seed"),
        (paddle.Tensor, "numel"),
        (paddle.Tensor, "type"),
        (paddle.Tensor, "is_sparse"),
        (paddle.nn, "AvgPool1d"),
        (paddle.nn, "AvgPool2d"),
        (paddle.nn, "AvgPool3d"),
        (paddle.nn, "BatchNorm1d"),
        (paddle.nn, "BatchNorm2d"),
        (paddle.nn, "BatchNorm3d"),
        (paddle.nn, "MultiheadAttention"),
    ]
    # Compat-only symbols must not be added to the paddle namespace.
    COMPAT_ONLY = [
        (paddle, "slogdet"),
    ]

    def setUp(self):
        # Dispatch is device-agnostic; run on CPU to dodge a DCU GPU-op hang
        # (attention/sdpa). Original device restored in tearDown.
        self._device = paddle.get_device()
        paddle.set_device('cpu')
        while TORCH_PROXY_FINDER in sys.meta_path or _PADDLE_NAMESPACE_SAVED:
            paddle.disable_compat()
        self._native = {(m, a): getattr(m, a) for (m, a) in self.EXISTING}
        self._scope = set(TORCH_PROXY_FINDER._local_enabled_scope)
        self._global = TORCH_PROXY_FINDER._globally_enabled

    def tearDown(self):
        while TORCH_PROXY_FINDER in sys.meta_path or _PADDLE_NAMESPACE_SAVED:
            paddle.disable_compat()
        TORCH_PROXY_FINDER._local_enabled_scope = set(self._scope)
        TORCH_PROXY_FINDER._globally_enabled = self._global
        paddle.set_device(self._device)

    def assertNativeRestored(self):
        for (m, a), native in self._native.items():
            self.assertIs(
                getattr(m, a), native, f"{m.__name__}.{a} not restored"
            )
        for m, a in self.COMPAT_ONLY:
            self.assertFalse(
                hasattr(m, a), f"{m.__name__}.{a} should not be added"
            )

    def assertAliased(self, paddle_attr, compat_attr):
        """Check the dispatcher or class proxy points to the compat API."""
        target = getattr(
            paddle_attr,
            "__compat_fn__",
            getattr(paddle_attr, "__compat_cls__", paddle_attr),
        )
        self.assertIs(target, compat_attr)


class TestTopLevelAlias(CompatNamespaceAliasBase):
    def test_public_level_parameter_is_minimal(self):
        enable_parameters = list(
            inspect.signature(paddle.enable_compat).parameters
        )
        self.assertEqual(enable_parameters[-1], "level")
        self.assertEqual(
            inspect.signature(paddle.enable_compat).parameters["level"].default,
            1,
        )
        self.assertNotIn(
            "level", inspect.signature(paddle.use_compat_guard).parameters
        )

    def test_invalid_level_has_no_side_effect(self):
        with self.assertRaisesRegex(ValueError, "Unsupported level: 4"):
            paddle.enable_compat(level=4)
        self.assertNotIn(TORCH_PROXY_FINDER, sys.meta_path)
        self.assertNativeRestored()

    def test_top_level_symbols_aliased_on_enable(self):
        paddle.enable_compat(level=2)
        self.assertAliased(paddle.sort, paddle.compat.sort)
        self.assertAliased(paddle.split, paddle.compat.split)
        self.assertAliased(paddle.min, paddle.compat.min)
        self.assertAliased(paddle.max, paddle.compat.max)
        self.assertAliased(paddle.unique, paddle.compat.unique)
        self.assertAliased(paddle.median, paddle.compat.median)
        self.assertAliased(paddle.nanmedian, paddle.compat.nanmedian)
        self.assertAliased(paddle.allclose, paddle.compat.allclose)
        self.assertAliased(paddle.equal, paddle.compat.equal)
        self.assertAliased(paddle.seed, paddle.compat.seed)
        paddle.disable_compat()
        self.assertNativeRestored()

    def test_level1_default_does_not_alias(self):
        """level=1 (the default) installs only the torch proxy and must NOT alias
        paddle.*; the namespace dispatch is opt-in via level=2."""
        paddle.enable_compat()  # default level=1
        try:
            self.assertIs(paddle.sort, self._native[(paddle, "sort")])
            self.assertIs(
                paddle.Tensor.numel, self._native[(paddle.Tensor, "numel")]
            )
            self.assertIs(
                paddle.Tensor.type, self._native[(paddle.Tensor, "type")]
            )
            self.assertIs(
                paddle.Tensor.is_sparse,
                self._native[(paddle.Tensor, "is_sparse")],
            )
            self.assertFalse(hasattr(paddle.sort, "__compat_fn__"))
            with self.assertRaises(TypeError):
                paddle.sort(
                    paddle.to_tensor([3.0, 1.0]), dim=-1
                )  # native: no dim=
        finally:
            paddle.disable_compat()
        self.assertNativeRestored()

    def test_class_type_compatibility_without_alias(self):
        constructor_args = {
            "Unfold": (1,),
            "Linear": (2, 2),
            "Softmax": (),
            "AvgPool1D": (1,),
            "AvgPool2D": (1,),
            "AvgPool3D": (1,),
            "BatchNorm1D": (2,),
            "BatchNorm2D": (2,),
            "BatchNorm3D": (2,),
            "SmoothL1Loss": (),
            "MultiheadAttention": (4, 1),
            "Categorical": (paddle.to_tensor([0.5, 0.5]),),
        }
        native_classes = {
            "Unfold": paddle.nn.Unfold,
            "Linear": paddle.nn.Linear,
            "Softmax": paddle.nn.Softmax,
            "AvgPool1D": paddle.nn.AvgPool1D,
            "AvgPool2D": paddle.nn.AvgPool2D,
            "AvgPool3D": paddle.nn.AvgPool3D,
            "BatchNorm1D": paddle.nn.BatchNorm1D,
            "BatchNorm2D": paddle.nn.BatchNorm2D,
            "BatchNorm3D": paddle.nn.BatchNorm3D,
            "SmoothL1Loss": paddle.nn.SmoothL1Loss,
            "MultiheadAttention": paddle.nn.MultiHeadAttention,
            "Categorical": paddle.distributions.Categorical,
        }
        compat_modules = (paddle.compat.nn, paddle.compat.distributions)
        compat_classes = {
            getattr(module, name)
            for module in compat_modules
            for name in module.__all__
            if isinstance(getattr(module, name), type)
        }
        self.assertEqual(
            {compat_cls.__name__ for compat_cls in compat_classes},
            set(native_classes),
        )

        for compat_cls in compat_classes:
            name = compat_cls.__name__
            with self.subTest(name=name):
                native_cls = native_classes[name]
                compat = compat_cls(*constructor_args[name])
                self.assertIsInstance(compat, native_cls)
                self.assertTrue(issubclass(compat_cls, native_cls))
                native = native_cls(*constructor_args[name])
                self.assertNotIsInstance(native, compat_cls)

    @with_level2
    def test_submodule_symbols_aliased(self):
        self.assertAliased(paddle.nn.Linear, paddle.compat.nn.Linear)
        self.assertAliased(paddle.nn.Softmax, paddle.compat.nn.Softmax)
        self.assertAliased(paddle.nn.Unfold, paddle.compat.nn.Unfold)
        self.assertAliased(
            paddle.nn.functional.pad, paddle.compat.nn.functional.pad
        )
        self.assertAliased(
            paddle.nn.functional.linear, paddle.compat.nn.functional.linear
        )
        self.assertAliased(
            paddle.distributions.categorical.Categorical,
            paddle.compat.distributions.categorical.Categorical,
        )

    @with_level2
    def test_aliased_signatures_are_torch_style(self):
        self.assertEqual(
            inspect.signature(paddle.sort),
            inspect.signature(paddle.compat.sort),
        )
        self.assertEqual(
            inspect.signature(paddle.nn.Linear),
            inspect.signature(paddle.compat.nn.Linear),
        )

    def test_compat_only_symbols_are_not_added(self):
        paddle.enable_compat(level=2)
        try:
            for module, name in self.COMPAT_ONLY:
                self.assertFalse(hasattr(module, name))
        finally:
            paddle.disable_compat()

    def test_all_avgpool_aliases(self):
        paddle.enable_compat(level=2)
        try:
            for a in ("AvgPool1D", "AvgPool2D", "AvgPool3D"):
                self.assertAliased(
                    getattr(paddle.nn, a), getattr(paddle.compat.nn, a)
                )
        finally:
            paddle.disable_compat()

    def test_torch_style_nn_class_aliases(self):
        aliases = (
            ("AvgPool1d", "AvgPool1D", (1,)),
            ("AvgPool2d", "AvgPool2D", (1,)),
            ("AvgPool3d", "AvgPool3D", (1,)),
            ("BatchNorm1d", "BatchNorm1D", (2,)),
            ("BatchNorm2d", "BatchNorm2D", (2,)),
            ("BatchNorm3d", "BatchNorm3D", (2,)),
            ("MultiheadAttention", "MultiHeadAttention", (4, 1)),
        )

        for alias, native, _ in aliases:
            self.assertIn(alias, paddle.nn.__all__)
            self.assertIs(getattr(paddle.nn, alias), getattr(paddle.nn, native))

        with level2_guard():
            for alias, _, args in aliases:
                compat_cls = getattr(paddle.compat.nn, alias)
                self.assertAliased(getattr(paddle.nn, alias), compat_cls)
                self.assertIs(
                    type(getattr(paddle.nn, alias)(*args)),
                    compat_cls,
                )

        for alias, native, _ in aliases:
            self.assertIs(getattr(paddle.nn, alias), getattr(paddle.nn, native))

    @with_level2
    def test_all_nn_functional_aliases(self):
        """Every public compat nn.functional symbol overrides its paddle target,
        including softmax/log_softmax/scaled_dot_product_attention (NOT no-ops)."""
        F, cF = paddle.nn.functional, paddle.compat.nn.functional
        for a in (
            "pad",
            "softmax",
            "log_softmax",
            "linear",
            "scaled_dot_product_attention",
            "unfold",
        ):
            self.assertAliased(getattr(F, a), getattr(cF, a))


class TestAliasBehavior(CompatNamespaceAliasBase):
    def test_sort_returns_namedtuple(self):
        t = paddle.to_tensor([[3.0, 1.0, 2.0], [9.0, 7.0, 8.0]])
        with level2_guard():
            out = paddle.sort(t, dim=-1)
            self.assertTrue(hasattr(out, "values"))
            self.assertTrue(hasattr(out, "indices"))
            self.assertEqual(out.values[0].tolist(), [1.0, 2.0, 3.0])
        # native again: returns a plain Tensor, not the compat namedtuple
        self.assertIs(paddle.sort, self._native[(paddle, "sort")])
        self.assertNotIsInstance(paddle.sort(t), tuple)

    @with_level2
    def test_min_max_no_recursion(self):
        """Reduce-all min/max self-call the native impl; must not RecursionError."""
        t = paddle.to_tensor([3.0, 1.0, 2.0])
        self.assertEqual(paddle.min(t).item(), 1.0)
        self.assertEqual(paddle.max(t).item(), 3.0)
        # dim form returns a namedtuple (values, indices)
        r = paddle.min(t, dim=0)
        self.assertEqual(r.values.item(), 1.0)
        self.assertEqual(r.indices.item(), 1)

    @with_level2
    def test_median_unique_allclose_no_recursion(self):
        t = paddle.to_tensor([[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]])
        med = paddle.median(t, dim=1)
        self.assertTrue(hasattr(med, "values"))
        u = paddle.unique(paddle.to_tensor([2, 3, 3, 1]))
        self.assertEqual(u.tolist(), [1, 2, 3])
        # compat.allclose returns a python bool
        res = paddle.allclose(t, t)
        self.assertIsInstance(res, bool)
        self.assertTrue(res)

    @with_level2
    def test_seed_no_recursion(self):
        s = paddle.seed()  # compat.seed takes no arg and returns an int
        self.assertIsInstance(s, int)

    @with_level2
    def test_aliased_wrappers_execute(self):
        """Broader behavioral coverage of the self-ref-fixed / aliased wrappers
        that were previously only checked by identity."""
        t = paddle.to_tensor([[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]])
        # nanmedian dim-path reaches the native paddle.nanmedian internally
        nm = paddle.nanmedian(
            paddle.to_tensor([[1.0, float("nan")], [3.0, 4.0]]), dim=0
        )
        self.assertTrue(hasattr(nm, "values"))
        # split: torch semantics (per-chunk size), returns a tuple
        parts = paddle.split(t, 1, dim=0)
        self.assertIsInstance(parts, tuple)
        self.assertEqual(len(parts), 2)
        # equal returns a python bool (like allclose)
        self.assertIsInstance(paddle.equal(t, t), bool)
        # nn.functional: pad / linear / unfold
        self.assertEqual(paddle.nn.functional.pad(t, [1, 1]).shape, [2, 5])
        y = paddle.nn.functional.linear(
            t, paddle.ones([4, 3]), paddle.zeros([4])
        )
        self.assertEqual(y.shape, [2, 4])
        img = paddle.randn([1, 1, 4, 4])
        self.assertEqual(paddle.nn.functional.unfold(img, 2).shape[:2], [1, 4])
        # nn.Unfold layer forward routes nn.functional.unfold to native
        self.assertEqual(paddle.nn.Unfold(kernel_size=2)(img).shape[0], 1)

    @with_level2
    def test_nn_functional_softmax_logsoftmax_sdpa_execute(self):
        """Direct execution of the aliased softmax/log_softmax/sdpa (these are
        genuinely aliased to torch-style compat APIs, not no-ops). After enable,
        paddle.nn.functional.softmax is the torch-style compat API, so it takes
        `dim=` and rejects the paddle-native `axis=`."""
        x = paddle.randn([2, 4])
        self.assertEqual(paddle.nn.functional.softmax(x, dim=-1).shape, [2, 4])
        self.assertEqual(
            paddle.nn.functional.log_softmax(x, dim=-1).shape, [2, 4]
        )
        # paddle-native axis= is rejected once aliased (torch-style contract)
        with self.assertRaises(TypeError):
            paddle.nn.functional.softmax(x, axis=-1)
        q = paddle.randn([2, 2, 5, 4])
        out = paddle.nn.functional.scaled_dot_product_attention(q, q, q)
        self.assertEqual(out.shape, [2, 2, 5, 4])

    def test_sdpa_numeric_correct_under_compat(self):
        """Regression for caller-aware dispatch: at level=2 paddle.nn.functional.softmax
        dispatches to the torch-style compat softmax for *external* code, but paddle's
        own sdpa math backend (``_math_attention``) calls F.softmax internally and must
        get the NATIVE softmax (last/key axis). fp32 forces the math backend, so a
        wrong-axis softmax would diverge from this hand-computed softmax(QK^T/sqrt(d))V
        reference."""
        import numpy as np

        rng = np.random.RandomState(0)
        x = rng.rand(2, 4, 8, 16).astype("float32")
        d = x.shape[-1]
        s = (x @ np.swapaxes(x, -1, -2)) / np.sqrt(d)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        ref = (e / e.sum(-1, keepdims=True)) @ x
        with level2_guard():
            q = paddle.to_tensor(x)
            out = paddle.nn.functional.scaled_dot_product_attention(q, q, q)
            np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-4)


class TestScopeAndLifecycle(CompatNamespaceAliasBase):
    def test_scoped_level2_enable_aliases(self):
        paddle.enable_compat(scope={"triton"}, level=2, silent=True)
        try:
            self.assertNotIn(TORCH_PROXY_FINDER, sys.meta_path)
            self.assertAliased(paddle.sort, paddle.compat.sort)
        finally:
            paddle.disable_compat()
        self.assertNativeRestored()

    def test_level1_repeated_enable_preserves_legacy_finders(self):
        paddle.enable_compat()
        paddle.enable_compat(scope={"triton"}, silent=True)
        self.assertEqual(sys.meta_path.count(TORCH_PROXY_FINDER), 2)
        paddle.disable_compat()
        self.assertEqual(sys.meta_path.count(TORCH_PROXY_FINDER), 1)
        paddle.disable_compat()
        self.assertNotIn(TORCH_PROXY_FINDER, sys.meta_path)
        self.assertNativeRestored()

    def test_registry_empty_after_disable(self):
        self.assertEqual(len(_PADDLE_NAMESPACE_SAVED), 0)
        paddle.enable_compat(level=2)
        self.assertGreater(len(_PADDLE_NAMESPACE_SAVED), 0)
        paddle.disable_compat()
        self.assertEqual(len(_PADDLE_NAMESPACE_SAVED), 0)


class TestTorchSurfaceUnderCompat(CompatNamespaceAliasBase):
    """torch.* reaches public compat APIs only at proxy-enabled levels."""

    @staticmethod
    def _drop_torch_modules():
        for name in [
            m
            for m in list(sys.modules)
            if m == "torch" or m.startswith("torch.")
        ]:
            del sys.modules[name]

    def test_level1_root_torch_apis_resolve_to_compat(self):
        paddle.enable_compat()
        try:
            self._drop_torch_modules()
            import torch

            self.assertIs(torch.sort, paddle.compat.sort)
            self.assertIs(torch.min, paddle.compat.min)
            self.assertIs(torch.unique, paddle.compat.unique)
        finally:
            self._drop_torch_modules()
            paddle.disable_compat()

    def test_root_torch_apis_resolve_to_compat_at_level3(self):
        paddle.enable_compat(level=3)
        try:
            self._drop_torch_modules()
            import torch

            self.assertIs(torch.sort, paddle.compat.sort)
            self.assertIs(torch.min, paddle.compat.min)
            self.assertIs(torch.unique, paddle.compat.unique)
            # nn.* overrides reach compat regardless of the alias.
            self.assertIs(torch.nn.Linear, paddle.compat.nn.Linear)
        finally:
            self._drop_torch_modules()
            paddle.disable_compat()

    def test_root_compat_only_api_is_registered_at_proxy_levels(self):
        for level in (1, 3):
            with self.subTest(level=level):
                paddle.enable_compat(level=level)
                try:
                    self._drop_torch_modules()
                    import torch

                    self.assertFalse(hasattr(paddle, "slogdet"))
                    self.assertIs(torch.slogdet, paddle.compat.slogdet)
                    result = torch.slogdet(paddle.eye(2))
                    self.assertEqual(result.sign.item(), 1.0)
                    self.assertEqual(result.logabsdet.item(), 0.0)
                finally:
                    self._drop_torch_modules()
                    paddle.disable_compat()


class TestLevel2InternalCallersUseNative(CompatNamespaceAliasBase):
    """level=2 redirects ``paddle.*`` to the torch-aligned compat APIs for the
    USER's code, but paddle's own internals call these same APIs (F.softmax /
    F.linear / paddle.max / ...) with native ``axis=``/``name=`` kwargs and native
    defaults. Caller-aware dispatch must route those internal calls to native, or
    composite native layers break under level=2."""

    @with_level2
    def test_native_composite_layers_run_under_level2(self):
        # Each of these calls F.linear / F.softmax / paddle.* internally with
        # native kwargs; under level=2 they must still work (internal -> native).
        x = paddle.randn([2, 5, 8])
        self.assertEqual(
            paddle.nn.TransformerEncoderLayer(8, 2, 16)(x).shape, [2, 5, 8]
        )
        mha = paddle.nn.MultiHeadAttention(
            8, 2
        )  # native (capital H), not compat
        self.assertEqual(mha(x, x, x).shape, [2, 5, 8])
        self.assertEqual(paddle.nn.LayerNorm(8)(x).shape, [2, 5, 8])

    @with_level2
    def test_external_caller_still_gets_compat(self):
        # This test module is not a ``paddle.*`` module, so it still sees the
        # torch-aligned compat API: torch-style ``dim=`` + namedtuple return, and
        # the native ``axis=`` kwarg is rejected by the torch-style contract.
        t = paddle.to_tensor([[3.0, 1.0, 2.0]])
        self.assertTrue(hasattr(paddle.sort(t, dim=-1), "values"))
        with self.assertRaises(TypeError):
            paddle.max(t, axis=1)

    def test_tensor_methods_caller_aware(self):
        """torch exposes max/min/sort/split/... as Tensor methods too; under
        level=2 ``x.max(dim=1)`` is torch-style for external callers and native
        for paddle-internal ``x.max(axis=1)``; disable restores the namespace."""
        native_max = paddle.Tensor.max
        native_split = paddle.Tensor.split
        native_numel = paddle.Tensor.numel
        native_type = paddle.Tensor.type
        native_is_sparse = paddle.Tensor.is_sparse
        t = paddle.to_tensor([[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]])
        with level2_guard():
            r = t.max(dim=1)  # external -> compat namedtuple
            self.assertTrue(hasattr(r, "values"))
            self.assertEqual(r.values.tolist(), [3.0, 6.0])
            self.assertIsInstance(t.split(1, dim=0), tuple)
            self.assertEqual(len(t.split(split_size=1, dim=0)), 2)
            self.assertEqual(
                len(t.split(split_size=[1, 1], dim=0)),
                2,
            )
            self.assertEqual(len(paddle.split(t, split_size=1, dim=0)), 2)
            self.assertEqual(t.numel(), 6)
            self.assertIs(type(t.numel()), int)
            self.assertIsInstance(paddle.numel(t), paddle.Tensor)
            self.assertEqual(paddle.empty([0, 3]).numel(), 0)
            self.assertEqual(t.cpu().type(), "torch.FloatTensor")
            self.assertEqual(t.type(paddle.float64).dtype, paddle.float64)
            self.assertEqual(t.type(paddle.DoubleTensor).dtype, paddle.float64)
            self.assertEqual(t.type("torch.DoubleTensor").dtype, paddle.float64)
            self.assertEqual(
                t.type("paddle.DoubleTensor").dtype, paddle.float64
            )
            self.assertEqual(t.type("torch.float64").dtype, paddle.float64)
            self.assertEqual(t.type("paddle.float64").dtype, paddle.float64)
            self.assertIs(t.type(paddle.float32), t)
            self.assertIs(t.type("torch.FloatTensor"), t)
            self.assertIs(t.type("paddle.FloatTensor"), t)
            self.assertIs(t.type(t.type()), t)
            with self.assertRaises(ValueError):
                t.type("float64")
            self.assertEqual(
                paddle.ones([1], dtype="int64").cpu().type(),
                "torch.LongTensor",
            )
            self.assertEqual(
                paddle.ones([1], dtype="float8_e4m3fn").cpu().type(),
                "torch.Float8_e4m3fnTensor",
            )
            self.assertEqual(
                paddle.ones([1], dtype="float8_e5m2").cpu().type(),
                "torch.Float8_e5m2Tensor",
            )
            if paddle.device.is_compiled_with_cuda():
                self.assertEqual(
                    paddle.ones([1]).cuda().type(), "torch.cuda.FloatTensor"
                )
            self.assertIs(t.is_sparse, False)
            coo = paddle.sparse.sparse_coo_tensor([[0], [1]], [1.0], [2, 2])
            csr = paddle.sparse.sparse_csr_tensor([0, 1, 1], [0], [1.0], [2, 2])
            self.assertIs(coo.is_sparse, True)
            self.assertIs(csr.is_sparse, False)
            # paddle-internal native-style call (simulated) stays native
            ns = {"__name__": "paddle.fake_internal", "t": t}
            exec(
                "internal_max = t.max(axis=1)\n"
                "internal_split = t.split(num_or_sections=2, axis=0)\n"
                "internal_numel = t.numel()\n"
                "internal_type = t.type\n"
                "internal_is_sparse = t.is_sparse()",
                ns,
            )
            self.assertIsInstance(ns["internal_numel"], paddle.Tensor)
            self.assertEqual(
                ns["internal_type"], native_type.__get__(t, paddle.Tensor)
            )
            self.assertIs(ns["internal_is_sparse"], False)
            cached_numel = t.numel
            cached_max = t.max
            ns["cached_max"] = cached_max
            exec("internal_cached_max = cached_max(axis=1)", ns)
            self.assertIsInstance(ns["internal_cached_max"], paddle.Tensor)
        self.assertIs(paddle.Tensor.max, native_max)  # restored on disable
        self.assertIs(paddle.Tensor.split, native_split)
        self.assertIs(paddle.Tensor.numel, native_numel)
        self.assertIs(paddle.Tensor.type, native_type)
        self.assertIs(paddle.Tensor.is_sparse, native_is_sparse)
        self.assertEqual(cached_numel(), 6)
        self.assertTrue(hasattr(cached_max(dim=1), "values"))

    @with_level2
    def test_tensor_type_edge_cases(self):
        t = paddle.ones([1])
        with mock.patch.object(
            paddle.Tensor, "to", autospec=True, return_value=t
        ) as tensor_to:
            self.assertIs(
                t.type(paddle.float64, **{"async": True}),
                t,
            )
            tensor_to.assert_called_once_with(
                t,
                device=None,
                dtype=paddle.float64,
                blocking=False,
            )
        with self.assertRaisesRegex(
            TypeError, "unexpected keyword argument 'invalid'"
        ):
            t.type(invalid=True)
        with self.assertRaisesRegex(ValueError, "invalid type"):
            t.type("paddle.UnknownTensor")

        with mock.patch.object(
            paddle.Tensor, "to", autospec=True, return_value=t
        ) as tensor_to:
            self.assertIs(t.type("paddle.cuda.DoubleTensor"), t)
            tensor_to.assert_called_once_with(
                t,
                device="gpu",
                dtype=paddle.float64,
                blocking=True,
            )

        for factory, expected_device in (
            (paddle.DoubleTensor, "cpu"),
            (paddle.cuda.DoubleTensor, "gpu"),
        ):
            with mock.patch.object(
                paddle.Tensor, "to", autospec=True, return_value=t
            ) as tensor_to:
                self.assertIs(t.type(factory), t)
                tensor_to.assert_called_once_with(
                    t,
                    device=expected_device,
                    dtype=paddle.float64,
                    blocking=True,
                )

        coo = paddle.sparse.sparse_coo_tensor(
            [[0], [0]], [1.0], [1, 1], place='cpu'
        )
        self.assertEqual(coo.type(), "torch.sparse.FloatTensor")
        self.assertEqual(t.type(np.float64).dtype, paddle.float64)

        # Round-trip: XPU device string → correct device parameter
        with mock.patch.object(
            paddle.Tensor, "to", autospec=True, return_value=t
        ) as tensor_to:
            self.assertIs(t.type("torch.xpu.FloatTensor"), t)
            tensor_to.assert_called_once_with(
                t,
                device="xpu",
                dtype=paddle.float32,
                blocking=True,
            )

        # Round-trip: custom device string → correct device parameter
        with mock.patch.object(
            paddle.Tensor, "to", autospec=True, return_value=t
        ) as tensor_to:
            self.assertIs(t.type("torch.npu.FloatTensor"), t)
            tensor_to.assert_called_once_with(
                t,
                device="npu",
                dtype=paddle.float32,
                blocking=True,
            )

        # Round-trip: xpu + sparse device string
        with mock.patch.object(
            paddle.Tensor, "to", autospec=True, return_value=t
        ) as tensor_to:
            self.assertIs(t.type("torch.xpu.sparse.FloatTensor"), t)
            tensor_to.assert_called_once_with(
                t,
                device="xpu",
                dtype=paddle.float32,
                blocking=True,
            )

        # Round-trip: same-device no-op for XPU string via mock
        with mock.patch.object(
            paddle.Tensor, "to", autospec=True, return_value=t
        ) as tensor_to:
            with mock.patch.object(
                type(t.place), "is_xpu_place", return_value=True
            ):
                t.type("torch.xpu.FloatTensor")
                tensor_to.assert_not_called()

    @with_level2
    def test_tensor_type_name_devices(self):
        class _FakePlace:
            def __init__(self, kind, custom_device_type=None):
                self.kind = kind
                self._custom_device_type = custom_device_type

            def is_gpu_place(self):
                return self.kind == "gpu"

            def is_xpu_place(self):
                return self.kind == "xpu"

            def is_custom_place(self):
                return self.kind == "custom"

            def custom_device_type(self):
                return self._custom_device_type

        class _FakeTensor:
            def __init__(self, place):
                self.dtype = paddle.float32
                self.place = place

            def is_sparse_coo(self):
                return False

        cases = [
            (_FakePlace("cpu"), "torch.FloatTensor"),
            (_FakePlace("gpu"), "torch.cuda.FloatTensor"),
            (_FakePlace("xpu"), "torch.xpu.FloatTensor"),
            (_FakePlace("custom", "NPU"), "torch.npu.FloatTensor"),
        ]
        for place, expected in cases:
            self.assertEqual(
                paddle.compat._tensor_type_name(_FakeTensor(place)), expected
            )

    @with_level2
    def test_tensor_descriptor_class_access(self):
        type_descriptor = inspect.getattr_static(paddle.Tensor, "type")
        sparse_descriptor = inspect.getattr_static(paddle.Tensor, "is_sparse")

        self.assertIs(paddle.Tensor.type, type_descriptor.__compat_fn__)
        self.assertIsInstance(paddle.Tensor.is_sparse, property)
        self.assertIs(
            paddle.Tensor.is_sparse.fget,
            sparse_descriptor.__compat_fn__,
        )

        ns = {"__name__": "paddle.fake_internal", "paddle": paddle}
        exec(
            "internal_type = paddle.Tensor.type\n"
            "internal_is_sparse = paddle.Tensor.is_sparse",
            ns,
        )
        self.assertIs(ns["internal_type"], type_descriptor.__native_fn__)
        self.assertIs(ns["internal_is_sparse"], sparse_descriptor.__native_fn__)

    def test_property_to_property_dispatch(self):
        class Native:
            @property
            def attr(self):
                return "native"

        class Compat:
            @property
            def attr(self):
                return "compat"

        native_attr = inspect.getattr_static(Native, "attr")
        compat_attr = inspect.getattr_static(Compat, "attr")
        Native.attr = api_dispatch.dispatch_property(native_attr, compat_attr)
        instance = Native()

        self.assertEqual(instance.attr, "compat")
        self.assertIs(Native.attr, compat_attr)

        ns = {
            "__name__": "paddle.fake_internal",
            "Native": Native,
            "x": instance,
        }
        exec("value = x.attr\nattr = Native.attr", ns)
        self.assertEqual(ns["value"], "native")
        self.assertIs(ns["attr"], native_attr)

    def test_missing_tensor_override_is_skipped(self):
        missing_attr = "__missing_tensor_compat_override__"
        self.assertIsNone(
            inspect.getattr_static(paddle.Tensor, missing_attr, None)
        )
        with (
            mock.patch.object(paddle.compat, "__all__", ()),
            mock.patch.dict(
                paddle.compat._TENSOR_API_OVERRIDES,
                {missing_attr: mock.sentinel.compat_fn},
                clear=True,
            ),
        ):
            api_dispatch._patch_tensor_methods()
        self.assertNotIn((paddle.Tensor, missing_attr), _PADDLE_NAMESPACE_SAVED)

    @with_level2
    def test_aliased_class_caller_aware(self):
        """Existing classes (Linear/...) become caller-aware proxies: external
        callers get the torch-aligned compat class, paddle-internal callers get
        native."""
        # external (this module) -> compat class (torch-style)
        self.assertIs(type(paddle.nn.Linear(2, 2)), paddle.compat.nn.Linear)
        with self.assertRaises(TypeError):
            paddle.nn.Linear(2, 2, weight_attr=False)  # native kwarg rejected
        # paddle-internal caller (simulated) -> native class, accepts native kwarg
        ns = {"__name__": "paddle.fake_internal", "paddle": paddle}
        exec("obj = paddle.nn.Linear(2, 2, weight_attr=False)", ns)
        self.assertIsNot(type(ns["obj"]), paddle.compat.nn.Linear)
        # isinstance / issubclass accept both forms
        self.assertIsInstance(paddle.nn.Linear(2, 2), paddle.nn.Linear)
        self.assertTrue(issubclass(paddle.compat.nn.Linear, paddle.nn.Linear))

    @with_level2
    def test_aliased_class_subclassing_is_torch_style(self):
        """A user subclass derived from the alias class under level=2 uses the
        torch-style (compat) constructor, not a silent fallback to native."""

        class MyLinear(paddle.nn.Linear):
            pass

        m = MyLinear(3, 4, bias=False)  # torch-style ``bias=`` must be accepted
        self.assertIsInstance(m, MyLinear)
        self.assertIsInstance(m, paddle.nn.Linear)
        self.assertTrue(issubclass(MyLinear, paddle.nn.Linear))


if __name__ == "__main__":
    unittest.main()
