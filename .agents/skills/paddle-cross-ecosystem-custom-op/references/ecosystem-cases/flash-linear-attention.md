# flash-linear-attention：控制面在 Python 包装层，内核零改动

纯 Triton 库没有 C++ 扩展，构建打包与 C++ compat 两层完全为空：diff 不含 setup/pyproject，也不触碰任何 `@triton.jit` 内核，库内实际改动约 160 行全部落在 Python 包装层，外加一套独立的 `tests/paddle/` 对拍测试。该 fork 采用「只适配子集」策略（分支名 `paddle/kda-compact` 即此意）：第一步就是把各级 `__init__.py` 的导出面从全库 50+ 算子裁到业务路径所需的 `chunk_kda`、`ShortConvolution`、`FusedRMSNormGated` 和 context-parallel 工具，从源头缩小需要验证的兼容范围。

### 第一落点

- 裁剪导出面：`fla/ops/__init__.py`（50+ 算子 → 3 个子模块）、`fla/modules/__init__.py`（24 个符号 → 2 个）等，并把 lazy optional import 改成 eager import（后者同时是 PaddleFleet 集成的前置要求）
- `fla/__init__.py` 作为 compat 入口：用 `paddle.compat.proxy._extend_torch_proxy_overrides` 覆盖 `torch.empty`——Triton 的 cache allocation 会调用 `torch.empty(..., device="cuda")`，覆盖函数剥掉 `device` kwarg 改用当前设备
- 各 autograd Function 的 backward 返回值按 PyLayer 语义重写

### 具体例子

PyTorch 的 `autograd.Function.backward` 要求对 forward 的每个参数返回一个梯度（非张量参数返回 None）；PyLayer 则要求返回值与实际传入的张量输入一一对应，None 的可选输入不占位。fork 的固定模板是 forward 里记 `ctx.has_*` 标志位、backward 里条件拼接（长期保留类别）：

```diff
--- fla/ops/kda/chunk.py
     def forward(ctx, q, k, v, g, beta, A_log, dt_bias, ...):
+        ctx.has_A_log = A_log is not None
+        ctx.has_dt_bias = dt_bias is not None
+        ctx.has_initial_state = initial_state is not None
+        ctx.has_cu_seqlens = cu_seqlens is not None
 ...
     def backward(ctx, do, dht):
-        return (dq.to(q), dk.to(k), dv.to(v), dg.to(g_input), db.to(beta_raw), dA, dbias, None, dh0,
-                None, None, None, None, None, None, None, None, None, None, None, None, None, None)
+        return (
+            (dq.to(q), dk.to(k), dv.to(v), dg.to(g_input), db.to(beta_raw))
+            + ((dA,) if ctx.has_A_log else ())
+            + ((dbias,) if ctx.has_dt_bias else ())
+            + ((dh0,) if ctx.has_initial_state else ())
+            + ((None,) if ctx.has_cu_seqlens else ())
+        )
```

同一模式机械地复制到 `fla/ops/kda/gate.py`、`fla/modules/conv/triton/ops.py`、`fla/modules/fused_norm_gate.py`——纯 Triton 库迁移中最高频、最可模板化的一类改动。compat 入口则展示了 proxy override 扩展点的用法，同时把上游的 lazy optional import 改成 eager import（override 属可吸纳类别；导出面裁剪与 eager import 长期保留）：

```diff
--- fla/__init__.py
+def _torch_compat_empty(*args, **kwargs):
+    if kwargs.get("device") == "cuda":
+        del kwargs["device"]
+    return paddle.empty(*args, **kwargs)
+
+paddle.compat.proxy._extend_torch_proxy_overrides(
+    {
+        "torch.empty": paddle.compat.proxy.RawOverriddenAttribute(
+            _torch_compat_empty
+        ),
+    }
+)
-_layers = _import_optional_public_module('fla.layers')
-_models = _import_optional_public_module('fla.models')
+from fla import modules, ops  # noqa: E402

-del _import_optional_public_module, _export_public_api, _layers, _models
+__all__ = ["modules", "ops"]
```

同文件里还有配套小改：`torch.is_inference_mode_enabled()` → `not paddle.is_grad_enabled()`、删除 `@torch.compiler.disable`。

### 优先查看的文件

- `fla/__init__.py`：compat 入口的最小写法——eager import 子集 + proxy override 修 Triton cache allocation 的 `torch.empty(device="cuda")`
- `fla/ops/kda/chunk.py`：autograd Function → PyLayer 的 backward 返回值裁剪全套模式
- `fla/utils/_device.py`：AMP 装饰器置空（`torch.amp.custom_fwd/custom_bwd` → no-op）、设备上下文的收敛写法
- `fla/ops/cp/comm.py`：`torch.distributed` 语义差异——`all_gather_into_tensor(async_op=)` 改为 `dist.all_gather(sync_op=True)` + `stack`
- `tests/paddle/test_kda.py`：`paddle.enable_compat(scope={"fla", "triton"}, silent=True)` 的作用域用法，numpy 造数 + eager 参考实现对拍

### 可复用结论

- 纯 Triton 库的迁移没有构建与 C++ 两层，控制面完全在 Python 包装层；`@triton.jit` 内核一行不改是可以达成的目标
- 大库先裁 `__init__.py` 导出面、只适配业务算子路径，是控制验证成本的第一步，也让后续 sync upstream 的冲突面最小
- PyLayer 的 backward 返回值 arity（可选输入不占位）是最高频的机械改动，可用 `ctx.has_*` 标志 + 条件拼接元组的固定模板批量处理
- 零散但可枚举的第二类落点：`torch.device` 类型注解需字符串化（proxy 类型在注解求值时不可用）、`div(rounding_mode='floor')` → `//`、`searchsorted` 的标量实参需先 `paddle.to_tensor`、`memory_format=torch.contiguous_format` 直接删除（多数属可吸纳类别，动手前先验证当前 compat 是否已覆盖）
