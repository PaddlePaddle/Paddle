# cudnn-frontend：控制面在 Python bindings 层

cudnn-frontend 是 header-only C++ 库 + Python bindings（上游 NVIDIA/cudnn-frontend）。迁移分支按上游版本号命名（`paddle/v1.24.0` → `paddle/v1.26.0`），base 是上游 main 上的版本 bump 提交。刨掉上游自身演进后，Paddle 适配只有 7 个提交、13 个文件，全部落在 `python/cudnn/`——`include/` 下的 header-only C++ 与 pybind 主体零改动。适配点集中在三类：dtype 映射的 feature-detect、stream/nvtx 等 runtime helper、以及为满足 PaddleFleet 集成前置要求的 eager import 改造。

### 第一落点

- `python/cudnn/datatypes.py`：dtype 映射表的 feature-detect 函数里 `import torch` → `import paddle as torch`（显式别名，不依赖 proxy 作用域）
- `python/cudnn/deepseek_sparse_attention/utils/runtime.py`：stream 裸指针、`nvtx.range` shim、`ExternalStream` 缺失时的 `get_stream_from_external` 回退
- `python/cudnn/__init__.py` 与 `DSANamespace`：lazy 的 `__getattr__` 符号加载改成 eager import（PaddleFleet 集成的前置要求）

### 具体例子

runtime helper 里的兼容 shim 全部带 `hasattr` 探测，compat 补齐后自动失效（可吸纳类别）：

```diff
--- python/cudnn/deepseek_sparse_attention/utils/runtime.py
+if hasattr(torch.cuda, "nvtx") and not hasattr(torch.cuda.nvtx, "range"):
+    @contextmanager
+    def _nvtx_range(msg: str) -> Iterator[None]:
+        torch.cuda.nvtx.range_push(msg)
+        try:
+            yield
+        finally:
+            torch.cuda.nvtx.range_pop()
+
+    torch.cuda.nvtx.range = _nvtx_range
 ...
-    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
+    return cuda.CUstream(torch.cuda.current_stream().stream_base.raw_stream)
```

eager import 改造直接对应 PaddleFleet 的硬约束——lazy 的 namespace 属性查找改成构造时全量加载（长期保留类别）：

```diff
--- python/cudnn/deepseek_sparse_attention/__init__.py
 class DSANamespace:
-    def __getattr__(self, name):
-        if name in _SYMBOLS:
-            return _load_symbol(name)
-        raise AttributeError(f"DSA has no attribute {name!r}")
+    # Make import all symbols eagerly
+    def __init__(self):
+        for symbol in _SYMBOLS:
+            setattr(self, _SYMBOLS[symbol][1], _load_symbol(symbol))
```

能力缺口的处理也停在单点：Paddle 暂无 FP4 dtype，`_is_fp4x2` 直接 `return False` 短路并注释原因（可吸纳类别：Paddle 支持 FP4 后删除）。

### 优先查看的文件

- `python/cudnn/datatypes.py`：看 dtype 映射的 feature-detect 别名写法
- `python/cudnn/deepseek_sparse_attention/utils/runtime.py`：看 nvtx/stream/ExternalStream 三个 shim
- `python/cudnn/deepseek_sparse_attention/utils/tensor_conversion.py`：看 `Tensor.dim_order` 缺失时按 stride 排序的 `_dim_order` 回退
- `python/cudnn/__init__.py`：看 eager import 的最小改造
- `python/cudnn/api_base.py`：看 `torch.device.Device` 类型判断与 FP4 短路

### 可复用结论

- header-only C++ + Python bindings 的库，适配面可以完全压在 Python 层，C++ 一行不动
- 迁移分支跟随上游版本号命名，拉新时基于新版本提交重放补丁开新分支（`paddle/v1.24.0` → `paddle/v1.26.0`），旧分支保留服务未升级的下游
- 比对这类分支时先在 fork 内定位 base 提交（`git log` 找 Compat 提交的父节点），用 `<base>...<迁移分支>` 做同仓 compare，避免把上游演进算进适配面
- 计划集成进 PaddleFleet 的库，eager import 改造在迁移阶段就一并做掉（lazy `__getattr__`、函数内 import 都要消除）
