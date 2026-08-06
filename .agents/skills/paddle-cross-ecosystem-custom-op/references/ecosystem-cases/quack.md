# quack：控制面在 PyLayer ctx shim 与 CuTe DSL 导入边界

quack 是纯 CuTe DSL（cutlass python）kernel 库，也是 supersonic-moe 的依赖。张量经 DLPack 进 kernel 的主路径在 Paddle compat 下开箱即用，完全没动；补丁集中在两处：一是 Python autograd 语义差异（PyLayer 的 ctx 缺 `needs_input_grad`、`saved_tensors`、`set_materialize_grads` 等属性），二是 CuTe DSL 特有的导入期陷阱——DSL 的 AST preprocessor 会反射 `torch` 命名空间的类型注解，compat 下 `torch.device` 是 ProxyModule 而非真正的类型。

### 第一落点

- `quack/__init__.py` 的 import-time shim 集合：`torch.compiler` 缺失兜底、`CustomOpDef.__call__`、PyLayerContext 属性补齐，全部带 `hasattr` 守卫（compat 层日后补齐即自动失效）
- PyLayer 语义差异：`needs_input_grad` 无原生对应，forward 里用各输入的 `stop_gradient` 记录到 ctx，再以 property 形式读回
- CuTe DSL runtime 边界：stream 用 `.stream_base.raw_stream` 取裸指针，device capability 查询前把 place 归一化为 int

### 具体例子

PyLayerContext 的属性补齐集中在 `quack/__init__.py`，逐条带 `hasattr` 守卫——compat 层日后补齐对应能力后 shim 自动短路，这就是删除条件（可吸纳类别）：

```diff
--- quack/__init__.py
+    # Paddle compat: ctx.saved_tensors (PyTorch property) -> ctx.saved_tensor() (Paddle method)
+    if not hasattr(_PyLayerCtxCls, "saved_tensors"):
+        _PyLayerCtxCls.saved_tensors = property(lambda self: self.saved_tensor())
+
+    # Paddle compat: ctx.set_materialize_grads (PyTorch) is not in Paddle PyLayer
+    if not hasattr(_PyLayerCtxCls, "set_materialize_grads"):
+        _PyLayerCtxCls.set_materialize_grads = lambda self, value: None
+
+    # Paddle compat: ctx.mark_non_differentiable (PyTorch) is not in Paddle PyLayer
+    if not hasattr(_PyLayerCtxCls, "mark_non_differentiable"):
+        _PyLayerCtxCls.mark_non_differentiable = lambda self, *tensors: None
```

CuTe DSL 特有的导入期陷阱：DSL 的 AST preprocessor 处理 `cutlass.torch` 里 `Optional[torch.device]` 这类注解时要求 `torch.device` 是真实类型，而 proxy 把它替换成了模块。解法是把 import 下沉为函数内惰性导入，模块级留注释说明原因——kernel 定义一行未改（长期保留类别，除非 compat 对类型反射提供支持）：

```diff
--- quack/gemm_sm100.py
-import cutlass.torch as cutlass_torch
+# cutlass.torch is lazy-imported in _get_cutlass_torch() below — importing it
+# at module scope crashes the CuTe-DSL AST preprocessor under Paddle compat
+# (torch.device is replaced with a ProxyModule that breaks typing.Optional).
 ...
 def run(...):
     if not torch.cuda.is_available():
         raise RuntimeError("GPU is required to run this example!")
+    import cutlass.torch as cutlass_torch  # lazy import — see module-level comment
```

同文件里还有 stream 裸指针的固定改法：`cuda.CUstream(torch_stream.cuda_stream)` → `cuda.CUstream(torch_stream.stream_base.raw_stream)`（可吸纳类别：compat 若对齐 `Stream.cuda_stream` 属性即可还原）。类似的边界问题出现在 autotuner 的子进程预编译 worker 上：主进程序列化张量元信息时 `str(dtype)` 得到 `"paddle.bfloat16"`，worker 侧的 `_dtype_map` 只认 `"torch.*"`，补丁在发送端做字符串归一化、接收端扩表，双向兜底。

### 优先查看的文件

- `quack/__init__.py`：看 import-time shim 如何集中收敛并逐条带 `hasattr` 守卫
- `quack/linear.py`：看 `_record_needs_input_grad` 与 backward 返回值数量契约的 PyLayer 适配
- `quack/gemm_sm100.py`：看 `cutlass.torch` 惰性导入与 current stream 裸指针获取
- `quack/autotuner.py`：看子进程预编译的 dtype 归一化与 best-effort 容错（worker 起不来就退回主进程编译）
- `quack/activation.py`：看新增算子如何保持纯增量（新函数 + 注册表加条目），不触碰已有 kernel

### 可复用结论

- 迁移分支按上游版本号命名（`paddle/v0.3.7`、历史 `paddle/v0.2.5`），base 固定在对应 release tag 上（ahead 4 / behind 0）：下游按版本 pin 依赖能精确对上，升级即在新 tag 上重放补丁栈开新分支，新旧分支可共存服务不同下游
- CuTe DSL 生态下张量进 kernel 走 DLPack，这条路径通常零改动；真正的边界是 stream 裸指针、device capability、以及 `cutlass.torch` 这类上游 torch 工具模块与 compat 代理的相互作用
- 即使要在内核/算法层加东西，也应保持纯增量，不触碰已有 kernel，rebase 能力不受影响
- 带 `hasattr` / `try` 守卫的 shim 自带删除条件：compat 层补齐后 shim 自动短路（可吸纳类别）
