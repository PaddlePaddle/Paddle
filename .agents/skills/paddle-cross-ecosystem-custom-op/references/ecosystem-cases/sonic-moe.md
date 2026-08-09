# sonic-moe：控制面在 import-time patch 与 Triton runtime wrapper

这类库的高频工作点通常是 Triton runtime wrapper、import 阶段的框架假设、stream 与 DLPack 边界。

### 第一落点

- import-time patch
- runtime wrapper
- stream / DLPack helper

### 具体例子

import-time patch 集中在包入口，且带「compat 已支持则不打 patch」的探测条件——这类补丁全部属于可吸纳类别，compat 补齐后应整体删除：

```diff
--- sonicmoe/__init__.py
+if not (hasattr(paddle.library.CustomOpDef, "__call__")
+        and inspect.isfunction(paddle.library.CustomOpDef.__call__)):
+    def __call__(self, *args, **kwargs):
+        return getattr(getattr(paddle.ops, self._namespace), self._name)(*args, **kwargs)
+
+    paddle.library.CustomOpDef.__call__ = __call__
...
+paddle.compat.proxy._extend_torch_proxy_overrides(
+    {
+        "torch.empty": paddle.compat.proxy.RawOverriddenAttribute(torch_compat_empty),
+    }
+)
```

Triton runtime 在 launch 阶段自己 `import torch` 做类型识别，代理机制覆盖不到，于是用 wrapper 只在 kernel 调用期间临时替换 `sys.modules["torch"]`，`finally` 里立即还原：

```diff
--- sonicmoe/triton_utils.py (新增)
+def swap_torch_guard(fn):
+    def wrapped_fn(*args, **kwargs):
+        torch_module = sys.modules["torch"]
+        sys.modules["torch"] = paddle
+        paddle.empty = torch_compat_empty
+        try:
+            return fn(*args, **kwargs)
+        finally:
+            sys.modules["torch"] = torch_module
+            paddle.empty = original_paddle_empty
```

另一个现实注脚：`sonicmoe/functional/__init__.py` 里 `ctx.saved_tensors` → `ctx.saved_tensor()`、backward 返回梯度个数按 PyLayer 语义重排等 autograd 桥接已经扩散进 functional 层——正对应「看 patch 是否已经扩散过深」的提示。

### 优先查看的文件

- `sonicmoe/__init__.py`：看 import-time patch 是否集中在入口
- `sonicmoe/triton_utils.py`：看 Triton runtime 隔离层
- `sonicmoe/utils.py`：看 stream、DLPack、wrapper 共用逻辑
- `sonicmoe/moe.py`：看业务入口如何串起 runtime 假设
- `sonicmoe/functional/`：看 patch 是否已经扩散过深

### 可复用结论

- import-time patch 的重量直接反映 compat 边界压力
- Triton 生态常从 wrapper 与 runtime 边界切入
- monkey patch 需要配合删除条件和 compat gap 判断

## supersonic-moe：控制面在 proxy override 与 PyLayer 梯度契约

supersonic-moe 是 sonic-moe 案例的后继（同一上游 Dao-AILab/sonic-moe 的新一轮适配），旧案例的「import-time patch / Triton runtime wrapper」结论仍然成立，这里只记录增量。增量有三点：compat 边界不再全靠手写 monkey patch，而是用 `paddle.compat.proxy._extend_torch_proxy_overrides` 这个官方扩展点注册 override；Triton wrapper 演化成作用域化的 torch 替换（只在 kernel launch 期间换掉 `sys.modules["torch"]`，`finally` 里恢复）；以及最大的一块 diff 落在 wrapper 挡不住的地方——`torch.autograd.Function` 到 PyLayer 的梯度返回契约差异，直接渗入了业务层文件。

### 第一落点

- `_extend_torch_proxy_overrides` 注册 proxy override（如 `torch.empty` 的 `device="cuda"` 参数适配）
- `wrap_triton_kernel` 作用域化 wrapper，在 Triton kernel 定义处逐个装饰
- PyLayer 梯度契约改写：`ctx.saved_tensors` → `ctx.saved_tensor()`，backward 返回值按张量输入动态组装（PyLayer 只要求返回张量输入对应的梯度，不接受为非张量参数补 None）

### 具体例子

PyLayer 梯度契约差异是整个 diff 里唯一成规模进入"算法文件"的部分：上游 backward 为 15 个 forward 参数逐一占位返回，迁移分支改成按条件组装——forward 时用 `ctx.has_*` 记录可选张量参数是否传入，backward 里据此决定占位个数，`db2` 为 None 时直接不占位。数值路径原样保留（长期保留类别：这是 PyLayer 语义差异，不随 compat 演进消失）：

```diff
--- sonicmoe/functional/__init__.py（_DownProjection.backward 结尾）
-        return None, dz, dw2, db2, ds, *[None] * 10
+        grads = []
+        grads.extend([None, dz, dw2])
+        if db2 is not None:
+            grads.append(db2)
+
+        if ctx.has_num_activated_expert_per_token_offset:
+            grads.extend([ds, *[None] * 5])
+        else:
+            grads.extend([ds, *[None] * 4])
+
+        return tuple(grads)
```

作用域化 torch swap 的核心是把替换窗口压缩到单次 kernel launch，`finally` 里立即还原（可吸纳类别：compat 代理覆盖 Triton runtime 的框架探测后应删除）：

```diff
--- sonicmoe/triton_utils.py（新增文件）
+def swap_torch_guard(fn):
+    def wrapped_fn(*args, **kwargs):
+        if "torch" not in sys.modules:
+            return fn(*args, **kwargs)
+        torch_module = sys.modules["torch"]
+        original_paddle_empty = paddle.empty
+        sys.modules["torch"] = paddle
+        paddle.empty = torch_compat_empty
+        try:
+            return fn(*args, **kwargs)
+        finally:
+            sys.modules["torch"] = torch_module
+            paddle.empty = original_paddle_empty
```

跨 fork 依赖也是控制面：上游 pin `quack-kernels==0.2.5`，迁移分支改成 pin PFCCLab/quack `paddle/v0.2.5` 分支头对应的 git commit——「上游版本号 ↔ 迁移分支名」一一对应，依赖边界可复现（长期保留类别）。

### 优先查看的文件

- `sonicmoe/triton_utils.py`：新增文件，看作用域化 torch 替换与恢复的实现
- `sonicmoe/__init__.py`：看 proxy override 扩展点用法与 `CustomOpDef.__call__` 补丁
- `sonicmoe/functional/__init__.py`：看 PyLayer 梯度契约差异如何渗入业务层
- `sonicmoe/quack_utils/gemm_gated.py`：看如何绕过 `cutlass_torch.current_stream()`，改用 `torch.cuda.current_stream().stream_base.raw_stream` 取裸 stream 指针
- `sonicmoe/jit.py`：看 Paddle `cpp_extension` 的参数名差异（`extra_cflags` → `extra_cxx_cflags`）与用 FileLock 取代 rank-0 编译 + barrier 的多卡 JIT 串行化

### 可复用结论

- compat 边界优先走 `paddle.compat.proxy` 的 override 扩展点；手写 patch 也应作用域化并带恢复逻辑（可吸纳类别：compat 覆盖后 override 应删除）
- autograd Function → PyLayer 的梯度返回契约是 wrapper 挡不住的差异，会成体系地进入业务层，评估工作量时要单独计入
- 依赖同生态的其他适配 fork 时，用 git commit pin 到对方的版本分支头，保持依赖边界可复现
- C++ compat header 有缺口时（如 `AT_DISPATCH_SWITCH`），可先在库内 header 里本地兜底并加守卫，同时把缺口记为 compat gap
