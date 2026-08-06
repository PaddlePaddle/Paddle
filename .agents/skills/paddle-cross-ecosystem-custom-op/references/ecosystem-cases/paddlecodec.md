# paddlecodec：控制面在 Python glue

这类库表面上是 C++ custom op，但实际迁移中更关键的是 Python 层对 `torch.ops`、`torch._dynamo`、buffer 创建、metadata 管理的依赖。

### 第一落点

- wrapper
- 薄 shim
- 私有 API 依赖边界

### 具体例子

薄 shim 的实物：上游只用 `torch._dynamo.disallow_in_graph` 屏蔽图优化，Paddle 没有对应物，就放一个恒等实现的假模块让调用点原样通过（可吸纳类别：compat 提供 `_dynamo` 兼容面后删除）：

```diff
--- src/torchcodec/_core/ops.py
+import types
+class FakeDynamo(types.ModuleType):
+    def disallow_in_graph(self, fn):
+        return fn
+torch._dynamo = FakeDynamo("torch._dynamo")
+torch._C._log_api_usage_once = lambda *args, **kwargs: None
 # Note: We use disallow_in_graph because PyTorch does constant propagation of
 # factory functions.
 create_from_file = torch._dynamo.disallow_in_graph(
```

C++ 扩展的链接层不走 `find_package(Torch)`，而是手工把 `TORCH_LIBRARIES` 指向 Paddle 的动态库并注入 compat 头文件目录——`TORCH_LIBRARY(torchcodec_ns, ...)` 算子注册代码一行不改（长期保留类别，属于构建栈差异）：

```diff
--- src/torchcodec/_core/CMakeLists.txt
-find_package(Torch REQUIRED)
+set(
+    TORCH_LIBRARIES
+    "${PADDLE_PATH}/base/libpaddle.so"
+    "${PADDLE_PATH}/libs/libcommon.so"
+    "${PADDLE_PATH}/libs/libphi_core.so"
+)
 ...
         "${TORCH_INSTALL_PREFIX}/include"
+        "${TORCH_INSTALL_PREFIX}/include/paddle/phi/api/include/compat"
+        "${TORCH_INSTALL_PREFIX}/include/paddle/phi/api/include/compat/torch/csrc/api/include"
```

C++ 侧其余改动全是单点：`torch::tensor(scalar)` → `torch::full({}, scalar)`、`C10_THROW_ERROR(IndexError, ...)` → `throw pybind11::index_error(...)`（可吸纳类别）。发行策略与 tilelang 相同：独立包名 `paddlecodec` 发 PyPI，README 维护 `paddlecodec ↔ paddle` 版本兼容表。

### 优先查看的文件

- `src/torchcodec/_core/ops.py`：看最厚的 Python glue 与 FakeDynamo shim
- `src/torchcodec/_core/CMakeLists.txt`：看 C++ 扩展最终如何链接到 Paddle 侧库
- `setup.py`：看打包入口的最小切换点
- `src/torchcodec/__init__.py`：看对外 API 形状
- `test_paddle/`：看当前验证覆盖了哪些用户路径

### 可复用结论

- Python glue 很厚时，薄 shim 是性价比最高的入口
- shim 要带边界与删除条件
- shim 数量开始扩散时，应回查 compat gap
