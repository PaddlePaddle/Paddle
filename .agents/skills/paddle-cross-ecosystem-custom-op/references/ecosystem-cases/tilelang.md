# tilelang-paddle：控制面在 adapter 与 runtime

这类库的主体是编译器或 DSL，跨框架迁移时最常遇到的控制点是 current device、current stream、DLPack、JIT runtime 初始化。

### 第一落点

- runtime adapter
- device helper
- stream helper
- DLPack bridge

### 具体例子

PyTorch 会把 libcudart 预加载进全局符号空间而 Paddle 不会，tilelang 的 C++ 库在 dlopen 时因此找不到 CUDA 符号。解法是在导入阶段自己补一次 `RTLD_GLOBAL` 预加载（长期保留类别，除非 Paddle 侧改变加载行为）：

```diff
--- tilelang/__init__.py
 def _lazy_load_lib():
+    # Preload cudart for frameworks like PaddlePaddle that don't pre-load it
+    # (unlike PyTorch which does)
+    _preload_libcudart()
 ...
+def _preload_libcudart() -> None:
+    for lib_path in cudart_paths:  # pip 包 → 系统 CUDA → SONAME 的回退链
+        try:
+            ctypes.CDLL(lib_path, mode=os.RTLD_GLOBAL)
+            return
+        except Exception:
+            continue
```

adapter 层的 current device 入口是另一个典型单点：绕开 `torch._C._cuda_getDevice` 私有 API，直接换成 Paddle 的等价物（可吸纳类别：compat 覆盖 `torch.cuda` device 查询后可还原）：

```diff
--- tilelang/jit/adapter/base.py
         if torch.cuda.is_available():
+            import paddle
+            return lambda: paddle.framework._current_expected_place()
             try:
                 torch.cuda._lazy_init()
                 current_device = torch._C._cuda_getDevice
```

零散配套：`engine/param.py` 里 dtype 字符串前缀判断 `"torch."` → `"paddle."`；`torch.randint(size=...)` → `paddle` 的 `shape=` 参数差异在 `utils/tensor.py` 逐一改写（均属可吸纳类别）。发行策略上 fork 直接以独立包名 `tilelang-paddle` 发 PyPI（`pyproject.toml` 改名、torch 依赖注释掉），用户侧 `paddle.enable_compat(scope={"tilelang"})` 后 `import tilelang` 即可用。

### 优先查看的文件

- `tilelang/__init__.py`：看导入阶段的 runtime preload 与环境准备
- `pyproject.toml`：看独立 PyPI 包名与依赖面的裁剪
- `tilelang/jit/adapter/base.py`：看 current device/current stream 的共用入口
- `tilelang/jit/adapter/tvm_ffi.py`：看 backend 如何把框架张量送进 FFI
- `tilelang/contrib/dlpack.py`：看跨框架张量协议边界
- `tests_paddle/`：看当前已验证的 backend 路径

### 可复用结论

- adapter 层通常就是第一轮补丁的位置
- DLPack、device、stream 是最稳定的观察点
- backend 较多时，先把主路径跑通，再逐步扩展
