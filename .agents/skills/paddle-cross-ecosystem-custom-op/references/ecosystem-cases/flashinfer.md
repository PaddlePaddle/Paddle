# flashinfer：控制面在 runtime feature gate 与 workaround 边界

这类高性能推理库的 kernel 主体通常比较稳定，跨框架迁移更常见的工作集中在 device 语义、custom op registration、通信路径以及 runtime feature gate。

### 第一落点

- feature gate
- 张量创建路径
- wrapper 与 registration fallback

### 具体例子

registration fallback 不是新写注册逻辑，而是用一个 feature gate 复用上游本来就有的「torch < 2.4 老式注册」分支：

```diff
--- flashinfer/utils.py
+def use_paddle_compatible_api() -> bool:
+    return os.environ.get("PADDLE_COMPATIBLE_API", "1").lower() in ["1", "on", "true"]
+
+
+if use_paddle_compatible_api() or torch.torch_version.TorchVersion(
+    torch.torch_version.__version__
+) < torch.torch_version.TorchVersion("2.4"):
     def register_custom_op(
         name: str,
```

这是「compat 尚未覆盖 `torch.library.custom_op` 时，把 Paddle 路径导向上游已有分支」的改动方式；compat 对齐 custom op 注册后，这个 gate 应删掉（可吸纳类别）。

张量创建路径上的 workaround 停在单个创建点，原始写法保留为注释、并注明是绕框架 bug——注释本身就是删除条件：

```diff
--- flashinfer/comm/trtllm_ar.py
     # Store workspace pointers in device tensor
-    workspace_tensor = torch.tensor(
-        workspace, dtype=torch.int64, device=torch.device("cuda")
-    )
+    # There is a bug in the paddle framework when device="CUDA".
+    # Currently, the bug is being avoided by changing the source code.
+    workspace_tensor = torch.tensor(workspace, dtype=torch.int64)
```

### 优先查看的文件

- `flashinfer/utils.py`：看 feature gate、device 判断、registration fallback
- `flashinfer/fused_moe/core.py`：看高频运行路径里的框架分支
- `flashinfer/comm/trtllm_ar.py`：看通信和张量创建路径
- `flashinfer/decode.py`：看用户入口如何把 device/place 一路传下去
- `tests/conftest.py`：看测试入口的 compat 范围与环境准备

### 可复用结论

- feature gate 是定位 runtime 分支的重要线索
- workaround 最适合停留在最小创建路径或最小 wrapper 层
- 同类 workaround 开始扩散时，要转向 compat gap 处理
