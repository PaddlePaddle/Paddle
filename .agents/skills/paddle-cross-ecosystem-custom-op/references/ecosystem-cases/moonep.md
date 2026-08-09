# MoonEP：控制面在分布式引导与测试启动器 glue

MoonEP 是 CuTe DSL 实现的 EP 通信库（上游 MoonshotAI/MoonEP，默认分支 `master`，迁移分支 `paddle`，ahead 5 / behind 2）。diff 里混着两条工作流，分析时要先分开：一条是 Paddle 适配（setup.py compat 开关、测试启动器 glue、stream/rank 的单点 fallback，量很小）；另一条是跨 tray fabric 通信支持（`csrc/bindings.cu` 新增 fabric 句柄 API、`moonep/buffer.py` 的 fabric 分支），它是功能增强而非框架适配，以运行时环境变量开关控制、与 POSIX fd 路径并存——与 HybridEP 的独立分支策略相反，这是「硬件变体收在主迁移分支内」的另一种做法。

### 第一落点

- `setup.py`：4 行 `paddle.enable_compat()` 前置（与 FlashMLA 相同模式），CUDA extension 构建链原样保留
- `tests/conftest.py`：测试引导从 torchrun 语义换到 `paddle.distributed.launch` 语义（rank 环境变量、`init_parallel_env`、`set_device`）
- 单点 fallback：stream 裸指针、`dist.get_global_rank` 缺失时的 group 方法回退、`.numel()` → `.shape.numel()`

### 具体例子

分布式测试引导是这类通信库固定要接的一层，workaround 带着明确的删除条件注释（可吸纳类别：compat 的 `torch.distributed` 代理补齐 `init_process_group` 后应还原）：

```diff
--- tests/conftest.py
+import paddle
+
+paddle.enable_compat(scope={"moonep", "tests"})
 ...
+    local_rank = _local_rank()
+    paddle.cuda.set_device(local_rank)
     if not dist.is_initialized():
-        dist.init_process_group(backend="nccl")
+        # TODO(Paddle compat): use init_process_group once the
+        # torch.distributed proxy provides it.
+        dist.init_parallel_env()
```

stream 裸指针的获取点收敛成一个 helper，先走公共属性、拿不到再走 fallback——这种「探测 + 注释删除条件」的写法让补丁自带过期机制（可吸纳类别）：

```diff
--- moonep/_common.py
+def current_cuda_stream_handle() -> int:
+    stream = torch.cuda.current_stream()
+    handle = getattr(stream, "cuda_stream", None)
+    if handle is not None:
+        return handle
+    # TODO(Paddle compat): remove this fallback once the proxy exposes the
+    # public PyTorch Stream.cuda_stream property.
+    return stream.__cuda_stream__()[1]
```

### 优先查看的文件

- `setup.py`：确认主构建入口只有 compat 开关
- `tests/conftest.py`：看 torchrun → `paddle.distributed.launch` 的引导 glue 与 rank 环境变量回退链
- `moonep/_common.py`：看 stream 句柄 helper 与 CuTe DSL 版本差异的处理
- `moonep/buffer.py`：看 fd/fabric 双路径如何用运行时开关并存、`_get_global_rank` 回退
- `csrc/bindings.cu`：看 fabric API 是纯增量注册（原 fd 路径入口不动）

### 可复用结论

- EP 通信库的 Paddle 适配面可以很小：build 一个开关 + 测试引导 glue + 少量单点 fallback；CuTe DSL kernel 与通信算法零改动
- 硬件变体不一定要开独立分支：与主路径能共存时（运行时开关、纯增量 API），收在主迁移分支内 sync 成本更低；无法共存（如 HybridEP 基线不同）才开变体分支
- 每个 fallback 都带 `TODO(Paddle compat)` 注释写明删除条件，是 workaround 收敛的标准姿势
- 分析 fork diff 前先分离「框架适配」与「功能增强」两条工作流，否则控制面判断会被功能改动带偏
