# DeepEP：控制面在分布式上下文

这类库最关键的对象是 ProcessGroup、communicator、stream、event 以及相关上下文。通信 kernel 通常不需要动，迁移工作主要集中在这些上下文如何获取、如何传入底层。

### 第一落点

- runtime glue
- distributed context bridge
- communicator / stream 初始化

### 具体例子

Buffer 初始化是 group → communicator 桥接的核心落点：PyTorch 的 `ProcessGroup` 换成 Paddle 的 `Group` 后，方法变属性（`rank()` → `rank`、`size()` → `world_size`），并把 `group.id` 显式传进 C++ runtime 以定位 comm context（可吸纳类别：compat 若代理 ProcessGroup 语义可部分还原）：

```diff
--- deep_ep/buffer.py
+from paddle.distributed.communication.group import Group

     def __init__(self,
-                 group: Optional[dist.ProcessGroup],
+                 group: Optional[Group],
 ...
         if group is not None:
-            self.rank = group.rank()
+            self.rank = group.rank
             self.group = group
-            self.group_size = group.size()
+            self.group_size = group.world_size
 ...
         self.runtime = deep_ep_cpp.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode, explicitly_destroy,
-                                          enable_shrink, use_fabric)
+                                          enable_shrink, use_fabric, group.id)
```

comm stream 上的 event 获取走了 Paddle 内部接口，但依赖被收敛在一个 helper 里——这正是"依赖内部接口时收敛在最小入口"的实例（长期保留类别，直到有公开等价物）：

```diff
--- deep_ep/utils.py
+def get_event_from_comm_stream(group_id: int) -> EventOverlap:
+    return EventOverlap(
+        event=paddle.base.core.get_event_handle_from_comm_stream(group_id)
+    )
```

build 侧的固定改法：`torch.utils.cpp_extension` → `paddle.utils.cpp_extension`，`TORCH_CUDA_ARCH_LIST` → `PADDLE_CUDA_ARCH_LIST`（并补了 nvidia-smi 自动探测），nvcc dlink 参数追加 `_get_cuda_arch_flags()`，另加 `-DPADDLE_WITH_CUDA`、`-DWITH_NVSHMEM` 等宏。

### 优先查看的文件

- `setup.py`：确认 build 入口如何声明分布式相关能力与 arch 解析
- `csrc/deep_ep.hpp`：看 runtime context、communicator、stream 成员如何组织
- `csrc/deep_ep.cpp`：看 communicator/context 初始化落点
- `deep_ep/buffer.py`：看 Python 侧 group、event、stream 如何传到底层
- `tests/utils.py`：看最小分布式测试如何起环境和 group

### 可复用结论

- 先把 distributed glue 和上下文边界对齐
- stream / event / communicator 初始化是一等公民，不能绕过
- 依赖 Paddle 分布式内部接口时，要把依赖收敛在最小入口

## HybridEP（DeepEP 变体分支）：控制面在通信引导与 JIT 构建链

HybridEP 是 DeepEP 同仓库内为跨 tray NVLink 域机型维护的变体，上游代码来自 deepseek-ai/DeepEP 的 `hybrid-ep` 分支，适配分支为 `hybrid-ep-paddle`，与 `paddle` 主迁移分支不共享提交历史。DeepEP 主扩展部分的 Paddle 补丁（comm context、allocator stream、`context_ring_id`）从 `paddle` 分支按模式移植；HybridEP 特有的 JIT 编译器、communicator 引导、NVLink 域探测才是这条分支新增的适配面。

### 第一落点

- 构建拆分：`setup.py` 收缩为入口，实际逻辑拆到 `setup_deep_ep.py` 与 `setup_hybrid_ep.py`（两个 extension 可独立构建），公共 arch 解析收进 `setup_utils.py`
- C++ 侧 distributed bridge：pybind 层的 `py::module_::import("torch.distributed")` 改为 import `paddle.distributed`
- JIT 运行时路径：新增 `deep_ep/runtime_paths.py`，运行时定位 CUDA toolkit 与 RDMA 库，让 wheel 脱离构建机环境

### 具体例子

HybridEP 的 buffer 初始化在 C++ 里回调 Python 的 distributed API 交换内存句柄，桥接点集中在少数 `py::module_::import` 调用处：

```diff
--- csrc/hybrid_ep/executor/executor.cu
-    auto torch_distributed = py::module_::import("torch.distributed");
+    auto paddle_distributed = py::module_::import("paddle.distributed");
     ...
-    auto group_size = process_group.attr("size")().cast<int>();
+    auto group_size = process_group.attr("world_size").cast<int>();
     ...
-        torch_distributed.attr("all_gather_into_tensor")(global_routing_map, local_routing_map, process_group);
+        paddle_distributed.attr("stream").attr("all_gather")(global_routing_map, local_routing_map, process_group, py::arg("sync_op") = true);
```

这是 runtime glue 重接：`ProcessGroup` 的方法名/属性差异（`size()` 对 `world_size`）和集合通信入口差异都收敛在 pybind 桥接点上（部分可吸纳：如果 compat 后续代理 `torch.distributed` 模块级 API，import 改写可以还原）。

路由元数据构造展示了 workaround 在分支内部的收敛过程——第一版用 `paddle.scatter_nd_add` 手工拼索引绕过 `scatter` 的 compat gap（约 15 行），后续提交收敛成 `paddle.put_along_axis` 三行：

```diff
--- deep_ep/hybrid_ep_buffer.py
-    routing_map = routing_map.scatter(1, topk_idx.to(torch.int64), 1).bool()
+    routing_map = paddle.put_along_axis(
+        routing_map,
+        topk_idx,
+        torch.ones(topk_idx.shape, device="cuda", dtype=torch.uint8),
+        axis=1,
+    ).bool()
```

直接调用 `paddle.*` 原生 API 属于显式的 Paddle 分支，`scatter` 的 compat gap 补齐后可以还原为上游写法（可吸纳类别）。

### 优先查看的文件

- `setup_hybrid_ep.py`：看 HybridEP extension 如何独立构建（JIT backend 代码拷贝、RDMA 依赖、`PADDLE_CUDA_ARCH_LIST`）
- `deep_ep/runtime_paths.py`：看 JIT 运行时如何定位 CUDA/RDMA，剥离构建环境假设
- `csrc/hybrid_ep/executor/executor.cu`：看 C++ 回调 Python distributed API 的桥接点
- `deep_ep/hybrid_ep_buffer.py`：看 NVLink 域大小探测（`PADDLE_LOCAL_SIZE`/`CUDA_VISIBLE_DEVICES` 回退链）与路由元数据构造
- `csrc/deep_ep.hpp`：看从 `paddle` 分支移植的 `comm_ctx`/`calc_ctx`/`SetAllocatorStreamForGPUContext` 模式

### 可复用结论

- 上游本身以分支形式维护变体时，fork 侧应镜像该分支并 1:1 开变体适配分支，比较基准始终是上游同名分支而不是 main
- 已在主迁移分支验证过的补丁模式直接按模式移植（cherry-pick/重放），不要跨不同 base 做 merge；变体特有的控制面才是新增工作量
- C++ 经由 pybind 回调 Python distributed API 的库，桥接点天然集中，替换成本远低于表面 diff 规模
- 两个 pybind extension 共存时给类加 `py::module_local()`，并把 setup 拆成可独立跳过的入口，保住"只装需要的那一半"的能力
