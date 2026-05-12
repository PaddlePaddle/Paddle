# 调试案例

## 案例：one_hot kernel CUDA error(9) 修复

### 问题描述
```bash
FLAGS_check_cuda_error=1 FLAGS_use_system_allocator=1 python test_one_hot_v2_op.py
```
报错：`CUDA error(9), invalid configuration argument`

### 根因分析
1. 测试用例 `TestOneHotOp_ZeroSize` 使用 `x_shape=[0, 10, 7, 3]`，即 `numel=0`
2. `one_hot_kernel.cu` 中代码顺序：
   ```cpp
   funcs::set_constant(dev_ctx, out, 0.0);  // 先调用 kernel
   if (numel == 0) return;                   // 后检查边界
   ```
3. `set_constant` 内部启动 CUDA kernel，numel=0 导致 grid size=0，触发 CUDA error(9)

### 修复方案
将边界检查移到 kernel 调用之前：
```cpp
if (numel == 0) return;  // 先检查边界
funcs::set_constant(dev_ctx, out, 0.0);  // 后调用 kernel
```

### 修复文件
- `paddle/phi/kernels/gpu/one_hot_kernel.cu`
- `paddle/phi/kernels/legacy/gpu/one_hot_kernel.cu`

### 经验总结
- GPU kernel 调用前必须检查 numel/shape 是否为空
- 同一算子可能有多个实现（主版本和 legacy），需同步修复
- 使用 `FLAGS_check_cuda_error=1` 可以将异步 CUDA 错误立即暴露

---

## 案例：tril_triu kernel CUDA error(9) 修复

### 问题描述
```bash
FLAGS_check_cuda_error=1 FLAGS_use_system_allocator=1 python test_tril_triu_op.py
```
6 个与 `ZeroSize` / `ZeroDim` 相关的测试用例失败，报错：`CUDA error(9), invalid configuration argument`

### 根因分析
1. 测试用例使用 `X.shape = [0, 3, 9, 4]`（numel=0）
2. `TrilTriuKernel` 和 `TrilTriuGradKernel` 使用 `ForRange` 调度 kernel
3. 当 `numel=0` 时，`ForRange` 以 `limit=0` 被调用，导致 `grid_size=0, block_size=0`
4. 虽然 `TrilKernel`/`TriuKernel` 有 `numel==0` 检查，但底层的 `TrilTriuKernel` 没有

### 修复方案
在 `TrilTriuKernel` 和 `TrilTriuGradKernel` 中添加提前返回：
```cpp
// 在 kernel 调用前添加
if (x.numel() == 0) {
  return;  // 提前返回，避免无效的 CUDA kernel 启动
}
```

### 修复文件
- `paddle/phi/kernels/impl/tril_triu_kernel_impl.h`（前向 kernel）
- `paddle/phi/kernels/impl/tril_triu_grad_kernel_impl.h`（反向 kernel）

### 与 one_hot 案例的关键差异

| 维度 | one_hot 案例 | tril_triu 案例 |
|------|-------------|---------------|
| 修复位置 | `.cu` 文件 | `.h` 头文件模板 |
| 修复范围 | 前向 kernel | 前向 + 反向 kernel |
| 入口函数 | 单一入口 | 多入口（tril/triu/tril_triu） |
| 编译验证 | 编译 .cu 即可 | 需重新编译所有引用该头文件的 .cu |

### 经验总结
- **前向和反向 kernel 要一并检查**：反向 kernel 往往复用相同的计算逻辑，同样存在边界问题
- **检查所有入口函数**：`TrilKernel`/`TriuKernel` 虽有检查，但它们调用的 `TrilTriuKernel` 没有
- **头文件修改需完整重编**：修改 `.h` 后需重新编译所有引用它的 `.cu`，并确保 Python 加载的 `.so` 是最新的
- **验证 Python 库路径**：`build/python/paddle/base/libpaddle.so` 可能与 `build/paddle/fluid/pybind/libpaddle.so` 不同步，需手动复制或重新链接

---

## 案例：CudaEvent::ElapsedTime CUDA error(400) sticky error 修复

### 问题描述
```bash
FLAGS_check_cuda_error=1 FLAGS_use_system_allocator=1 python test/compat/test_event_stream_apis.py
```
`test_event_stream_timing_functionality` 在 `paddle.randn()` 时报错：`CUDA error(400), invalid resource handle`。

### 根因分析

**直接原因**：`paddle/phi/api/profiler/event.cc` 的 `CudaEvent::ElapsedTime()` 存在两个缺陷：
1. `cudaEventSynchronize()` 的返回值**未被检查**（直接丢弃）
2. 错误路径上**未调用 `cudaGetLastError()` 清除 CUDA last error**

**触发序列**：
1. `test_event_stream_error_handling` 对**未 record 的** Event 调用 `elapsed_time()`
2. `cudaEventSynchronize(unrecorded_event)` 返回 `cudaErrorInvalidResourceHandle`(400)，但返回值被忽略
3. CUDA runtime 的 last error 被设置为 400
4. `cudaEventElapsedTime` 也失败，`PADDLE_ENFORCE_GPU_SUCCESS` 抛出 C++ 异常
5. Python `try/except` 捕获异常，但 **CUDA last error 未被清除**（sticky error 残留）
6. 后续 `FLAGS_check_cuda_error=1` 的 `CUDAErrorCheck` 调用 `cudaGetLastError()` 检测到残留错误 400
7. 此后所有 CUDA 操作报错

**问题代码**：
```cpp
// paddle/phi/api/profiler/event.cc (修复前)
float CudaEvent::ElapsedTime(CudaEvent *end_event) {
  float milliseconds = 0;
  cudaEventSynchronize(end_event->GetRawCudaEvent());  // ← 返回值未检查！
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventElapsedTime(       // ← 异常路径未清除 last error
      &milliseconds, event_, end_event->GetRawCudaEvent()));
  return milliseconds;
}
```

### 最小复现（不依赖 unittest）
```python
import paddle
paddle.device.set_device('gpu:0')
event1 = paddle.device.Event()
event2 = paddle.device.Event()
try:
    event1.elapsed_time(event2)  # 未 record 的 event → CUDA error(400)
except Exception:
    pass  # Python 捕获了异常，但 CUDA last error 仍残留

# 后续任何 CUDA 操作都会失败（在 FLAGS_check_cuda_error=1 下）
stream = paddle.device.Stream(device='gpu:0')
with paddle.device.stream_guard(stream):
    x = paddle.randn([100, 100])  # ← CUDA error(400)!
```

### 修复方案

**C++ 修复**（`paddle/phi/api/profiler/event.cc`）：
- 检查 `cudaEventSynchronize` 返回值
- 错误路径调用 `cudaGetLastError()` 清除 CUDA last error

```cpp
float CudaEvent::ElapsedTime(CudaEvent *end_event) {
  float milliseconds = 0;
  gpuError_t sync_err = cudaEventSynchronize(end_event->GetRawCudaEvent());
  if (sync_err != cudaSuccess) {
    cudaGetLastError();  // 清除 CUDA last error
    PADDLE_ENFORCE_GPU_SUCCESS(sync_err);
  }
  gpuError_t elapsed_err = cudaEventElapsedTime(
      &milliseconds, event_, end_event->GetRawCudaEvent());
  if (elapsed_err != cudaSuccess) {
    cudaGetLastError();  // 清除 CUDA last error
    PADDLE_ENFORCE_GPU_SUCCESS(elapsed_err);
  }
  return milliseconds;
}
```

**测试修复**（`test/compat/test_event_stream_apis.py`）：
- 不再对未 record 的 events 调用 `elapsed_time`（这是 CUDA 层的未定义行为）
- 改为先 record events 后再调用 `elapsed_time`

### 修复文件
- `paddle/phi/api/profiler/event.cc`（C++ 核心修复）
- `test/compat/test_event_stream_apis.py`（测试修复）

### 调试过程中的关键手段

| 手段 | 具体操作 | 效果 |
|------|---------|------|
| 二分测试 | 通过逐步去掉/保留 unittest 中各测试，定位到 `test_event_stream_error_handling` | 从 7 个测试缩小到 1 个关键测试 |
| 最小化复现 | 将 unittest 三类交互抽象为 10 行脚本 | 确认了根因链条 |
| 手动清除 CUDA error | 在 Python 中用 ctypes 调 `cudaGetLastError()` | 验证了 sticky error 假设 |
| 测试执行顺序分析 | 用 `unittest.TestLoader` 打印测试顺序 | 发现错误只在特定测试序列下出现 |

### 经验总结

1. **CUDA API 返回值必须全部检查**：即使是 `cudaEventSynchronize` 这类"辅助"调用，忽略返回值也会在 CUDA runtime 中留下残留错误
2. **CUDA error 清除机制**：
   - CUDA runtime 的 last error 需通过 `cudaGetLastError()` 显式清除
   - `PADDLE_ENFORCE_GPU_SUCCESS` 只检查传入的错误码，**不会**调用 `cudaGetLastError()` 来清除 runtime 残留
   - Python `try/except` 捕获 C++ 异常后，CUDA last error 仍残留
3. **`FLAGS_check_cuda_error=1` 的放大效应**：该 flag 使每个算子前后都调用 `cudaDeviceSynchronize()` + `cudaGetLastError()`，能检测到之前任何残留的错误——即使错误发生在完全不相关的代码路径上
4. **跨测试状态污染**：unittest 中一个测试产生的 CUDA sticky error 可以影响后续所有测试，问题表现为"看似无关的测试随机失败"
5. **最小复现的缩减策略**：对于仅在特定测试序列下出现的 bug，应关注测试执行顺序、逐个删除测试来二分定位"污染源"测试

---

## 案例：RecordedGpuMallocHelper::Free CUDA error(3) fork safety 修复

### 问题描述
```bash
FLAGS_check_cuda_error=1 FLAGS_use_system_allocator=1 python test/legacy_test/test_newprofiler.py
```
`TestTimerOnly::test_with_dataloader` 失败，DataLoader worker 子进程报错：`CUDA error(3), initialization error`，随后 abort。

### 关键现象

- **全部测试一起跑才出现**：单独运行 `test_with_dataloader` 通过
- **需要 `FLAGS_use_system_allocator=1`**：默认分配器下不触发（因为有缓存池，不会立即 `cudaFree`）
- **错误发生在 DataLoader worker 子进程中**（fork 出来的进程）

### 根因分析

**触发链**：
1. `TestProfiler::test_profiler` 在主进程中初始化了 CUDA（创建了 GPU tensor）
2. `TestTimerOnly::test_with_dataloader` 使用 `DataLoader(num_workers=2)` fork 子进程
3. 子进程继承了父进程中 GPU tensor 的 `shared_ptr<Allocation>` 引用
4. 子进程中 GC 回收 tensor 时，触发 `DenseTensor::~DenseTensor()`
5. 析构链：`CUDAAllocator::FreeImpl` -> `RecordedGpuFree` -> `RecordedGpuMallocHelper::Free`
6. `Free` 方法构造 `CUDADeviceGuard(dev_id_)` -> `GetCurrentDeviceId()` -> `cudaGetDevice()`
7. fork 后子进程中 CUDA context 不可用，`cudaGetDevice()` 返回 error 3
8. `PADDLE_ENFORCE_GPU_SUCCESS` 将此视为致命错误并 abort

**代码位置**：
- 崩溃点：`paddle/phi/backends/gpu/cuda/cuda_info.cc:179` — `GetCurrentDeviceId()` 中的 `PADDLE_ENFORCE_GPU_SUCCESS(cudaGetDevice(&device_id))`
- 问题入口：`paddle/phi/core/platform/device/gpu/gpu_info.cc:338` — `RecordedGpuMallocHelper::Free()` 中的 `CUDADeviceGuard guard(dev_id_)`

### 修复方案

在 `RecordedGpuMallocHelper::Free()` 和 `FreeAsync()` 中，在 `CUDADeviceGuard` 之前添加 CUDA context 可用性检查：

```cpp
{
  int device_id;
  auto device_err = cudaGetDevice(&device_id);
  if (device_err == cudaErrorInitializationError ||
      device_err == cudaErrorNoDevice ||
      device_err == cudaErrorInsufficientDriver) {
    cudaGetLastError();  // 清除 sticky error
    return;              // 跳过释放，由 OS/driver 回收
  }
}
CUDADeviceGuard guard(dev_id_);  // 现在安全了
```

### 修复文件

- `paddle/phi/core/platform/device/gpu/gpu_info.cc`（`RecordedGpuMallocHelper::Free` 和 `FreeAsync`）

### 调试过程中的关键踩坑点

| 踩坑点 | 说明 | 解决方法 |
|--------|------|---------|
| .so 未同步 | `ninja phi_gpu` 编译了新 `.so`，但 Python 加载的 `build/python/paddle/libs/libphi_core.so` 是旧版本 | 手动 `cp build/paddle/phi/libphi_core.so build/python/paddle/libs/` |
| 行号不变判断法 | 修改代码后错误消息中行号没变（仍显示 :179），暴露了旧 `.so` 问题 | 利用行号作为判断 .so 是否更新的 indicator |
| 调用链穷举 | `GetCurrentDeviceId` 被多处调用，需要确认实际触发路径 | 在崩溃函数中加 `backtrace_symbols_fd` 临时日志 |

### 经验总结

1. **"单独通过，一起失败"的 bug 优先检查跨测试副作用**：前一个测试初始化了 CUDA context，后一个测试 fork 了子进程，两者组合导致问题
2. **CUDA fork safety 是底层框架必须处理的边界条件**：任何可能在 fork 后子进程中调用的 CUDA API，都需要做 context 可用性检查
3. **`FLAGS_use_system_allocator=1` 绕过了缓存池**：使问题在正常路径下隐藏的 bug 暴露出来（默认分配器有缓存，不会每次都 `cudaFree`）
4. **增量编译后必须验证 .so 部署**：Paddle 的构建产物和 Python 加载路径不同，`ninja` 只更新了前者，需要手动同步后者
5. **修复必须覆盖所有并行路径**：`Free` 和 `FreeAsync` 都需要添加保护，不能只修一处

---

## 案例：put_along_axis (scatter) CUDA Graph 模式下内存越界修复

### 问题描述
```bash
compute-sanitizer --tool memcheck --target-processes all python test_put_along_axis.py >run.log 2>&1
```
`compute-sanitizer` 报告大量 `Invalid __global__ atomic of size 4 bytes` 错误，出错 kernel 为 `phi::funcs::PickWinnersScatterKernel<long>`，访问地址远超分配大小（偏移数十亿字节）。**不使用 CUDA Graph 时完全正常**。

### 关键现象

- 仅在 CUDA Graph `graph.replay()` 阶段触发，capture 阶段无报错
- 越界访问发生在 `atomicMax(&winners[replace_index_self], ...)` 处
- `replace_index_self` 的值异常巨大，说明上游 `ComputeOffset` 的输入数据（`shape_strides`）是垃圾值
- Host backtrace 中出现 `cudaGraphLaunch` → `CUDAGraph::Replay()`，确认是 graph replay 触发

### 根因分析

**问题代码**（`paddle/phi/kernels/funcs/gather_scatter_functor.cu`，`gpu_gather_scatter_functor::operator()` 中）：
```cpp
DenseTensor shape_stride_dev;
shape_stride_dev.Resize({3 * ndim});
dev_ctx.Alloc<int64_t>(&shape_stride_dev);
{  // deallocate host once the copy is done
    DenseTensor shape_stride_host;
    shape_stride_host.Resize({3 * ndim});
    dev_ctx.template HostAlloc<int64_t>(&shape_stride_host);
    int64_t* host_data = shape_stride_host.data<int64_t>();
    // ... 填充 host_data ...
    phi::Copy(dev_ctx, shape_stride_host, dev_ctx.GetPlace(), false, &shape_stride_dev);
}  // ← shape_stride_host 在此处析构，pinned memory 被释放
```

**触发链**：
1. `phi::Copy` 在 CUDA Graph capture 期间被录制为 `cudaMemcpyAsync(H2D)` 节点
2. CUDA Graph 录制的是 H2D memcpy 的**源地址指针**（host pinned memory 地址），而非数据内容
3. `shape_stride_host` 是局部变量，在 `{}` 作用域结束后析构，pinned memory 被释放
4. `graph.replay()` 时，CUDA runtime 从**已释放的 host 地址**读取垃圾数据到 device
5. 垃圾 `shape_strides` → `ComputeOffset` 计算出错误的偏移 → `atomicMax` 严重越界 → CUDA error 719

**影响范围**：文件中共有 **7 处**完全相同模式的 H2D 拷贝（前向 1 处 + 反向 6 处），均存在 CUDA Graph 不兼容问题。

### 修复方案

参照 Paddle 已有的 `concat_and_split_functor.cu` 中的做法，使用 `RestoreHostMemIfCapturingCUDAGraph` 在 CUDA Graph 捕获期间对 host 数据做快照，确保 graph replay 时 H2D memcpy 的源地址仍然有效。

```cpp
#include "paddle/phi/backends/gpu/cuda/cuda_graph_with_memory_pool.h"

// 修复后：
{
    DenseTensor shape_stride_host;
    shape_stride_host.Resize({3 * ndim});
    dev_ctx.template HostAlloc<int64_t>(&shape_stride_host);
    int64_t* host_data = shape_stride_host.data<int64_t>();
    // ... 填充 host_data ...
    auto* restored =
        phi::backends::gpu::RestoreHostMemIfCapturingCUDAGraph(
            host_data, 3 * ndim);
    phi::backends::gpu::GpuMemcpyAsync(
        shape_stride_dev.data<int64_t>(),
        restored,
        3 * ndim * sizeof(int64_t),
        phi::gpuMemcpyHostToDevice,
        stream);
}
```

`RestoreHostMemIfCapturingCUDAGraph` 的原理：
- 非 CUDA Graph 模式：直接返回原指针，零开销
- CUDA Graph capture 模式：在堆上分配一份 host 数据快照（`new uint8_t[nbytes]` + `memcpy`），通过 `AddPostResetCallbackIfCapturingCUDAGraph` 注册回调在 graph 重置时释放，确保 graph 整个生命周期内源地址有效

### 修复文件
- `paddle/phi/kernels/funcs/gather_scatter_functor.cu`（7 处 H2D 拷贝全部修复）

### 验证结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| compute-sanitizer 错误数 | 大量 `Invalid __global__ atomic` | **0 errors** |
| CUDA Graph replay | CUDA error 719 (launch failure) | **PASS** |

### 经验总结

1. **CUDA Graph 不兼容 H2D memcpy 中使用临时 host 内存**：CUDA Graph 录制的是地址而非数据，如果 host 源地址在 replay 时已失效，GPU 会读取垃圾数据。这不会在 capture 阶段报错，只会在 replay 时产生难以预料的内存越界
2. **`RestoreHostMemIfCapturingCUDAGraph` 是 Paddle 标准的 CUDA Graph 安全 H2D 模式**：任何在 CUDA Graph capture 期间需要 H2D memcpy 且 host 源为临时变量的场景，都应使用此函数做快照保护
3. **同文件中重复模式需全部修复**：本案例中 7 处完全相同的代码模式均存在同一问题，不能只修前向不修反向
4. **compute-sanitizer 是定位 CUDA Graph 内存问题的利器**：普通运行不报错（垃圾偏移可能"碰巧"落在合法范围内），但 `compute-sanitizer --tool memcheck` 能精确检测到越界访问
5. **非 CUDA Graph 正常 + CUDA Graph 异常 → 优先排查 H2D/D2H memcpy 和 host 内存生命周期**：这是 CUDA Graph 模式最常见的兼容性问题类别

---

## 案例：put_along_axis CUDA Graph 模式下 Python API 层隐式 D2H 同步导致 error(906)

### 问题描述
```bash
PYTHONPATH=build/python python test_put_along_axis.py
```
在 CUDA Graph capture 区间内调用 `paddle.put_along_axis(x, index, value, axis=1)` 时，抛出 `CUDA error(906): cudaErrorStreamCaptureImplicit`。

### 关键现象

- 不使用 CUDA Graph 时完全正常
- 错误栈指向 Python API 层（`manipulation.py`），而非 C++ kernel
- 错误信息：`operation would make the legacy stream depend on a capturing blocking stream`

### 根因分析

**问题代码**（`python/paddle/tensor/manipulation.py`，`put_along_axis` 函数中）：
```python
if (paddle.in_dynamic_mode() and indices.numel() == 0) or (
    not paddle.in_dynamic_mode() and 0 in indices.shape
):
    return paddle.assign(arr)
```

**触发链**：
1. `indices.numel()` 返回一个 0-d GPU Tensor
2. `== 0` 触发 `Tensor.__eq__` → 返回 boolean GPU Tensor
3. `if` 语句触发 `Tensor.__bool__()` → `__nonzero__()`
4. `__nonzero__()` 内部调用 `np.array(self)` → `self.numpy(False)`
5. `numpy()` 需要 GPU→CPU 数据拷贝（D2H memcpy + stream sync on legacy stream）
6. CUDA Graph capture 期间，legacy stream 上的 D2H 操作违反 capture 约束 → error 906

**影响范围**：`put_along_axis` 和 `put_along_axis_`（inplace 版本）均受影响。

### 修复方案

将 `indices.numel() == 0` 替换为 `0 in indices.shape`。`shape` 是 host 端 Python tuple，不触发任何 GPU 同步：

```python
# 修复前（触发 D2H sync，CUDA Graph 不兼容）
if (paddle.in_dynamic_mode() and indices.numel() == 0) or (
    not paddle.in_dynamic_mode() and 0 in indices.shape
):

# 修复后（纯 host 端操作，CUDA Graph safe）
if 0 in indices.shape:
```

### 修复文件
- `python/paddle/tensor/manipulation.py`（`put_along_axis` 和 `put_along_axis_` 两处）

### 验证结果

| 测试 | 修复前 | 修复后 |
|------|--------|--------|
| `put_along_axis` 普通模式 | PASS | PASS |
| `put_along_axis_` (inplace) 普通模式 | PASS | PASS |
| `put_along_axis` CUDA Graph capture + replay | **CUDA error(906)** | PASS |
| `put_along_axis_` (inplace) CUDA Graph capture + replay | **CUDA error(906)** | PASS |

### 经验总结

1. **CUDA Graph 不兼容 Python API 层的隐式 D2H 同步**：`Tensor.__bool__()`、`Tensor.__nonzero__()`、`Tensor.numpy()` 等方法会触发 GPU→CPU 数据拷贝，在 CUDA Graph capture 期间使用会导致 error 906。这类问题的根因在 Python 层而非 C++ kernel 层，容易被忽略
2. **`tensor.numel() == 0` 是常见的 CUDA Graph 不兼容模式**：`numel()` 返回 GPU Tensor → `== 0` 触发 `__bool__` → `numpy()` → D2H sync。应替换为 `0 in tensor.shape`（host 端 tuple 操作，零 GPU 开销）
3. **CUDA Graph 调试时 `FLAGS_check_cuda_error=1` 不可用**：该 flag 会在每个算子前后插入 `cudaDeviceSynchronize()`，这在 capture 期间本身就会触发 error 906，不能用于定位 CUDA Graph 相关问题。应直接运行并观察原始错误栈
4. **排查 CUDA Graph error 906 的优先级**：先检查 Python API 层是否有隐式 D2H 同步（`numpy()`、`__bool__()`、`item()`、`tolist()` 等），再检查 C++ 层的 H2D/D2H memcpy 和 stream 使用
