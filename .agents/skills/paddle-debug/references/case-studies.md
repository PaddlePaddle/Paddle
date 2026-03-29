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
