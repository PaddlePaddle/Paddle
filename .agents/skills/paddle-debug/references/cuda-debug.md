# CUDA / GPU 调试技巧

## 常用调试环境变量

| 环境变量 | 作用 |
|---------|------|
| `FLAGS_check_cuda_error=1` | 启用 CUDA 同步错误检查，将异步错误立即暴露 |
| `FLAGS_use_system_allocator=1` | 使用系统内存分配器，便于排查显存相关问题 |
| `CUDA_LAUNCH_BLOCKING=1` | 强制 CUDA kernel 同步执行，便于定位出错的 kernel |
| `FLAGS_cudnn_exhaustive_search=0` | 关闭 cuDNN 算法搜索，排除算法选择导致的不确定性 |
| `FLAGS_cudnn_deterministic=1` | 启用 cuDNN 确定性模式 |
| `NCCL_DEBUG=INFO` | 启用 NCCL 调试日志（分布式训练问题排查） |

## 常见 CUDA 错误码

| 错误码 | 名称 | 常见原因 |
|--------|------|----------|
| `CUDA error(1)` | cudaErrorInvalidValue | 非法参数值，如空指针、越界 |
| `CUDA error(2)` | cudaErrorMemoryAllocation | 显存分配失败，OOM |
| `CUDA error(3)` | cudaErrorInitializationError | CUDA 初始化失败，常见于 fork 后的子进程中（CUDA context 不可用） |
| `CUDA error(4)` | cudaErrorLaunchFailure | kernel 启动失败 |
| `CUDA error(9)` | cudaErrorInvalidConfiguration | kernel 配置无效（grid/block size 为 0 或超限） |
| `CUDA error(11)` | cudaErrorInvalidValue | 非法设备指针或参数 |
| `CUDA error(35)` | cudaErrorInsufficientDriver | CUDA driver 版本不足或异常 |
| `CUDA error(100)` | cudaErrorNoDevice | 无可用 GPU 设备 |
| `CUDA error(400)` | cudaErrorInvalidResourceHandle | 无效的 CUDA 资源句柄（stream/event 未初始化或已销毁） |
| `CUDA error(700)` | cudaErrorIllegalAddress | 非法内存访问 |
| `CUDA error(719)` | cudaErrorLaunchFailure | kernel 执行期间出错 |

## 边界条件检查要点

GPU kernel 对边界条件特别敏感，常见需要检查的场景：

- **空 Tensor / numel=0**：确保在调用 kernel 前检查，避免 grid size 为 0
- **0-d Tensor（标量）**：shape=() 时 numel=1，但处理逻辑可能与高维不同
- **极小 batch**：batch=1 或某个维度为 1 时，可能触发特殊分支
- **极大 shape**：可能导致 int32 溢出或超出 GPU 线程限制
- **数值边界**：NaN、Inf、极大/极小值可能导致数值不稳定

## 调试步骤示例

1. 启用错误检查环境变量复现问题：
   ```bash
   FLAGS_check_cuda_error=1 FLAGS_use_system_allocator=1 python reproduce.py
   ```

2. 分析错误栈，定位到具体 kernel 或 API 调用

3. 检查该路径的边界条件处理：
   - 是否在调用 kernel 前检查了 numel == 0？
   - 是否正确处理了 0-d tensor？
   - grid/block size 计算是否可能为 0？

4. 在关键位置添加日志，打印 shape、numel、配置参数

5. 修复后使用相同环境变量验证

## CUDA Sticky Error（残留错误）机制

### 什么是 CUDA Sticky Error

CUDA runtime 维护一个 "last error" 状态。当 CUDA API 调用失败时，错误会被记录到这个状态中。这个残留错误**不会自动清除**，必须通过 `cudaGetLastError()` 显式读取并清除。

如果一个错误未被清除，后续所有依赖 `cudaGetLastError()` 检查的操作都会检测到它——即使后续操作本身没有问题。

### 为什么这在 Paddle 中特别重要

Paddle 的 `FLAGS_check_cuda_error=1` 模式下，每个算子前后都会调用：
```cpp
void CUDAErrorCheck(const std::string& check_tag) {
    cudaDeviceSynchronize();   // 同步所有 pending 操作
    cudaGetLastError();        // 检查并清除 last error → 如果非0则抛异常
}
```
因此，之前任何未清除的 CUDA error 都会在下一个算子执行时被检测到，表现为"看似无关的代码报错"。

### 常见 Sticky Error 产生场景

| 场景 | 原因 | 表现 |
|------|------|------|
| 未检查的 CUDA API 返回值 | 忽略了 `cudaEventSynchronize` 等 API 的返回值 | 错误残留，后续操作失败 |
| try/catch 捕获 CUDA 异常 | C++ 异常被捕获但未调用 `cudaGetLastError()` | 异常虽被处理，错误仍残留 |
| 异步 kernel 失败 | kernel 在异步执行中出错，直到同步时才被发现 | 错误在首次同步点被报告 |

### 防御编码规范

```cpp
// 错误示例：忽略返回值
cudaEventSynchronize(event);  // ← 返回值丢弃，错误残留

// 正确示例 1：使用 PADDLE_ENFORCE_GPU_SUCCESS 检查
PADDLE_ENFORCE_GPU_SUCCESS(cudaEventSynchronize(event));

// 正确示例 2：手动检查并清除（适用于允许失败的场景）
gpuError_t err = cudaEventSynchronize(event);
if (err != cudaSuccess) {
    cudaGetLastError();  // 清除 last error
    // 进行错误处理...
}

// 注意：PADDLE_ENFORCE_GPU_SUCCESS 不会调用 cudaGetLastError()，
// 如果需要在抛异常前清除 last error，必须手动调用。
```

### 调试 Sticky Error 的方法

1. **确认是否为残留错误**：在报错点之前手动插入 `cudaGetLastError()` 调用，如果能清除错误且后续正常，则说明是残留
2. **二分测试定位污染源**：逐步删减之前执行的测试/操作，找到产生残留错误的源头
3. **Python 层快速验证**：
   ```python
   import ctypes
   libcudart = ctypes.CDLL("libcudart.so")
   err = libcudart.cudaGetLastError()
   print(f"CUDA last error: {err}")  # 0 表示无错误
   ```

## CUDA Fork Safety（进程 fork 与 CUDA 的冲突）

### 背景

CUDA runtime 的 context（包括 GPU 内存句柄、stream、event 等）**在 `fork()` 后不可用**。这是 NVIDIA CUDA 的设计约束：fork 出的子进程继承了父进程的虚拟地址空间，但 CUDA driver 内部状态在子进程中是无效的。

任何在 fork 后的子进程中调用 CUDA API（如 `cudaGetDevice`、`cudaFree`、`cudaSetDevice`）都会返回 `cudaErrorInitializationError`(3)。

### 典型触发场景

```
主进程: 初始化 CUDA (创建 GPU tensor / 运行 GPU 测试)
  |
  |-- fork --> DataLoader worker 子进程
                |
                |-- 继承了父进程中 GPU tensor 的 shared_ptr
                |-- GC 回收时触发 DenseTensor::~DenseTensor()
                |-- 析构链调用 cudaFree / cudaGetDevice
                |-- CUDA error(3)! ABORT!
```

**关键条件组合**（三个条件同时满足才触发）：
1. 父进程已初始化 CUDA（任何 GPU 操作都算）
2. 使用了 `fork` 方式创建子进程（如 `DataLoader(num_workers>0)`）
3. 子进程中触发了 GPU 内存释放（继承的 GPU tensor 被 GC 回收）

### Paddle 中的具体调用链

```
DenseTensor::~DenseTensor()
  -> ~shared_ptr<phi::Allocation>()
    -> AllocationDeleter()
      -> CUDAAllocator::FreeImpl()
        -> RecordedGpuFree()
          -> RecordedGpuMallocHelper::Free()
            -> CUDADeviceGuard(dev_id_)
              -> GetCurrentDeviceId()
                -> cudaGetDevice()  <-- error 3!
```

### Paddle 中的内存分配器路径

不同 FLAGS 配置下，GPU 内存释放走不同路径：

| 配置 | 分配器链 | Free 路径 |
|------|---------|-----------|
| 默认 | StatAllocator -> RetryAllocator -> StreamSafeCUDA -> AutoGrowth -> CUDAAllocator | 缓存池管理，不立即 cudaFree |
| `FLAGS_use_system_allocator=1` | CUDAAllocator (直接) | 每次都调用 cudaFree |

`FLAGS_use_system_allocator=1` 下更容易触发此问题，因为每个 tensor 释放都直接走 `cudaFree`。

### 修复模式

在调用 CUDA API 前先做 context 可用性检查：

```cpp
// 在 cudaFree / CUDADeviceGuard 之前添加
{
  int device_id;
  auto err = cudaGetDevice(&device_id);
  if (err == cudaErrorInitializationError ||  // fork 后
      err == cudaErrorNoDevice ||             // 无 GPU
      err == cudaErrorInsufficientDriver) {   // driver 异常
    cudaGetLastError();  // 清除 sticky error
    return;              // 跳过释放，由 OS 回收
  }
}
```

跳过 `cudaFree` 是安全的：fork 后子进程中的 GPU 内存不属于该进程，进程退出时 OS/driver 自动回收。

### 排查要点

1. **确认是否为 fork 问题**：检查错误是否发生在子进程（DataLoader worker）中，且父进程之前有 GPU 操作
2. **单独运行不复现**：fork safety 问题通常只在多测试/多进程场景下出现——单独运行某个测试通过，全部一起跑才失败
3. **穷举 CUDA API 调用路径**：修复时要覆盖所有可能的 CUDA API 调用点（`Free`、`FreeAsync` 等），不能只修一处
4. **验证 .so 实际加载**：确保 Python 加载的 `.so` 是重新编译后的版本（行号对比法——如果错误消息中的行号和修改后的源码行号不一致，说明加载了旧 `.so`）
