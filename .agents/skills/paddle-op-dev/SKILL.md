---
name: paddle-op-dev
description: PaddlePaddle (飞桨) C++ 算子开发指南。提供从 YAML 配置、InferMeta 函数、Kernel 实现、Python API 封装、单元测试到编译验证的完整算子开发流程指导。在以下场景使用此 skill：(1) 为 Paddle 框架新增 C++ 算子 (2) 修改或调试已有 Paddle 算子 (3) 编写算子的 YAML 配置、InferMeta、Kernel、Python API 或单元测试 (4) 理解 Paddle 算子开发架构和流程 (5) 编译 Paddle 并验证算子正确性
---

# Paddle 算子开发

## 架构概览

```
Python API (paddle.xxx)
    │
    ▼ (YAML 自动生成的调度代码)
算子 InferMeta ──→ 推导输出 shape/dtype
    │
    ▼
算子 Kernel ──→ 实际计算（CPU/GPU 分别实现）
```

YAML 配置是连接 Python API 与底层 Kernel 的桥梁，框架编译时自动生成调度代码。

## 开发流程

新增算子 `xxx` 需完成以下 6 步：

### 步骤 1：YAML 算子定义

在 `paddle/phi/ops/yaml/ops.yaml` 和 `backward.yaml` 中配置前向和反向算子。

关键配置项：`op`(名称)、`args`(输入)、`output`(输出)、`infer_meta`(推导函数)、`kernel`(计算函数)、`backward`(反向算子)。

快速示例：
```yaml
# ops.yaml
- op: trace
  args: (Tensor x, int offset = 0, int axis1 = 0, int axis2 = 1)
  output: Tensor(out)
  infer_meta:
    func: TraceInferMeta
  kernel:
    func: trace
  backward: trace_grad
```

详细配置规则见 [references/yaml-config.md](references/yaml-config.md)。

### 步骤 2：InferMeta 函数

在 `paddle/phi/infermeta/` 下实现，按输入 Tensor 个数分文件（unary/binary/multiary）。

职责：推导输出 shape 和 dtype，检查输入合法性。

函数签名模式：
```cpp
void XxxInferMeta(const MetaTensor& x, int attr, MetaTensor* out) {
  // PADDLE_ENFORCE_XX 检查输入
  out->set_dims(...);
  out->set_dtype(x.dtype());
}
```

详细规则和示例见 [references/infermeta.md](references/infermeta.md)。

### 步骤 3：Kernel 实现

在 `paddle/phi/kernels/` 下实现。设备无关的放根目录，设备相关的分 `cpu/` 和 `gpu/` 子目录。

文件结构：
- `xxx_kernel.h` — 前向声明
- `cpu/xxx_kernel.cc` / `gpu/xxx_kernel.cu` — 设备实现
- `xxx_grad_kernel.h` + 对应设备实现 — 反向

Kernel 函数模式：
```cpp
template <typename T, typename Context>
void XxxKernel(const Context& dev_ctx, const DenseTensor& x,
               int attr, DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  // 计算逻辑
}
// 注册
PD_REGISTER_KERNEL(xxx, CPU, ALL_LAYOUT, phi::XxxKernel, float, double) {}
```

**参考 PyTorch 实现**：Paddle 算子的计算逻辑与 PyTorch (ATen) 往往相似。实现 Kernel 时可参考 PyTorch 对应算子的实现（位于 `aten/src/ATen/native/` 目录），理解核心算法后适配为 Paddle 的 API 风格（DenseTensor、dev_ctx 等）。注意不要直接复制代码，需根据 Paddle 框架规范进行适配。

详细规则和示例见 [references/kernel-dev.md](references/kernel-dev.md)。

### 步骤 4：Python API 封装

在 `python/paddle/` 对应子目录中实现 Python API。

关键要素：
- 动态图：`_C_ops.xxx(args...)`
- 静态图：`LayerHelper` + `append_op`
- 完整的 docstring（含可运行的 Examples）

详细规则和示例见 [references/python-api.md](references/python-api.md)。

### 步骤 5：单元测试

在 `test/legacy_test/test_xxx_op.py` 中编写，继承 `OpTest`。

测试要点：
- `setUp()` 设置 `op_type`、`python_api`、`inputs`、`attrs`、`outputs`
- `test_check_output(check_pir=True)` 验证前向
- `test_check_grad(['X'], 'Out', check_pir=True)` 验证反向
- 多 Case 继承基类重写 `init_config()`

详细规则和示例见 [references/unit-test.md](references/unit-test.md)。

### 步骤 6：编译与验证

完成代码开发后，需要编译 Paddle 并运行测试来验证算子的正确性。

#### 6.1 增量编译

在 Paddle 的 `build` 目录下执行增量编译。新增算子通常只需重新编译 phi 相关目标，无需全量编译。

```bash
cd build

# 增量编译（推荐使用 ninja，速度更快）
ninja -j$(nproc)

# 或使用 make
make -j$(nproc)
```

如果只修改了 Kernel/InferMeta 等 C++ 文件，可以缩小编译范围：

```bash
# 只编译 phi 库（覆盖 Kernel + InferMeta）
ninja phi -j$(nproc)

# 编译完成后重新安装 Python 包
pip install -e . --no-build-isolation
# 或
cd python && pip install -e .
```

#### 6.2 运行单元测试

```bash
# 方式 1：直接运行测试文件
python test/legacy_test/test_xxx_op.py

# 方式 2：使用 pytest（支持更灵活的筛选）
python -m pytest test/legacy_test/test_xxx_op.py -v

# 方式 3：通过 ctest 运行（在 build 目录下）
cd build && ctest -R test_xxx_op -V
```

#### 6.3 验证要点

- **前向正确性**：`test_check_output` 通过，算子输出与 NumPy 参考实现一致
- **反向正确性**：`test_check_grad` 通过，梯度通过数值微分法校验
- **PIR 模式**：确认 `check_pir=True` 下测试通过
- **多设备验证**：有 GPU 环境时确认 CPU 和 GPU 结果一致

#### 6.4 GPU 算子调试

GPU 算子出现 CUDA 错误时，使用以下环境变量辅助定位：

```bash
# 启用 CUDA 同步错误检查 + 系统内存分配器
FLAGS_check_cuda_error=1 FLAGS_use_system_allocator=1 python test/legacy_test/test_xxx_op.py

# 强制 CUDA kernel 同步执行，定位出错的 kernel
CUDA_LAUNCH_BLOCKING=1 python test/legacy_test/test_xxx_op.py
```

常见问题：
- `CUDA error(9)` — kernel 配置无效，检查 grid/block size 是否为 0（通常由空 Tensor 触发）
- `CUDA error(700)` — 非法内存访问，检查数组越界或空指针
- `CUDA error(2)` — 显存不足，减小测试数据规模或检查是否有显存泄漏

#### 6.5 常见编译错误排查

| 现象 | 排查方向 |
|---|---|
| 找不到 `XxxInferMeta` 符号 | 检查 InferMeta 函数是否在 `.h` 中声明、YAML 中函数名是否拼写一致 |
| 找不到 `xxx` kernel | 检查 `PD_REGISTER_KERNEL` 注册名是否与 YAML `kernel:func` 一致 |
| Python 端 `_C_ops.xxx` 不存在 | 确认 YAML 配置正确且已重新编译，`pip install -e .` 已执行 |
| 参数数量不匹配 | 对照 YAML `args` 与 InferMeta/Kernel 函数签名的参数列表 |

## 文件清单速查

| 内容 | 文件位置 |
|---|---|
| 前向 YAML | `paddle/phi/ops/yaml/ops.yaml` |
| 反向 YAML | `paddle/phi/ops/yaml/backward.yaml` |
| InferMeta | `paddle/phi/infermeta/{unary,binary,multiary}.{h,cc}` |
| Kernel 头文件 | `paddle/phi/kernels/xxx_kernel.h` |
| CPU Kernel | `paddle/phi/kernels/cpu/xxx_kernel.cc` |
| GPU Kernel | `paddle/phi/kernels/gpu/xxx_kernel.cu` |
| Python API | `python/paddle/` 对应子目录 |
| 单元测试 | `test/legacy_test/test_xxx_op.py` |

## 显存优化

- **inplace**：输出复用输入显存，YAML 中配置 `inplace: (x -> out)`
- **no_need_buffer**：反向不需要前向变量的内存数据时配置，提前释放内存
- **减少反向无关变量**：反向 args 中只包含实际需要的前向变量

## 参考文档

- [官方文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html)
