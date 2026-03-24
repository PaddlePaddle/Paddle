---
name: paddle-phi-kernel
description: "Use when working with Paddle's PHI kernel system: registering new kernels, debugging kernel selection/dispatch, understanding code auto-generation from YAML, or implementing operator decomposition via the combination mechanism."
---

# PHI Kernel 体系总览

## Kernel 两级 map 结构

PHI kernel 在 `KernelFactory` 单例中以两级 `flat_hash_map` 存储：

```
api_name (string)  →  KernelKeyMap
                        ├── KernelKey_1  →  Kernel_1
                        ├── KernelKey_2  →  Kernel_2
                        └── ...
```

- 第一级：算子名称（如 `"add"`, `"matmul"`）→ `KernelKeyMap`
- 第二级：`KernelKey` → `Kernel` 对象（包含函数指针、参数元信息）

## KernelKey 哈希结构

`KernelKey` 由 Backend、DataLayout、DataType 三要素组成，哈希编码为 32-bit 整数：

```
bit:  31 ─────── 20 | 19 ─── 12 | 11 ── 8 | 7 ── 0
      Extended       DataType    Layout    Backend
```

- `Backend`(bit 0-7): CPU / GPU / GPUDNN / XPU / OneDNN 等
- `DataLayout`(bit 8-11): NCHW / NHWC / ANY 等
- `DataType`(bit 12-19): FLOAT32 / FLOAT16 / BFLOAT16 / INT64 等
- `Extended`(bit 20-31): 保留位，用于未来扩展

## 代码生成管线概览

PHI 体系通过 YAML 驱动代码生成，覆盖三个子系统：

| 子系统 | 生成器 | 产出物 |
|--------|--------|--------|
| C++ API 层 | `api_gen.py` | `paddle/phi/api/lib/api.cc` |
| 动态图函数层 | `eager_gen.py` | `dygraph_functions.cc`, `nodes.cc` |
| Python-C 映射层 | `python_c_gen.py` | `eager_op_function.cc` |

输入 YAML：`ops.yaml`, `backward.yaml`, `fused_ops.yaml`, `sparse_ops.yaml`, `op_compat.yaml`

## 组合算子机制

将 1061 个原生算子分解为约 200 个基础算子（primitive operators），降低分布式 / 编译器 / 新硬件适配成本。

- **前向分解**：`DecompInterface` → `call_decomp_rule()` → `composite.h` 实现
- **反向分解**（VJP）：`VjpInterface` → `call_vjp()` → `details.h` 实现
- **CustomVJP**：为 sigmoid、log_softmax 等数值敏感算子提供手写反向

## 什么场景看什么文件

| 场景 | 参考文档 |
|------|----------|
| 注册新 kernel / 理解宏展开 | [references/kernel-registration.md](references/kernel-registration.md) |
| 调试 kernel 选择 / 分派失败 | [references/kernel-selection.md](references/kernel-selection.md) |
| 理解 YAML → 代码生成流程 | [references/codegen-pipeline.md](references/codegen-pipeline.md) |
| 开发组合算子 / VJP | [references/combination-mechanism.md](references/combination-mechanism.md) |

## 社区源码链接（L3）

- PHI kernels 根目录: `paddle/phi/kernels/`
- Kernel 注册宏: `paddle/phi/core/kernel_registry.h`
- KernelFactory: `paddle/phi/core/kernel_factory.h`, `kernel_factory.cc`
- YAML 定义: `paddle/phi/ops/yaml/ops.yaml`, `backward.yaml`
- 代码生成脚本: `paddle/phi/api/yaml/generator/`
- 组合算子前向: `paddle/fluid/primitive/composite/composite.h`
- 组合算子反向: `paddle/fluid/primitive/rule/vjp/details.h`
