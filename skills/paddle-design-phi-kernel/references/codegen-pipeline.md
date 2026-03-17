# 代码自动生成管线详解

## 概述

Paddle PHI 体系采用 YAML 驱动的代码生成方式，从算子定义文件自动生成 C++ API、动态图函数、Python-C 绑定等多层代码。这一设计使得新增算子只需编写 YAML 描述和 kernel 实现，中间的粘合代码全部自动生成。

## YAML 定义文件

所有算子元信息以 YAML 格式维护，位于 `paddle/phi/ops/yaml/` 目录下：

| 文件 | 内容 |
|------|------|
| `ops.yaml` | 前向算子定义（输入、输出、属性、kernel 映射、infer_meta） |
| `backward.yaml` | 反向算子定义 |
| `fused_ops.yaml` | 融合算子（如 fused_attention、fused_feedforward） |
| `sparse_ops.yaml` | 稀疏算子（SparseCoo / SparseCsr） |
| `op_compat.yaml` | 新旧算子名称 / 参数名映射，保证兼容性 |
| `op_version.yaml` | 算子版本管理，记录不兼容变更 |

### ops.yaml 单条记录示例

```yaml
- op: add
  args: (Tensor x, Tensor y)
  output: Tensor(out)
  infer_meta:
    func: ElementwiseInferMeta
  kernel:
    func: add
    data_type: x
  backward: add_grad
```

各字段含义：
- `op`：算子名称
- `args`：输入和属性声明（用括号包裹，逗号分隔）
- `output`：输出声明
- `infer_meta.func`：形状推导函数名（对应 `paddle/phi/infermeta/` 下的 C++ 函数）
- `kernel.func`：PHI kernel 函数名
- `kernel.data_type`：kernel DataType 推断来源（此处取输入 `x` 的 dtype）
- `backward`：关联的反向算子名

## 新动态图 CodeGen（三层架构）

### 第一层：C++ API 生成

**生成器**：`paddle/phi/api/yaml/generator/api_gen.py`

**输入**：`ops.yaml`, `backward.yaml`, `op_compat.yaml`

**产出**：
- `paddle/phi/api/lib/api.cc`（前向 API）
- `paddle/phi/api/lib/api.h`（头文件）
- `paddle/phi/api/lib/backward_api.cc`（反向 API）

**生成内容**：每个算子对应一个 C++ 函数，内部包含完整的 kernel 选择 + 数据准备 + kernel 调用逻辑：

```cpp
// 自动生成的 paddle::experimental::add()
Tensor add(const Tensor& x, const Tensor& y) {
  // 1. ParseKernelKeyByInputArgs → KernelKey
  // 2. SelectKernelOrThrowError → Kernel
  // 3. PrepareData（TransDataPlace/Type/Layout）
  // 4. InferMeta（形状推导）
  // 5. kernel_fn(dev_ctx, x, y, out)
  return out;
}
```

### 第二层：动态图函数生成

**生成器**：`paddle/fluid/eager/auto_code_generator/generator/eager_gen.py`

**输入**：与第一层相同的 YAML 文件

**产出**：
- `paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.cc`
- `paddle/fluid/eager/api/generated/eager_generated/backwards/nodes.cc`（反向 Node 类）

**生成内容**：
- 前向函数：调用 C++ API 层 + 构建反向计算图（创建 GradNode、保存前向 Tensor）
- 反向 Node 类：继承 `egr::GradNodeBase`，实现 `operator()()` 调用反向 C++ API

### 第三层：Python-C 绑定生成

**生成器**：`paddle/fluid/pybind/generator/python_c_gen.py`

**输入**：同上 YAML + 第二层生成的函数签名

**产出**：
- `paddle/fluid/pybind/eager_op_function.cc`

**生成内容**：将动态图函数包装为 Python 可调用对象，处理：
- Python 对象到 C++ Tensor 的转换
- 属性类型解析（int / float / list / string）
- 关键字参数和默认值
- 错误消息和类型检查

## 静态图 CodeGen

静态图使用 Jinja2 模板引擎，脚本位于 `paddle/fluid/operators/generator/`：

| 脚本 | 功能 |
|------|------|
| `parse_op.py` | 解析 YAML 为内部 OpDef 数据结构 |
| `cross_validate.py` | 校验 ops.yaml 与 op_compat.yaml 一致性 |
| `generate_op.py` | 从 Jinja 模板生成 `generated_op.cc` |

模板文件位于 `paddle/fluid/operators/generator/templates/`，包含：
- `op.cc.j2`：OpMaker、InferShape、GetExpectedKernelType
- `op.h.j2`：Op 类声明

生成的文件注册到 fluid 的 `OpRegistry` 中，主要用于静态图执行和 save/load 兼容。

## CMake 构建集成

代码生成在 CMake configure 或 build 阶段触发，使用以下 CMake 命令：

### execute_process（configure 阶段）

```cmake
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${API_GEN_PY}
    --api_yaml_path ${OPS_YAML}
    --api_header_path ${API_HEADER}
    --api_source_path ${API_SOURCE}
)
```

在 `cmake ..` 时立即执行，适用于生成文件不频繁变化的场景。

### add_custom_command + add_custom_target（build 阶段）

```cmake
add_custom_command(
  OUTPUT ${API_SOURCE}
  COMMAND ${PYTHON_EXECUTABLE} ${API_GEN_PY} ...
  DEPENDS ${OPS_YAML} ${API_GEN_PY}
)
add_custom_target(api_gen ALL DEPENDS ${API_SOURCE})
```

在 `make` 时按依赖关系触发，只有 YAML 或生成器脚本修改后才重新生成。

### copy_if_different

```cmake
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TMP_FILE} ${FINAL_FILE}
)
```

先生成到临时文件，再与目标文件比较。内容相同则不覆盖，避免触发不必要的重编译。

## 关键文件路径汇总

| 层级 | 生成器脚本 | 产出目录 |
|------|-----------|---------|
| C++ API | `paddle/phi/api/yaml/generator/api_gen.py` | `paddle/phi/api/lib/` |
| 动态图函数 | `paddle/fluid/eager/auto_code_generator/generator/eager_gen.py` | `paddle/fluid/eager/api/generated/` |
| Python-C | `paddle/fluid/pybind/generator/python_c_gen.py` | `paddle/fluid/pybind/` |
| 静态图 | `paddle/fluid/operators/generator/generate_op.py` | `paddle/fluid/operators/generated/` |
| YAML 定义 | — | `paddle/phi/ops/yaml/` |

## 调试与开发建议

1. **修改 YAML 后重新生成**：执行 `make api_gen` 或完整 `make` 即可触发
2. **查看生成结果**：到 `build/` 目录下对应路径查看 `.cc` 文件
3. **调试生成器**：直接用 Python 运行生成器脚本，添加 `--help` 查看参数
4. **新增算子**：只需在 `ops.yaml` + `backward.yaml` 添加条目，编写 kernel 和 infer_meta，其余全部自动生成
5. **兼容旧算子**：在 `op_compat.yaml` 中添加新旧名称映射
