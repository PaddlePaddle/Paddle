# 静态图 Program 构建

## 静态图 vs 动态图

Paddle 静态图采用 **compile-then-execute（先编译、后执行）** 范式：

1. **编译阶段**：Python 代码不直接执行计算，而是通过 `append_op` 等接口向 Program 中追加算子描述（OpDesc），构建完整的计算图。
2. **执行阶段**：将构建好的 Program 交给 Executor，由 C++ 端创建真实的算子对象并在设备上调度执行。

与之对比，动态图（eager mode）在每一行 Python 代码执行时立即计算并返回结果，调试直观但难以做全局优化。静态图的优势在于：编译期可对全图进行依赖分析、算子融合、内存复用等优化。

## Program 的两大角色

Paddle 静态图中，同时存在两个 Program：

| Program | 用途 | 包含的算子类型 |
|---------|------|--------------|
| **main_program** | 承载完整训练逻辑 | 前向算子 + 反向算子 + 优化器算子 |
| **startup_program** | 参数初始化 | 初始化算子（fill_constant、uniform_random 等） |

典型训练流程：先用 Executor 运行 `startup_program` 完成参数初始化，再循环运行 `main_program` 进行训练迭代。

## 核心类层次

```
ProgramDesc
 ├─ proto_: framework::proto::ProgramDesc    ← protobuf 序列化描述
 └─ blocks_: vector<BlockDesc*>
      └─ BlockDesc
           ├─ proto_: framework::proto::BlockDesc
           ├─ ops_: deque<OpDesc*>           ← 算子描述列表（有序）
           └─ vars_: map<string, VarDesc*>   ← 变量描述字典
                ├─ OpDesc
                │   ├─ type_: string         ← 算子类型名（如 "matmul_v2"）
                │   ├─ inputs_: map<string, vector<string>>  ← 输入端口 → 变量名列表
                │   ├─ outputs_: map<string, vector<string>> ← 输出端口 → 变量名列表
                │   └─ attrs_: map<string, Attribute>        ← 属性字典
                └─ VarDesc
                    ├─ name_: string
                    ├─ type_: VarType        ← LOD_TENSOR, SELECTED_ROWS 等
                    ├─ shape_: vector<int64>
                    ├─ dtype_: DataType
                    └─ persistable_: bool    ← 参数 = true, 中间变量 = false
```

**BlockDesc** 是逻辑作用域。Block 0 为全局 block；控制流算子（`while_op`、`cond`）会创建子 block（Block 1, 2, ...），子 block 通过 `parent_idx` 指向父 block。

## 图构建调用链示例：nn.Linear

以 `paddle.nn.Linear(in_features=784, out_features=10)` 为例，说明 Python 端如何将一个层的定义转化为 Program 中的 OpDesc / VarDesc。

### 阶段一：`Linear.__init__` — 创建参数

```
Linear.__init__(in_features=784, out_features=10)
  │
  ├─ self.weight = self.create_parameter(shape=[784, 10], ...)
  │     └─ Block.create_parameter(...)
  │           └─ Block.create_var(name="linear_0.w_0", shape=[784,10],
  │                               dtype=float32, persistable=True)
  │                 └─ 创建 VarDesc，写入 BlockDesc.vars_
  │
  └─ self.bias = self.create_parameter(shape=[10], ...)
        └─ 同上，创建 "linear_0.b_0" VarDesc
```

此时 startup_program 中同步追加了初始化算子（如 `uniform_random`），用于在执行时初始化这些参数的真实值。

### 阶段二：`Linear.forward` — 追加计算算子

```
Linear.forward(input)
  │
  ├─ paddle.nn.functional.linear(input, weight, bias)
  │     └─ _matmul_static(input, weight)
  │           └─ LayerHelper.append_op(type="matmul_v2",
  │                 inputs={"X": input, "Y": weight},
  │                 outputs={"Out": out_var})
  │                 └─ Block.append_op(...)
  │                       ├─ 创建 OpDesc(type="matmul_v2")
  │                       ├─ 设置 inputs / outputs / attrs
  │                       └─ 追加到 BlockDesc.ops_ 尾部
  │
  └─ elementwise_add(matmul_out, bias)
        └─ Block.append_op(type="elementwise_add", ...)
```

每次 `append_op` 调用都会为输出创建新的 VarDesc（中间变量，`persistable=False`），并将 OpDesc 追加到当前 Block 的算子列表中。

### 阶段三：`optimizer.minimize()` — 追加反向与优化器算子

```
optimizer.minimize(loss)
  │
  ├─ backward(loss)
  │     ├─ 从 loss 所在的 OpDesc 反向遍历 main_program 中的算子列表
  │     ├─ 为每个前向 OpDesc 创建对应的反向 OpDesc（如 matmul_v2_grad）
  │     └─ 生成梯度变量 VarDesc（如 "linear_0.w_0@GRAD"）
  │
  └─ _apply_optimize(optimizer_ops)
        ├─ 为每个参数追加优化器算子（如 adam）
        │   OpDesc(type="adam",
        │     inputs={"Param": w, "Grad": w@GRAD, "Moment1": ..., "Moment2": ...},
        │     outputs={"ParamOut": w, ...})
        └─ 追加到 main_program 的 Block 0
```

最终 `main_program` 的算子序列为：`前向算子 → 反向算子 → 优化器算子`，形成完整的训练迭代描述。

## Scope：运行时变量容器

Scope 是 Executor 执行时存放真实 Variable 对象的**多叉树结构**：

```
Root Scope (全局)
 ├─ "linear_0.w_0"  → Variable (persistable, 常驻)
 ├─ "linear_0.b_0"  → Variable (persistable, 常驻)
 └─ Child Scope (每次 Executor.run 可创建)
      ├─ "tmp_0"    → Variable (中间变量, 执行后可回收)
      ├─ "tmp_1"    → Variable
      └─ ...
```

- **Root Scope** 持有 persistable 参数，跨迭代保留。
- **Child Scope** 持有中间变量，迭代结束后可被 GC 回收，实现内存复用。
- 查找变量时沿 parent 指针向上搜索，类似编程语言的作用域链。

## 关键源码路径

| 模块 | 路径 |
|------|------|
| ProgramDesc / BlockDesc | `paddle/fluid/framework/program_desc.cc`, `block_desc.cc` |
| OpDesc / VarDesc | `paddle/fluid/framework/op_desc.cc`, `var_desc.cc` |
| Proto 定义 | `paddle/fluid/framework/framework.proto` |
| Python 层 Program / Block | `python/paddle/base/framework.py` |
| LayerHelper (append_op 入口) | `python/paddle/base/layer_helper_base.py` |
| Scope | `paddle/fluid/framework/scope.cc` |
| backward 实现 | `python/paddle/autograd/backward_utils.py` |
