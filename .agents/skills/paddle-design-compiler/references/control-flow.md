# PIR 控制流设计

PIR 通过 Region 嵌套实现结构化控制流，避免传统 CFG 中的 phi 节点和复杂跳转。

## Block 与 Region 基础

### Block

```cpp
class Block {
  std::list<Operation *> ops_;       // Operation 链表
  std::vector<BlockArgument> args_;  // Block 参数（类似 MLIR 的 BlockArgument）
  Region *parent_region_;            // 所属 Region
};
```

- Block 内的 Operation 按顺序排列
- 最后一个 Operation 必须是 **terminator**（如 `cf.yield`）
- `BlockArgument` 是 Block 的输入 Value，由外层 Operation 定义

### Region

```cpp
class Region {
  std::vector<Block *> blocks_;
  Operation *parent_op_;  // 所属 Operation
};
```

Region 是 Block 的有序容器。作用域规则：Region 内部定义的 Value **不能**被外部引用，但内部可以**捕获**外部 Value（类似闭包语义）。

## 控制流 Op 分布

控制流相关 Op 分布在两个 Dialect 中：

| Op 名称 | 所属 Dialect | 用途 |
|---------|-------------|------|
| `pd_op.if` | PaddleDialect | 条件分支 |
| `pd_op.while` | PaddleDialect | 循环 |
| `cf.has_elements` | PaddleDialect（定义在 control_flow_op.h） | 判断 Stack 是否有元素（反向 while 条件） |
| `pd_op.assert` | PaddleDialect | 断言 |
| `cf.yield` | ControlFlowDialect | Region 通用终止符，返回值给外层 Operation |
| `cf.stack_create` | ControlFlowDialect | 创建 Stack（含 inlet/outlet 两端） |
| `cf.tuple_push` | ControlFlowDialect | 向 Stack 的 inlet 端压入值 |
| `cf.tuple_pop` | ControlFlowDialect | 从 Stack 的 outlet 端弹出值 |

## IfOp：条件分支

```
%result = pd_op.if(%condition) {
  // Region[0]: true_region (then branch)
  %a = pd_op.relu(%x)
  cf.yield(%a)
} else {
  // Region[1]: false_region (else branch)
  %b = pd_op.tanh(%x)
  cf.yield(%b)
}
```

### IfOp 结构

- **输入**：1 个 bool 类型的 condition Value
- **Region 数量**：固定 2 个
  - Region[0]：`true_region`（then 分支）
  - Region[1]：`false_region`（else 分支）
- **输出**：`cf.yield` 返回的值，两个分支的返回类型必须一致
- **terminator**：每个分支的最后一个 Op 必须是 `cf.yield`

### 源码定义

```
paddle/fluid/pir/dialect/operator/ir/control_flow_op.h — IfOp
  ├── cond()         → 获取条件 Value
  ├── true_region()  → Region[0]
  ├── false_region() → Region[1]
  ├── true_block()   → Region[0] 的 Block
  └── false_block()  → Region[1] 的 Block
```

## WhileOp：循环

```
// 语义等价于：
// outputs = inputs
// while(cond) {
//   cond, outputs = body(outputs)
// }

%results = pd_op.while(%cond, %init_val) {
  // Region[0]: body
  ^bb(%iter_arg):
    %new_val = pd_op.add(%iter_arg, %step)
    %new_cond = pd_op.less_than(%new_val, %limit)
    cf.yield(%new_cond, %new_val)   // 第一个返回值更新 cond，其余更新 loop_vars
}
```

### WhileOp 结构

- **输入**：cond Value + loop_vars（循环初始值）
- **Region 数量**：固定 1 个
  - Region[0]：body（循环体）
- **body BlockArgument**：与 loop_vars 一一对应
- **body terminator**：`cf.yield(new_cond, new_values...)`
  - 第一个返回值更新循环条件
  - 其余返回值更新 loop_vars，成为下一次迭代的 BlockArgument
  - 当 cond 为 false 时退出循环，最后的 new_values 成为 WhileOp 的输出

### 数据流循环

```
cond, init_vals → WhileOp
                    └── body Region
                        ├── BlockArgs = loop_vars
                        ├── ... 计算 ...
                        └── cf.yield(new_cond, new_vals)
                              │
                              ├── new_cond=true → 回到 body（new_vals → BlockArgs）
                              └── new_cond=false → 退出（new_vals → WhileOp outputs）
```

## 反向支持：Stack 机制

控制流的反向求导面临一个核心问题：前向执行中的局部变量（如循环体内的中间 Tensor）在反向时可能需要使用，但由于 Region 作用域限制，反向 Region 无法直接访问前向 Region 的 Value。

PIR 使用 **Stack 机制**（基于 `cf.stack_create` / `cf.tuple_push` / `cf.tuple_pop`）解决此问题。

### Stack 创建

```
(%stack, %inlet, %outlet) = cf.stack_create()
```

`cf.stack_create` 创建一个 Stack 容器，返回三个值：
- `%stack`：Stack 引用（用于 `cf.has_elements` 查询）
- `%inlet`：入口端（用于 `cf.tuple_push` 压入）
- `%outlet`：出口端（用于 `cf.tuple_pop` 弹出）

### 三步构造过程

#### Step 1：修改前向——Push 中间变量

在前向控制流中插入 `cf.tuple_push` 操作，将反向需要的中间变量压入 Stack：

```
// 前向 WhileOp body（修改后）
^bb(%x):
  %y = pd_op.relu(%x)
  cf.tuple_push(%inlet, %x)    // 保存 x 供反向使用
  cf.tuple_push(%inlet, %y)    // 保存 y 供反向使用
  cf.yield(%new_cond, %y)
```

#### Step 2：构造反向——Pop 中间变量

反向控制流中通过 `cf.tuple_pop` 按 LIFO 顺序取出前向保存的变量：

```
// 反向 WhileOp body
^bb(%dy):
  %y = cf.tuple_pop(%outlet)   // 后入先出：先 pop y
  %x = cf.tuple_pop(%outlet)   // 再 pop x
  %dx = pd_op.relu_grad(%x, %y, %dy)
  cf.yield(%has_elements, %dx)
```

反向 WhileOp 的循环条件通过 `cf.has_elements(%stack)` 判断 Stack 是否还有元素。

#### Step 3：剪枝——移除未使用的 Op

反向图构建完成后，执行 DCE（Dead Code Elimination）Pass，移除前向中不被反向使用的 `tuple_push` 以及对应的 `stack_create`，减少不必要的内存开销。

### Stack 机制的优势

| 特性 | 说明 |
|------|------|
| 作用域安全 | Stack 在控制流 Op 外部创建，对所有子 Region 可见 |
| LIFO 语义 | 天然匹配循环反向的逆序访问模式 |
| 可剪枝 | 未使用的 Stack 可在编译期移除 |
| 统一处理 | IfOp 和 WhileOp 使用相同的 Stack 机制 |

## 关键源码路径

| 文件 | 说明 |
|------|------|
| `paddle/fluid/pir/dialect/operator/ir/control_flow_op.h` | IfOp、WhileOp、HasElementsOp、AssertOp 定义 |
| `paddle/pir/include/dialect/control_flow/ir/cf_op.h` | YieldOp、StackCreateOp、TuplePushOp、TuplePopOp 定义 |
| `paddle/pir/include/dialect/control_flow/ir/cf_type.h` | StackType 类型定义 |
