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

## 辅助类型与 Op

控制流 Dialect (`cf`) 定义了以下辅助类型和 Op：

| 名称 | 用途 |
|------|------|
| `cf.StackType` | 栈类型，用于前反向传值（LIFO 语义） |
| `cf.CreateStackOp` | 创建一个空栈 |
| `cf.PushBackOp` | 向栈中压入一个 Tensor |
| `cf.PopBackOp` | 从栈中弹出一个 Tensor |
| `cf.YieldOp` | Region 的通用终止符，返回值给外层 Operation |
| `cf.CondYieldOp` | 条件终止符，携带 bool 条件和返回值 |

## IfOp：条件分支

```
%result = cf.if(%condition) {
  // Region[0]: then branch
  %a = pd_op.relu(%x)
  cf.yield(%a)
} else {
  // Region[1]: else branch
  %b = pd_op.tanh(%x)
  cf.yield(%b)
}
```

### IfOp 结构

- **输入**：1 个 bool 类型的 condition Value
- **Region 数量**：2 或 3 个
  - Region[0]：then 分支（必需）
  - Region[1]：else 分支（必需）
  - Region[2]：init 分支（可选，用于初始化前反向共享变量）
- **输出**：`cf.yield` 返回的值，两个分支的返回类型必须一致
- **terminator**：每个分支的最后一个 Op 必须是 `cf.yield`

### 带 init Region 的 IfOp（反向场景）

```
%result, %stack = cf.if(%condition) [init] {
  // Region[2]: init
  %s = cf.create_stack()
  cf.yield(%s)
} then {
  // Region[0]: then
  %a = pd_op.relu(%x)
  cf.push_back(%stack, %a)    // 保存中间变量供反向使用
  cf.yield(%a)
} else {
  // Region[1]: else
  %b = pd_op.tanh(%x)
  cf.push_back(%stack, %b)
  cf.yield(%b)
}
```

## WhileOp：循环

```
%results = cf.while(%init_val) {
  // Region[1]: cond（循环条件）
  ^bb(%iter_arg):
    %c = pd_op.less_than(%iter_arg, %limit)
    cf.cond_yield(%c, %iter_arg)   // 条件 + 要传递给 body 的值
} do {
  // Region[2]: body（循环体）
  ^bb(%body_arg):
    %new_val = pd_op.add(%body_arg, %step)
    cf.yield(%new_val)             // 值回传给 cond 的 BlockArgument
}
```

### WhileOp 结构

- **输入**：循环初始值（映射为 cond Region 的 BlockArgument）
- **Region 数量**：2 或 3 个
  - Region[0]：init 分支（可选，同 IfOp）
  - Region[1]：cond 分支（循环条件判断）
  - Region[2]：body 分支（循环体）
- **cond terminator**：`cf.cond_yield(bool_condition, pass_values...)`
  - 如果 condition 为 true → 进入 body，`pass_values` 成为 body 的 BlockArgument
  - 如果 condition 为 false → 退出循环，`pass_values` 成为 WhileOp 的输出
- **body terminator**：`cf.yield(new_values...)`，新值回传给 cond 的 BlockArgument，开始下一次迭代

### 数据流循环

```
init_val → cond BlockArg → cond_yield values → body BlockArg
                ↑                                    │
                └────── cf.yield(new_values) ────────┘
```

## 反向支持：Stack 机制

控制流的反向求导面临一个核心问题：前向执行中的局部变量（如循环体内的中间 Tensor）在反向时可能需要使用，但由于 Region 作用域限制，反向 Region 无法直接访问前向 Region 的 Value。

PIR 使用 **Stack 机制**（LIFO 栈）解决此问题。

### 三步构造过程

#### Step 1：修改前向——Push 中间变量

在前向控制流中插入 `cf.push_back` 操作，将反向需要的中间变量压入 Stack：

```
// 前向 WhileOp body（修改后）
^bb(%x):
  %y = pd_op.relu(%x)
  cf.push_back(%stack, %x)    // 保存 x 供反向使用
  cf.push_back(%stack, %y)    // 保存 y 供反向使用
  cf.yield(%y)
```

#### Step 2：构造反向——Pop 中间变量

反向控制流中通过 `cf.pop_back` 按 LIFO 顺序取出前向保存的变量：

```
// 反向 WhileOp body
^bb(%dy):
  %y = cf.pop_back(%stack)    // 后入先出：先 pop y
  %x = cf.pop_back(%stack)    // 再 pop x
  %dx = pd_op.relu_grad(%x, %y, %dy)
  cf.yield(%dx)
```

LIFO 语义保证了在循环场景下，反向迭代（逆序）取出的变量与前向迭代（正序）的变量正确对应。

#### Step 3：剪枝——移除未使用的 Op

反向图构建完成后，执行 DCE（Dead Code Elimination）Pass，移除前向中不被反向使用的 `push_back` 以及对应的 `create_stack`，减少不必要的内存开销。

### Stack 机制的优势

| 特性 | 说明 |
|------|------|
| 作用域安全 | Stack 在 init Region 中创建，对所有子 Region 可见 |
| LIFO 语义 | 天然匹配循环反向的逆序访问模式 |
| 可剪枝 | 未使用的 Stack 可在编译期移除 |
| 统一处理 | IfOp 和 WhileOp 使用相同的 Stack 机制 |
