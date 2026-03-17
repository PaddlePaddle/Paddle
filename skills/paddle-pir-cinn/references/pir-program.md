# PIR Program 与模型结构

## 设计原则

1. **严格 SSA**：每个 Value 有且仅有一个定义点（def-site），所有使用点（use-site）通过 use-chain 链接
2. **Pimpl 模式**：用户侧 `Value`/`OpResult`/`OpOperand` 只是轻量句柄，实际数据存于 `*Impl` 类
3. **三层架构**：面向用户的 API 层 → Pimpl 实现层 → 连续内存布局层

## Value 系统

### ValueImpl：SSA Value 的基础

```cpp
class ValueImpl {
  Type type_;              // 值的类型
  OpOperandImpl *first_user_;  // use-list 链表头
};
```

每个 `ValueImpl` 维护一个侵入式链表，串联所有使用该 Value 的 `OpOperandImpl`。

### OpResultImpl：Operation 的输出

Operation 的输出结果分为两种存储方式：

- **Inline Result**（index 0-5）：`OpInlineResultImpl` 直接存储在 Operation 前方的连续内存中，将 result index 编码在对象自身（利用低位 bit）
- **Out-of-line Result**（index >= 6）：`OpOutOfLineResultImpl` 存储在更前方的内存区域，持有显式的 `result_index_` 字段

```cpp
class OpInlineResultImpl : public ValueImpl {
  // index 编码在 ValueImpl 前方的内存标记中
  // 通过地址偏移可反向定位到所属 Operation
};

class OpOutOfLineResultImpl : public ValueImpl {
  uint32_t result_index_;
};
```

这种设计的关键优势：给定一个 `OpResult`，可以通过地址运算直接找到所属 `Operation`，无需额外指针。

### OpOperandImpl：Operation 的输入

```cpp
class OpOperandImpl {
  ValueImpl *source_;          // 指向被使用的 Value
  OpOperandImpl *next_user_;   // use-list 中的下一个使用者
  OpOperandImpl **back_addr_;  // 指向前一个节点的 next_user_ 指针（双向链表的"反向指针"）
  Operation *owner_;           // 所属 Operation
};
```

四个字段构成侵入式双向链表：

- `source_` → 指向定义点的 `ValueImpl`
- `next_user_` + `back_addr_` → 链表的前后链接
- `owner_` → 可快速回溯到使用该 Value 的 Operation

**遍历 use-chain**：从 `ValueImpl::first_user_` 出发，沿 `next_user_` 遍历所有使用者，每个使用者通过 `owner_` 获取所属 Operation。

## Operation 内存布局

Operation 采用连续内存分配，所有关联数据在一次 `malloc` 中完成：

```
低地址 ──────────────────────────────────────────── 高地址
[OpOutOfLineResults | OpInlineResults | Operation | OpOperands]
                                       ↑ this 指针
```

- `Operation` 的 `this` 指针位于中间
- 向低地址方向：先是 InlineResults (最多6个)，再是 OutOfLineResults
- 向高地址方向：紧跟 OpOperands 数组

### Operation 核心字段

```cpp
class Operation {
  DictionaryAttribute attrs_;   // 属性字典（sorted pairs, binary search）
  OpInfo info_;                 // 指向 OpInfoImpl 的指针
  uint32_t num_results_;        // 输出数量
  uint32_t num_operands_;       // 输入数量
  uint32_t num_regions_;        // Region 数量
  Block *parent_block_;         // 所属 Block
  Region *regions_;             // Region 数组（动态分配）
};
```

### OpInfo 与 OpInfoImpl

```cpp
class OpInfoImpl {
  InterfaceMap interface_map_;   // concept-model 多态
  HasTraitFunction has_trait_;
  VerifyFunction verify_;
  std::vector<InterfaceValue> interface_set_;
};
```

`OpInfo` 是 `OpInfoImpl*` 的轻量包装。`InterfaceMap` 内部按 `TypeID` 排序，查找 Interface 实现时用二分搜索定位 `Concept*`，再通过函数指针调用——这就是 **concept-model 多态** 的核心机制，替代 C++ 虚函数。

## 权重与参数管理

PIR Program 通过 `hash_map<StrAttribute, Variable*>` 管理模型权重：

```
Program
├── computation graph (ModuleOp → Block → Operations)
└── weights: {"linear.weight" → Variable*, "linear.bias" → Variable*, ...}
```

两个专用 Op 桥接计算图与权重：

- **`builtin.get_parameter("linear.weight")`** → 从权重表读取，产生一个 `Value`
- **`builtin.set_parameter(value, "linear.weight")`** → 将计算结果写回权重表

这种设计将权重存储与计算图解耦，便于序列化和分布式场景下的参数分片。

## 模型嵌套结构

```
Program
└── ModuleOp (顶层 Operation)
    └── Region[0]
        └── Block[0]
            ├── builtin.get_parameter("w")  → %0
            ├── pd_op.matmul(%input, %0)    → %1
            ├── cf.if(%cond)                → %2
            │   ├── Region[0] (then)
            │   │   └── Block[0]
            │   │       ├── pd_op.relu(%1) → %3
            │   │       └── cf.yield(%3)
            │   └── Region[1] (else)
            │       └── Block[0]
            │           ├── pd_op.tanh(%1) → %4
            │           └── cf.yield(%4)
            └── builtin.set_parameter(%2, "out")
```

嵌套规则：`Operation` → `Region` → `Block` → `Operation`，支持任意深度。Region 约束 Value 的作用域——内部 Region 可使用外部 Value（capture），但外部不能使用内部 Value。

## Alias/Inplace 机制

PIR 通过 **view 语义** 处理 Tensor 别名和原地操作：

- `v_tensor` 类型：表示"view tensor"，与原始 Tensor 共享底层存储
- `InplaceTrait`：标记原地操作的 Op（如 `pd_op.relu_`）
- View Ops：`reshape`、`slice`、`transpose` 等产生 `v_tensor`，不拷贝数据

编译器在做 buffer 分配和内存优化时，需要追踪 view 关系以避免错误的内存复用。

## ProgramTranslator：旧 IR 到 PIR 的翻译

`ProgramTranslator` 负责将旧的 `ProgramDesc`（protobuf 描述的静态图）翻译为 `pir::Program`：

### 翻译流程

```
ProgramDesc (旧 IR)
  │
  ├─ 遍历每个 Block 中的 OpDesc
  │   │
  │   ├─ OpTranslator::Translate(op_desc)
  │   │   ├── 查找特化翻译器 (special handlers)
  │   │   │   例如: while → WhileOp, conditional_block → IfOp
  │   │   └── 回退通用翻译器 (general handler)
  │   │       根据 OpDesc 属性构造对应的 pir::Operation
  │   │
  │   └─ 维护 VarName → pir::Value 映射表
  │
  └─ 处理 sub_block 递归翻译（控制流 Op）
```

### OpTranslator 策略

- **通用处理器**：按 OpDesc 的 inputs/outputs/attrs 一一映射，适用于大部分算术算子
- **特化处理器**：处理语义差异较大的 Op，如：
  - `while` → `cf.while`（需要构造 Region 和 BlockArgument）
  - `fetch` → `builtin.set_parameter`
  - `feed` → `builtin.get_parameter`
  - `conditional_block` → `cf.if`

特化处理器通过注册表 `OpTranslator::special_handlers_` 管理，翻译时优先查找特化处理器，找不到则走通用路径。
