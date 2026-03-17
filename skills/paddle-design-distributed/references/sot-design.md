# PaddleSOT（Symbolic Opcode Translator）设计详解

## 背景与动机

Paddle 的动转静（dy2st）最初基于 **AST Transformer**：将 Python 源码解析为 AST，通过语法树变换生成静态图代码。这种方式存在根本性局限：

1. **numpy/Tensor 互操作**：AST 层面无法区分 `x[0]` 是 numpy 索引还是 Tensor 索引
2. **控制流边界**：嵌套函数、闭包、装饰器等场景难以正确转换
3. **动态 shape**：`-1` 等动态形状信息在 AST 层面无法推断
4. **第三方库**：调用 numpy、scipy 等库的代码无法直接转换

PaddleSOT 采用 **字节码级别** 的方案：不分析源码，而是在 Python VM 执行字节码时进行拦截和模拟，精确追踪每个操作的语义。

## PEP 523：自定义帧求值

Python 3.6+ 通过 PEP 523 提供了自定义帧求值的能力：

```c
// Python 解释器状态中的钩子
PyInterpreterState.eval_frame = custom_eval_frame_func;
```

当 Python 执行每一帧（函数调用）时，会调用 `eval_frame` 函数。PaddleSOT 在此注入自己的逻辑：

1. **检查 Cache**：该帧是否已有编译结果？Guard 条件是否满足？
2. **命中**：直接执行缓存的 StaticFunction
3. **未命中**：启动 OpcodeExecutor 模拟执行

## OpcodeExecutor：字节码模拟执行

OpcodeExecutor 是 SOT 的核心，它 **模拟 Python VM 的字节码执行**，但不真正执行计算，而是：

### Variable 体系

将 Python 对象包装为 Variable，在模拟层面追踪：

| Variable 类型 | 包装对象 | 说明 |
|--------------|---------|------|
| `TensorVariable` | `paddle.Tensor` | 核心，记录到 FunctionGraph |
| `ConstantVariable` | `int`, `float`, `str`, `None` | 常量直接内联 |
| `ContainerVariable` | `list`, `dict`, `tuple` | 递归追踪容器中的元素 |
| `CallableVariable` | 函数 / 方法 | 模拟函数调用行为 |
| `ModuleVariable` | `nn.Layer` | 追踪模块的子层和参数 |

### Tracker 系统

每个 Variable 持有一个 **Tracker**，记录其来源（provenance）。Tracker 形成 DAG 结构：

```
x = args[0]           → LocalTracker("x")
y = x.shape[0]        → GetAttrTracker(LocalTracker("x"), "shape") → GetItemTracker(..., 0)
z = paddle.add(x, y)  → DummyTracker()  # 计算结果由 FunctionGraph 追踪
```

**Tracker 的核心用途**：生成 Guard。从叶节点回溯到根节点，即可生成一条检查链，验证运行时输入是否满足编译假设。

### FunctionGraph 与 StatementIR

OpcodeExecutor 在模拟执行过程中，将 Tensor 相关操作记录到 **FunctionGraph** 中。FunctionGraph 最终输出 **StatementIR**。

StatementIR 包含 4 种语句类型：

| 语句类型 | 含义 | 示例 |
|---------|------|------|
| `call_api` | 调用 Paddle API | `paddle.add(x, y)` |
| `call_method` | 调用 Tensor 方法 | `x.reshape([2, 3])` |
| `call_sir` | 调用子 StatementIR（嵌套子图） | 内联函数的子图 |
| `call_layer` | 调用 nn.Layer 的 forward | `self.linear(x)` |

StatementIR 最终被转换为 **StaticFunction**（静态图可执行单元），缓存供后续复用。

## OpcodeInlineExecutor：跨函数子图融合

当 OpcodeExecutor 遇到函数调用时，会创建 **OpcodeInlineExecutor** 进入被调用函数内部模拟执行。这使得 SOT 能够：

- 跨函数边界追踪 Tensor 操作
- 将多个函数的操作融合到同一个子图中
- 避免不必要的子图切分

## Sub-graph Fallback 场景

当 OpcodeExecutor 遇到无法模拟的操作时，会触发 **sub-graph fallback**：将当前已收集的子图编译执行，无法模拟的部分交给 Python 原生执行，随后开始新的子图收集。

| 缩写 | 全称 | 场景 |
|------|------|------|
| **DDCF** | Data-Dependent Control Flow | 控制流条件依赖 Tensor 值（如 `if x.sum() > 0`） |
| **UNSPS** | Unsupported Simulation | 无法模拟的 Python 操作（如某些 C 扩展） |
| **CDBL** | Custom Blacklist | 用户或框架标记的不转换函数 |
| **UNIMP** | Unimplemented Opcode | 尚未实现模拟的字节码指令 |

Fallback 是 SOT 的安全兜底机制：**任何无法处理的情况都不会导致报错**，而是退化为部分子图编译 + 部分 Python 执行。

## Guard 与 Cache

### Guard 定义

Guard 是一个 `Callable[[FrameType], bool]`，判断当前帧的输入是否满足编译时的假设。

```python
# 概念示例
def guard(frame: FrameType) -> bool:
    x = frame.f_locals["x"]
    return isinstance(x, paddle.Tensor) and x.shape == [2, 3] and x.dtype == paddle.float32
```

### Guard 生成

每个 Variable 的 Tracker 链自动生成 sub-guard：

1. 回溯 Tracker DAG 到根节点（函数参数）
2. 每一步生成一个检查条件（类型检查、shape 检查、值检查等）
3. 所有 sub-guard 通过 **AND** 组合成完整 Guard

### Cache 机制

每个帧维护一个 `(Guard, StaticFunction)` 列表：

```
frame_cache = [
    (guard_1, compiled_fn_1),
    (guard_2, compiled_fn_2),
    ...
]
```

执行时依次检查 Guard，首个满足的即命中缓存，直接执行对应的 StaticFunction。

## SideEffect 模块

Python 代码可能修改全局变量、修改可变对象（list.append 等）。SOT 的 **SideEffect** 模块负责：

1. **记录**：在模拟执行过程中，记录所有对全局变量和可变对象的修改
2. **回放**：在生成的 StaticFunction 执行后，按记录顺序回放这些副作用

这确保了动转静的语义等价性：编译后的代码和原始 Python 代码产生相同的副作用。

## 使用方式

```python
import paddle

@paddle.jit.to_static(fallback=True)
def train_step(net, x, label):
    pred = net(x)
    loss = paddle.nn.functional.cross_entropy(pred, label)
    loss.backward()
    return loss

# fallback=True 启用 SOT 模式（字节码级别转换 + 自动 fallback）
# fallback=False 使用传统 AST Transformer
```

**关键路径**：
- eval_frame 入口：`python/paddle/jit/sot/opcode_translator/eval_frame_callback.py`
- OpcodeExecutor：`python/paddle/jit/sot/opcode_translator/executor/opcode_executor.py`
- Variable 体系：`python/paddle/jit/sot/opcode_translator/executor/variables/`
- Tracker：`python/paddle/jit/sot/opcode_translator/executor/tracker.py`
- Guard：`python/paddle/jit/sot/opcode_translator/executor/guard.py`
- FunctionGraph：`python/paddle/jit/sot/opcode_translator/executor/function_graph.py`
