# 复数自动微分与 Wirtinger 导数

本文档说明 Paddle 中复数 Tensor 的自动微分原理，基于 Wirtinger Calculus（维尔廷格微积分），解释框架如何处理非全纯（non-holomorphic）函数的梯度。

## 问题背景

深度学习中的 loss 函数是**实值**的（输出为 R），但中间计算可能涉及复数（C）。例如傅里叶变换、复数注意力机制等。

**核心困难**：常见的 loss 函数是 C→R 映射（如 `|z|^2`），不满足柯西-黎曼方程，因此不是全纯函数（holomorphic），传统复变微积分的导数定义不适用。

## Wirtinger Calculus 基础

### 核心思想

将复数 z = a + bi 及其共轭 z* = a - bi 视为**两个独立变量**，定义两个偏导数：

| 名称 | 定义 | 符号 |
|------|------|------|
| Wirtinger 导数 | ∂f/∂z = (1/2)(∂f/∂a - i·∂f/∂b) | 对 z 的偏导 |
| 共轭 Wirtinger 导数 | ∂f/∂z* = (1/2)(∂f/∂a + i·∂f/∂b) | 对 z* 的偏导 |

其中 a = Re(z)，b = Im(z)。

### 关键性质

- 若 f 是全纯函数，则 ∂f/∂z* = 0，∂f/∂z 就是传统的复数导数
- 若 f 是反全纯函数（anti-holomorphic），则 ∂f/∂z = 0
- 对一般的可微函数，两个偏导数都非零

### 与实数梯度的关系

对实值函数 L: C→R，实数梯度为：

```
∂L/∂a = 2·Re(∂L/∂z*) = 2·Re(∂L/∂z)
∂L/∂b = 2·Im(∂L/∂z*) = -2·Im(∂L/∂z)
```

## 梯度下降中的应用

### 优化公式

对复数参数 z 的梯度下降更新规则：

```
z_{n+1} = z_n - 2·α · (∂L/∂z*)
```

其中 α 是学习率。因子 2 来源于 Wirtinger 导数的定义（含 1/2 系数）。

**等价形式**：也可以写为 `z_{n+1} = z_n - α · ∇_z L`，其中 ∇_z L = 2·(∂L/∂z*)。

### 为什么用 ∂L/∂z* 而非 ∂L/∂z？

对实值 loss L，∂L/∂z* 给出的方向是**最速下降方向**。直觉上，∂L/∂z* 编码了实部和虚部的梯度信息，使得沿其负方向移动能最快减小 L。

## 链式法则

设复合函数 L = L(s(z))，其中 s: C→C，L: C→R。链式法则为：

```
∂L/∂z* = (∂L/∂s)* · (∂s/∂z*) + (∂L/∂s) · (∂s/∂z)*

等价地：
∂L/∂z* = conj(output_grad) · (∂s/∂z*) + output_grad · conj(∂s/∂z)
```

其中 `output_grad = ∂L/∂s`，是上游传来的梯度。

**注意**：这比实数链式法则复杂——需要同时用到 output_grad 及其共轭，以及 s 对 z 和 z* 的偏导数。

## 各框架约定

不同深度学习框架在自动微分中计算的 Wirtinger 导数不同：

| 框架 | 反向传播计算的量 | 说明 |
|------|----------------|------|
| **PyTorch** | ∂L/∂z* (共轭 Wirtinger 导数) | `.grad` 存储的是 ∂L/∂z*，直接用于梯度下降 |
| **TensorFlow** | ∂L/∂z* (共轭 Wirtinger 导数) | 与 PyTorch 一致 |
| **JAX** | ∂L/∂z (Wirtinger 导数) | 需取共轭后才能用于梯度下降 |
| **Paddle** | ∂L/∂z* (共轭 Wirtinger 导数) | 与 PyTorch/TF 一致 |

**Paddle 的选择**：计算 ∂L/∂z*，与 PyTorch 保持一致。`.grad` 中存储的值可直接用于优化器更新。

## 特殊情况简化

### C→R 函数（如 loss 函数）

当 s: C→R（输出为实数），有 ∂L/∂s 为实数，因此 `conj(output_grad) = output_grad`。链式法则简化为：

```
∂L/∂z* = output_grad · [(∂s/∂z*) + conj(∂s/∂z)]
        = output_grad · ∂s/∂a    (其中 a = Re(z))
```

**实际含义**：对 C→R 函数，共轭 Wirtinger 导数退化为对实部的普通偏导数乘以 output_grad，与实数情况类似。

### R→C 函数

当 s: R→C（输入为实数），z = z*，两个 Wirtinger 导数合并：

```
∂L/∂z* = conj(output_grad) · (∂s/∂z*) + output_grad · conj(∂s/∂z)
```

由于输入为实数，最终需要取实部：

```
∂L/∂x = 2·Re(∂L/∂z*) = 2·Re[conj(output_grad) · (∂s/∂x)/2 + output_grad · conj((∂s/∂x)/2)]
       = 2·Re[output_grad · conj(∂s/∂x)]    (简化后)
```

### C→C 全纯函数

当 s 是全纯函数时，∂s/∂z* = 0，链式法则简化为：

```
∂L/∂z* = output_grad · conj(∂s/∂z)
```

这与实数反向传播形式一致——只需要传统导数的共轭。

## Paddle 中的实现

在 Paddle 的反向 Kernel 实现中，对涉及复数的算子，反向公式需遵循上述链式法则。具体体现为：

1. **grad kernel 接收的 `out_grad`**：是 ∂L/∂s*（上游传来的共轭 Wirtinger 导数）
2. **grad kernel 计算并返回**：∂L/∂z*（本层的共轭 Wirtinger 导数）
3. **对复数乘法等算子**：需要显式使用 `conj()` 操作

### 示例：复数乘法的反向

前向：`s = x * y`（逐元素复数乘法，全纯）

反向（计算 ∂L/∂x* 和 ∂L/∂y*）：
```
∂L/∂x* = out_grad · conj(y)    (因为 ∂s/∂x = y，全纯)
∂L/∂y* = out_grad · conj(x)    (因为 ∂s/∂y = x，全纯)
```

## 调试提示

- **梯度校验**：使用 `paddle.autograd.gradcheck` 时，复数 Tensor 的有限差分需要在实部和虚部两个方向分别扰动
- **梯度共轭问题**：如果梯度的虚部符号反了，检查是否混淆了 ∂L/∂z 和 ∂L/∂z*
- **与 PyTorch 对比**：Paddle 与 PyTorch 约定一致，可直接对比 `.grad` 数值
