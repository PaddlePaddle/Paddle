# Inplace 操作与版本追踪

本文档说明 Paddle 中 Inplace 操作的实现机制、API 约定、叶 Tensor 约束以及版本追踪系统。

## 三种操作类型

| 类型 | 说明 | 示例 |
|------|------|------|
| **Normal** | 创建新 Tensor 存放结果 | `y = paddle.add(x, other)` |
| **Inplace** | 结果直接写入输入 Tensor 的内存 | `x.add_(other)` |
| **View** | 与原 Tensor 共享内存，shape 可能不同 | `y = x.reshape(...)` |

## Inplace 操作的优势

- **减少内存占用**：不分配新的存储空间，原地修改
- **避免分配开销**：省去内存分配和释放的时间
- **适用场景**：参数更新（`param.add_(grad)`）、激活函数原地替换等

## API 约定

Paddle 的 Inplace 操作遵循**下划线后缀**约定：

| 普通版本 | Inplace 版本 |
|---------|-------------|
| `paddle.add(x, y)` | `paddle.add_(x, y)` 或 `x.add_(y)` |
| `paddle.scale(x, s)` | `x.scale_(s)` |
| `paddle.relu(x)` | `paddle.nn.functional.relu_(x)` |
| `paddle.zero_like(x)` | `x.zero_()` |

Inplace 方法返回修改后的自身 Tensor（方便链式调用），但底层内存已被原地修改。

## 叶 Tensor 约束

**规则：不能对需要梯度的叶 Tensor 执行 inplace 操作。**

```python
x = paddle.randn([3, 4])
x.stop_gradient = False  # x 是叶 Tensor，需要梯度

x.add_(1.0)  # RuntimeError! 不允许
```

**原因**：叶 Tensor 被 `TensorWrapper` 保存用于反向计算。如果原地修改了叶 Tensor 的值，反向计算时读到的就是被修改后的错误值，导致梯度错误。

**非叶 Tensor 可以 inplace**：
```python
x = paddle.randn([3, 4])
x.stop_gradient = False
y = x * 2        # y 是非叶 Tensor
y.add_(1.0)      # 允许，但会触发版本检查
```

## 版本追踪机制

为了检测 inplace 修改带来的数据一致性问题，Paddle 使用**版本号**（inplace version）追踪 Tensor 的修改历史。

### 核心组件

```
Tensor
  └─ impl_ (TensorBase)
       └─ inplace_version_counter_  ← 每次 inplace 操作 +1

TensorWrapper（保存前向 Tensor 用于反向）
  └─ inplace_version_snapshot_     ← 保存时记录版本号
```

### 工作流程

1. **保存时快照**：`TensorWrapper` 构造时，记录当前 Tensor 的 `inplace_version` 到 `inplace_version_snapshot_`。

2. **Inplace 操作递增版本号**：
   ```cpp
   // 每个 inplace 操作（如 add_）在执行后调用
   tensor.bump_inplace_version();
   // 内部：inplace_version_counter_++
   ```

3. **反向恢复时校验**：`TensorWrapper::recover()` 被调用时（反向计算需要该 Tensor），比较当前版本与快照：
   ```cpp
   void TensorWrapper::recover() {
       if (tensor.current_inplace_version() != inplace_version_snapshot_) {
           PADDLE_THROW(
               "Tensor has been modified by inplace operation. "
               "Its version is %d but expected %d.",
               tensor.current_inplace_version(),
               inplace_version_snapshot_);
       }
       // 版本一致，安全返回 Tensor
   }
   ```

4. **版本不一致时报错**：抛出异常，提示用户某个 Tensor 在前向保存后被 inplace 修改，反向计算不安全。

### 示例

```python
x = paddle.randn([3, 4])
x.stop_gradient = False
y = x * 2            # y 的 TensorWrapper 保存了 x，版本快照 = 0
x.add_(1.0)          # x.inplace_version = 1（但 x 是叶 Tensor，实际会报错）

# 若绕过叶 Tensor 检查：
# y.backward() 时 TensorWrapper.recover() 发现版本 1 != 快照 0，报错
```

## 调试提示

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `Leaf Tensor that requires grad has been used in an inplace operation` | 对需要梯度的叶 Tensor 做了 inplace | 改用 normal 版本，或先 `.detach()` |
| `Tensor version mismatch` | Tensor 在前向保存后被 inplace 修改 | 找到修改该 Tensor 的 inplace 操作，改为 normal 版本 |
| 梯度数值不正确 | 可能有未检测到的 inplace 修改 | 检查所有带 `_` 后缀的操作 |
