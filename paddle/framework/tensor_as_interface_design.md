# LODTensor 及周边设计
本设计尝试综合之前的讨论和目前实现的现状，提供一个简单但相对有扩展性的版本，其中有以下一些需要注意的点：

- 基本设计完全按照8-11（上周五）讨论的方案，只有如下变化
  - 由于目前只有2种tensor，难以确保将来是否有其他类似 `LOD` 角色的Tensor，因此本设计不会将 `LODTensor` 直接作为 Interface，而是将最基础的Tensor概念作为Interface
- 目前实现的Tensor会更名为 `DenseTensor`
- 目前的 LODTensor 会更名为 `LODDenseTensor`
- 为了实现简单，LODDenseTensor 会继承自DenseTensor，将来可以改成组合
- 更复杂的类型问题，将推迟到有了2个以上Tensor 类型的时候考虑

## Tensor 作为Interface
`Tensor interface` 直接对应其他平台的相同概念，但LODTensor无法； 稠密的为 DenseTensor，稀疏的为 SparseTensor。


**为什么不用 LODTensor 作为 Interface？**

- 目前只有两个 Tensor，LODTensor
  - 如果将LODTensor作为interface，难确保将来不会出现和`LOD`类似的语义
  - 但 `Tensor` 作为基础概念作为 interface会更合理一些
  - **两者带来的实现代价类似**

## LODDenseTensor 

LODDenseTensor 会直接沿用目前的实现，即直接继承 `DenseTensor`，这里主要是为了更简单地实现 Tensor Interface规定的接口。

完整跑通之后，如果有必要可以增加代码改写为组合的实现。

## 现有代码迁移
### Tensor替换为 DenseTensor
现有所有类似以 `Tensor` 作为模板参数的操作，比如 `GetMutable<Tensor>` 全部替换为 `GetMutable<DenseTensor>`
### LODTensor 替换为 LODDenseTensor
### 在Variable中添加Clone接口

```c++
class Variable {
 public:

  // ...
  template <typename T>
  void Clone(const Variable& other) {
    holder_.reset(new PlaceholderImpl<T>(other.Clone()));
  }
  // ...
};
```

### 所有的InferShape里在最开始添加var 的Clone，以实现类型传递

```c++
T* VarClone(const Variable& var){
  return var.Get<Tensor>()->Clone();
}

void InferShape() {
  // ...
  // transfer LOD if input_var is a LODDenseTensor
  // user should be aware that each output' tensor type is the same as input
  output_var.Clone(input_var);
  // ...
}
```

### 添加VarGetMutable函数支持Variable的继承关系

目前只有两种 tensor，因此直接裸写 if-else 实现

为了把 Tensor的实现与 Variable 解耦，另外增加一个函数

```c++
template <typename T>
bool IsInherienceOf(const Variable& child_type_var) {
  // several if-else
  if (xxxx) return true;
  return false;
}

template <typename T>
T* VarGetMutable(Variable* var, bool enable_inherience = true) {
  if (enable_inherience && IsInherienceOf<T>(var)) {
    return var.Get<T>();
  }
  return var.GetMutable<T>();
}
```
## 未来扩展
两个维度，如果有新的tensor 类型，只需要如下两步支持
- 继承 `Tensor interface` 实现新类型tensor
- 在 `VarGetMutable` 中增加 if-else 支持新类型的继承关系
