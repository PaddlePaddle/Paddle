# PIR 类型系统与可扩展组件

## 动机：统一三套断裂的类型系统

Paddle 旧体系存在三套相互独立的类型系统——`framework::proto::VarType`（静态图）、`DataType/DataLayout`（PHI 算子库）、以及推理阶段的类型表示。它们之间通过大量 `switch-case` 硬编码转换，既难以维护也无法扩展。PIR 参考 MLIR 设计理念，用一套统一的类型系统替代三者，核心目标：

1. **可扩展**：新增类型无需修改核心框架代码
2. **高性能**：类型相等性通过指针比较完成，O(1) 复杂度
3. **模块化**：类型归属于 Dialect，不同 Dialect 独立注册

## 类型系统核心类

### TypeID：类型的唯一标识

```cpp
class TypeID {
  const StorageUniquer::TypeIDAllocator *storage_;  // 静态变量指针
};
```

每种类型在编译期通过 `TypeIDAllocator::Get<T>()` 获得一个唯一的 `TypeID`。实现技巧：利用 C++ 模板函数内的 `static` 局部变量地址天然唯一——不同类型实例化不同的函数，地址自然不同。

### AbstractType：类型的元信息

```cpp
struct AbstractType {
  TypeID type_id_;
  Dialect &dialect_;            // 所属 Dialect
  InterfaceMap interface_map_;  // 该类型实现的 Interface 集合
  HasTraitFunction has_trait_;  // Trait 查询函数
};
```

`AbstractType` 在 `IRContext` 中按 `TypeID` 注册一次，是类型的"类描述符"。

### TypeStorage：类型实例的存储

```cpp
class TypeStorage : public StorageManager::StorageBase {
  AbstractType *abstract_type_;
};
```

无参数类型（如 `Float32Type`）全局只有一个 `TypeStorage` 实例；带参数类型（如 `DenseTensorType`）按参数值 hash 唯一化存储（uniquing），相同参数返回同一指针。

### Type：面向用户的轻量包装

```cpp
class Type {
  TypeStorage *storage_{nullptr};
};
```

`Type` 是值语义的轻量对象，仅包装一个指向 `TypeStorage` 的指针。两个 Type 相等 <==> 指针相等。用户通过 `Type::isa<T>()` 和 `Type::dyn_cast<T>()` 做类型判断和转换。

### TypeBase 模板：定义具体类型

```cpp
template <typename ConcreteT, typename BaseT, typename StorageT>
class TypeBase : public BaseT {
  // ConcreteT: 具体类型类 (如 DenseTensorType)
  // BaseT:     基类 (通常是 Type)
  // StorageT:  存储类 (如 DenseTensorTypeStorage)
};
```

## 参数化类型示例：DenseTensorType

```cpp
struct DenseTensorTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<Type, DDim, DataLayout, LoD, size_t>;

  static std::size_t hashFunc(const KeyTy &key) { ... }
  static DenseTensorTypeStorage *construct(IRContext *ctx, const KeyTy &key) { ... }
  bool operator==(const KeyTy &key) const { ... }

  Type dtype_;
  DDim dims_;
  DataLayout layout_;
  LoD lod_;
  size_t offset_;
};
```

获取或创建实例：

```cpp
auto dt = DenseTensorType::get(ctx, dtype, dims, layout, lod, offset);
```

`IRContext` 内部通过 `StorageManager` 管理所有 `TypeStorage`，以 hash 表做 uniquing——如果已存在相同参数的实例直接返回指针，否则 `construct()` 创建并缓存。

## 类型使用方式

```cpp
Type t = some_value.type();

// 类型判断
if (t.isa<DenseTensorType>()) { ... }

// 类型转换（失败返回 nullptr 语义的空 Type）
if (auto dt = t.dyn_cast<DenseTensorType>()) {
    auto dims = dt.dims();
}

// 相等性
if (t1 == t2) { ... }  // 指针比较，O(1)
```

## Attribute 系统

Attribute 与 Type 共享相同的 uniquing 基础设施，区别在于语义：Type 描述 Value 的类型，Attribute 描述 Operation 的常量属性。

### AttributeStorage

结构与 `TypeStorage` 对称：`AbstractAttribute` + `AttributeStorage` + `Attribute` 包装。

### DictionaryAttribute

PIR 中每个 `Operation` 的属性集合用 `DictionaryAttribute` 存储：

```cpp
class DictionaryAttributeStorage : public AttributeStorage {
  std::vector<NamedAttribute> data_;  // 按 key 排序的 (StrAttribute, Attribute) 对
};
```

查找时使用**二分搜索**，插入时保持有序。`NamedAttribute` 是 `std::pair<StrAttribute, Attribute>`。

## 可扩展组件：Trait 与 Interface

### Trait：静态标记

Trait 是无状态的编译期标记，用于表示 Operation 或 Type 的某种静态属性：

```cpp
// 定义
template <typename ConcreteOp>
class ReadOnlyTrait : public OpTraitBase<ConcreteOp, ReadOnlyTrait> {};

// 查询
if (op->HasTrait<ReadOnlyTrait>()) { ... }
```

常见 Trait：`InplaceTrait`、`SideEffectTrait`、`SameOperandsAndResultTypeTrait`。

### Interface：动态分派

Interface 采用 **concept-model** 模式实现多态，避免虚函数表的开销：

```cpp
// Concept（纯虚接口）
struct InferShapeInterface::Concept {
  void (*infer_shape)(Operation *);
};

// Model（具体实现）
template <typename ConcreteOp>
struct InferShapeInterface::Model : Concept {
  static void infer_shape(Operation *op) {
    ConcreteOp::InferShape(op);
  }
};
```

`OpInfo` 内部持有 `InterfaceMap`（按 `TypeID` 索引的 concept 指针数组），调用时通过 `TypeID` 查找到对应 `Concept*`，再调用函数指针——一次间接跳转，无虚表开销。

## Dialect：模块化容器

```cpp
class Dialect {
  std::string name_;
  TypeID dialect_id_;
  IRContext *context_;
};
```

每个 Dialect 是 Type、Attribute、Op 定义的**注册容器**，通过 `IRContext::RegisterDialect<T>()` 加载。

### 核心 Dialect

| Dialect | 职责 | 典型内容 |
|---------|------|---------|
| `BuiltinDialect` | PIR 内置基础类型 | `Float32Type`, `Int64Type`, `VectorType`, `DenseTensorType` |
| `PaddleDialect` | Paddle 算子定义 | `pd_op.matmul`, `pd_op.relu`, `pd_op.conv2d` 等 |
| `CinnDialect` | CINN 编译器专用 | `cinn_op.group`, `cinn_op.yield`, `cinn_op.generate_shape` |
| `ControlFlowDialect` | 控制流 | `cf.if`, `cf.while`, `cf.yield`, `cf.cond_yield` |

Dialect 的模块化设计使得第三方可以定义自己的 Dialect 并注册到 `IRContext`，无需修改 PIR 核心代码。
