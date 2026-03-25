# PHI Kernel 注册机制详解

## 概述

PHI kernel 的注册完全通过宏完成，利用 C++ 静态初始化机制在 `main()` 之前将所有 kernel 注册到 `KernelFactory` 单例中。本文档详细展开宏的展开链路、核心类和关键设计。

## 宏展开链路

用户侧调用从 `PD_REGISTER_KERNEL` 开始，逐步展开：

```
PD_REGISTER_KERNEL(add, GPU, ALL_LAYOUT, phi::AddKernel, float, double, ...)
  └── _PD_REGISTER_KERNEL(add, GPU, ALL_LAYOUT, phi::AddKernel, ARGS_DEF, {}, float, double, ...)
        ├── PD_STATIC_ASSERT_GLOBAL_NAMESPACE(...)    // 编译期断言：必须在全局命名空间
        ├── _PD_REGISTER_2TA_KERNEL(...)               // Two-Template-Argument 变体
        │     └── PD_KERNEL_REGISTRAR_INIT(...)
        │           ├── PD_NARGS(float, double, ...)   // 计算变参数量
        │           └── _PD_KERNEL_REGISTRAR_INIT_N(N, ...)  // 按 N 展开
        │                 └── _PD_CREATE_REGISTRAR_OBJECT(add, GPU, ALL_LAYOUT,
        │                       phi::AddKernel, float, ADD_kernel_reg_0, ...)
        └── 为每个 DataType 创建一个 KernelRegistrar 静态对象
```

### 关键宏说明

**`PD_STATIC_ASSERT_GLOBAL_NAMESPACE`**：使用 `static_assert` 确保宏在全局命名空间调用，否则编译报错。这是因为注册依赖静态初始化，放在函数或命名空间内会导致顺序不确定或遗漏。

**`PD_NARGS`**：利用可变参数宏技巧计算传入的 DataType 数量，最大支持约 15 个类型。展开结果用于选择对应的 `_PD_KERNEL_REGISTRAR_INIT_N` 版本。

**`_PD_CREATE_REGISTRAR_OBJECT`**：核心创建步骤，为每个 DataType 生成一个 `static KernelRegistrar` 对象。变量名通过拼接算子名、序号和后缀保证全局唯一。

## KernelRegistrar 类

`KernelRegistrar` 是实际执行注册的工具类，位于 `paddle/phi/core/kernel_registry.h`。

### 构造函数

```cpp
KernelRegistrar(const char* kernel_name_cstr,
                BackendType backend,
                DataLayout data_layout,
                DataType dtype,
                KernelArgsParseFn args_parse_fn,
                ArgsDefFn args_def_fn,
                KernelFn kernel_fn,
                void* variadic_kernel_fn);
```

参数说明：
- `kernel_name_cstr`：算子名称字符串
- `backend`, `data_layout`, `dtype`：构成 `KernelKey` 的三要素
- `args_parse_fn`：由 `KernelArgsParseFunctor` 生成的参数解析函数
- `args_def_fn`：用户自定义参数属性回调（即宏中的 `{}` body）
- `kernel_fn`：kernel 函数指针（经 PHI_KERNEL 包装后的统一签名）
- `variadic_kernel_fn`：原始函数指针，用于变参 kernel

### ConstructKernel() 方法

构造函数内部调用 `ConstructKernel()`，完成以下步骤：

1. 通过 `KernelFactory::Instance().InsertKernel()` 将 kernel 写入两级 map
2. 调用 `args_parse_fn` 解析模板参数，填充 `Kernel` 对象的 `args_def` 列表
3. 调用用户提供的 `args_def_fn`（如果非空），允许用户修改参数的 Backend 属性
4. 设置 `kernel_fn` 和 `variadic_kernel_fn`

## KernelFactory 单例

```cpp
class KernelFactory {
  static KernelFactory& Instance();

  // 两级存储
  using KernelKeyMap = paddle::flat_hash_map<KernelKey, Kernel, KernelKey::Hash>;
  paddle::flat_hash_map<std::string, KernelKeyMap> kernels_;

  // 核心方法
  void InsertKernel(const std::string& name, const KernelKey& key, const Kernel& kernel);
  const Kernel& SelectKernelOrThrowError(const std::string& name, const KernelKey& key) const;
  const KernelKeyMap* GetKernelKeyMap(const std::string& name) const;
};
```

- `InsertKernel()`：写入时如果 key 已存在会打印 warning 但仍覆盖（后注册的优先）
- `SelectKernelOrThrowError()`：查找失败时抛出详细的 `phi::enforce` 异常，包含可用 kernel 列表

## KernelArgsParseFunctor

```cpp
template <typename KernelFunc>
struct KernelArgsParseFunctor;

template <typename Return, typename... Args>
struct KernelArgsParseFunctor<Return (*)(Args...)> {
  static void Parse(const KernelKey& key, Kernel* kernel) {
    // 对每个 Args... 通过 type_traits 判断：
    //   - const DenseTensor&     → InputArgDef
    //   - DenseTensor*           → OutputArgDef
    //   - const Scalar&          → AttributeArgDef
    //   - DataType / Place / ... → AttributeArgDef
    // 依次追加到 kernel->args_def 中
  }
};
```

该模板元编程机制使得注册时不需要手动声明参数列表——编译器从 kernel 函数签名自动提取。

## PHI_KERNEL 与 PHI_VARIADIC_KERNEL

```cpp
#define PHI_KERNEL(kernel_fn) \
  ::phi::KernelImpl<decltype(&kernel_fn), &kernel_fn>::Compute
```

`KernelImpl` 将具有具体参数签名的 kernel 函数包装成统一的 `void(KernelContext*)` 签名。运行时从 `KernelContext` 中按顺序取出 input / output / attribute，再调用真正的 kernel 函数。

`PHI_VARIADIC_KERNEL` 用于参数数量不固定的 kernel（如 `concat`），其内部使用 `std::vector<const DenseTensor*>` 接收变长输入。

## 用户自定义注册 Hook

宏展开中的 `{}` 占位符（即 `args_def_fn`）允许用户在注册时修改参数属性：

```cpp
PD_REGISTER_KERNEL(my_kernel, GPU, ALL_LAYOUT, phi::MyKernel, float) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
}
```

常见用途：
- `SetBackend(ALL_BACKEND)`：声明输入不需要 TransDataPlace（接受任意 backend）
- `SetDataType(...)`：固定某个输出的数据类型，不跟随 kernel dtype
- `SetDataLayout(...)`：强制某个输入 / 输出的 layout

## 关键源码路径

| 文件 | 说明 |
|------|------|
| `paddle/phi/core/kernel_registry.h` | 所有注册宏定义 + KernelRegistrar |
| `paddle/phi/core/kernel_factory.h` | KernelFactory、KernelKey、Kernel 类 |
| `paddle/phi/core/kernel_factory.cc` | KernelFactory 实现 |
| `paddle/phi/core/kernel_context.h` | KernelContext，运行时参数容器 |
| `paddle/phi/kernels/add_kernel.h` | 典型 kernel 头文件示例 |
| `paddle/phi/kernels/gpu/add_kernel.cu` | 典型 GPU kernel 实现 |
| `paddle/phi/kernels/cpu/add_kernel.cc` | 典型 CPU kernel 注册示例 |
