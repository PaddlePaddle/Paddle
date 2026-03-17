# Python-C++ 互操作桥接

## 架构概览

Paddle 的核心计算逻辑用 C++ 实现，编译为共享库 `core_avx.so`（或 `core_noavx.so`），Python 层在其上提供用户友好的 API。C++ 与 Python 的桥接采用两种机制：

| 机制 | 适用场景 | 性能 |
|------|---------|------|
| **pybind11** | 低频调用 API（配置、初始化等） | 一般（有封装开销） |
| **Python/C API** | 高频调用 API（Tensor 操作、算子调用） | 高（直接操作 PyObject） |

## pybind11 绑定

pybind11 是 C++ 与 Python 互操作的轻量级库，Paddle 用它绑定低频调用的 API。

### 基本用法

```cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(core_avx, m) {
    m.doc() = "PaddlePaddle C++ core";

    // 绑定函数
    m.def("init_glog", &InitGLOG, "Initialize GLOG");

    // 绑定类
    py::class_<DeviceContext>(m, "DeviceContext")
        .def(py::init<>())
        .def("get_place", &DeviceContext::GetPlace);

    // 子模块
    auto eager = m.def_submodule("eager", "Eager mode bindings");
    eager.def("run_backward", &RunBackward);
}
```

### GIL 管理

pybind11 默认持有 GIL。对于耗时的 C++ 计算，需要释放 GIL 以允许其他 Python 线程执行：

```cpp
m.def("heavy_compute", [](Tensor& x) {
    py::gil_scoped_release release;  // 释放 GIL
    return HeavyCompute(x);
    // 函数返回时自动重新获取 GIL
});
```

## Python/C API 绑定（高频路径）

对于 Tensor 类型和算子调用等热路径，pybind11 的封装开销不可接受。Paddle 直接使用 **Python/C API** 构建 Tensor 类型。

### 核心文件

| 文件 | 职责 |
|------|------|
| `paddle/fluid/pybind/eager.cc` | Tensor 类型定义（PyHeapTypeObject） |
| `paddle/fluid/pybind/eager_method.cc` | Tensor 实例方法（如 `numpy()`, `backward()`） |
| `paddle/fluid/pybind/eager_properties.cc` | Tensor 属性（如 `shape`, `dtype`, `grad`） |
| `paddle/fluid/pybind/eager_functions.cc` | 通用函数（如 `to_tensor()`） |
| `paddle/fluid/pybind/eager_op_function.cc` | 算子函数（**自动生成**） |
| `paddle/fluid/pybind/eager_utils.cc` | 类型转换工具 |

### Tensor 类型定义

Paddle 的 Tensor Python 类型直接通过 `PyHeapTypeObject` 创建：

```cpp
// eager.cc 中的关键字段
PyTypeObject* p_tensor_type;

void BindEager(pybind11::module* module) {
    auto heap_type = reinterpret_cast<PyHeapTypeObject*>(
        PyType_Type.tp_alloc(&PyType_Type, 0));
    auto type = &heap_type->ht_type;

    type->tp_name = "paddle.Tensor";
    type->tp_basicsize = sizeof(TensorObject);
    type->tp_dealloc = (destructor)TensorDealloc;
    type->tp_methods = variable_methods;      // 实例方法表
    type->tp_getset = variable_properties;    // 属性表
    type->tp_init = TensorInit;
    type->tp_new = TensorNew;

    PyType_Ready(type);
    p_tensor_type = type;
}
```

其中 `TensorObject` 是 C++ 结构体，包含 `PyObject_HEAD` 和 `paddle::Tensor tensor` 成员。

### 添加 Tensor 方法

在 `eager_method.cc` 中定义方法实现，并注册到方法表：

```cpp
// 方法实现
static PyObject* tensor_method_numpy(TensorObject* self,
                                      PyObject* args,
                                      PyObject* kwargs) {
    // 实现逻辑...
    return ToPyObject(result);
}

// 方法表（eager_method.cc 末尾）
PyMethodDef variable_methods[] = {
    {"numpy",
     (PyCFunction)(void (*)(void))tensor_method_numpy,
     METH_VARARGS | METH_KEYWORDS,
     "Convert Tensor to numpy array"},
    // ... 更多方法
    {nullptr, nullptr, 0, nullptr}  // 哨兵
};
```

### 添加 Tensor 属性

在 `eager_properties.cc` 中定义 getter/setter，注册到属性表：

```cpp
// getter 实现
static PyObject* tensor_properties_get_shape(TensorObject* self, void* closure) {
    auto shape = self->tensor.shape();
    return ToPyObject(shape);
}

// 属性表
PyGetSetDef variable_properties[] = {
    {"shape",
     (getter)tensor_properties_get_shape,
     nullptr,  // setter（nullptr 表示只读）
     "Tensor shape",
     nullptr},
    // ... 更多属性
    {nullptr, nullptr, nullptr, nullptr, nullptr}  // 哨兵
};
```

## 类型转换

Python 对象和 C++ 对象之间的转换是桥接层的基础设施，集中在 `eager_utils.cc`：

### PyObject → C++（CastPyArg2* 系列）

```cpp
paddle::Tensor CastPyArg2Tensor(PyObject* obj, Py_ssize_t arg_pos);
int CastPyArg2Int(PyObject* obj, const std::string& op_type, Py_ssize_t arg_pos);
std::vector<int64_t> CastPyArg2Longs(PyObject* obj, ...);
paddle::Place CastPyArg2Place(PyObject* obj, ...);
```

### C++ → PyObject（ToPyObject 重载）

```cpp
PyObject* ToPyObject(const paddle::Tensor& value);
PyObject* ToPyObject(int value);
PyObject* ToPyObject(const std::vector<int64_t>& value);
PyObject* ToPyObject(const phi::DataType& dtype);
```

## 自动代码生成

算子的 Python-C 绑定代码量大且模式固定，因此采用自动生成：

**生成器**：`paddle/fluid/eager/auto_code_generator/generator/python_c_gen.py`

**流程**：
1. 读取 `paddle/fluid/operators/generator/parsed_ops/legacy_api.yaml`（算子定义）
2. 根据模板生成 `eager_op_function.cc`
3. 每个算子生成一个 `eager_api_{op_name}` 函数

**生成的代码模式**：

```cpp
static PyObject* eager_api_matmul(PyObject* self, PyObject* args, PyObject* kwargs) {
    // 1. 解析参数：CastPyArg2Tensor, CastPyArg2Bool 等
    // 2. 释放 GIL
    // 3. 调用 C++ API：matmul_ad_func(x, y, transpose_x, transpose_y)
    // 4. 获取 GIL
    // 5. 返回结果：ToPyObject(result)
}
```

## pybind11 与 Python/C API 的共存

pybind11 本身是 Python/C API 的封装。两者可以互操作：

```cpp
// pybind11 对象 → PyObject*
py::object obj = ...;
PyObject* raw = obj.ptr();

// PyObject* → pybind11 对象
PyObject* raw = ...;
py::handle h(raw);
py::object obj = py::reinterpret_borrow<py::object>(h);
```

Paddle 中 pybind11 负责模块初始化和低频 API，Python/C API 负责 Tensor 类型和高频算子调用，两者在 `PYBIND11_MODULE` 入口处汇合。
