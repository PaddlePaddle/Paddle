/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <Python.h>

#include <string>
#include <vector>

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/scope_guard.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/operators/py_func_op.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace pybind {

extern PyTypeObject* p_tensor_type;

extern PyTypeObject* g_vartype_pytype;
extern PyTypeObject* g_place_pytype;
extern PyTypeObject* g_cudaplace_pytype;
extern PyTypeObject* g_cpuplace_pytype;
extern PyTypeObject* g_xpuplace_pytype;
extern PyTypeObject* g_npuplace_pytype;
extern PyTypeObject* g_cudapinnedplace_pytype;
extern PyTypeObject* g_framework_tensor_pytype;
extern PyTypeObject* g_framework_lodtensorarray_pytype;

int TensorDtype2NumpyDtype(phi::DataType dtype) {
  switch (dtype) {
    case phi::DataType::BOOL:
      return pybind11::detail::npy_api::NPY_BOOL_;
    case phi::DataType::INT8:
      return pybind11::detail::npy_api::NPY_INT8_;
    case phi::DataType::UINT8:
      return pybind11::detail::npy_api::NPY_UINT8_;
    case phi::DataType::INT16:
      return pybind11::detail::npy_api::NPY_INT16_;
    case phi::DataType::INT32:
      return pybind11::detail::npy_api::NPY_INT32_;
    case phi::DataType::INT64:
      return pybind11::detail::npy_api::NPY_INT64_;
    case phi::DataType::FLOAT16:
      return pybind11::detail::NPY_FLOAT16_;
    case phi::DataType::FLOAT32:
      return pybind11::detail::npy_api::NPY_FLOAT_;
    case phi::DataType::FLOAT64:
      return pybind11::detail::npy_api::NPY_DOUBLE_;
    case phi::DataType::COMPLEX64:
      return pybind11::detail::NPY_COMPLEX64;
    case phi::DataType::COMPLEX128:
      return pybind11::detail::NPY_COMPLEX128;
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Unknow phi::DataType, the int value = %d.",
          static_cast<int>(dtype)));
      return 0;
  }
}

bool PyObject_CheckLongOrConvertToLong(PyObject** obj) {
  if ((PyLong_Check(*obj) && !PyBool_Check(*obj))) {
    return true;
  }

  if (std::string((reinterpret_cast<PyTypeObject*>((*obj)->ob_type))->tp_name)
          .find("numpy") != std::string::npos) {
    auto to = PyNumber_Long(*obj);
    if (to) {
      *obj = to;
      return true;
    }
  }

  return false;
}

bool PyObject_CheckFloatOrConvertToFloat(PyObject** obj) {
  // sometimes users provide PyLong or numpy.int64 but attr is float
  if (PyFloat_Check(*obj) || PyLong_Check(*obj)) {
    return true;
  }
  if (std::string((reinterpret_cast<PyTypeObject*>((*obj)->ob_type))->tp_name)
          .find("numpy") != std::string::npos) {
    auto to = PyNumber_Float(*obj);
    if (to) {
      *obj = to;
      return true;
    }
  }
  return false;
}

bool PyObject_CheckStr(PyObject* obj) { return PyUnicode_Check(obj); }

bool CastPyArg2AttrBoolean(PyObject* obj, ssize_t arg_pos) {
  if (obj == Py_None) {
    return false;  // To be compatible with QA integration testing. Some
                   // test case pass in None.
  } else if (obj == Py_True) {
    return true;
  } else if (obj == Py_False) {
    return false;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "bool, but got %s",
        arg_pos + 1, (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
  }
}

int CastPyArg2AttrInt(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_CheckLongOrConvertToLong(&obj)) {
    return static_cast<int>(PyLong_AsLong(obj));
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "int, but got %s",
        arg_pos + 1, (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
  }
}

int64_t CastPyArg2AttrLong(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_CheckLongOrConvertToLong(&obj)) {
    return (int64_t)PyLong_AsLong(obj);  // NOLINT
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "long, but got %s",
        arg_pos + 1, (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
  }
}

float CastPyArg2AttrFloat(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_CheckFloatOrConvertToFloat(&obj)) {
    return static_cast<float>(PyFloat_AsDouble(obj));
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "float, but got %s",
        arg_pos + 1, (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
  }
}

std::string CastPyArg2AttrString(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_CheckStr(obj)) {
    Py_ssize_t size;
    const char* data;
    data = PyUnicode_AsUTF8AndSize(obj, &size);
    return std::string(data, static_cast<size_t>(size));
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "str, but got %s",
        arg_pos + 1, (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
    return "";
  }
}

paddle::experimental::Tensor CastPyArg2Tensor(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type))) {
    return reinterpret_cast<TensorObject*>(obj)->tensor;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "EagerVariable, but got %s",
        arg_pos + 1, reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

std::shared_ptr<imperative::VarBase> CastPyArg2VarBase(PyObject* obj,
                                                       ssize_t arg_pos) {
  return py::cast<std::shared_ptr<imperative::VarBase>>(obj);
}

std::vector<paddle::experimental::Tensor> CastPyArg2VectorOfTensor(
    PyObject* obj, ssize_t arg_pos) {
  std::vector<paddle::experimental::Tensor> result;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_IsInstance(item,
                              reinterpret_cast<PyObject*>(p_tensor_type))) {
        result.emplace_back(reinterpret_cast<TensorObject*>(item)->tensor);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of Tensor, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name, i));
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_IsInstance(item,
                              reinterpret_cast<PyObject*>(p_tensor_type))) {
        result.emplace_back(reinterpret_cast<TensorObject*>(item)->tensor);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of Tensor, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name, i));
      }
    }
  } else if (obj == Py_None) {
    return {};
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "list or tuple, but got %s",
        arg_pos + 1, reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return result;
}

std::vector<int> CastPyArg2VectorOfInt(PyObject* obj, size_t arg_pos) {
  std::vector<int> result;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckLongOrConvertToLong(&item)) {
        result.emplace_back(static_cast<int>(PyLong_AsLong(item)));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of int, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name, i));
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckLongOrConvertToLong(&item)) {
        result.emplace_back(static_cast<int>(PyLong_AsLong(item)));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of bool, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name, i));
      }
    }
  } else if (obj == Py_None) {
    return {};
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "list or tuple, but got %s",
        arg_pos + 1, reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return result;
}

platform::Place CastPyArg2Place(PyObject* obj, ssize_t arg_pos) {
  platform::Place place;
  if (PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(g_place_pytype))) {
    place = ::pybind11::handle(obj).cast<platform::Place>();
  } else if (PyObject_IsInstance(
                 obj, reinterpret_cast<PyObject*>(g_cudaplace_pytype))) {
    place = ::pybind11::handle(obj).cast<platform::CUDAPlace>();
  } else if (PyObject_IsInstance(
                 obj, reinterpret_cast<PyObject*>(g_cpuplace_pytype))) {
    place = ::pybind11::handle(obj).cast<platform::CPUPlace>();
  } else if (PyObject_IsInstance(
                 obj, reinterpret_cast<PyObject*>(g_xpuplace_pytype))) {
    place = ::pybind11::handle(obj).cast<platform::XPUPlace>();
  } else if (PyObject_IsInstance(
                 obj, reinterpret_cast<PyObject*>(g_npuplace_pytype))) {
    place = ::pybind11::handle(obj).cast<platform::NPUPlace>();
  } else if (PyObject_IsInstance(
                 obj, reinterpret_cast<PyObject*>(g_cudapinnedplace_pytype))) {
    place = ::pybind11::handle(obj).cast<platform::CUDAPinnedPlace>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "one of(Place,CUDAPlace,CPUPlace,XPUPlace,NPUPlace,CUDAPinnedPlace), "
        "but got %s",
        arg_pos + 1, reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return place;
}

framework::Tensor CastPyArg2FrameworkTensor(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_IsInstance(
          obj, reinterpret_cast<PyObject*>(g_framework_tensor_pytype))) {
    return ::pybind11::handle(obj).cast<framework::Tensor>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "EagerVariable, but got %s",
        arg_pos + 1, reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

std::vector<framework::Tensor> CastPyArg2VectorOfTensorBase(PyObject* obj,
                                                            ssize_t arg_pos) {
  std::vector<framework::LoDTensor> result;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_IsInstance(
              item, reinterpret_cast<PyObject*>(g_framework_tensor_pytype))) {
        result.emplace_back(
            ::pybind11::handle(item).cast<framework::LoDTensor>());
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of LoDTensor, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name, i));
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_IsInstance(
              item, reinterpret_cast<PyObject*>(g_framework_tensor_pytype))) {
        result.emplace_back(
            ::pybind11::handle(item).cast<framework::LoDTensor>());
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of LoDTensor, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name, i));
      }
    }
  } else if (PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(
                                          g_framework_lodtensorarray_pytype))) {
    return ::pybind11::handle(obj).cast<framework::LoDTensorArray>();
  } else if (obj == Py_None) {
    return {};
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "list or tuple, but got %s",
        arg_pos + 1, reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return result;
}

paddle::framework::proto::VarType::Type CastPyArg2ProtoType(PyObject* obj,
                                                            ssize_t arg_pos) {
  paddle::framework::proto::VarType::Type dtype;
  if (PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(g_vartype_pytype))) {
    dtype =
        ::pybind11::handle(obj).cast<paddle::framework::proto::VarType::Type>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "one of core.VarDesc.VarType, "
        "but got %s",
        arg_pos + 1, reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return dtype;
}

PyObject* ToPyObject(bool value) {
  if (value) {
    Py_INCREF(Py_True);
    return Py_True;
  } else {
    Py_INCREF(Py_False);
    return Py_False;
  }
}

PyObject* ToPyObject(int value) { return PyLong_FromLong(value); }

PyObject* ToPyObject(int64_t value) { return PyLong_FromLongLong(value); }

PyObject* ToPyObject(float value) { return PyLong_FromDouble(value); }

PyObject* ToPyObject(double value) { return PyLong_FromDouble(value); }

PyObject* ToPyObject(const char* value) { return PyUnicode_FromString(value); }

PyObject* ToPyObject(const std::string& value) {
  return PyUnicode_FromString(value.c_str());
}

PyObject* ToPyObject(const paddle::experimental::Tensor& value) {
  PyObject* obj = p_tensor_type->tp_alloc(p_tensor_type, 0);
  if (obj) {
    auto v = reinterpret_cast<TensorObject*>(obj);
    new (&(v->tensor)) paddle::experimental::Tensor();
    v->tensor = value;
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "tp_alloc return null, can not new a PyObject."));
  }
  return obj;
}

PyObject* ToPyObject(const std::vector<bool>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<int>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<int64_t>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, (Py_ssize_t)i, ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<float>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<double>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<paddle::experimental::Tensor>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyObject* obj = p_tensor_type->tp_alloc(p_tensor_type, 0);
    if (obj) {
      auto v = reinterpret_cast<TensorObject*>(obj);
      new (&(v->tensor)) paddle::experimental::Tensor();
      v->tensor = value[i];
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "tp_alloc return null, can not new a PyObject."));
    }
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), obj);
  }

  return result;
}

PyObject* ToPyObject(const platform::Place& value) {
  auto obj = ::pybind11::cast(value);
  obj.inc_ref();
  return obj.ptr();
}

PyObject* ToPyObject(const paddle::framework::proto::VarType::Type& dtype) {
  auto obj = ::pybind11::cast(dtype);
  obj.inc_ref();
  return obj.ptr();
}

PyObject* ToPyObject(const paddle::framework::proto::VarType& type) {
  auto obj = ::pybind11::cast(type);
  obj.inc_ref();
  return obj.ptr();
}

PyObject* ToPyObject(const paddle::framework::LoDTensor* value) {
  auto obj = ::pybind11::cast(value, py::return_value_policy::reference);
  obj.inc_ref();
  return obj.ptr();
}

PyObject* ToPyObject(const void* value) {
  if (value == nullptr) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  PADDLE_THROW(
      platform::errors::Fatal("ToPyObject do not support void* with value."));
}

PyObject* ToPyObject(
    const std::unordered_map<std::string, std::vector<std::string>>& value) {
  PyObject* dict = PyDict_New();
  for (const auto map_iter : value) {
    // Convert Key
    PyObject* key_string = PyUnicode_FromString(map_iter.first.c_str());
    if (!key_string) {
      PADDLE_THROW(
          platform::errors::Fatal("Unable to convert std::string to PyObject"));
    }

    // Convert Val
    PyObject* py_list = PyList_New(0);
    for (const auto vector_iter : map_iter.second) {
      PyObject* val_string = PyUnicode_FromString(vector_iter.c_str());
      if (!val_string) {
        PADDLE_THROW(platform::errors::Fatal(
            "Unable to convert std::string to PyObject"));
      }

      if (PyList_Append(py_list, val_string) != 0) {
        PADDLE_THROW(
            platform::errors::Fatal("Unable to append string to py_list"));
      }
    }

    if (PyDict_SetItem(dict, key_string, py_list) != 0) {
      PADDLE_THROW(
          platform::errors::Fatal("Unable to set key:value for py_dict"));
    }
  }

  return dict;
}

// For Final State Dygraph,
// We directly use paddle::optional(Tensor) as dispensable Tensor
paddle::optional<paddle::experimental::Tensor> GetOptionalTensorFromArgs(
    const std::string& op_type, const std::string& arg_name, PyObject* args,
    ssize_t arg_idx, bool dispensable) {
  PyObject* obj = PyTuple_GET_ITEM(args, arg_idx);

  if (PyTuple_Check(obj)) {
    obj = PyTuple_GET_ITEM(obj, 0);
  }

  if (obj == nullptr || obj == Py_None) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be Tensor, but got None",
          op_type, arg_name, arg_idx));
    }
    return {};
  }

  return paddle::make_optional<paddle::experimental::Tensor>(
      reinterpret_cast<TensorObject*>(obj)->tensor);
}

// For Intermediate State Dygraph,
// we use an uninitialized Tensor to represent dispensable Tensor
paddle::experimental::Tensor& GetTensorFromArgs(const std::string& op_type,
                                                const std::string& arg_name,
                                                PyObject* args, ssize_t arg_idx,
                                                bool dispensable) {
  PyObject* obj = PyTuple_GET_ITEM(args, arg_idx);

  if (PyTuple_Check(obj)) {
    obj = PyTuple_GET_ITEM(obj, 0);
  }

  if (obj == nullptr || obj == Py_None) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be Tensor, but got None",
          op_type, arg_name, arg_idx));
    }
    static paddle::experimental::Tensor emptytensor;
    return emptytensor;
  }

  return reinterpret_cast<TensorObject*>(obj)->tensor;
}

std::vector<paddle::experimental::Tensor> GetTensorListFromArgs(
    const std::string& op_type, const std::string& arg_name, PyObject* args,
    ssize_t arg_idx, bool dispensable) {
  PyObject* list = PyTuple_GET_ITEM(args, arg_idx);

  if (list == nullptr) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensor, but got "
          "None",
          op_type, arg_name, arg_idx));
    }
    return {};
  }

  std::vector<paddle::experimental::Tensor> result;

  if (PyList_Check(list)) {
    Py_ssize_t len = PyList_Size(list);
    result.reserve(static_cast<size_t>(len));
    if (len == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensors, but got "
          "empty list",
          op_type, arg_name, arg_idx));
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      result.emplace_back(
          reinterpret_cast<TensorObject*>(PyList_GetItem(list, i))->tensor);
    }
  } else if (PyTuple_Check(list)) {
    Py_ssize_t len = PyTuple_Size(list);
    result.reserve(static_cast<size_t>(len));
    if (len == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensors, but got "
          "empty list",
          op_type, arg_name, arg_idx));
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      result.emplace_back(
          reinterpret_cast<TensorObject*>(PyTuple_GetItem(list, i))->tensor);
    }
  } else if (list == Py_None) {
    return {};
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be list of Tensors, but got "
        "%s",
        op_type, arg_name, arg_idx,
        (reinterpret_cast<PyTypeObject*>(list->ob_type))->tp_name));
  }

  return result;
}

paddle::experimental::Tensor* GetTensorPtrFromArgs(const std::string& op_type,
                                                   const std::string& arg_name,
                                                   PyObject* args,
                                                   ssize_t arg_idx,
                                                   bool dispensable) {
  PyObject* obj = PyTuple_GET_ITEM(args, arg_idx);

  if (PyTuple_Check(obj)) {
    obj = PyTuple_GET_ITEM(obj, 0);
  }

  if (obj == nullptr || obj == Py_None) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be Tensor, but got None",
          op_type, arg_name, arg_idx));
    }
    static paddle::experimental::Tensor emptytensor;
    return &emptytensor;
  }

  return &(reinterpret_cast<TensorObject*>(obj)->tensor);
}

std::vector<paddle::experimental::Tensor*> GetTensorPtrListFromArgs(
    const std::string& op_type, const std::string& arg_name, PyObject* args,
    ssize_t arg_idx, bool dispensable) {
  PyObject* list = PyTuple_GET_ITEM(args, arg_idx);

  if (list == nullptr) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensor, but got "
          "None",
          op_type, arg_name, arg_idx));
    }
    return {};
  }

  std::vector<paddle::experimental::Tensor*> result;

  if (PyList_Check(list)) {
    Py_ssize_t len = PyList_Size(list);
    if (len == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensors, but got "
          "empty list",
          op_type, arg_name, arg_idx));
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      result.emplace_back(
          &(reinterpret_cast<TensorObject*>(PyList_GetItem(list, i))->tensor));
    }
  } else if (PyTuple_Check(list)) {
    Py_ssize_t len = PyTuple_Size(list);
    if (len == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensors, but got "
          "empty list",
          op_type, arg_name, arg_idx));
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      result.emplace_back(
          &(reinterpret_cast<TensorObject*>(PyTuple_GetItem(list, i))->tensor));
    }
  } else if (list == Py_None) {
    return {};
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be list of Tensors, but got "
        "%s",
        op_type, arg_name, arg_idx,
        (reinterpret_cast<PyTypeObject*>(list->ob_type))->tp_name));
  }

  return result;
}

bool PyCheckInteger(PyObject* obj) {
#if PY_VERSION_HEX < 0x03000000
  return (PyLong_Check(obj) || PyInt_Check(obj)) && !PyBool_Check(obj);
#else
  return PyLong_Check(obj) && !PyBool_Check(obj);
#endif
}

bool IsNumpyType(PyObject* obj) {
  // It is not a good way to judge the type of obj by its type'name. Maybe using
  // `PyArray_IsScalar` will be better. However, this interface cannot be used
  // by including pybind11, and it needs to compile with numpy.
  auto type_name = std::string(Py_TYPE(obj)->tp_name);
  return type_name == "numpy.int64" || type_name == "numpy.longlong" ||
         type_name == "numpy.int32" || type_name == "numpy.int16";
}

bool PyCheckTensor(PyObject* obj) {
  return (py::isinstance<imperative::VarBase>(obj)) ||
         (PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type)));
}

Py_ssize_t GetSliceIndexFromTensor(const phi::DenseTensor& tensor) {
  if (tensor.numel() == 1) {
    if (framework::TransToProtoVarType(tensor.type()) ==
        framework::proto::VarType::INT32) {
      return static_cast<Py_ssize_t>(operators::GetValue<int32_t>(&tensor));
    } else if (framework::TransToProtoVarType(tensor.type()) ==
               framework::proto::VarType::INT64) {
      return static_cast<Py_ssize_t>(operators::GetValue<int64_t>(&tensor));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, the type of tensor in slice indices only allows "
          "int32 and int64, please check the type of index tensor."));
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Currently, tensor in slice indices only allows 1 element, "
        "but received %d.",
        tensor.numel()));
  }
}

Py_ssize_t GetSliceIndexFromPyObject(PyObject* obj) {
  if (py::isinstance<imperative::VarBase>(obj)) {
    VLOG(6) << "Call GetSliceIndexFromTensor in Imperative";
    return GetSliceIndexFromTensor(
        py::cast<std::shared_ptr<imperative::VarBase>>(obj)
            ->Var()
            .Get<framework::LoDTensor>());
  } else if (PyObject_IsInstance(obj,
                                 reinterpret_cast<PyObject*>(p_tensor_type))) {
    VLOG(6) << "Call GetSliceIndexFromTensor in Eager";
    paddle::experimental::Tensor tensor = CastPyArg2Tensor(obj, 0);
    PADDLE_ENFORCE_EQ(
        tensor.initialized(), true,
        paddle::platform::errors::InvalidArgument(
            "We can only support initialized tensor in slice, however we got "
            "uninitialized tensor %s, please check your code.",
            tensor.name()));
    return GetSliceIndexFromTensor((*static_cast<phi::DenseTensor*>(
        CastPyArg2Tensor(obj, 0).impl().get())));
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "We should only get paddle::experimental::Tensor or VarBase in this "
        "method, when you reach this means we got another type index."));
  }
}

// NOTE(zhiqiu): Revised version of PySlice_GetIndices. From:
// https://github.com/python/cpython/blob/8d21aa21f2cbc6d50aab3f420bb23be1d081dac4/Objects/sliceobject.c#L103
// Original PySlice_GetIndices return wrong result when
// slice_item contains long int, such as arr[:180L].
// NOT sure why this happens !!!
// Besides, PySlice_GetIndices cannot raise error when float in slice item.
// So, I make a revised version of PySlice_GetIndices, named to
// _PySlice_GetIndices. Try to use _PySlice_Unpack which is more robust than
// PySlice_GetIndices in the future.
int _PySlice_GetIndices(PySliceObject* r, Py_ssize_t length, Py_ssize_t* start,
                        Py_ssize_t* stop, Py_ssize_t* step) {
  /* XXX support long ints */
  if (r->step == Py_None) {
    *step = 1;
  } else {
    if (PyCheckInteger(r->step) || IsNumpyType(r->step)) {
      *step = PyLong_AsLong(r->step);
    } else if (PyCheckTensor(r->step)) {
      *step = GetSliceIndexFromPyObject(r->step);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, slice indices only allows None, integers, "
          "tensor(int) and numpy(int) in slice item, but received %s.",
          std::string(Py_TYPE(r->step)->tp_name)));
    }
  }
  if (r->start == Py_None) {
    *start = *step < 0 ? length - 1 : 0;
  } else {
    if (PyCheckInteger(r->start) || IsNumpyType(r->start)) {
      *start = PyLong_AsLong(r->start);
    } else if (PyCheckTensor(r->start)) {
      *start = GetSliceIndexFromPyObject(r->start);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, slice indices only allows None, integers, "
          "tensor(int) and numpy(int) in slice item, but received %s.",
          std::string(Py_TYPE(r->start)->tp_name)));
    }
    if (*start < 0) *start += length;
    *start = std::max(*start, static_cast<Py_ssize_t>(0));
  }
  if (r->stop == Py_None) {
    *stop = *step < 0 ? -1 : length;
  } else {
    if (PyCheckInteger(r->stop) || IsNumpyType(r->stop)) {
      *stop = PyLong_AsLong(r->stop);
    } else if (PyCheckTensor(r->stop)) {
      *stop = GetSliceIndexFromPyObject(r->stop);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, slice indices only allows None, integers, "
          "tensor(int) and numpy(int) in slice item, but received %s.",
          std::string(Py_TYPE(r->stop)->tp_name)));
    }
    if (0 < *step && *stop < 0) *stop += length;
    *stop = std::min(*stop, length);
  }
  if (*stop > length) return -1;
  if (*start >= length) return -1;
  if (*step == 0) return -1;
  return 0;
}

void ParseIndexingSlice(
    framework::LoDTensor* tensor, PyObject* _index,
    std::vector<int>* slice_axes, std::vector<int>* slice_starts,
    std::vector<int>* slice_ends, std::vector<int>* slice_strides,
    std::vector<int>* decrease_axis, std::vector<int>* none_axes,
    std::vector<int>* infer_flags, std::vector<int>* list_select_idxs,
    bool* list_select_flag) {
  // We allow indexing by Integers, Slices, Ellipsis, None, tuples of those
  // types, and list of Bool and Integers.
  // wrap to tuple

  // NOTE(zhiqiu): PyTuple_Pack increases refcount.
  PyObject* index = !PyTuple_Check(_index) ? PyTuple_Pack(1, _index) : _index;
  DEFINE_PADDLE_SCOPE_GUARD([index, _index]() {
    if (!PyTuple_Check(_index)) {
      Py_DECREF(index);
      VLOG(4) << "Call Py_DECREF";
    }
  });
  PADDLE_ENFORCE_EQ(
      tensor->IsInitialized(), true,
      platform::errors::InvalidArgument("tensor has not been initialized"));
  const auto& shape = tensor->dims();
  const int rank = shape.size();
  const int size = PyTuple_GET_SIZE(index);

  // specified_dims is the number of dimensions which indexed by Interger,
  // Slices.
  int specified_dims = 0;
  int ell_count = 0;
  for (int dim = 0; dim < size; ++dim) {
    PyObject* slice_item = PyTuple_GetItem(index, dim);
    if (PyCheckInteger(slice_item) || PySlice_Check(slice_item)) {
      specified_dims++;
    } else if (slice_item == Py_Ellipsis) {
      ell_count++;
    }
  }

  PADDLE_ENFORCE_LE(ell_count, 1,
                    platform::errors::InvalidArgument(
                        "An index can only have a single ellipsis ('...')"));
  int none_count = 0;
  for (int i = 0, dim = 0; i < size; ++i) {
    PyObject* slice_item = PyTuple_GetItem(index, i);

    infer_flags->push_back(1);
    int dim_len = shape[dim];
    if (PyCheckInteger(slice_item) || IsNumpyType(slice_item)) {
      // integer, PyLong_AsLong supports both int and long
      int start = static_cast<int>(PyLong_AsLong(slice_item));
      auto s_t = start;
      start = start < 0 ? start + dim_len : start;
      if (start >= dim_len || start < 0) {
        std::string str_error_message =
            "The starting index " + std::to_string(s_t) +
            " of slice is out of bounds in tensor " + std::to_string(dim) +
            "-th axis, it shound be in the range of [" +
            std::to_string(-dim_len) + ", " + std::to_string(dim_len) + ")";
        // py::index_error is corresponding to IndexError in Python
        // Used to indicate out of bounds access in __getitem__, __setitem__
        throw py::index_error(str_error_message);
      }
      slice_axes->push_back(dim);
      slice_starts->push_back(start);
      slice_ends->push_back(start + 1);
      slice_strides->push_back(1);
      decrease_axis->push_back(dim);
      dim++;
    } else if (PySlice_Check(slice_item)) {
      // slice item
      Py_ssize_t start, end, step;
      PySliceObject* p = reinterpret_cast<PySliceObject*>(slice_item);
      _PySlice_GetIndices(p, dim_len, &start, &end, &step);

      // :: or : or 0:dim_len:1
      if (start == 0 && end == dim_len && step == 1) {
        dim++;
        continue;
      }
      slice_axes->push_back(dim);
      slice_starts->push_back(start);
      slice_ends->push_back(end);
      slice_strides->push_back(step);
      dim++;
    } else if (slice_item == Py_Ellipsis) {
      dim += rank - specified_dims;
    } else if (slice_item == Py_None) {
      none_axes->push_back(dim + none_count);
      none_count++;
    } else if (PyList_Check(slice_item)) {
      *list_select_flag = true;
      PADDLE_ENFORCE_EQ(
          size, 1,
          platform::errors::InvalidArgument(
              "When index contains a list, its length is excepted to 1, "
              "but received %d",
              size));
      bool all_bool = true;
      int list_size = PyList_GET_SIZE(slice_item);
      for (int j = 0; j < list_size; ++j) {
        PyObject* list_item = PyList_GetItem(slice_item, j);
        if (PyCheckInteger(list_item)) {
          all_bool = false;
        } else if (!PyBool_Check(list_item)) {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Only support int or bool in index list."));
        }
      }
      if (all_bool) {
        PADDLE_ENFORCE_EQ(
            list_size, shape[0],
            platform::errors::InvalidArgument(
                "The dimension of bool index doesn't match indexed array along "
                "dimension 0, the target dimension is %d, but received %d.",
                shape[0], list_size));

        for (int j = 0; j < list_size; ++j) {
          PyObject* list_item = PyList_GetItem(slice_item, j);
          if (list_item == Py_True) {
            list_select_idxs->push_back(j);
          }
        }
      } else {
        for (int j = 0; j < list_size; ++j) {
          PyObject* list_item = PyList_GetItem(slice_item, j);
          if (PyCheckInteger(list_item)) {
            list_select_idxs->push_back(
                static_cast<int>(PyLong_AsLong(list_item)));
          } else if (list_item == Py_True) {
            list_select_idxs->push_back(1);
          } else {
            list_select_idxs->push_back(0);
          }
        }
      }

    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, Tensor.__indices__() only allows indexing "
          "by Integers, Slices, Ellipsis, None, tuples of these types "
          "and list of Bool and Integers, but received "
          "%s in %dth slice item",
          std::string(Py_TYPE(slice_item)->tp_name), i + 1));
    }
  }

  // valid_index is the number of dimensions exclude None index
  const int valid_indexs = size - none_axes->size() - ell_count;
  PADDLE_ENFORCE_EQ(valid_indexs <= rank, true,
                    platform::errors::InvalidArgument(
                        "Too many indices (%d) for tensor of dimension %d.",
                        valid_indexs, rank));
}

}  // namespace pybind
}  // namespace paddle
