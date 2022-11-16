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

#include "paddle/fluid/pybind/eager_utils.h"

#include <Python.h>

#include <string>
#include <vector>

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/scope_guard.h"
#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/operators/py_func_op.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace pybind {

extern PyTypeObject* p_tensor_type;
extern PyTypeObject* p_string_tensor_type;

extern PyTypeObject* g_framework_scope_pytype;
extern PyTypeObject* g_vartype_pytype;
extern PyTypeObject* g_place_pytype;
extern PyTypeObject* g_cudaplace_pytype;
extern PyTypeObject* g_cpuplace_pytype;
extern PyTypeObject* g_xpuplace_pytype;
extern PyTypeObject* g_npuplace_pytype;
extern PyTypeObject* g_cudapinnedplace_pytype;
extern PyTypeObject* g_customplace_pytype;
extern PyTypeObject* g_framework_tensor_pytype;
extern PyTypeObject* g_framework_lodtensorarray_pytype;
extern PyTypeObject* g_custom_op_kernel_ctx_pytype;
extern PyTypeObject* g_jit_function_pytype;

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
    case phi::DataType::BFLOAT16:
      return pybind11::detail::NPY_UINT16_;
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
    case phi::DataType::PSTRING:
      return pybind11::detail::npy_api::NPY_UNICODE_;
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Unknow phi::DataType, the int value = %d.",
          static_cast<int>(dtype)));
      return 0;
  }
}

bool PyObject_CheckLongOrConvertToLong(PyObject** obj) {
  if (PyLong_Check(*obj) && !PyBool_Check(*obj)) {
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
                   // test cases pass in None.
  } else if (obj == Py_True) {
    return true;
  } else if (obj == Py_False) {
    return false;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "bool, but got %s",
        arg_pos + 1,
        (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
  }
}

int CastPyArg2AttrInt(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_CheckLongOrConvertToLong(&obj)) {
    return static_cast<int>(PyLong_AsLong(obj));
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "int, but got %s",
        arg_pos + 1,
        (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
  }
}

int64_t CastPyArg2AttrLong(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_CheckLongOrConvertToLong(&obj)) {
    return (int64_t)PyLong_AsLong(obj);  // NOLINT
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "long, but got %s",
        arg_pos + 1,
        (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
  }
}

size_t CastPyArg2AttrSize_t(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_CheckLongOrConvertToLong(&obj)) {
    return PyLong_AsSize_t(obj);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "long, but got %s",
        arg_pos + 1,
        (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
  }
}

float CastPyArg2AttrFloat(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_CheckFloatOrConvertToFloat(&obj)) {
    return static_cast<float>(PyFloat_AsDouble(obj));
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "float, but got %s",
        arg_pos + 1,
        (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
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
        arg_pos + 1,
        (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
    return "";
  }
}

bool IsEagerTensor(PyObject* obj) {
  return PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type));
}

paddle::experimental::Tensor CastPyArg2Tensor(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type)) ||
      PyObject_IsInstance(obj,
                          reinterpret_cast<PyObject*>(p_string_tensor_type))) {
    return reinterpret_cast<TensorObject*>(obj)->tensor;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "Tensor, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

std::shared_ptr<imperative::VarBase> CastPyArg2VarBase(PyObject* obj,
                                                       ssize_t arg_pos) {
  return py::cast<std::shared_ptr<imperative::VarBase>>(obj);
}

std::shared_ptr<jit::Function> CastPyArg2JitFunction(PyObject* obj,
                                                     ssize_t arg_pos) {
  if (PyObject_IsInstance(obj,
                          reinterpret_cast<PyObject*>(g_jit_function_pytype))) {
    return ::pybind11::handle(obj).cast<std::shared_ptr<jit::Function>>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "BaseEngine, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
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
      } else if (item == Py_None) {
        // emplace empty Tensor for None
        result.emplace_back();
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of Tensor, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
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
      } else if (item == Py_None) {
        // emplace empty Tensor for None
        result.emplace_back();
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of Tensor, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
  } else if (obj == Py_None) {
    return {};
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "list or tuple, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return result;
}

std::vector<int> CastPyArg2VectorOfInt(PyObject* obj, size_t arg_pos) {
  std::vector<int> result;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GET_ITEM(obj, i);
      if (PyObject_CheckLongOrConvertToLong(&item)) {
        result.emplace_back(static_cast<int>(PyLong_AsLong(item)));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of int, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GET_ITEM(obj, i);
      if (PyObject_CheckLongOrConvertToLong(&item)) {
        result.emplace_back(static_cast<int>(PyLong_AsLong(item)));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of int, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
  } else if (obj == Py_None) {
    return {};
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "list or tuple, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return result;
}

std::vector<size_t> CastPyArg2VectorOfSize_t(PyObject* obj, size_t arg_pos) {
  std::vector<size_t> result;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckLongOrConvertToLong(&item)) {
        result.emplace_back(PyLong_AsSize_t(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of int, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "list, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return result;
}

std::vector<std::vector<size_t>> CastPyArg2VectorOfVectorOfSize_t(
    PyObject* obj, size_t arg_pos) {
  std::vector<std::vector<size_t>> result;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      result.emplace_back(CastPyArg2VectorOfSize_t(item, arg_pos));
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "list but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
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
  } else if (PyObject_IsInstance(
                 obj, reinterpret_cast<PyObject*>(g_customplace_pytype))) {
    place = ::pybind11::handle(obj).cast<platform::CustomPlace>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "one "
        "of(Place,CUDAPlace,CPUPlace,XPUPlace,NPUPlace,CUDAPinnedPlace,"
        "CustomPlace), "
        "but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return place;
}

phi::DenseTensor CastPyArg2FrameworkTensor(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_IsInstance(
          obj, reinterpret_cast<PyObject*>(g_framework_tensor_pytype))) {
    return ::pybind11::handle(obj).cast<phi::DenseTensor>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "DenseTensor, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

std::vector<phi::DenseTensor> CastPyArg2VectorOfTensorBase(PyObject* obj,
                                                           ssize_t arg_pos) {
  std::vector<phi::DenseTensor> result;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_IsInstance(
              item, reinterpret_cast<PyObject*>(g_framework_tensor_pytype))) {
        result.emplace_back(::pybind11::handle(item).cast<phi::DenseTensor>());
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of LoDTensor, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_IsInstance(
              item, reinterpret_cast<PyObject*>(g_framework_tensor_pytype))) {
        result.emplace_back(::pybind11::handle(item).cast<phi::DenseTensor>());
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument (position %d) must be "
            "list of LoDTensor, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
  } else if (PyObject_IsInstance(obj,
                                 reinterpret_cast<PyObject*>(
                                     g_framework_lodtensorarray_pytype))) {
    for (auto& tensor :
         (::pybind11::handle(obj).cast<framework::LoDTensorArray>())) {
      result.emplace_back(tensor);
    }
  } else if (obj == Py_None) {
    return {};
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "list or tuple, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
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
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return dtype;
}

std::unordered_map<std::wstring, int> CastPyArg2Vocab(PyObject* obj,
                                                      ssize_t arg_pos) {
  if (PyDict_Check(obj)) {
    return ::pybind11::handle(obj)
        .cast<std::unordered_map<std::wstring, int>>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be dict, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

std::vector<std::string> CastPyArg2Strings(PyObject* obj, ssize_t arg_pos) {
  if (PyList_Check(obj)) {
    return ::pybind11::handle(obj).cast<std::vector<std::string>>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be list, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

paddle::CustomOpKernelContext CastPyArg2CustomOpKernelContext(PyObject* obj,
                                                              ssize_t arg_pos) {
  if (PyObject_IsInstance(
          obj, reinterpret_cast<PyObject*>(g_custom_op_kernel_ctx_pytype))) {
    return ::pybind11::handle(obj).cast<paddle::CustomOpKernelContext>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "one of(Place,CUDAPlace,CPUPlace,XPUPlace,NPUPlace,CUDAPinnedPlace), "
        "but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
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

PyObject* ToPyObject(uint32_t value) { return PyLong_FromUnsignedLong(value); }

PyObject* ToPyObject(int64_t value) { return PyLong_FromLongLong(value); }

PyObject* ToPyObject(size_t value) { return PyLong_FromSize_t(value); }

PyObject* ToPyObject(float value) { return PyLong_FromDouble(value); }

PyObject* ToPyObject(double value) { return PyLong_FromDouble(value); }

PyObject* ToPyObject(const char* value) { return PyUnicode_FromString(value); }

PyObject* ToPyObject(const std::string& value) {
  return PyUnicode_FromString(value.c_str());
}

PyObject* ToPyObject(const paddle::experimental::Tensor& value,
                     bool return_py_none_if_not_initialize) {
  if (return_py_none_if_not_initialize && !value.initialized()) {
    RETURN_PY_NONE
  }
  PyObject* obj = nullptr;
  if (value.initialized() && value.is_string_tensor()) {
    // In order to return the core.eager.StringTensor, there is need
    // to use p_string_tensor_type to create a python obj.
    obj = p_string_tensor_type->tp_alloc(p_string_tensor_type, 0);
  } else {
    obj = p_tensor_type->tp_alloc(p_tensor_type, 0);
  }
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

PyObject* ToPyObject(const paddle::experimental::Tensor& value,
                     PyObject* args,
                     const std::map<ssize_t, ssize_t>& inplace_var_idx_map) {
  if (!inplace_var_idx_map.empty() && inplace_var_idx_map.count(0)) {
    return ToPyObject(args, inplace_var_idx_map.at(0));
  } else {
    return ToPyObject(value);
  }
}

PyObject* ToPyObject(PyObject* args, ssize_t arg_idx) {
  // For inplace op, directly return the input PyObject of the inplace tensor.
  // [Parameter]
  // args: Input PyObject.
  // arg_idx: Index of inplace PyObject in input args. Used to find the input
  // inplace PyObject.
  PyObject* obj = PyTuple_GET_ITEM(args, arg_idx);
  Py_INCREF(obj);
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

PyObject* ToPyObject(const std::vector<size_t>& value) {
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

PyObject* ToPyObject(const std::vector<std::vector<size_t>>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<paddle::experimental::Tensor>& value,
                     bool return_py_none_if_not_initialize) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    if (!value[i].initialized() && return_py_none_if_not_initialize) {
      Py_INCREF(Py_None);
      PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), Py_None);
    } else {
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
  }

  return result;
}

PyObject* ToPyObject(
    const std::vector<std::vector<paddle::experimental::Tensor>>& value,
    bool return_py_none_if_not_initialize) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result,
                    static_cast<Py_ssize_t>(i),
                    ToPyObject(value[i], return_py_none_if_not_initialize));
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

PyObject* ToPyObject(const phi::DenseTensor* value) {
  auto obj = ::pybind11::cast(value, py::return_value_policy::reference);
  obj.inc_ref();
  return obj.ptr();
}

PyObject* ToPyObject(const phi::SelectedRows* value) {
  auto obj = ::pybind11::cast(value, py::return_value_policy::reference);
  obj.inc_ref();
  return obj.ptr();
}

PyObject* ToPyObject(const void* value) {
  if (value == nullptr) {
    RETURN_PY_NONE
  }
  PADDLE_THROW(
      platform::errors::Fatal("ToPyObject do not support void* with value."));
}

PyObject* ToPyObject(
    const std::unordered_map<std::string, std::vector<std::string>>& value) {
  PyObject* dict = PyDict_New();
  for (const auto& map_iter : value) {
    // Convert Key
    PyObject* key_string = PyUnicode_FromString(map_iter.first.c_str());
    if (!key_string) {
      PADDLE_THROW(
          platform::errors::Fatal("Unable to convert std::string to PyObject"));
    }

    // Convert Val
    PyObject* py_list = PyList_New(0);
    for (const auto& vector_iter : map_iter.second) {
      PyObject* val_string = PyUnicode_FromString(vector_iter.c_str());
      if (!val_string) {
        PADDLE_THROW(platform::errors::Fatal(
            "Unable to convert std::string to PyObject"));
      }

      if (PyList_Append(py_list, val_string) != 0) {
        PADDLE_THROW(
            platform::errors::Fatal("Unable to append string to py_list"));
      }
      Py_DECREF(val_string);
    }

    if (PyDict_SetItem(dict, key_string, py_list) != 0) {
      PADDLE_THROW(
          platform::errors::Fatal("Unable to set key:value for py_dict"));
    }
    Py_DECREF(py_list);
    Py_DECREF(key_string);
  }

  return dict;
}

PyObject* ToPyObject(const std::unordered_map<std::wstring, int>& value) {
  PyObject* dict = PyDict_New();
  for (const auto& map_iter : value) {
    // Convert Key
    PyObject* key_string =
        PyUnicode_FromWideChar(map_iter.first.c_str(), map_iter.first.size());
    if (!key_string) {
      PADDLE_THROW(platform::errors::Fatal(
          "Unable to convert std::wstring to PyObject"));
    }

    // Convert Val
    PyObject* py_int = PyLong_FromLong(map_iter.second);

    if (PyDict_SetItem(dict, key_string, py_int) != 0) {
      PADDLE_THROW(
          platform::errors::Fatal("Unable to set key:value for py_dict"));
    }
  }
  return dict;
}

// For Final State Dygraph,
// We directly use paddle::optional(Tensor) as dispensable Tensor
paddle::optional<paddle::experimental::Tensor> GetOptionalTensorFromArgs(
    const std::string& op_type,
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
          op_type,
          arg_name,
          arg_idx));
    }
    return paddle::none;
  }

  if (PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type))) {
    return paddle::make_optional<paddle::experimental::Tensor>(
        reinterpret_cast<TensorObject*>(obj)->tensor);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be Tensor, but got %s",
        op_type,
        arg_name,
        arg_idx,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

static paddle::experimental::Tensor& GetTensorFromPyObject(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* obj,
    ssize_t arg_idx,
    bool dispensable) {
  if (PyTuple_Check(obj)) {
    obj = PyTuple_GET_ITEM(obj, 0);
  }

  if (obj == nullptr || obj == Py_None) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be Tensor, but got None",
          op_type,
          arg_name,
          arg_idx));
    }
    static paddle::experimental::Tensor emptytensor;
    return emptytensor;
  }

  if (PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type))) {
    return reinterpret_cast<TensorObject*>(obj)->tensor;
  } else if (PyObject_IsInstance(
                 obj, reinterpret_cast<PyObject*>(p_string_tensor_type))) {
    return reinterpret_cast<TensorObject*>(obj)->tensor;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be Tensor, but got %s",
        op_type,
        arg_name,
        arg_idx,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

// For Intermediate State Dygraph,
// we use an uninitialized Tensor to represent dispensable Tensor
paddle::experimental::Tensor& GetTensorFromArgs(const std::string& op_type,
                                                const std::string& arg_name,
                                                PyObject* args,
                                                ssize_t arg_idx,
                                                bool dispensable) {
  PyObject* obj = PyTuple_GET_ITEM(args, arg_idx);
  return GetTensorFromPyObject(op_type, arg_name, obj, arg_idx, dispensable);
}

std::vector<paddle::experimental::Tensor> GetTensorListFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable) {
  PyObject* list = PyTuple_GET_ITEM(args, arg_idx);

  if (list == nullptr) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensor, but got "
          "None",
          op_type,
          arg_name,
          arg_idx));
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
          op_type,
          arg_name,
          arg_idx));
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
          op_type,
          arg_name,
          arg_idx));
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
        op_type,
        arg_name,
        arg_idx,
        (reinterpret_cast<PyTypeObject*>(list->ob_type))->tp_name));
  }

  return result;
}

paddle::optional<std::vector<paddle::experimental::Tensor>>
GetOptionalTensorListFromArgs(const std::string& op_type,
                              const std::string& arg_name,
                              PyObject* args,
                              ssize_t arg_idx,
                              bool dispensable) {
  PyObject* list = PyTuple_GET_ITEM(args, arg_idx);

  if (list == nullptr || list == Py_None) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensor, but got "
          "None",
          op_type,
          arg_name,
          arg_idx));
    }
    return paddle::none;
  }

  std::vector<paddle::experimental::Tensor> result;

  if (PyList_Check(list)) {
    Py_ssize_t len = PyList_Size(list);
    result.reserve(static_cast<size_t>(len));
    if (len == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensors, but got "
          "empty list",
          op_type,
          arg_name,
          arg_idx));
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
          op_type,
          arg_name,
          arg_idx));
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      result.emplace_back(
          reinterpret_cast<TensorObject*>(PyTuple_GetItem(list, i))->tensor);
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be list of Tensors, but got "
        "%s",
        op_type,
        arg_name,
        arg_idx,
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
          op_type,
          arg_name,
          arg_idx));
    }
    static paddle::experimental::Tensor emptytensor;
    return &emptytensor;
  }

  if (PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type))) {
    return &(reinterpret_cast<TensorObject*>(obj)->tensor);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be Tensor, but got %s",
        op_type,
        arg_name,
        arg_idx,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

std::vector<paddle::experimental::Tensor*> GetTensorPtrListFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable) {
  PyObject* list = PyTuple_GET_ITEM(args, arg_idx);

  if (list == nullptr) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of Tensor, but got "
          "None",
          op_type,
          arg_name,
          arg_idx));
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
          op_type,
          arg_name,
          arg_idx));
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
          op_type,
          arg_name,
          arg_idx));
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
        op_type,
        arg_name,
        arg_idx,
        (reinterpret_cast<PyTypeObject*>(list->ob_type))->tp_name));
  }

  return result;
}

std::vector<paddle::experimental::Tensor*> GetTensorPtrListFromPyObject(
    PyObject* obj) {
  std::vector<paddle::experimental::Tensor*> result;

  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    if (len == 0) {
      PADDLE_THROW(
          platform::errors::InvalidArgument("The list of Tensor is empty."));
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      result.emplace_back(
          &(reinterpret_cast<TensorObject*>(PyList_GetItem(obj, i))->tensor));
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    if (len == 0) {
      PADDLE_THROW(
          platform::errors::InvalidArgument("The tuple of Tensor is empty."));
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      result.emplace_back(
          &(reinterpret_cast<TensorObject*>(PyTuple_GetItem(obj, i))->tensor));
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The PyObject must be list of Tensors, but got "
        "%s",
        (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
  }

  return result;
}

std::vector<paddle::experimental::Tensor> GetTensorListFromPyObject(
    PyObject* obj) {
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
            "argument must be "
            "list of Tensor, but got %s at pos %d",
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
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
            "argument must be "
            "list of Tensor, but got %s at pos %d",
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument must be "
        "list or tuple, but got %s",
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return result;
}

paddle::experimental::Tensor& GetTensorFromPyObject(PyObject* obj) {
  if (!IsEagerTensor(obj)) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument must be "
        "Tensor, but got %s",
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return reinterpret_cast<TensorObject*>(obj)->tensor;
}

paddle::experimental::Scalar CastNumpy2Scalar(PyObject* obj,
                                              const std::string& op_type,
                                              ssize_t arg_pos) {
  PyTypeObject* type = obj->ob_type;
  auto type_name = std::string(type->tp_name);
  VLOG(4) << "type_name: " << type_name;
  if (type_name == "numpy.ndarray" && PySequence_Check(obj)) {
    PyObject* item = nullptr;
    item = PySequence_GetItem(obj, 0);
    if (PyObject_CheckFloatOrToFloat(&item)) {
      float value = static_cast<float>(PyFloat_AsDouble(item));
      return paddle::experimental::Scalar(value);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument (position %d) is numpy.ndarry, the inner elements "
          "must be "
          "numpy.float32/float64 now, but got %s",
          op_type,
          arg_pos + 1,
          type_name));  // NOLINT
    }
  } else if (type_name == "numpy.float64") {
    double value = CastPyArg2Double(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (type_name == "numpy.float32") {
    float value = CastPyArg2Float(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (type_name == "numpy.int64") {
    int64_t value = CastPyArg2Long(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (type_name == "numpy.int32" || type_name == "numpy.intc") {
    int value = CastPyArg2Int(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "numpy.float32/float64, numpy.int32/int64, but got %s",
        op_type,
        arg_pos + 1,
        type_name));  // NOLINT
  }
}

paddle::experimental::Scalar CastPyArg2Scalar(PyObject* obj,
                                              const std::string& op_type,
                                              ssize_t arg_pos) {
  if (obj == Py_None) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "int, float, bool or Tensor, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  // obj could be: int, float, bool, paddle.Tensor
  PyTypeObject* type = obj->ob_type;
  auto type_name = std::string(type->tp_name);
  VLOG(4) << "type_name: " << type_name;
  if (PyBool_Check(obj)) {
    bool value = CastPyArg2Boolean(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (PyLong_Check(obj)) {
    int64_t value = CastPyArg2Long(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (PyFloat_Check(obj)) {
    double value = CastPyArg2Double(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (IsEagerTensor(obj)) {
    paddle::experimental::Tensor& value = GetTensorFromPyObject(
        op_type, "" /*arg_name*/, obj, arg_pos, false /*dispensable*/);
    return paddle::experimental::Scalar(value);
  } else if (type_name.find("numpy") != std::string::npos) {
    return CastNumpy2Scalar(obj, op_type, arg_pos);
  } else if (PyComplex_Check(obj)) {
    auto value = CastPyArg2Complex(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (PyObject_CheckLongOrToLong(&obj)) {
    int value = CastPyArg2Int(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (PyObject_CheckString(obj)) {
    std::string value = CastPyArg2String(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "int, float, bool or Tensor, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  // Fake a Scalar
  return paddle::experimental::Scalar(1.0);
}

std::vector<phi::Scalar> CastPyArg2ScalarArray(PyObject* obj,
                                               const std::string& op_type,
                                               ssize_t arg_pos) {
  if (obj == Py_None) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "a list of int, float, or bool, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  PyTypeObject* type = obj->ob_type;
  auto type_name = std::string(type->tp_name);
  VLOG(4) << "type_name: " << type_name;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    item = PyList_GetItem(obj, 0);
    if (PyObject_CheckFloatOrToFloat(&item)) {
      std::vector<phi::Scalar> value;
      for (Py_ssize_t i = 0; i < len; i++) {
        item = PyList_GetItem(obj, i);
        value.emplace_back(phi::Scalar{PyFloat_AsDouble(item)});
      }
      return value;
    } else if (PyObject_CheckLongOrToLong(&item)) {
      std::vector<phi::Scalar> value;
      for (Py_ssize_t i = 0; i < len; i++) {
        item = PyList_GetItem(obj, i);
        value.emplace_back(
            phi::Scalar{static_cast<int64_t>(PyLong_AsLong(item))});
      }
      return value;
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "a list of int, float, or bool, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  // Fake a ScalarArray
  return std::vector<phi::Scalar>({phi::Scalar(1.0)});
}

paddle::experimental::IntArray CastPyArg2IntArray(PyObject* obj,
                                                  const std::string& op_type,
                                                  ssize_t arg_pos) {
  // In case of IntArray, only two possible PyObjects:
  // 1. list of int
  // 2. Tensor
  if (obj == Py_None) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "list or Tensor, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  // obj could be: int, float, bool, paddle.Tensor
  PyTypeObject* type = obj->ob_type;
  auto type_name = std::string(type->tp_name);
  if (type_name == "list" || type_name == "tuple" ||
      type_name == "numpy.ndarray") {
    std::vector<int64_t> value = CastPyArg2Longs(obj, op_type, arg_pos);
    return paddle::experimental::IntArray(value);
  } else if (type_name == "paddle.Tensor" || type_name == "Tensor") {
    paddle::experimental::Tensor& value = GetTensorFromPyObject(
        op_type, "" /*arg_name*/, obj, arg_pos, false /*dispensable*/);
    return paddle::experimental::IntArray(value);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "list or Tensor, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }

  // Fake a IntArray
  return paddle::experimental::IntArray({1});
}

paddle::framework::Scope* CastPyArg2ScopePtr(PyObject* obj) {
  if (PyObject_IsInstance(
          obj, reinterpret_cast<PyObject*>(g_framework_scope_pytype))) {
    return ::pybind11::handle(obj).cast<paddle::framework::Scope*>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "PyObject can not be cast into framework::Scope"));
  }
}

std::vector<paddle::framework::Scope*> GetScopePtrListFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable) {
  PyObject* list = PyTuple_GET_ITEM(args, arg_idx);
  if (list == nullptr) {
    if (!dispensable) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of scope, but got "
          "None",
          op_type,
          arg_name,
          arg_idx));
    }
  }

  std::vector<paddle::framework::Scope*> result;
  if (PyList_Check(list)) {
    Py_ssize_t len = PyList_Size(list);
    if (len == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of scope, but got "
          "empty list",
          op_type,
          arg_name,
          arg_idx));
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      result.emplace_back(CastPyArg2ScopePtr(PyList_GetItem(list, i)));
    }
  } else if (PyTuple_Check(list)) {
    Py_ssize_t len = PyTuple_Size(list);
    if (len == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "%s(): argument '%s' (position %d) must be list of scope, but got "
          "empty list",
          op_type,
          arg_name,
          arg_idx));
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      result.emplace_back(CastPyArg2ScopePtr(PyList_GetItem(list, i)));
    }
  } else if (list == Py_None) {
    return {};
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument '%s' (position %d) must be list of Tensors, but got "
        "%s",
        op_type,
        arg_name,
        arg_idx,
        (reinterpret_cast<PyTypeObject*>(list->ob_type))->tp_name));
  }
  return result;
}

paddle::Place CastPyArg2Place(PyObject* obj,
                              const std::string& op_type,
                              ssize_t arg_pos) {
  return CastPyArg2Place(obj, arg_pos);
}

paddle::DataType CastPyArg2DataType(PyObject* obj,
                                    const std::string& op_type,
                                    ssize_t arg_pos) {
  if (obj == Py_None) {
    return paddle::experimental::DataType::UNDEFINED;
  }

  framework::proto::VarType::Type type = CastPyArg2ProtoType(obj, arg_pos);
  return framework::TransToPhiDataType(type);
}

paddle::experimental::Tensor PyTensorHook::operator()(
    const paddle::experimental::Tensor& var) {
  py::gil_scoped_acquire gil;
  VLOG(3) << "Call PyTensorHook for var " << var.name();

  PyObject* res = nullptr;
  try {
    PyObject* p_tmp_var = ToPyObject(var);
    res = PyObject_CallFunctionObjArgs(py_func_, p_tmp_var, nullptr);
    Py_DECREF(p_tmp_var);
  } catch (platform::EnforceNotMet& e) {
    throw std::move(e);
  } catch (std::exception& e) {
    PADDLE_THROW(platform::errors::Unavailable(
        "Hook function of Tensor raises an exception: %s.", e.what()));
  } catch (...) {
    PADDLE_THROW(platform::errors::Fatal(
        "Hook function of Tensor raises an unknown exception."));
  }

  PADDLE_ENFORCE_NOT_NULL(res,
                          paddle::platform::errors::External(
                              pybind11::detail::error_string().c_str()));
  if (res == Py_None) {
    return var;
  }
  auto res_tensor = reinterpret_cast<TensorObject*>(res)->tensor;
  Py_DECREF(res);
  return res_tensor;
}

void PyVoidHook::operator()() {
  py::gil_scoped_acquire gil;
  VLOG(3) << "Call PyVoidHook";

  try {
    PyObject_CallFunctionObjArgs(py_func_, nullptr);
  } catch (platform::EnforceNotMet& e) {
    throw std::move(e);
  } catch (std::exception& e) {
    PADDLE_THROW(platform::errors::Unavailable(
        "Hook function of Tensor raises an exception: %s.", e.what()));
  } catch (...) {
    PADDLE_THROW(platform::errors::Fatal(
        "Hook function of Tensor raises an unknown exception."));
  }
}

}  // namespace pybind
}  // namespace paddle
