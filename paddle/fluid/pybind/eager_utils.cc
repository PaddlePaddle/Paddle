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
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/operators/py_func_op.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"

namespace paddle {
namespace pybind {

extern PyTypeObject* p_eager_tensor_type;

extern PyTypeObject* g_vartype_pytype;
extern PyTypeObject* g_place_pytype;
extern PyTypeObject* g_cudaplace_pytype;
extern PyTypeObject* g_cpuplace_pytype;
extern PyTypeObject* g_xpuplace_pytype;
extern PyTypeObject* g_npuplace_pytype;
extern PyTypeObject* g_cudapinnedplace_pytype;
extern PyTypeObject* g_framework_tensor_pytype;
extern PyTypeObject* g_framework_lodtensorarray_pytype;

int TensorDtype2NumpyDtype(pten::DataType dtype) {
  switch (dtype) {
    case pten::DataType::BOOL:
      return pybind11::detail::npy_api::NPY_BOOL_;
    case pten::DataType::INT8:
      return pybind11::detail::npy_api::NPY_INT8_;
    case pten::DataType::UINT8:
      return pybind11::detail::npy_api::NPY_UINT8_;
    case pten::DataType::INT16:
      return pybind11::detail::npy_api::NPY_INT16_;
    case pten::DataType::INT32:
      return pybind11::detail::npy_api::NPY_INT32_;
    case pten::DataType::INT64:
      return pybind11::detail::npy_api::NPY_INT64_;
    case pten::DataType::FLOAT16:
      return pybind11::detail::NPY_FLOAT16_;
    case pten::DataType::FLOAT32:
      return pybind11::detail::npy_api::NPY_FLOAT_;
    case pten::DataType::FLOAT64:
      return pybind11::detail::npy_api::NPY_DOUBLE_;
    case pten::DataType::COMPLEX64:
      return pybind11::detail::NPY_COMPLEX64;
    case pten::DataType::COMPLEX128:
      return pybind11::detail::NPY_COMPLEX128;
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Unknow pten::DataType, the int value = %d.",
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

egr::EagerTensor CastPyArg2EagerTensor(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_IsInstance(obj,
                          reinterpret_cast<PyObject*>(p_eager_tensor_type))) {
    return reinterpret_cast<EagerTensorObject*>(obj)->eager_tensor;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "argument (position %d) must be "
        "EagerTensor, but got %s",
        arg_pos + 1, reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

std::vector<egr::EagerTensor> CastPyArg2VectorOfEagerTensor(PyObject* obj,
                                                            ssize_t arg_pos) {
  std::vector<egr::EagerTensor> result;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_IsInstance(
              item, reinterpret_cast<PyObject*>(p_eager_tensor_type))) {
        result.emplace_back(
            reinterpret_cast<EagerTensorObject*>(item)->eager_tensor);
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
      if (PyObject_IsInstance(
              item, reinterpret_cast<PyObject*>(p_eager_tensor_type))) {
        result.emplace_back(
            reinterpret_cast<EagerTensorObject*>(item)->eager_tensor);
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
        "EagerTensor, but got %s",
        arg_pos + 1, reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

std::vector<framework::Tensor> CastPyArg2VectorOfTensor(PyObject* obj,
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

PyObject* ToPyObject(const egr::EagerTensor& value) {
  PyObject* obj = p_eager_tensor_type->tp_alloc(p_eager_tensor_type, 0);
  if (obj) {
    auto v = reinterpret_cast<EagerTensorObject*>(obj);
    new (&(v->eager_tensor)) egr::EagerTensor();
    v->eager_tensor = value;
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

PyObject* ToPyObject(const std::vector<egr::EagerTensor>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyObject* obj = p_eager_tensor_type->tp_alloc(p_eager_tensor_type, 0);
    if (obj) {
      auto v = reinterpret_cast<EagerTensorObject*>(obj);
      new (&(v->eager_tensor)) egr::EagerTensor();
      v->eager_tensor = value[i];
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
  auto obj = ::pybind11::cast(value, py::return_value_policy::copy);
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

egr::EagerTensor& GetEagerTensorFromArgs(const std::string& op_type,
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
    static egr::EagerTensor emptytensor;
    return emptytensor;
  }

  return reinterpret_cast<EagerTensorObject*>(obj)->eager_tensor;
}

std::vector<egr::EagerTensor> GetEagerTensorListFromArgs(
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

  std::vector<egr::EagerTensor> result;

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
          reinterpret_cast<EagerTensorObject*>(PyList_GetItem(list, i))
              ->eager_tensor);
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
          reinterpret_cast<EagerTensorObject*>(PyTuple_GetItem(list, i))
              ->eager_tensor);
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

egr::EagerTensor* GetEagerTensorPtrFromArgs(const std::string& op_type,
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
    static egr::EagerTensor emptytensor;
    return &emptytensor;
  }

  return &(reinterpret_cast<EagerTensorObject*>(obj)->eager_tensor);
}

std::vector<egr::EagerTensor*> GetEagerTensorPtrListFromArgs(
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

  std::vector<egr::EagerTensor*> result;

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
          &(reinterpret_cast<EagerTensorObject*>(PyList_GetItem(list, i))
                ->eager_tensor));
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
          &(reinterpret_cast<EagerTensorObject*>(PyTuple_GetItem(list, i))
                ->eager_tensor));
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
}  // namespace pybind
}  // namespace paddle
