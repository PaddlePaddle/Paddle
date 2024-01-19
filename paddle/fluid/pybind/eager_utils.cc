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
#include "paddle/common/exception.h"
#include "paddle/pir/core/value.h"
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

#include <string>
#include <vector>

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
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
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/fluid/pybind/pir.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/placement_types.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/flags.h"
#include "paddle/pir/core/attribute.h"

PHI_DECLARE_bool(check_nan_inf);
PHI_DECLARE_int32(check_nan_inf_level);
namespace paddle {
namespace pybind {

extern PyTypeObject* p_tensor_type;
extern PyTypeObject* p_string_tensor_type;

extern PyTypeObject* g_framework_scope_pytype;
extern PyTypeObject* g_ir_value_pytype;
extern PyTypeObject* g_vartype_pytype;
extern PyTypeObject* g_data_type_pytype;
extern PyTypeObject* g_place_pytype;
extern PyTypeObject* g_cudaplace_pytype;
extern PyTypeObject* g_cpuplace_pytype;
extern PyTypeObject* g_xpuplace_pytype;
extern PyTypeObject* g_cudapinnedplace_pytype;
extern PyTypeObject* g_customplace_pytype;
extern PyTypeObject* g_framework_tensor_pytype;
extern PyTypeObject* g_framework_lodtensorarray_pytype;
extern PyTypeObject* g_jit_function_pytype;
extern PyTypeObject* g_tensor_dist_attr_pytype;
extern PyTypeObject* g_process_mesh_pytype;
extern PyTypeObject* g_placement_shard_pytype;
extern PyTypeObject* g_placement_replicated_pytype;
extern PyTypeObject* g_placement_partial_pytype;

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

void ConvertToDistTensor(Tensor* x, const phi::distributed::ProcessMesh* mesh) {
  if (!x->defined()) {
    return;
  }
  if (x->is_dist_tensor()) {
    PADDLE_ENFORCE_EQ(
        std::dynamic_pointer_cast<phi::distributed::DistTensor>(x->impl())
            ->process_mesh(),
        *mesh,
        platform::errors::InvalidArgument(
            "Input %s has different mesh. However all inputs should "
            "have the same mesh.",
            x->name()));
    return;
  } else {
    PADDLE_ENFORCE_EQ(
        phi::DenseTensor::classof(x->impl().get()),
        true,
        platform::errors::InvalidArgument(
            "Failed to convert input %s impl to phi::distributed::DistTensor "
            "as it's not phi::DenseTensor.",
            x->name()));
    phi::distributed::Placements placements;
    for (int64_t i = 0; i < mesh->ndim(); ++i) {
      placements.emplace_back(std::make_shared<phi::distributed::Replicate>());
    }

    auto dense_t = std::static_pointer_cast<phi::DenseTensor>(x->impl());
    // auto parallel in dygraph doesn't support strided kernel.
    if (!dense_t->meta().is_contiguous()) {
      *dense_t = paddle::experimental::Trans2Contiguous(*dense_t);
    }
    x->set_impl(std::make_shared<phi::distributed::DistTensor>(
        dense_t, *mesh, placements));
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

bool PyObject_CheckIRValue(PyObject* obj) {
  return PyObject_TypeCheck(obj, g_ir_value_pytype);
}

bool PyObject_CheckIRVectorOfValue(PyObject* obj) {
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    // if obj is [], parse it as std::vector<scalar>
    if (len == 0) {
      return false;
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (!PyObject_CheckIRValue(item)) {
        return false;
      }
    }
    return true;
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    if (len == 0) {
      return false;
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (!PyObject_CheckIRValue(item)) {
        return false;
      }
    }
    return true;
  } else if (PyObject_TypeCheck(obj, g_ir_value_pytype)) {
    return true;
  } else {
    return false;
  }
}

bool CastPyArg2AttrBoolean(PyObject* obj, ssize_t arg_pos) {
  if (obj == Py_None || obj == Py_False) {
    return false;  // To be compatible with QA integration testing. Some
                   // test cases pass in None.
  } else if (obj == Py_True) {
    return true;
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
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
    PADDLE_THROW(platform::errors::InvalidType(
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
    PADDLE_THROW(platform::errors::InvalidType(
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
    PADDLE_THROW(platform::errors::InvalidType(
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
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be "
        "float, but got %s",
        arg_pos + 1,
        (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
  }
}

std::string CastPyArg2AttrString(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_CheckStr(obj)) {
    Py_ssize_t size = 0;
    const char* data = nullptr;
    data = PyUnicode_AsUTF8AndSize(obj, &size);
    return std::string(data, static_cast<size_t>(size));
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be "
        "str, but got %s",
        arg_pos + 1,
        (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
    return "";
  }
}

std::shared_ptr<imperative::VarBase> CastPyArg2VarBase(PyObject* obj,
                                                       ssize_t arg_pos) {
  return py::cast<std::shared_ptr<imperative::VarBase>>(obj);
}

void SetPythonStack() {
  if (FLAGS_check_nan_inf && FLAGS_check_nan_inf_level == 0) {
    VLOG(4) << "this is SetPythonStack";
    pybind11::gil_scoped_acquire gil;
    PyObject* mod = PyImport_ImportModule("traceback");
    PyObject* traceback_list = PyObject_CallMethod(mod, "format_stack", "");
    std::string str = "";
    for (Py_ssize_t i = 0; i < PyList_Size(traceback_list); i++) {
      PyObject* line = PyList_GetItem(traceback_list, i);
      str += py::str(PyUnicode_AsUTF8(line));
    }
    std::string last = str + egr::Controller::Instance().GetPythonStack();
    egr::Controller::Instance().SetPythonStack(last);
  }
}

std::shared_ptr<jit::Function> CastPyArg2JitFunction(PyObject* obj,
                                                     ssize_t arg_pos) {
  if (PyObject_TypeCheck(obj, g_jit_function_pytype)) {
    return ::pybind11::handle(obj).cast<std::shared_ptr<jit::Function>>();
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be "
        "BaseEngine, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

std::vector<paddle::Tensor> CastPyArg2VectorOfTensor(
    PyObject* obj, ssize_t arg_pos, const phi::distributed::ProcessMesh* mesh) {
  std::vector<paddle::Tensor> result;
  const phi::distributed::ProcessMesh* local_mesh = mesh;
  int mesh_start_index = -1;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_TypeCheck(item, p_tensor_type)) {
        paddle::Tensor& tensor = reinterpret_cast<TensorObject*>(item)->tensor;
        if (local_mesh) {
          ConvertToDistTensor(&tensor, local_mesh);
        } else {
          if (tensor.defined() && tensor.is_dist_tensor()) {
            local_mesh =
                &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                      tensor.impl())
                      ->process_mesh());
            mesh_start_index = i;
          }
        }
        result.emplace_back(tensor);
      } else if (item == Py_None) {
        // emplace empty Tensor for None
        result.emplace_back();
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "argument (position %d) must be "
            "list of Tensor, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
    for (Py_ssize_t i = 0; i < mesh_start_index; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_TypeCheck(item, p_tensor_type)) {
        paddle::Tensor& tensor = reinterpret_cast<TensorObject*>(item)->tensor;
        ConvertToDistTensor(&tensor, local_mesh);
        result[i] = tensor;
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_TypeCheck(item, p_tensor_type)) {
        paddle::Tensor& tensor = reinterpret_cast<TensorObject*>(item)->tensor;
        if (local_mesh) {
          ConvertToDistTensor(&tensor, local_mesh);
        } else {
          if (tensor.defined() && tensor.is_dist_tensor()) {
            local_mesh =
                &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                      tensor.impl())
                      ->process_mesh());
            mesh_start_index = i;
          }
        }
        result.emplace_back(tensor);
      } else if (item == Py_None) {
        // emplace empty Tensor for None
        result.emplace_back();
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "argument (position %d) must be "
            "list of Tensor, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
    for (Py_ssize_t i = 0; i < mesh_start_index; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_TypeCheck(item, p_tensor_type)) {
        paddle::Tensor& tensor = reinterpret_cast<TensorObject*>(item)->tensor;
        ConvertToDistTensor(&tensor, local_mesh);
        result[i] = tensor;
      }
    }
  } else if (obj == Py_None) {
    return {};
  } else if (PyObject_TypeCheck(obj, p_tensor_type)) {
    return {reinterpret_cast<TensorObject*>(obj)->tensor};
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
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
        PADDLE_THROW(platform::errors::InvalidType(
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
        PADDLE_THROW(platform::errors::InvalidType(
            "argument (position %d) must be "
            "list of int, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
  } else if (obj == Py_None) {
    return {};
  } else if (PyObject_CheckLongOrConvertToLong(&obj)) {
    return {static_cast<int>(PyLong_AsLong(obj))};
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be "
        "list or tuple, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return result;
}

std::vector<int64_t> CastPyArg2VectorOfInt64(PyObject* obj, size_t arg_pos) {
  std::vector<int64_t> result;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GET_ITEM(obj, i);
      if (PyObject_CheckLongOrConvertToLong(&item)) {
        result.emplace_back(static_cast<int64_t>(PyLong_AsLong(item)));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
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
        result.emplace_back(static_cast<int64_t>(PyLong_AsLong(item)));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "argument (position %d) must be "
            "list of int, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
  } else if (obj == Py_None) {
    return {};
  } else if (PyObject_CheckLongOrConvertToLong(&obj)) {
    return {static_cast<int64_t>(PyLong_AsLong(obj))};
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
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
        PADDLE_THROW(platform::errors::InvalidType(
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
        result.emplace_back(PyLong_AsSize_t(item));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "argument (position %d) must be "
            "list of size_t, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
  } else if (obj == Py_None) {
    return {};
  } else if (PyObject_CheckLongOrConvertToLong(&obj)) {
    return {PyLong_AsSize_t(obj)};
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be "
        "list of size_t, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return result;
}

std::vector<float> CastPyArg2VectorOfFloat(PyObject* obj, size_t arg_pos) {
  std::vector<float> result;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckFloatOrConvertToFloat(&item)) {
        result.emplace_back(static_cast<float>(PyFloat_AsDouble(item)));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "argument (position %d) must be "
            "list of float, but got %s at pos %d",
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
      if (PyObject_CheckFloatOrConvertToFloat(&item)) {
        result.emplace_back(static_cast<float>(PyFloat_AsDouble(item)));
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "argument (position %d) must be "
            "list of float, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
  } else if (obj == Py_None) {
    return {};
  } else if (PyObject_CheckFloatOrConvertToFloat(&obj)) {
    return {static_cast<float>(PyFloat_AsDouble(obj))};
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be "
        "list of float, but got %s",
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
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be "
        "list but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return result;
}

platform::Place CastPyArg2Place(PyObject* obj, ssize_t arg_pos) {
  platform::Place place;
  if (PyObject_TypeCheck(obj, g_place_pytype)) {
    place = ::pybind11::handle(obj).cast<platform::Place>();
  } else if (PyObject_TypeCheck(obj, g_cudaplace_pytype)) {
    place = ::pybind11::handle(obj).cast<platform::CUDAPlace>();
  } else if (PyObject_TypeCheck(obj, g_cpuplace_pytype)) {
    place = ::pybind11::handle(obj).cast<platform::CPUPlace>();
  } else if (PyObject_TypeCheck(obj, g_xpuplace_pytype)) {
    place = ::pybind11::handle(obj).cast<platform::XPUPlace>();
  } else if (PyObject_TypeCheck(obj, g_cudapinnedplace_pytype)) {
    place = ::pybind11::handle(obj).cast<platform::CUDAPinnedPlace>();
  } else if (PyObject_TypeCheck(obj, g_customplace_pytype)) {
    place = ::pybind11::handle(obj).cast<platform::CustomPlace>();
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be "
        "one "
        "of(Place,CUDAPlace,CPUPlace,XPUPlace,CUDAPinnedPlace,"
        "CustomPlace), "
        "but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return place;
}

using phi::distributed::TensorDistAttr;
TensorDistAttr CastPyArg2DistAttr(PyObject* obj, ssize_t arg_pos) {
#ifdef PADDLE_WITH_DISTRIBUTE
  if (PyObject_IsInstance(
          obj, reinterpret_cast<PyObject*>(g_tensor_dist_attr_pytype))) {
    return ::pybind11::handle(obj).cast<TensorDistAttr>();
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be "
        "TensorDistAttr, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
#else
  PADDLE_THROW(platform::errors::Unavailable(
      "The parsing of `DistAttr` is not supported in the current "
      "PaddlePaddle, please recompile and installPaddlePaddle with the option "
      "of `WITH_DISTRIBUTE=ON`."));
#endif
}

using phi::distributed::ProcessMesh;
ProcessMesh CastPyArg2ProcessMesh(PyObject* obj, ssize_t arg_pos) {
#ifdef PADDLE_WITH_DISTRIBUTE
  if (PyObject_IsInstance(obj,
                          reinterpret_cast<PyObject*>(g_process_mesh_pytype))) {
    return ::pybind11::handle(obj).cast<ProcessMesh>();
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be "
        "ProcessMesh, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
#else
  PADDLE_THROW(platform::errors::Unavailable(
      "The parsing of `ProcessMesh` is not supported in the current "
      "PaddlePaddle, please recompile and installPaddlePaddle with the option "
      "of `WITH_DISTRIBUTE=ON`."));
#endif
}

phi::DenseTensor CastPyArg2FrameworkTensor(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_TypeCheck(obj, g_framework_tensor_pytype)) {
    return ::pybind11::handle(obj).cast<phi::DenseTensor>();
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
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
      if (PyObject_TypeCheck(item, g_framework_tensor_pytype)) {
        result.emplace_back(::pybind11::handle(item).cast<phi::DenseTensor>());
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
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
      if (PyObject_TypeCheck(item, g_framework_tensor_pytype)) {
        result.emplace_back(::pybind11::handle(item).cast<phi::DenseTensor>());
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "argument (position %d) must be "
            "list of LoDTensor, but got %s at pos %d",
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
  } else if (PyObject_TypeCheck(obj, g_framework_lodtensorarray_pytype)) {
    for (auto& tensor :
         (::pybind11::handle(obj).cast<framework::LoDTensorArray>())) {
      result.emplace_back(tensor);
    }
  } else if (obj == Py_None) {
    return {};
  } else if (PyObject_TypeCheck(obj, g_framework_tensor_pytype)) {
    return {::pybind11::handle(obj).cast<phi::DenseTensor>()};
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be "
        "list or tuple, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return result;
}

using phi::distributed::Partial;
using phi::distributed::Placement;
using phi::distributed::Placements;
using phi::distributed::Replicate;
using phi::distributed::Shard;
Placements CastPyArg2VectorOfPlacement(PyObject* obj, ssize_t arg_pos) {
  Placements result;
  auto check_and_emplace = [&](PyObject* item, ssize_t i) {
    if (PyObject_TypeCheck(item, g_placement_shard_pytype)) {
      result.emplace_back(
          std::make_shared<Shard>(::pybind11::handle(item).cast<Shard>()));
    } else if (PyObject_TypeCheck(item, g_placement_replicated_pytype)) {
      result.emplace_back(std::make_shared<Replicate>(
          ::pybind11::handle(item).cast<Replicate>()));
    } else if (PyObject_TypeCheck(item, g_placement_partial_pytype)) {
      result.emplace_back(
          std::make_shared<Partial>(::pybind11::handle(item).cast<Partial>()));
    } else {
      PADDLE_THROW(platform::errors::InvalidType(
          "argument (position %d) must be list of Placement, but got %s at pos "
          "%d",
          arg_pos + 1,
          reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
          i));
    }
  };

  if (PyList_Check(obj) || PyTuple_Check(obj)) {
    Py_ssize_t len = PyObject_Size(obj);
    for (Py_ssize_t i = 0; i < len; i++) {
      PyObject* item =
          PyList_Check(obj) ? PyList_GetItem(obj, i) : PyTuple_GetItem(obj, i);
      check_and_emplace(item, i);
    }
  } else if (obj == Py_None) {
    return {};
  } else {
    check_and_emplace(obj, 0);
  }
  return result;
}

paddle::framework::proto::VarType::Type CastPyArg2ProtoType(PyObject* obj,
                                                            ssize_t arg_pos) {
  paddle::framework::proto::VarType::Type dtype;
  if (PyObject_TypeCheck(obj, g_vartype_pytype)) {
    dtype =
        ::pybind11::handle(obj).cast<paddle::framework::proto::VarType::Type>();
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be "
        "one of core.VarDesc.VarType, "
        "but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return dtype;
}

paddle::DataType CastPyArg2DataTypeDirectly(PyObject* obj,
                                            const std::string& op_type,
                                            ssize_t arg_pos) {
  if (obj == Py_None) {
    return phi::DataType::UNDEFINED;
  }

  paddle::DataType dtype;
  if (PyObject_TypeCheck(obj, g_data_type_pytype)) {
    dtype = ::pybind11::handle(obj).cast<paddle::DataType>();
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s: argument (position %d) must be "
        "one of paddle::DataType, "
        "but got %s",
        op_type,
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
  return dtype;
}

paddle::framework::Vocab CastPyArg2Vocab(PyObject* obj, ssize_t arg_pos) {
  if (PyDict_Check(obj)) {
    paddle::framework::Vocab vocab;
    vocab = ::pybind11::handle(obj)
                .cast<std::unordered_map<std::wstring, std::int32_t>>();
    return vocab;
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be dict, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
}

std::vector<std::string> CastPyArg2VectorOfString(PyObject* obj,
                                                  ssize_t arg_pos) {
  if (PyList_Check(obj)) {
    return ::pybind11::handle(obj).cast<std::vector<std::string>>();
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "argument (position %d) must be list, but got %s",
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

PyObject* ToPyObject(float value) { return PyFloat_FromDouble(value); }

PyObject* ToPyObject(double value) { return PyFloat_FromDouble(value); }

PyObject* ToPyObject(const char* value) { return PyUnicode_FromString(value); }

PyObject* ToPyObject(const std::string& value) {
  return PyUnicode_FromString(value.c_str());
}

PyObject* ToPyObject(const paddle::Tensor& value,
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

PyObject* ToPyObject(const std::vector<paddle::Tensor>& value,
                     bool return_py_none_if_not_initialize) {
// NOTE(liuyuanle): I encountered a bug(access violation) in windows. ref to
// https://stackoverflow.com/questions/55598839/how-to-fix-access-violation-error-when-returning-pyobject-from-c-function-usin
#ifdef _WIN32
  PyGILState_STATE gstate = PyGILState_Ensure();
#endif
  PyObject* result = PyList_New((Py_ssize_t)value.size());
#ifdef _WIN32
  PyGILState_Release(gstate);
#endif

  for (size_t i = 0; i < value.size(); i++) {
    if (!value[i].initialized() && return_py_none_if_not_initialize) {
      Py_INCREF(Py_None);
      PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), Py_None);
    } else {
      PyObject* obj = p_tensor_type->tp_alloc(p_tensor_type, 0);
      if (obj) {
        auto v = reinterpret_cast<TensorObject*>(obj);
        new (&(v->tensor)) paddle::Tensor();
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

PyObject* ToPyObject(const std::vector<std::vector<paddle::Tensor>>& value,
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

PyObject* ToPyObject(const pir::Value& value) {
  auto obj = ::pybind11::cast(value);
  obj.inc_ref();
  return obj.ptr();
}

PyObject* ToPyObject(const std::vector<pir::Value>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const phi::distributed::DistTensor* value) {
#ifdef PADDLE_WITH_DISTRIBUTE
  auto obj = ::pybind11::cast(value, py::return_value_policy::reference);
  obj.inc_ref();
  return obj.ptr();
#else
  PADDLE_THROW(platform::errors::Unavailable(
      "DistTensor to PyObject is not supported in the current "
      "PaddlePaddle, please recompile and installPaddlePaddle with the option "
      "of `WITH_DISTRIBUTE=ON`."));
#endif
}

PyObject* ToPyObject(const phi::distributed::TensorDistAttr* value) {
#ifdef PADDLE_WITH_DISTRIBUTE
  auto obj = ::pybind11::cast(value, py::return_value_policy::reference);
  obj.inc_ref();
  return obj.ptr();
#else
  PADDLE_THROW(platform::errors::Unavailable(
      "TensorDistAttr to PyObject is not supported in the current "
      "PaddlePaddle, please recompile and installPaddlePaddle with the option "
      "of `WITH_DISTRIBUTE=ON`."));
#endif
}

PyObject* ToPyObject(const phi::distributed::ProcessMesh* value) {
#ifdef PADDLE_WITH_DISTRIBUTE
  auto obj = ::pybind11::cast(value, py::return_value_policy::reference);
  obj.inc_ref();
  return obj.ptr();
#else
  PADDLE_THROW(platform::errors::Unavailable(
      "ProcessMesh to PyObject is not supported in the current "
      "PaddlePaddle, please recompile and installPaddlePaddle with the option "
      "of `WITH_DISTRIBUTE=ON`."));
#endif
}

PyObject* ToPyObject(const phi::distributed::Placement& value) {
  auto obj = ::pybind11::cast(value, py::return_value_policy::reference);
  obj.inc_ref();
  return obj.ptr();
}

PyObject* ToPyObject(const phi::distributed::Placements& values) {
#ifdef PADDLE_WITH_DISTRIBUTE
  PyObject* result = PyList_New((Py_ssize_t)values.size());

  for (size_t i = 0; i < values.size(); i++) {
    auto& value = values[i];
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(*value));
  }

  return result;
#else
  PADDLE_THROW(platform::errors::Unavailable(
      "Placements to PyObject is not supported in the current "
      "PaddlePaddle, please recompile and install PaddlePaddle with the option "
      "of `WITH_DISTRIBUTE=ON`."));
#endif
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

PyObject* ToPyObject(const std::unordered_map<int, int>& value) {
  PyObject* dict = PyDict_New();
  for (const auto& map_iter : value) {
    // Convert Key
    PyObject* key = ToPyObject(map_iter.first);
    // Convert Value
    PyObject* value = ToPyObject(map_iter.second);

    if (!key || !value) {
      PADDLE_THROW(
          platform::errors::Fatal("Unable to convert int to PyObject"));
    }

    if (PyDict_SetItem(dict, key, value) != 0) {
      PADDLE_THROW(
          platform::errors::Fatal("Unable to set key:value for py_dict"));
    }
  }
  return dict;
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

PyObject* ToPyObject(const paddle::framework::Vocab& value) {
  PyObject* dict = PyDict_New();
  for (const auto& map_iter : value) {
    // Convert Key
    PyObject* key_string = PyUnicode_FromWideChar(
        map_iter.first.c_str(), map_iter.first.size());  // NOLINT
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
paddle::optional<paddle::Tensor> GetOptionalTensorFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable,
    const phi::distributed::ProcessMesh* mesh) {
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

  if (PyObject_TypeCheck(obj, p_tensor_type)) {
    if (mesh) {
      ConvertToDistTensor(&(reinterpret_cast<TensorObject*>(obj)->tensor),
                          mesh);
    }
    return paddle::make_optional<paddle::Tensor>(
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

PyObject* ToPyObject(std::shared_ptr<egr::GradNodeBase> grad_node) {
  py::object py_obj = py::cast(grad_node, py::return_value_policy::reference);
  PyObject* py_grad_node = py_obj.release().ptr();
  Py_INCREF(py_grad_node);
  return py_grad_node;
}

static paddle::Tensor& GetTensorFromPyObject(const std::string& op_type,
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
    static paddle::Tensor emptytensor;
    return emptytensor;
  }

  if (PyObject_TypeCheck(obj, p_tensor_type) ||
      PyObject_TypeCheck(obj, p_string_tensor_type)) {
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
paddle::Tensor& GetTensorFromArgs(const std::string& op_type,
                                  const std::string& arg_name,
                                  PyObject* args,
                                  ssize_t arg_idx,
                                  bool dispensable) {
  PyObject* obj = PyTuple_GET_ITEM(args, arg_idx);
  return GetTensorFromPyObject(op_type, arg_name, obj, arg_idx, dispensable);
}

std::vector<paddle::Tensor> GetTensorListFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable,
    const phi::distributed::ProcessMesh* mesh) {
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

  std::vector<paddle::Tensor> result;
  const phi::distributed::ProcessMesh* local_mesh = nullptr;
  int mesh_start_index = -1;

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
      paddle::Tensor& tensor =
          reinterpret_cast<TensorObject*>(PyList_GetItem(list, i))->tensor;
      if (local_mesh) {
        ConvertToDistTensor(&tensor, local_mesh);
      } else {
        if (tensor.defined() && tensor.is_dist_tensor()) {
          local_mesh =
              &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                    tensor.impl())
                    ->process_mesh());
          mesh_start_index = i;
        }
      }
      result.emplace_back(tensor);
    }
    for (Py_ssize_t i = 0; i < mesh_start_index; i++) {
      paddle::Tensor& tensor =
          reinterpret_cast<TensorObject*>(PyList_GetItem(list, i))->tensor;
      ConvertToDistTensor(&tensor, local_mesh);
      result[i] = tensor;
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
      paddle::Tensor& tensor =
          reinterpret_cast<TensorObject*>(PyTuple_GetItem(list, i))->tensor;
      if (local_mesh) {
        ConvertToDistTensor(&tensor, local_mesh);
      } else {
        if (tensor.defined() && tensor.is_dist_tensor()) {
          local_mesh =
              &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                    tensor.impl())
                    ->process_mesh());
          mesh_start_index = i;
        }
      }
      result.emplace_back(tensor);
    }
    for (Py_ssize_t i = 0; i < mesh_start_index; i++) {
      paddle::Tensor& tensor =
          reinterpret_cast<TensorObject*>(PyTuple_GetItem(list, i))->tensor;
      ConvertToDistTensor(&tensor, local_mesh);
      result[i] = tensor;
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

paddle::optional<std::vector<paddle::Tensor>> GetOptionalTensorListFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable,
    const phi::distributed::ProcessMesh* mesh) {
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

  std::vector<paddle::Tensor> result;
  const phi::distributed::ProcessMesh* local_mesh = nullptr;
  int mesh_start_index = -1;

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
      paddle::Tensor& tensor =
          reinterpret_cast<TensorObject*>(PyList_GetItem(list, i))->tensor;
      if (local_mesh) {
        ConvertToDistTensor(&tensor, local_mesh);
      } else {
        if (tensor.defined() && tensor.is_dist_tensor()) {
          local_mesh =
              &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                    tensor.impl())
                    ->process_mesh());
          mesh_start_index = i;
        }
      }
      result.emplace_back(tensor);
    }
    for (Py_ssize_t i = 0; i < mesh_start_index; i++) {
      paddle::Tensor& tensor =
          reinterpret_cast<TensorObject*>(PyList_GetItem(list, i))->tensor;
      ConvertToDistTensor(&tensor, local_mesh);
      result[i] = tensor;
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
      paddle::Tensor& tensor =
          reinterpret_cast<TensorObject*>(PyTuple_GetItem(list, i))->tensor;
      if (local_mesh) {
        ConvertToDistTensor(&tensor, local_mesh);
      } else {
        if (tensor.defined() && tensor.is_dist_tensor()) {
          local_mesh =
              &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                    tensor.impl())
                    ->process_mesh());
          mesh_start_index = i;
        }
      }
      result.emplace_back(tensor);
    }
    for (Py_ssize_t i = 0; i < mesh_start_index; i++) {
      paddle::Tensor& tensor =
          reinterpret_cast<TensorObject*>(PyTuple_GetItem(list, i))->tensor;
      ConvertToDistTensor(&tensor, local_mesh);
      result[i] = tensor;
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

paddle::Tensor* GetTensorPtrFromArgs(const std::string& op_type,
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
    static paddle::Tensor emptytensor;
    return &emptytensor;
  }

  if (PyObject_TypeCheck(obj, p_tensor_type)) {
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

std::vector<paddle::Tensor*> GetTensorPtrListFromArgs(
    const std::string& op_type,
    const std::string& arg_name,
    PyObject* args,
    ssize_t arg_idx,
    bool dispensable,
    const phi::distributed::ProcessMesh* mesh) {
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

  std::vector<paddle::Tensor*> result;
  const phi::distributed::ProcessMesh* local_mesh = mesh;
  int mesh_start_index = -1;

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
      paddle::Tensor* tensor =
          &(reinterpret_cast<TensorObject*>(PyList_GetItem(list, i))->tensor);
      if (local_mesh) {
        ConvertToDistTensor(tensor, local_mesh);
      } else {
        if (tensor->defined() && tensor->is_dist_tensor()) {
          local_mesh =
              &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                    tensor->impl())
                    ->process_mesh());
          mesh_start_index = i;
        }
      }
      result.emplace_back(tensor);
    }
    for (Py_ssize_t i = 0; i < mesh_start_index; i++) {
      paddle::Tensor* tensor =
          &(reinterpret_cast<TensorObject*>(PyList_GetItem(list, i))->tensor);
      ConvertToDistTensor(tensor, local_mesh);
      result[i] = tensor;
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
      paddle::Tensor* tensor =
          &(reinterpret_cast<TensorObject*>(PyTuple_GetItem(list, i))->tensor);
      if (local_mesh) {
        ConvertToDistTensor(tensor, local_mesh);
      } else {
        if (tensor->defined() && tensor->is_dist_tensor()) {
          local_mesh =
              &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                    tensor->impl())
                    ->process_mesh());
          mesh_start_index = i;
        }
      }
      result.emplace_back(tensor);
    }
    for (Py_ssize_t i = 0; i < mesh_start_index; i++) {
      paddle::Tensor* tensor =
          &(reinterpret_cast<TensorObject*>(PyTuple_GetItem(list, i))->tensor);
      ConvertToDistTensor(tensor, local_mesh);
      result[i] = tensor;
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

std::vector<paddle::Tensor*> GetTensorPtrListFromPyObject(PyObject* obj) {
  std::vector<paddle::Tensor*> result;
  const phi::distributed::ProcessMesh* local_mesh = nullptr;
  int mesh_start_index = -1;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    if (len == 0) {
      PADDLE_THROW(
          platform::errors::InvalidArgument("The list of Tensor is empty."));
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      paddle::Tensor* tensor =
          &(reinterpret_cast<TensorObject*>(PyList_GetItem(obj, i))->tensor);
      if (local_mesh) {
        ConvertToDistTensor(tensor, local_mesh);
      } else {
        if (tensor->defined() && tensor->is_dist_tensor()) {
          local_mesh =
              &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                    tensor->impl())
                    ->process_mesh());
          mesh_start_index = i;
        }
      }
      result.emplace_back(tensor);
    }
    for (Py_ssize_t i = 0; i < mesh_start_index; i++) {
      paddle::Tensor* tensor =
          &(reinterpret_cast<TensorObject*>(PyList_GetItem(obj, i))->tensor);
      ConvertToDistTensor(tensor, local_mesh);
      result[i] = tensor;
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    if (len == 0) {
      PADDLE_THROW(
          platform::errors::InvalidArgument("The tuple of Tensor is empty."));
    }
    for (Py_ssize_t i = 0; i < len; i++) {
      paddle::Tensor* tensor =
          &(reinterpret_cast<TensorObject*>(PyTuple_GetItem(obj, i))->tensor);
      if (local_mesh) {
        ConvertToDistTensor(tensor, local_mesh);
      } else {
        if (tensor->defined() && tensor->is_dist_tensor()) {
          local_mesh =
              &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                    tensor->impl())
                    ->process_mesh());
          mesh_start_index = i;
        }
      }
      result.emplace_back(tensor);
    }
    for (Py_ssize_t i = 0; i < mesh_start_index; i++) {
      paddle::Tensor* tensor =
          &(reinterpret_cast<TensorObject*>(PyTuple_GetItem(obj, i))->tensor);
      ConvertToDistTensor(tensor, local_mesh);
      result[i] = tensor;
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The PyObject must be list of Tensors, but got "
        "%s",
        (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name));
  }

  return result;
}

std::vector<paddle::Tensor> GetTensorListFromPyObject(PyObject* obj,
                                                      bool allow_none) {
  std::vector<paddle::Tensor> result;
  const phi::distributed::ProcessMesh* local_mesh = nullptr;
  int mesh_start_index = -1;

  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_TypeCheck(item, p_tensor_type)) {
        paddle::Tensor& tensor = reinterpret_cast<TensorObject*>(item)->tensor;
        if (local_mesh) {
          ConvertToDistTensor(&tensor, local_mesh);
        } else {
          if (tensor.defined() && tensor.is_dist_tensor()) {
            local_mesh =
                &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                      tensor.impl())
                      ->process_mesh());
            mesh_start_index = i;
          }
        }
        result.emplace_back(tensor);
      } else if (allow_none && (item == Py_None)) {
        VLOG(4) << "Got None in Tensor list: " << i;
        result.emplace_back();
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument must be "
            "list of Tensor, but got %s at pos %d",
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
    for (Py_ssize_t i = 0; i < mesh_start_index; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_TypeCheck(item, p_tensor_type)) {
        paddle::Tensor& tensor = reinterpret_cast<TensorObject*>(item)->tensor;
        ConvertToDistTensor(&tensor, local_mesh);
        result.emplace_back(tensor);
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_TypeCheck(item, p_tensor_type)) {
        paddle::Tensor& tensor = reinterpret_cast<TensorObject*>(item)->tensor;
        if (local_mesh) {
          ConvertToDistTensor(&tensor, local_mesh);
        } else {
          if (tensor.defined() && tensor.is_dist_tensor()) {
            local_mesh =
                &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                      tensor.impl())
                      ->process_mesh());
            mesh_start_index = i;
          }
        }
        result.emplace_back(tensor);
      } else if (allow_none && (item == Py_None)) {
        VLOG(4) << "Got None in Tensor list: " << i;
        result.emplace_back();
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "argument must be "
            "list of Tensor, but got %s at pos %d",
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name,
            i));
      }
    }
    for (Py_ssize_t i = 0; i < mesh_start_index; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_TypeCheck(item, p_tensor_type)) {
        paddle::Tensor& tensor = reinterpret_cast<TensorObject*>(item)->tensor;
        ConvertToDistTensor(&tensor, local_mesh);
        result.emplace_back(tensor);
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

paddle::Tensor& UnSafeGetTensorFromPyObject(PyObject* obj) {
  return reinterpret_cast<TensorObject*>(obj)->tensor;
}

paddle::Tensor CreateTensorFromVarDesc(
    const paddle::framework::VarDesc& var_desc) {
  auto tensor = paddle::Tensor();

  auto dtype = var_desc.GetDataType();
  std::vector<int64_t> dims = var_desc.GetShape();

  auto var_type = var_desc.GetType();

  auto ddims = common::make_ddim(dims);
  tensor.set_name(var_desc.Name());
  auto autograd_meta = egr::EagerUtils::autograd_meta(&tensor);
  autograd_meta->SetPersistable(false);
  autograd_meta->SetStopGradient(var_desc.StopGradient());

  if (var_type == paddle::framework::proto::VarType::LOD_TENSOR) {
    // TODO(jiabin): Maybe support LOD later
    std::shared_ptr<phi::DenseTensor> dense_tensor = nullptr;
    if (dims.size() == 1 && dims[0] == 0) {
      std::shared_ptr<phi::Allocation> allocation_ptr = nullptr;
      dense_tensor = std::make_shared<phi::DenseTensor>(
          allocation_ptr,
          phi::DenseTensorMeta(paddle::framework::TransToPhiDataType(dtype),
                               ddims));
    } else {
      // TODO(dev): we need enhance check for ddims.
      dense_tensor = std::make_shared<phi::DenseTensor>(
          std::make_shared<phi::Allocation>(),
          phi::DenseTensorMeta(paddle::framework::TransToPhiDataType(dtype),
                               ddims));
    }
    tensor.set_impl(dense_tensor);
  } else if (var_type == paddle::framework::proto::VarType::SELECTED_ROWS) {
    std::shared_ptr<phi::SelectedRows> selected_rows_tensor =
        std::make_shared<phi::SelectedRows>();
    tensor.set_impl(selected_rows_tensor);
  }

  if (!autograd_meta->GetMutableGradNode()) {
    autograd_meta->SetGradNode(
        std::make_shared<egr::GradNodeAccumulation>(autograd_meta));
  }

  return tensor;
}

PyObject* GetEmptyTensorsWithVarDesc(PyObject* self, PyObject* args) {
  std::vector<paddle::Tensor> result;
  std::unordered_map<std::string, paddle::Tensor> out_tensor_map;

  auto var_desc_list = PyTuple_GetItem(args, 0);

  if (PyList_Check(var_desc_list)) {
    Py_ssize_t len = PyList_Size(var_desc_list);
    for (Py_ssize_t i = 0; i < len; i++) {
      auto var_desc = PyObjectCast<paddle::framework::VarDesc>(
          PyList_GetItem(var_desc_list, i));
      auto var_name = var_desc.Name();
      if (out_tensor_map.find(var_name) == out_tensor_map.end()) {
        paddle::Tensor tensor = CreateTensorFromVarDesc(var_desc);
        out_tensor_map[var_name] = tensor;
        result.emplace_back(tensor);
      } else {
        result.emplace_back(out_tensor_map[var_name]);
      }
    }
  } else if (PyTuple_Check(var_desc_list)) {
    Py_ssize_t len = PyTuple_Size(var_desc_list);
    for (Py_ssize_t i = 0; i < len; i++) {
      auto var_desc = PyObjectCast<paddle::framework::VarDesc>(
          PyTuple_GetItem(var_desc_list, i));
      auto var_name = var_desc.Name();
      if (out_tensor_map.find(var_name) == out_tensor_map.end()) {
        paddle::Tensor tensor = CreateTensorFromVarDesc(var_desc);
        out_tensor_map[var_name] = tensor;
        result.emplace_back(tensor);
      } else {
        result.emplace_back(out_tensor_map[var_name]);
      }
    }
  } else if (var_desc_list != Py_None) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Argument of CreateTensorsWithVarDesc must be list of VarDesc, but got "
        "%s",
        (reinterpret_cast<PyTypeObject*>(var_desc_list->ob_type))->tp_name));
  }
  return ToPyObject(result);
}

paddle::Tensor CreateTensorFromValue(const pir::Value& value) {
  auto tensor = paddle::Tensor();

  auto dims = phi::vectorize(GetValueDims(value));
  auto ddims = phi::make_ddim(dims);
  auto autograd_meta = egr::EagerUtils::autograd_meta(&tensor);
  autograd_meta->SetPersistable(false);
  autograd_meta->SetStopGradient(GetValueBoolAttr(value, kAttrStopGradients));

  if (value.type().isa<paddle::dialect::DenseTensorType>()) {
    // TODO(jiabin): Maybe support LOD later
    std::shared_ptr<phi::DenseTensor> dense_tensor = nullptr;
    auto dtype = paddle::dialect::TransToPhiDataType(
        value.type().dyn_cast<paddle::dialect::DenseTensorType>().dtype());

    if (dims.size() == 1 && dims[0] == 0) {
      std::shared_ptr<phi::Allocation> allocation_ptr = nullptr;
      dense_tensor = std::make_shared<phi::DenseTensor>(
          allocation_ptr, phi::DenseTensorMeta(dtype, ddims));
    } else {
      // TODO(dev): we need enhance check for ddims.
      dense_tensor = std::make_shared<phi::DenseTensor>(
          std::make_shared<phi::Allocation>(),
          phi::DenseTensorMeta(dtype, ddims));
    }
    tensor.set_impl(dense_tensor);
  } else if (value.type().isa<paddle::dialect::SelectedRowsType>()) {
    std::shared_ptr<phi::SelectedRows> selected_rows_tensor =
        std::make_shared<phi::SelectedRows>();
    tensor.set_impl(selected_rows_tensor);
  }

  if (!autograd_meta->GetMutableGradNode()) {
    autograd_meta->SetGradNode(
        std::make_shared<egr::GradNodeAccumulation>(autograd_meta));
  }

  return tensor;
}

PyObject* GetEmpytyTensorsWithValue(PyObject* self, PyObject* args) {
  std::vector<paddle::Tensor> result;
  std::unordered_map<pir::Value, paddle::Tensor> out_tensor_map;

  auto value_list = PyTuple_GetItem(args, 0);

  if (PyList_Check(value_list)) {
    Py_ssize_t len = PyList_Size(value_list);
    for (Py_ssize_t i = 0; i < len; i++) {
      auto value = PyObjectCast<pir::Value>(PyList_GetItem(value_list, i));
      if (out_tensor_map.find(value) == out_tensor_map.end()) {
        paddle::Tensor tensor = CreateTensorFromValue(value);
        out_tensor_map[value] = tensor;
        result.emplace_back(tensor);
      } else {
        result.emplace_back(out_tensor_map[value]);
      }
    }
  } else if (PyTuple_Check(value_list)) {
    Py_ssize_t len = PyTuple_Size(value_list);
    for (Py_ssize_t i = 0; i < len; i++) {
      auto value = PyObjectCast<pir::Value>(PyTuple_GetItem(value_list, i));
      if (out_tensor_map.find(value) == out_tensor_map.end()) {
        paddle::Tensor tensor = CreateTensorFromValue(value);
        out_tensor_map[value] = tensor;
        result.emplace_back(tensor);
      } else {
        result.emplace_back(out_tensor_map[value]);
      }
    }
  } else if (value_list != Py_None) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Argument of GetTensorsWithValueInArgs must be list of Value, "
        "but got "
        "%s",
        (reinterpret_cast<PyTypeObject*>(value_list->ob_type))->tp_name));
  }

  return ToPyObject(result);
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
          "%s(): argument (position %d) is numpy.ndarray, the inner elements "
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
  } else if (type_name == "numpy.float16") {
    float16 value = CastPyArg2Float16(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (type_name == "numpy.int64") {
    int64_t value = CastPyArg2Long(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (type_name == "numpy.int32" || type_name == "numpy.intc") {
    int value = CastPyArg2Int(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (type_name == "numpy.complex64") {
    phi::dtype::complex<float> value = CastPyArg2Complex(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (type_name == "numpy.complex128") {
    phi::dtype::complex<double> value =
        CastPyArg2Complex128(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "%s(): argument (position %d) must be "
        "numpy.float32/float64, numpy.int32/int64, numpy.complex64/complex128, "
        "but got %s",
        op_type,
        arg_pos + 1,
        type_name));  // NOLINT
  }
}

PyObject* CastPyArg2ValuePreHook(PyObject* obj) {
  PyObject* hook = static_op_arg_pre_cast_hook_get();
  if (hook == Py_None) {
    return obj;
  }
  Py_INCREF(obj);
  PyObject* result = PyObject_CallFunction(hook, "O", obj);
  PADDLE_ENFORCE(result,
                 paddle::platform::errors::Fatal(
                     "Call static_op_arg_pre_cast_hook failed."));
  Py_DECREF(obj);
  return result;
}

pir::Value CastPyArg2Value(PyObject* obj,
                           const std::string& op_type,
                           size_t arg_pos) {
  obj = CastPyArg2ValuePreHook(obj);
  if (PyObject_TypeCheck(obj, g_ir_value_pytype)) {
    return ::pybind11::handle(obj).cast<pir::Value>();
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "Value, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
}

paddle::optional<pir::Value> CastPyArg2OptionalValue(PyObject* obj,
                                                     const std::string& op_type,
                                                     size_t arg_pos) {
  if (obj == nullptr || obj == Py_None) {
    return paddle::none;
  }
  return paddle::make_optional<pir::Value>(
      CastPyArg2Value(obj, op_type, arg_pos));
}

std::vector<pir::Value> CastPyArg2VectorOfValue(PyObject* obj,
                                                const std::string& op_type,
                                                size_t arg_pos) {
  std::vector<pir::Value> value_list;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      item = CastPyArg2ValuePreHook(item);
      if (PyObject_TypeCheck(item, g_ir_value_pytype)) {
        value_list.emplace_back(::pybind11::handle(item).cast<pir::Value>());
      } else if (item == Py_None) {
        continue;
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "vector<Value>, but got vector<%s>",
            op_type,
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)
                ->tp_name));  // NOLINT
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      item = CastPyArg2ValuePreHook(item);
      if (PyObject_TypeCheck(item, g_ir_value_pytype)) {
        value_list.emplace_back(::pybind11::handle(item).cast<pir::Value>());
      } else if (item == Py_None) {
        continue;
      } else {
        PADDLE_THROW(platform::errors::InvalidType(
            "%s(): argument (position %d) must be "
            "vector<Value>, but got vector<%s>",
            op_type,
            arg_pos + 1,
            reinterpret_cast<PyTypeObject*>(item->ob_type)
                ->tp_name));  // NOLINT
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "Vector<>, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
  return value_list;
}

paddle::optional<std::vector<pir::Value>> CastPyArg2OptionalVectorOfValue(
    PyObject* obj, const std::string& op_type, size_t arg_pos) {
  if (obj == nullptr || obj == Py_None) {
    return paddle::none;
  }
  return paddle::make_optional<std::vector<pir::Value>>(
      CastPyArg2VectorOfValue(obj, op_type, arg_pos));
}

paddle::experimental::Scalar CastPyArg2Scalar(PyObject* obj,
                                              const std::string& op_type,
                                              ssize_t arg_pos) {
  if (obj == Py_None) {
    PADDLE_THROW(platform::errors::InvalidType(
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
  } else if (PyCheckTensor(obj)) {
    paddle::Tensor& value = GetTensorFromPyObject(
        op_type, "" /*arg_name*/, obj, arg_pos, false /*dispensable*/);
    return paddle::experimental::Scalar(value);
  } else if (type_name.find("numpy") != std::string::npos) {
    return CastNumpy2Scalar(obj, op_type, arg_pos);
  } else if (PyComplex_Check(obj)) {
    auto value = CastPyArg2Complex128(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (PyObject_CheckLongOrToLong(&obj)) {
    int value = CastPyArg2Int(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else if (PyObject_CheckString(obj)) {
    std::string value = CastPyArg2String(obj, op_type, arg_pos);
    return paddle::experimental::Scalar(value);
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
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
    PADDLE_THROW(platform::errors::InvalidType(
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
    } else if (PyObject_CheckComplexOrToComplex(&item)) {
      std::vector<phi::Scalar> value;
      for (Py_ssize_t i = 0; i < len; i++) {
        item = PyList_GetItem(obj, i);
        Py_complex v = PyComplex_AsCComplex(item);
        value.emplace_back(phi::Scalar{std::complex<double>(v.real, v.imag)});
      }
      return value;
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "a list of int, float, complex, or bool, but got %s",
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
  if (obj == Py_None) {
    return paddle::experimental::IntArray({});
  }

  // obj could be: int, float, bool, paddle.Tensor
  PyTypeObject* type = obj->ob_type;
  auto type_name = std::string(type->tp_name);
  if (type_name == "list" || type_name == "tuple" ||
      type_name == "numpy.ndarray") {
    std::vector<int64_t> value = CastPyArg2Longs(obj, op_type, arg_pos);
    return paddle::experimental::IntArray(value);
  } else if (type_name == "paddle.Tensor" || type_name == "Tensor") {
    paddle::Tensor& value = GetTensorFromPyObject(
        op_type, "" /*arg_name*/, obj, arg_pos, false /*dispensable*/);
    return paddle::experimental::IntArray(value);
  } else if (PyObject_CheckLongOrConvertToLong(&obj)) {
    return paddle::experimental::IntArray(
        {static_cast<int64_t>(PyLong_AsLong(obj))});
  } else {
    PADDLE_THROW(platform::errors::InvalidType(
        "%s(): argument (position %d) must be "
        "list or int, but got %s",
        op_type,
        arg_pos + 1,
        ((PyTypeObject*)obj->ob_type)->tp_name));  // NOLINT
  }
  // Fake a IntArray
  return paddle::experimental::IntArray({1});
}

paddle::framework::Scope* CastPyArg2ScopePtr(PyObject* obj) {
  if (PyObject_TypeCheck(obj, g_framework_scope_pytype)) {
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
    return phi::DataType::UNDEFINED;
  }

  framework::proto::VarType::Type type = CastPyArg2ProtoType(obj, arg_pos);
  return framework::TransToPhiDataType(type);
}

paddle::Tensor PyTensorHook::operator()(const paddle::Tensor& var) {
  py::gil_scoped_acquire gil;
  VLOG(3) << "Call PyTensorHook for var " << var.name();

  PyObject* res = nullptr;
  try {
    bool return_py_none_if_not_initialize = true;
    if (var.defined() && !var.initialized()) {
      return_py_none_if_not_initialize = !var.is_dist_tensor();
    }
    PyObject* p_tmp_var = ToPyObject(var, return_py_none_if_not_initialize);
    res = PyObject_CallFunctionObjArgs(py_func_, p_tmp_var, nullptr);
    Py_DECREF(p_tmp_var);
  } catch (platform::EnforceNotMet& e) {
    throw e;
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
    throw e;
  } catch (std::exception& e) {
    PADDLE_THROW(platform::errors::Unavailable(
        "Hook function of Tensor raises an exception: %s.", e.what()));
  } catch (...) {
    PADDLE_THROW(platform::errors::Fatal(
        "Hook function of Tensor raises an unknown exception."));
  }
}

PyObjectHolder::PyObjectHolder(PyObject* ptr) { ptr_ = ptr; }

PyObjectHolder::~PyObjectHolder() {  // NOLINT
  ::pybind11::gil_scoped_acquire gil;
  Py_XDECREF(ptr_);
}

void* PyObjectHolder::get() { return reinterpret_cast<void*>(ptr_); }

void PyObjectHolder::reset(void* ptr) {
  if (ptr_) {
    ::pybind11::gil_scoped_acquire gil;
    Py_XDECREF(ptr_);
  }
  ptr_ = reinterpret_cast<PyObject*>(ptr);
}

void PyObjectHolder::inc_ref() {
  ::pybind11::gil_scoped_acquire gil;
  Py_XINCREF(ptr_);
}
void PyObjectHolder::dec_ref() {
  ::pybind11::gil_scoped_acquire gil;
  Py_XDECREF(ptr_);
}

PackHook::PackHook(PyObject* hook) : hook_(hook) { Py_INCREF(hook_); }

PackHook::~PackHook() {  // NOLINT
  ::pybind11::gil_scoped_acquire gil;
  Py_DECREF(hook_);
}

std::shared_ptr<egr::PyObjectHolderBase> PackHook::operator()(
    const paddle::Tensor& tensor) {
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  ::pybind11::gil_scoped_acquire gil;
  auto args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, paddle::pybind::ToPyObject(tensor));
  PyObject* ret = PyObject_Call(hook_, args, nullptr);
  PADDLE_ENFORCE_NOT_NULL(ret,
                          paddle::platform::errors::External(
                              pybind11::detail::error_string().c_str()));
  Py_XDECREF(args);
  egr::Controller::Instance().SetHasGrad(grad_tmp);
  return std::make_shared<PyObjectHolder>(ret);
}

void* PackHook::operator()(void* py_tensor) {
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  ::pybind11::gil_scoped_acquire gil;
  auto args = PyTuple_New(1);
  Py_INCREF(reinterpret_cast<PyObject*>(py_tensor));
  PyTuple_SET_ITEM(args, 0, reinterpret_cast<PyObject*>(py_tensor));
  PyObject* ret = PyObject_Call(hook_, args, nullptr);
  PADDLE_ENFORCE_NOT_NULL(ret,
                          paddle::platform::errors::External(
                              pybind11::detail::error_string().c_str()));
  Py_XDECREF(args);
  egr::Controller::Instance().SetHasGrad(grad_tmp);
  return reinterpret_cast<void*>(ret);
}

UnPackHook::UnPackHook(PyObject* hook) : hook_(hook) { Py_INCREF(hook_); }

UnPackHook::~UnPackHook() {  // NOLINT
  ::pybind11::gil_scoped_acquire gil;
  Py_DECREF(hook_);
}

paddle::Tensor UnPackHook::operator()(
    std::shared_ptr<egr::PyObjectHolderBase> packed_value) {
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  ::pybind11::gil_scoped_acquire gil;
  auto args = PyTuple_New(1);
  Py_INCREF(reinterpret_cast<PyObject*>(packed_value->get()));
  PyTuple_SET_ITEM(args, 0, reinterpret_cast<PyObject*>(packed_value->get()));
  PyObject* ret = PyObject_Call(hook_, args, nullptr);
  PADDLE_ENFORCE_NOT_NULL(ret,
                          paddle::platform::errors::External(
                              pybind11::detail::error_string().c_str()));
  Py_XDECREF(args);
  egr::Controller::Instance().SetHasGrad(grad_tmp);

  PADDLE_ENFORCE_EQ(paddle::pybind::PyCheckTensor(ret),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "paddle.autograd.saved_tensors_hooks only one pair "
                        "of hooks is allowed at a time."));

  auto tensor = reinterpret_cast<paddle::pybind::TensorObject*>(ret)->tensor;
  Py_XDECREF(ret);
  return tensor;
}

void* UnPackHook::operator()(void* packed_value, void* other) {
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  ::pybind11::gil_scoped_acquire gil;
  auto args = PyTuple_New(1);
  Py_INCREF(reinterpret_cast<PyObject*>(packed_value));
  PyTuple_SET_ITEM(args, 0, reinterpret_cast<PyObject*>(packed_value));
  PyObject* ret = PyObject_Call(hook_, args, nullptr);
  PADDLE_ENFORCE_NOT_NULL(ret,
                          paddle::platform::errors::External(
                              pybind11::detail::error_string().c_str()));
  Py_XDECREF(args);
  egr::Controller::Instance().SetHasGrad(grad_tmp);

  PADDLE_ENFORCE_EQ(paddle::pybind::PyCheckTensor(ret),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "paddle.autograd.saved_tensors_hooks only one pair "
                        "of hooks is allowed at a time."));

  return reinterpret_cast<void*>(ret);
}

/* ------------------ for SetStaticOpArgPreCastHook ----------------------- */

static Py_tss_t static_op_arg_pre_cast_hook_key = {0, 0};

inline static PyObject* static_op_arg_pre_cast_hook_get() {
  void* result = PyThread_tss_get(&static_op_arg_pre_cast_hook_key);
  if (result == NULL) {
    return Py_None;
  } else {
    return reinterpret_cast<PyObject*>(result);
  }
}

inline static void static_op_arg_pre_cast_hook_set(PyObject* obj) {
  PyThread_tss_set(&static_op_arg_pre_cast_hook_key, obj);
}

static PyObject* set_static_op_arg_pre_cast_hook(PyObject* new_callback,
                                                 PyThreadState* tstate) {
  PyObject* old_callback = static_op_arg_pre_cast_hook_get();
  Py_INCREF(new_callback);
  static_op_arg_pre_cast_hook_set(new_callback);

  return old_callback;
}

PyObject* SetStaticOpArgPreCastHook(PyObject* dummy, PyObject* callback) {
  if (callback != Py_None && !PyCallable_Check(callback)) {
    VLOG(7) << "callback is not a callable or none, invalid arguments.";
    Py_INCREF(Py_None);
    return Py_None;
  }
  return set_static_op_arg_pre_cast_hook(callback, PyThreadState_GET());
}

PyMODINIT_FUNC PyInit__static_op_arg_pre_cast_hook() {
  auto result = PyThread_tss_create(&static_op_arg_pre_cast_hook_key);
  VLOG(7) << "Set PyThread_tss_create return: " << result;

  Py_INCREF(Py_None);
  static_op_arg_pre_cast_hook_set(Py_None);
  return NULL;
}

/* ------------------ for auto parallel ----------------------- */

void DistTensorTypeParser::operator()(const Tensor& x) {
  if (x.defined() && x.is_dist_tensor()) {
    *mesh = &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(x.impl())
                  ->process_mesh());
    result = true;
  }
}

void DistTensorTypeParser::operator()(const paddle::optional<Tensor>& x) {
  if (x) {
    if (x.get_ptr()->defined() && x.get_ptr()->is_dist_tensor()) {
      *mesh = &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                    x.get_ptr()->impl())
                    ->process_mesh());
      result = true;
    }
  }
}

void DistTensorTypeParser::operator()(const std::vector<Tensor>& x) {
  if (!x.empty()) {
    for (auto& t : x) {
      if (t.defined() && t.is_dist_tensor()) {
        *mesh =
            &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(t.impl())
                  ->process_mesh());
        result = true;
        break;
      }
    }
  }
}

void DistTensorTypeParser::operator()(
    const paddle::optional<std::vector<Tensor>>& x) {
  if (x) {
    if (!(x.get_ptr()->empty())) {
      for (auto& t : *(x.get_ptr())) {
        if (t.defined() && t.is_dist_tensor()) {
          *mesh = &(
              std::dynamic_pointer_cast<phi::distributed::DistTensor>(t.impl())
                  ->process_mesh());
          result = true;
          break;
        }
      }
    }
  }
}

void DistTensorConverter::convert(Tensor* x) { ConvertToDistTensor(x, mesh); }

void DistTensorConverter::operator()(Tensor* x) {
  DistTensorConverter::convert(x);
}

void DistTensorConverter::operator()(paddle::optional<Tensor>* x) {
  if (*x) {
    DistTensorConverter::convert(x->get_ptr());
  }
}

void DistTensorConverter::operator()(std::vector<Tensor>* x) {
  if (!x->empty()) {
    for (auto& t : *x) {
      DistTensorConverter::convert(&t);
    }
  }
}

void DistTensorConverter::operator()(paddle::optional<std::vector<Tensor>>* x) {
  if (*x) {
    if (!(x->get_ptr()->empty())) {
      for (auto& t : *(x->get_ptr())) {
        if (!t.is_dist_tensor()) {
          DistTensorConverter::convert(&t);
        }
      }
    }
  }
}

static PyMethodDef EagerUtilMethods[] = {
    {"create_empty_tensors_with_var_descs",
     (PyCFunction)(void (*)(void))GetEmptyTensorsWithVarDesc,
     METH_VARARGS,
     "GetEmptyTensorsWithVarDesc"},
    {"create_empty_tensors_with_values",
     (PyCFunction)(void (*)(void))GetEmpytyTensorsWithValue,
     METH_VARARGS,
     "GetEmpytyTensorsWithValue."},
    {"set_static_op_arg_pre_cast_hook",
     (PyCFunction)SetStaticOpArgPreCastHook,
     METH_O,
     "Set hook for pre cast a static OP argument."},
    {nullptr, nullptr, 0, nullptr}};

void BindEagerUtils(PyObject* module) {
  PyInit__static_op_arg_pre_cast_hook();
  if (PyModule_AddFunctions(module, EagerUtilMethods) < 0) {
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle error in BindEagerUtils(PyModule_AddFunctions)."));
    return;
  }
}

}  // namespace pybind
}  // namespace paddle
