/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
// disable numpy compile error

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <Python.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "pybind11/detail/internals.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/memory/allocation/mmap_allocator.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace pybind {

extern PyTypeObject* p_tensor_type;

bool PyCheckTensor(PyObject* obj) {
  return PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type));
}

static bool PyCheckInteger(PyObject* obj) {
#if PY_VERSION_HEX < 0x03000000
  return (PyLong_Check(obj) || PyInt_Check(obj)) && !PyBool_Check(obj);
#else
  return PyLong_Check(obj) && !PyBool_Check(obj);
#endif
}

static bool IsNumpyType(PyObject* obj) {
  // It is not a good way to judge the type of obj by its type'name. Maybe using
  // `PyArray_IsScalar` will be better. However, this interface cannot be used
  // by including pybind11, and it needs to compile with numpy.
  auto type_name = std::string(Py_TYPE(obj)->tp_name);
  return type_name == "numpy.int64" || type_name == "numpy.longlong" ||
         type_name == "numpy.int32" || type_name == "numpy.int16";
}

std::set<phi::DataType> _supported_int_dtype_{DataType::UINT8,
                                              DataType::INT8,
                                              DataType::INT16,
                                              DataType::INT32,
                                              DataType::INT64,
                                              DataType::BOOL};
std::set<phi::DataType> _complex_dtypes{
    DataType::COMPLEX64,
    DataType::COMPLEX128,
};

// _supported_promote_complex_types_
//     '__add__',
//     '__radd__',
//     '__sub__',
//     '__rsub__',
//     '__mul__',
//     '__rmul__',
//     '__div__',
//     '__truediv__',
//     '__rdiv__',
//     '__rtruediv__',
//     '__matmul__',

void SetDevice(paddle::platform::Place place) {
  if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    phi::backends::gpu::SetDeviceId(place.device);
    VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
            << " from " << static_cast<int>(place.device);
#else
    PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
  }

  if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
    phi::DeviceManager::SetDevice(place);
    VLOG(1) << "CurrentDeviceId: "
            << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from "
            << static_cast<int>(place.device);
#else
    PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use "
        "CustomPlace."));
#endif
  }
}

// scalar func only support add, radd, sub, rsub, mul, rmul, div, truediv.
// this function will update gradually.
paddle::experimental::Tensor CallScalarFuction(
    const paddle::experimental::Tensor& self_tensor,
    float other,
    std::string op_type) {
  paddle::experimental::Tensor ret;
  if (op_type == "add" || op_type == "radd") {
    ret = scale_ad_func(self_tensor, phi::Scalar(1.0), other, true);
  } else if (op_type == "sub") {
    ret = scale_ad_func(self_tensor, phi::Scalar(1.0), -other, true);

  } else if (op_type == "rsub") {
    ret = scale_ad_func(self_tensor, phi::Scalar(-1.0), other, true);
  }

  return ret;
}

static PyObject* tensor__add__method(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__add__ or __radd_ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY
  VLOG(6) << "Running Eager tensor__add__method";
  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::experimental::Tensor ret;
  paddle::experimental::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    float other = 0.0;
    if (PyFloat_Check(other_obj)) {
      other = CastPyArg2AttrFloat(other_obj, 0);
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other = static_cast<float>(CastPyArg2AttrInt(other_obj, 0));
    }

    {
      eager_gil_scoped_release guard;
      ret = CallScalarFuction(self_tensor, other, "add");
    }
    return ToPyObject(ret);
  }

  // 2. create or get tensor for other_obj
  paddle::experimental::Tensor other_tensor;
  if (!PyCheckTensor(other_obj)) {
    paddle::experimental::Scalar value =
        CastPyArg2Scalar(other_obj, "__add__", 0);
    {
      eager_gil_scoped_release guard;
      other_tensor =
          full_ad_func(self_tensor.shape(), value, self_tensor.dtype(), place);
    }
  } else {
    other_tensor = CastPyArg2Tensor(other_obj, 0);
  }

  // 3. promote types or unify right var type to left var
  phi::DataType lhs_dtype = self_tensor.dtype();
  phi::DataType rhs_dtype = other_tensor.dtype();
  if (lhs_dtype != rhs_dtype) {
    // note: only op_type in _supported_promote_complex_types_ should promote
    // dtype
    if (_complex_dtypes.find(lhs_dtype) != _complex_dtypes.end() ||
        _complex_dtypes.find(rhs_dtype) != _complex_dtypes.end()) {
      phi::DataType promote_dtype =
          framework::TransToPhiDataType(framework::PromoteTypesIfComplexExists(
              framework::TransToProtoVarType(lhs_dtype),
              framework::TransToProtoVarType(rhs_dtype)));
      if (lhs_dtype != promote_dtype) {
        // cast
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, promote_dtype);
      }
      if (rhs_dtype != promote_dtype) {
        eager_gil_scoped_release guard;
        other_tensor = cast_ad_func(other_tensor, promote_dtype);
      }
    } else {
      LOG(WARNING)
          << "The dtype of left and right Tensor are not the same, left "
             "dtype is "
          << lhs_dtype << ", but right dtype is " << rhs_dtype
          << ", the right dtype will convert to " << lhs_dtype;
      eager_gil_scoped_release guard;
      other_tensor = cast_ad_func(other_tensor, lhs_dtype);
    }
  }

  // 4. calculation
  VLOG(6) << "Calling add_ad_func in tensor__add__method";

  {
    eager_gil_scoped_release guard;
    ret = add_ad_func(self_tensor, other_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__sub__method(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__sub__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY
  VLOG(6) << "Running Eager tensor__sub__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::experimental::Tensor ret;
  paddle::experimental::Tensor self_tensor = self->tensor;

  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);
  // 1. scalar exists cases
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    float other = 0.0;
    if (PyFloat_Check(other_obj)) {
      other = CastPyArg2AttrFloat(other_obj, 0);
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other = static_cast<float>(CastPyArg2AttrInt(other_obj, 0));
    }
    {
      eager_gil_scoped_release guard;
      ret = CallScalarFuction(self_tensor, other, "sub");
    }

    return ToPyObject(ret);
  }
  // 2. create or get tensor for other_obj
  paddle::experimental::Tensor other_tensor;
  if (!PyCheckTensor(other_obj)) {
    paddle::experimental::Scalar value =
        CastPyArg2Scalar(other_obj, "__sub__", 0);
    {
      eager_gil_scoped_release guard;
      other_tensor =
          full_ad_func(self_tensor.shape(), value, self_tensor.dtype(), place);
    }
  } else {
    other_tensor = CastPyArg2Tensor(other_obj, 0);
  }

  // 3. promote types or unify right var type to left var
  phi::DataType lhs_dtype = self_tensor.dtype();
  phi::DataType rhs_dtype = other_tensor.dtype();
  if (lhs_dtype != rhs_dtype) {
    if (_complex_dtypes.find(lhs_dtype) != _complex_dtypes.end() ||
        _complex_dtypes.find(rhs_dtype) != _complex_dtypes.end()) {
      phi::DataType promote_dtype =
          framework::TransToPhiDataType(framework::PromoteTypesIfComplexExists(
              framework::TransToProtoVarType(lhs_dtype),
              framework::TransToProtoVarType(rhs_dtype)));
      if (lhs_dtype != promote_dtype) {
        // cast
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, promote_dtype);
      }
      if (rhs_dtype != promote_dtype) {
        eager_gil_scoped_release guard;
        other_tensor = cast_ad_func(other_tensor, promote_dtype);
      }
    } else {
      LOG(WARNING)
          << "The dtype of left and right Tensor are not the same, left "
             "dtype is "
          << lhs_dtype << ", but right dtype is " << rhs_dtype
          << ", the right dtype will convert to " << lhs_dtype;
      eager_gil_scoped_release guard;
      other_tensor = cast_ad_func(other_tensor, lhs_dtype);
    }
  }
  // 4. calculation
  VLOG(6) << "Calling subtract_ad_func in tensor__sub__method";
  {
    eager_gil_scoped_release guard;
    ret = subtract_ad_func(self_tensor, other_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__rsub__method(TensorObject* self,
                                      PyObject* args,
                                      PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__rsub__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY
  VLOG(1) << "Running Eager tensor__rsub__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::experimental::Tensor ret;
  paddle::experimental::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    float other = 0.0;
    if (PyFloat_Check(other_obj)) {
      other = CastPyArg2AttrFloat(other_obj, 0);
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other = static_cast<float>(CastPyArg2AttrInt(other_obj, 0));
    }
    {
      eager_gil_scoped_release guard;
      ret = CallScalarFuction(self_tensor, other, "rsub");
    }
    return ToPyObject(ret);
  }

  // 2. create or get tensor for other_obj
  paddle::experimental::Tensor other_tensor;
  if (!PyCheckTensor(other_obj)) {
    paddle::experimental::Scalar value =
        CastPyArg2Scalar(other_obj, "__rsub__", 0);
    {
      eager_gil_scoped_release guard;
      other_tensor =
          full_ad_func(self_tensor.shape(), value, self_tensor.dtype(), place);
    }
  } else {
    other_tensor = CastPyArg2Tensor(other_obj, 0);
  }

  // 3. promote types or unify right var type to left var
  phi::DataType lhs_dtype = self_tensor.dtype();
  phi::DataType rhs_dtype = other_tensor.dtype();
  if (lhs_dtype != rhs_dtype) {
    if (_complex_dtypes.find(lhs_dtype) != _complex_dtypes.end() ||
        _complex_dtypes.find(rhs_dtype) != _complex_dtypes.end()) {
      phi::DataType promote_dtype =
          framework::TransToPhiDataType(framework::PromoteTypesIfComplexExists(
              framework::TransToProtoVarType(lhs_dtype),
              framework::TransToProtoVarType(rhs_dtype)));
      if (lhs_dtype != promote_dtype) {
        // cast
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, promote_dtype);
      }
      if (rhs_dtype != promote_dtype) {
        eager_gil_scoped_release guard;
        other_tensor = cast_ad_func(other_tensor, promote_dtype);
      }
    } else {
      LOG(WARNING)
          << "The dtype of left and right Tensor are not the same, left "
             "dtype is "
          << lhs_dtype << ", but right dtype is " << rhs_dtype
          << ", the right dtype will convert to " << lhs_dtype;
      eager_gil_scoped_release guard;
      other_tensor = cast_ad_func(other_tensor, lhs_dtype);
    }
  }

  // 4. calculation
  VLOG(6) << "Calling subtract_ad_func in tensor__rsub__method";
  {
    eager_gil_scoped_release guard;
    ret = subtract_ad_func(other_tensor, self_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef math_op_patch_methods[] = {
    {"__add__",
     (PyCFunction)(void (*)(void))tensor__add__method,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"__radd__",
     (PyCFunction)(void (*)(void))tensor__add__method,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"__sub__",
     (PyCFunction)(void (*)(void))tensor__sub__method,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"__rsub__",
     (PyCFunction)(void (*)(void))tensor__rsub__method,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {NULL, NULL, 0, NULL}};

}  // namespace pybind
}  // namespace paddle
