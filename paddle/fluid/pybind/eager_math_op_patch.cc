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
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

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
#include "paddle/common/ddim.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/memory/allocation/mmap_allocator.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/phi/common/type_promotion.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace pybind {

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

static bool IsNumpyArray(PyObject* obj) {
  auto type_name = std::string(Py_TYPE(obj)->tp_name);
  return type_name == "numpy.ndarray";
}

void InitTensorWithNumpyValue(const py::object& array,
                              const paddle::platform::Place& place,
                              Tensor* self,
                              bool zero_copy = false) {
  PADDLE_ENFORCE_EQ(
      self->defined(),
      true,
      paddle::platform::errors::Fatal(
          "Calling InitTensorWithNumpyValue of Eager Tensor without "
          "EmptyTensorInitializer is "
          "forbidden. Please check your code and make sure you new a "
          "eager tensor before init it with NumPy."));
  phi::DenseTensor* impl_ptr =
      static_cast<phi::DenseTensor*>(self->impl().get());
  if (platform::is_cpu_place(place)) {
    SetTensorFromPyArray<platform::CPUPlace>(impl_ptr, array, place, zero_copy);
  } else if (platform::is_xpu_place(place)) {
    SetTensorFromPyArray<platform::XPUPlace>(impl_ptr, array, place, zero_copy);
  } else if (platform::is_gpu_place(place)) {
    SetTensorFromPyArray<platform::CUDAPlace>(
        impl_ptr, array, place, zero_copy);
  } else if (platform::is_cuda_pinned_place(place)) {
    SetTensorFromPyArray<platform::CUDAPinnedPlace>(
        impl_ptr, array, place, zero_copy);
  } else if (platform::is_custom_place(place)) {
    SetTensorFromPyArray<platform::CustomPlace>(
        impl_ptr, array, place, zero_copy);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Place should be one of "
        "CPUPlace/XPUPlace/CUDAPlace/CUDAPinnedPlace/CustomPlace"));
  }
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
    VLOG(6) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
            << " from " << static_cast<int>(place.device);
#else
    PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
  }

  if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
    phi::DeviceManager::SetDevice(place);
    VLOG(6) << "CurrentDeviceId: "
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
paddle::Tensor CallScalarFuction(const paddle::Tensor& self_tensor,
                                 double other,
                                 std::string op_type) {
  paddle::Tensor ret;
  if (op_type == "add" || op_type == "radd") {
    ret = scale_ad_func(
        self_tensor, phi::Scalar(1.0), static_cast<float>(other), true);
  } else if (op_type == "sub") {
    ret = scale_ad_func(
        self_tensor, phi::Scalar(1.0), static_cast<float>(-other), true);

  } else if (op_type == "rsub") {
    ret = scale_ad_func(
        self_tensor, phi::Scalar(-1.0), static_cast<float>(other), true);
  } else if (op_type == "mul") {
    ret = scale_ad_func(self_tensor, phi::Scalar(other), 0.0, true);
  } else if (op_type == "div") {
    ret = scale_ad_func(self_tensor, phi::Scalar(1.0 / other), 0.0, true);
  } else if (op_type == "pow") {
    ret = pow_ad_func(self_tensor, other);
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

  paddle::Tensor ret;
  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    double other = 0.0;
    if (PyFloat_Check(other_obj)) {
      other = CastPyArg2Double(other_obj, "__add__", 0);
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other = CastPyArg2Double(other_obj, "__add__", 0);
    }

    {
      eager_gil_scoped_release guard;
      ret = CallScalarFuction(self_tensor, other, "add");
    }
    return ToPyObject(ret);
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__add__", 0);
      {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    self_tensor.dtype(),
                                    self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

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

  paddle::Tensor ret;

  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);
  // 1. scalar exists cases
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    double other = 0.0;
    if (PyFloat_Check(other_obj)) {
      other = CastPyArg2Double(other_obj, "__sub__", 0);
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other = CastPyArg2Double(other_obj, "__sub__", 0);
    }
    {
      eager_gil_scoped_release guard;
      ret = CallScalarFuction(self_tensor, other, "sub");
    }

    return ToPyObject(ret);
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__sub__", 0);
      {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    self_tensor.dtype(),
                                    self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

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
  VLOG(4) << "Running Eager tensor__rsub__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;
  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    double other = 0.0;
    if (PyFloat_Check(other_obj)) {
      other = CastPyArg2Double(other_obj, "__rsub__", 0);
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other = CastPyArg2Double(other_obj, "__rsub__", 0);
    }
    {
      eager_gil_scoped_release guard;
      ret = CallScalarFuction(self_tensor, other, "rsub");
    }
    return ToPyObject(ret);
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (PyCheckTensor(other_obj)) {
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    auto& self_tensor_ref = self->tensor;
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__rsub__", 0);
      {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    self_tensor.dtype(),
                                    self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling subtract_ad_func in tensor__rsub__method";
  {
    eager_gil_scoped_release guard;
    ret = subtract_ad_func(other_tensor, self_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__mul__method(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__mul__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY
  VLOG(6) << "Running Eager tensor__mul__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;

  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    double other = 0.0;
    if (PyFloat_Check(other_obj)) {
      other = CastPyArg2Double(other_obj, "__mul__", 0);
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other = CastPyArg2Double(other_obj, "__mul__", 0);
    }
    {
      eager_gil_scoped_release guard;
      ret = CallScalarFuction(self_tensor, other, "mul");
    }
    return ToPyObject(ret);
  }

  // 2. create or get tensor for other_obj
  // if lhs or rhs input is tensor, we need to inplace cast it to dist_tensor
  // if one of the input is numpy or scalar, no need to do inplace cast.
  paddle::Tensor other_tensor;
  if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__mul__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    DataType::COMPLEX64,
                                    self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    self_tensor.dtype(),
                                    self_tensor.place());
      }
      const phi::distributed::ProcessMesh* mesh = nullptr;
      if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
        ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
      }
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling multiply_ad_func in tensor__mul__method";
  {
    eager_gil_scoped_release guard;
    ret = multiply_ad_func(self_tensor, other_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__div__method(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__div__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY

  VLOG(6) << "Running Eager tensor__div__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;

  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    double other = 0.0;
    if (PyFloat_Check(other_obj)) {
      other = CastPyArg2Double(other_obj, "__div__", 0);
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other = CastPyArg2Double(other_obj, "__div__", 0);
    }
    if (_supported_int_dtype_.find(self_tensor.dtype()) !=
        _supported_int_dtype_.end()) {
      eager_gil_scoped_release guard;
      self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
    }
    {
      eager_gil_scoped_release guard;
      ret = CallScalarFuction(self_tensor, other, "div");
    }
    return ToPyObject(ret);
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__div__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    self_tensor.dtype(),
                                    self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling divide_ad_func in tensor__div__method";
  {
    eager_gil_scoped_release guard;
    ret = divide_ad_func(self_tensor, other_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__rdiv__method(TensorObject* self,
                                      PyObject* args,
                                      PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__rdiv__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);
  EAGER_TRY

  VLOG(6) << "Running Eager tensor__rdiv__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;

  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  // there is no scalar_div function for __rdiv__ and __rtruediv__
  double other_double = 0.0;
  bool has_other_double = false;
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    if (PyFloat_Check(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__rdiv__", 0);
      has_other_double = true;
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__rdiv__", 0);
      has_other_double = true;
    }
    if (_supported_int_dtype_.find(self_tensor.dtype()) !=
        _supported_int_dtype_.end()) {
      eager_gil_scoped_release guard;
      self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
    }
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (has_other_double) {
    eager_gil_scoped_release guard;
    other_tensor = full_ad_func(self_tensor.shape(),
                                phi::Scalar(other_double),
                                self_tensor.dtype(),
                                place);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  } else if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__rdiv__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    self_tensor.dtype(),
                                    self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling divide_ad_func in tensor__rdiv__method";
  {
    eager_gil_scoped_release guard;
    ret = divide_ad_func(other_tensor, self_tensor);
  }
  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__gt__method(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__gt__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY
  VLOG(4) << "Running Eager tensor__gt__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;
  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  // there is no scalar function for __gt__ now
  double other_double = 0.0;
  bool has_other_double = false;
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    if (PyFloat_Check(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__gt__", 0);
      has_other_double = true;
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__gt__", 0);
      has_other_double = true;
    }
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (has_other_double) {
    eager_gil_scoped_release guard;
    other_tensor = full_ad_func(self_tensor.shape(),
                                phi::Scalar(other_double),
                                self_tensor.dtype(),
                                place);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  } else if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__gt__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    self_tensor.dtype(),
                                    self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling greater_than_ad_func in tensor__gt__method";
  {
    eager_gil_scoped_release guard;
    ret = greater_than_ad_func(self_tensor, other_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__ge__method(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__ge__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY
  VLOG(4) << "Running Eager tensor__ge__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;
  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  // there is no scalar function for __ge__ now
  double other_double = 0.0;
  bool has_other_double = false;
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    if (PyFloat_Check(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__ge__", 0);
      has_other_double = true;
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__ge__", 0);
      has_other_double = true;
    }
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (has_other_double) {
    eager_gil_scoped_release guard;
    other_tensor = full_ad_func(self_tensor.shape(),
                                phi::Scalar(other_double),
                                self_tensor.dtype(),
                                place);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  } else if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__ge__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    self_tensor.dtype(),
                                    self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling greater_equal_ad_func in tensor__ge__method";
  {
    eager_gil_scoped_release guard;
    ret = greater_equal_ad_func(self_tensor, other_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__mod__method(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__mod__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);
  EAGER_TRY

  VLOG(6) << "Running Eager tensor__mod__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;

  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  // there is no scalar_mod function for __mod__ now
  float other_double = 0.0f;
  bool has_other_double = false;
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    if (PyFloat_Check(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__mod__", 0);  // NOLINT
      has_other_double = true;
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__mod__", 0);  // NOLINT
      has_other_double = true;
    }
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (has_other_double) {
    eager_gil_scoped_release guard;
    other_tensor = full_ad_func(self_tensor.shape(),
                                phi::Scalar(other_double),
                                self_tensor.dtype(),
                                self_tensor.place());
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  } else if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__mod__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    self_tensor.dtype(),
                                    self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling remainder_ad_func in tensor__mod__method";
  {
    eager_gil_scoped_release guard;
    ret = remainder_ad_func(self_tensor, other_tensor);
  }
  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__matmul__method(TensorObject* self,
                                        PyObject* args,
                                        PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__matmul__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);
  EAGER_TRY

  VLOG(6) << "Running Eager tensor__matmul__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;
  paddle::Tensor self_tensor = self->tensor;

  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  // there is no scalar_matmul function for __matmul__ now
  float other_double = 0.0f;
  bool has_other_double = false;
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    if (PyFloat_Check(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__matmul__", 0);  // NOLINT
      has_other_double = true;
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__matmul__", 0);  // NOLINT
      has_other_double = true;
    }
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (has_other_double) {
    eager_gil_scoped_release guard;
    other_tensor = full_ad_func({1},
                                phi::Scalar(other_double),
                                self_tensor.dtype(),
                                self_tensor.place());
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  } else if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__matmul__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, self_tensor.dtype(), self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling matmul_ad_func in tensor__matmul__method";
  {
    eager_gil_scoped_release guard;
    ret = matmul_ad_func(self_tensor, other_tensor, false, false);
  }
  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__lt__method(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__lt__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY
  VLOG(4) << "Running Eager tensor__lt__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;
  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  // there is no scalar function for __lt__ now
  float other_double = 0.0f;
  bool has_other_double = false;
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    if (PyFloat_Check(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__lt__", 0);  // NOLINT
      has_other_double = true;
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__lt__", 0);  // NOLINT
      has_other_double = true;
    }
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (has_other_double) {
    eager_gil_scoped_release guard;
    other_tensor = full_ad_func(self_tensor.shape(),
                                phi::Scalar(other_double),
                                self_tensor.dtype(),
                                self_tensor.place());
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  } else if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__lt__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    self_tensor.dtype(),
                                    self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling less_than_ad_func in tensor__lt__method";
  {
    eager_gil_scoped_release guard;
    ret = less_than_ad_func(self_tensor, other_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__le__method(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__le__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY
  VLOG(4) << "Running Eager tensor__le__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;
  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  // there is no scalar function for __le__ now
  float other_double = 0.0f;
  bool has_other_double = false;
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    if (PyFloat_Check(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__le__", 0);  // NOLINT
      has_other_double = true;
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__le__", 0);  // NOLINT
      has_other_double = true;
    }
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (has_other_double) {
    eager_gil_scoped_release guard;
    other_tensor = full_ad_func(self_tensor.shape(),
                                phi::Scalar(other_double),
                                self_tensor.dtype(),
                                self_tensor.place());
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  } else if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__le__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    self_tensor.dtype(),
                                    self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling less_equal_ad_func in tensor__le__method";
  {
    eager_gil_scoped_release guard;
    ret = less_equal_ad_func(self_tensor, other_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__floordiv__method(TensorObject* self,
                                          PyObject* args,
                                          PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "floordiv pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);
  EAGER_TRY
  VLOG(6) << "Running Eager tensor__floordiv__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;
  paddle::Tensor self_tensor = self->tensor;

  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases or not
  // there is no scalar case for floordiv, but alse need to cast self_tensor
  // in need.
  double other_double = 0.0;
  bool has_other_double = false;
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    if (PyFloat_Check(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__floordiv__", 0);
      has_other_double = true;
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__floordiv__", 0);
      has_other_double = true;
    }
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (has_other_double) {
    eager_gil_scoped_release guard;
    other_tensor = full_ad_func(self_tensor.shape(),
                                phi::Scalar(other_double),
                                self_tensor.dtype(),
                                self_tensor.place());
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  } else if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__floordiv__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, self_tensor.dtype(), self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling floor_divide_ad_func in tensor__floordiv__method";
  {
    eager_gil_scoped_release guard;
    ret = floor_divide_ad_func(self_tensor, other_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__pow__method(TensorObject* self,
                                     PyObject* args,
                                     PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "pow pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY
  VLOG(6) << "Running Eager tensor__pow__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;
  paddle::Tensor self_tensor = self->tensor;

  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    double other = 0.0;
    if (PyFloat_Check(other_obj)) {
      other = CastPyArg2Double(other_obj, "__pow__", 0);
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other = CastPyArg2Double(other_obj, "__pow__", 0);
    }
    {
      eager_gil_scoped_release guard;
      ret = CallScalarFuction(self_tensor, other, "pow");
    }
    return ToPyObject(ret);
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__pow__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, self_tensor.dtype(), self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling elementwise_pow_ad_func in tensor__pow__method";
  {
    eager_gil_scoped_release guard;
    ret = elementwise_pow_ad_func(self_tensor, other_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__rpow__method(TensorObject* self,
                                      PyObject* args,
                                      PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__rpow__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY
  VLOG(6) << "Running Eager tensor__rpow__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;
  paddle::Tensor self_tensor = self->tensor;

  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases or not
  // there is no scalar case for rpow, but alse need to cast self_tensor in
  // need.
  double other_double = 0.0;
  bool has_other_double = false;
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    if (PyFloat_Check(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__rpow__", 0);
      has_other_double = true;
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__rpow__", 0);
      has_other_double = true;
    }
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (has_other_double) {
    eager_gil_scoped_release guard;
    other_tensor = full_ad_func(self_tensor.shape(),
                                phi::Scalar(other_double),
                                self_tensor.dtype(),
                                self_tensor.place());
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  } else if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__rpow__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor = full_ad_func(self_tensor.shape(),
                                    value,
                                    self_tensor.dtype(),
                                    self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling elementwise_pow_ad_func in tensor__rpow__method";
  {
    eager_gil_scoped_release guard;
    ret = elementwise_pow_ad_func(other_tensor, self_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__ne__method(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__ne__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY
  VLOG(6) << "Running Eager tensor__ne__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;
  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  // there is no scalar function for __ne__ now
  double other_double = 0.0;
  bool has_other_double = false;
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    if (PyFloat_Check(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__ne__", 0);
      has_other_double = true;
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__ne__", 0);
      has_other_double = true;
    }
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (has_other_double) {
    eager_gil_scoped_release guard;
    other_tensor = full_ad_func(self_tensor.shape(),
                                phi::Scalar(other_double),
                                self_tensor.dtype(),
                                self_tensor.place());
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  } else if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__ne__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, self_tensor.dtype(), self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling not_equal_ad_func in tensor__ne__method";
  {
    eager_gil_scoped_release guard;
    ret = not_equal_ad_func(self_tensor, other_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* tensor__eq__method(TensorObject* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "__eq__ pybind_patch_func",
      paddle::platform::TracerEventType::UserDefined,
      1);

  EAGER_TRY
  VLOG(6) << "Running Eager tensor__eq__method";

  // Set Device ID
  auto place = egr::Controller::Instance().GetExpectedPlace();
  SetDevice(place);

  paddle::Tensor ret;
  paddle::Tensor self_tensor = self->tensor;
  PyObject* other_obj = PyTuple_GET_ITEM(args, 0);

  // 1. scalar exists cases
  // there is no scalar function for __eq__ now
  double other_double = 0.0;
  bool has_other_double = false;
  if (PyFloat_Check(other_obj) || PyCheckInteger(other_obj) ||
      IsNumpyType(other_obj)) {
    if (PyFloat_Check(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__eq__", 0);
      has_other_double = true;
      if (_supported_int_dtype_.find(self_tensor.dtype()) !=
          _supported_int_dtype_.end()) {
        eager_gil_scoped_release guard;
        self_tensor = cast_ad_func(self_tensor, DataType::FLOAT32);
      }
    } else if (PyCheckInteger(other_obj) || IsNumpyType(other_obj)) {
      other_double = CastPyArg2Double(other_obj, "__eq__", 0);
      has_other_double = true;
    }
  }

  // 2. create or get tensor for other_obj
  paddle::Tensor other_tensor;
  if (has_other_double) {
    eager_gil_scoped_release guard;
    other_tensor = full_ad_func(self_tensor.shape(),
                                phi::Scalar(other_double),
                                self_tensor.dtype(),
                                self_tensor.place());
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  } else if (PyCheckTensor(other_obj)) {
    auto& self_tensor_ref = self->tensor;
    auto& other_tensor_ref = CastPyArg2Tensor(other_obj, 0);
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor_ref, other_tensor_ref)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor_ref, other_tensor_ref);
    }
    self_tensor = self_tensor_ref;
    other_tensor = other_tensor_ref;
  } else {
    if (IsNumpyArray(other_obj)) {
      py::object numpy_value = py::object(py::handle(other_obj), true);
      other_tensor = paddle::Tensor(place);
      InitTensorWithNumpyValue(numpy_value, place, &other_tensor);
    } else {
      paddle::experimental::Scalar value =
          CastPyArg2Scalar(other_obj, "__eq__", 0);
      if (PyComplex_Check(other_obj)) {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, DataType::COMPLEX64, self_tensor.place());
      } else {
        eager_gil_scoped_release guard;
        other_tensor =
            full_ad_func({1}, value, self_tensor.dtype(), self_tensor.place());
      }
    }
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (InputsContainDistTensor(&mesh, self_tensor, other_tensor)) {
      ConvertAllInputsToDistTensor(mesh, self_tensor, other_tensor);
    }
  }

  // 3. type promotion will be automatically performed in add_ad_func

  // 4. calculation
  VLOG(6) << "Calling equal_ad_func in tensor__eq__method";
  {
    eager_gil_scoped_release guard;
    ret = equal_ad_func(self_tensor, other_tensor);
  }

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef math_op_patch_methods[] = {  // NOLINT
    {"__add__",
     (PyCFunction)(void (*)())tensor__add__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__radd__",
     (PyCFunction)(void (*)())tensor__add__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__sub__",
     (PyCFunction)(void (*)())tensor__sub__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__rsub__",
     (PyCFunction)(void (*)())tensor__rsub__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__mul__",
     (PyCFunction)(void (*)())tensor__mul__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__rmul__",
     (PyCFunction)(void (*)())tensor__mul__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__div__",
     (PyCFunction)(void (*)())tensor__div__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__truediv__",
     (PyCFunction)(void (*)())tensor__div__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__rdiv__",
     (PyCFunction)(void (*)())tensor__rdiv__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__rtruediv__",
     (PyCFunction)(void (*)())tensor__rdiv__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__floordiv__",
     (PyCFunction)(void (*)())tensor__floordiv__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__pow__",
     (PyCFunction)(void (*)())tensor__pow__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__rpow__",
     (PyCFunction)(void (*)())tensor__rpow__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__mod__",
     (PyCFunction)(void (*)())tensor__mod__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__matmul__",
     (PyCFunction)(void (*)())tensor__matmul__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__gt__",
     (PyCFunction)(void (*)())tensor__gt__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__ge__",
     (PyCFunction)(void (*)())tensor__ge__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__lt__",
     (PyCFunction)(void (*)())tensor__lt__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__le__",
     (PyCFunction)(void (*)())tensor__le__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__eq__",
     (PyCFunction)(void (*)())tensor__eq__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__ne__",
     (PyCFunction)(void (*)())tensor__ne__method,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace pybind
}  // namespace paddle
