// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// custom arg mapper function.
// The function here will be called by the functions in
// paddle/fluid/pybind/static_op_function.cc and
// paddle/fluid/pybind/eager_op_function.cc. Mainly used to customize the args
// parser from PyObject *args and PyObject *kwargs

#include "paddle/fluid/pybind/args_mapper.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
namespace paddle {
namespace pybind {
bool CheckBool(PyObject* obj) {
  if (obj == Py_None || obj == Py_False || obj == Py_True) {
    return true;
  }
  return false;
}
void ArgSumMapper(PyObject* args,
                  PyObject* kwargs,
                  Tensor* x,
                  paddle::experimental::IntArray* axis,
                  phi::DataType* dtype,
                  bool* keepdim) {
  // Get Total Params count and check validity if needed
  int nargs = args ? static_cast<int>(PyTuple_Size(args)) : 0;
  int remaining_kwargs = kwargs ? static_cast<int>(PyDict_Size(kwargs)) : 0;
  const int max_args = 4;
  CheckParamsCount(nargs, remaining_kwargs, max_args);

  // Get EagerTensors from args
  *x = GetTensorFromArgsOrKWArgs("sum",
                                 "x",
                                 args,
                                 0,
                                 kwargs,
                                 {"input", "x"},
                                 nargs,
                                 &remaining_kwargs,
                                 false);

  // Parse Attributes if needed
  PyObject* axis_obj = GetItemFromArgsOrKWArgs(
      args, 1, kwargs, {"dim", "axis"}, nargs, &remaining_kwargs);
  *axis = CastPyArg2IntArray(axis_obj, "sum", 1, {});

  PyObject* py_obj_1 = GetItemFromArgsOrKWArgs(
      args, 2, kwargs, {"dtype", "keepdim"}, nargs, &remaining_kwargs);
  PyObject* py_obj_2 = GetItemFromArgsOrKWArgs(
      args, 3, kwargs, {"keepdim", "dtype"}, nargs, &remaining_kwargs);

  if (py_obj_1 == nullptr && py_obj_2 == nullptr) {
    // No parameters, use default values
    *dtype = phi::DataType::UNDEFINED;
    *keepdim = false;
  } else if (py_obj_1 != nullptr && py_obj_2 == nullptr) {
    // There is only one parameter, it needs to be determined whether it is
    // dtype or keepdim
    if (CheckBool(py_obj_1)) {
      *keepdim = CastPyArg2Boolean(py_obj_1, "sum", 2, false);
      *dtype = phi::DataType::UNDEFINED;
    } else {
      *dtype = CastPyArg2DataType(py_obj_1, "sum", 2, phi::DataType::UNDEFINED);
      *keepdim = false;
    }
  } else {
    // Both parameters are not null ptr
    bool is_keepdim1 = CheckBool(py_obj_1);
    bool is_keepdim2 = CheckBool(py_obj_2);

    if (is_keepdim1 && !is_keepdim2) {
      *keepdim = CastPyArg2Boolean(py_obj_1, "sum", 2, false);
      *dtype = CastPyArg2DataType(py_obj_2, "sum", 3, phi::DataType::UNDEFINED);
    } else if (!is_keepdim1 && is_keepdim2) {
      *dtype = CastPyArg2DataType(py_obj_1, "sum", 2, phi::DataType::UNDEFINED);
      *keepdim = CastPyArg2Boolean(py_obj_2, "sum", 3, false);
    } else {
      // Both are judged as keepdim, or neither is
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Invalid arguments for paddle.sum(): One of the last two arguments "
          "must be a boolean (keepdim), and the other must be a dtype."));
    }
  }

  // Check Reminding Params validity if needed
  CheckRemainingParamsValidity(args, kwargs, remaining_kwargs, nargs);
}
void ArgSumMapper(PyObject* args,
                  PyObject* kwargs,
                  pir::Value* x,
                  pir::Value* axis,
                  phi::DataType* dtype,
                  bool* keepdim) {
  // Get Total Params count and check validity if needed
  int nargs = args ? static_cast<int>(PyTuple_Size(args)) : 0;
  int remaining_kwargs = kwargs ? static_cast<int>(PyDict_Size(kwargs)) : 0;
  const int max_args = 4;
  CheckParamsCount(nargs, remaining_kwargs, max_args);

  // Get Value from args
  PyObject* x_obj = GetItemFromArgsOrKWArgs(
      args, 0, kwargs, {"input", "x"}, nargs, &remaining_kwargs);
  *x = CastPyArg2Value(x_obj, "sum", 0, false);

  // Parse Attributes
  PyObject* axis_obj = GetItemFromArgsOrKWArgs(
      args, 1, kwargs, {"axis", "dim"}, nargs, &remaining_kwargs);
  PyObject* py_obj_1 = GetItemFromArgsOrKWArgs(
      args, 2, kwargs, {"dtype", "keepdim"}, nargs, &remaining_kwargs);
  PyObject* py_obj_2 = GetItemFromArgsOrKWArgs(
      args, 3, kwargs, {"keepdim", "dtype"}, nargs, &remaining_kwargs);

  // Parse input_out if needed
  Check_PIR_not_support_out(kwargs);

  // Check for mutable attrs
  if (axis_obj && PyObject_CheckIRValue(axis_obj)) {
    *axis = CastPyArg2Value(axis_obj, "sum", 1);
  } else if (axis_obj && PyObject_CheckIRVectorOfValue(axis_obj)) {
    std::vector<pir::Value> axis_tmp =
        CastPyArg2VectorOfValue(axis_obj, "sum", 1);
    *axis = paddle::dialect::stack(axis_tmp, /*axis*/ 0);
  } else if (axis_obj && PyObject_CheckIRVectorOfValueOrLong(axis_obj)) {
    std::vector<pir::Value> axis_tmp =
        CastPyArg2VectorOfValueOrLong(axis_obj, "sum", 1);
    *axis = paddle::dialect::stack(axis_tmp, /*axis*/ 0);
  } else {
    std::vector<int64_t> axis_tmp = CastPyArg2Longs(axis_obj, "sum", 1, {});
    *axis = paddle::dialect::full_int_array(
        axis_tmp, phi::DataType::INT64, phi::CPUPlace());
  }

  if (py_obj_1 == nullptr && py_obj_2 == nullptr) {
    // No parameters, use default values
    *dtype = phi::DataType::UNDEFINED;
    *keepdim = false;
  } else if (py_obj_1 != nullptr && py_obj_2 == nullptr) {
    // There is only one parameter, it needs to be determined whether it is
    // dtype or keepdim
    if (CheckBool(py_obj_1)) {
      *keepdim = CastPyArg2Boolean(py_obj_1, "sum", 2, false);
      *dtype = phi::DataType::UNDEFINED;
    } else {
      *dtype = CastPyArg2DataType(py_obj_1, "sum", 2, phi::DataType::UNDEFINED);
      *keepdim = false;
    }
  } else {
    // Both parameters are not null ptr
    bool is_keepdim1 = CheckBool(py_obj_1);
    bool is_keepdim2 = CheckBool(py_obj_2);

    if (is_keepdim1 && !is_keepdim2) {
      *keepdim = CastPyArg2Boolean(py_obj_1, "sum", 2, false);
      *dtype = CastPyArg2DataType(py_obj_2, "sum", 3, phi::DataType::UNDEFINED);
    } else if (!is_keepdim1 && is_keepdim2) {
      *dtype = CastPyArg2DataType(py_obj_1, "sum", 2, phi::DataType::UNDEFINED);
      *keepdim = CastPyArg2Boolean(py_obj_2, "sum", 3, false);
    } else {
      // Both are judged as keepdim, or neither is
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Invalid arguments for paddle.sum(): One of the last two arguments "
          "must be a boolean (keepdim), and the other must be a dtype."));
    }
  }
}
}  // namespace pybind

}  // namespace paddle
