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
void ArgMaxMinMapper(PyObject* args,
                     PyObject* kwargs,
                     Tensor* x,
                     paddle::experimental::Scalar* axis,
                     bool* keepdims,
                     bool* flatten,
                     phi::DataType* dtype) {
  // The python params are (x, axis,keepdim,dtype,name) which  haven't flatten
  // The _C_ops params are (x, axis,keepdim,flatten,dtype) which have flatten
  // but haven't name We should parse the python params and convert them to the
  // _C_ops params
  int nargs = args ? static_cast<int>(PyTuple_Size(args)) : 0;
  int remaining_kwargs = kwargs ? static_cast<int>(PyDict_Size(kwargs)) : 0;
  // python params count only consider the python params(x, axis, keepdim,
  // dtype), not include the name
  const int max_args = 4;
  CheckParamsCount(nargs, remaining_kwargs, max_args);

  VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);
  // Get EagerTensors from args
  *x = GetTensorFromArgsOrKWArgs("argmax",
                                 "x",
                                 args,
                                 0,
                                 kwargs,
                                 {"x", "input"},
                                 nargs,
                                 &remaining_kwargs,
                                 false);

  // Parse Attributes if needed

  PyObject* axis_obj = GetItemFromArgsOrKWArgs(
      args, 1, kwargs, {"axis", "dim"}, nargs, &remaining_kwargs);
  /**
      flatten = False
      if axis is None:
          flatten = True
          axis = 0
  */
  *flatten = false;
  if (axis_obj == Py_None || axis_obj == nullptr) {
    *flatten = true;
    *axis = 0;
  } else {
    *axis = CastPyArg2Scalar(axis_obj, "argmax", 1);
  }
  PyObject* keepdims_obj = GetItemFromArgsOrKWArgs(
      args, 2, kwargs, {"keepdim", "keepdims"}, nargs, &remaining_kwargs);
  *keepdims = CastPyArg2Boolean(keepdims_obj, "argmax", 2, false);

  PyObject* dtype_obj = GetItemFromArgsOrKWArgs(
      args, 3, kwargs, {"dtype"}, nargs, &remaining_kwargs);
  /**
     if dtype is None:
          raise ValueError(
         "the value of 'dtype' in argmax could not be None, but received None")
  */
  PADDLE_ENFORCE_NE(
      dtype_obj,
      Py_None,
      phi::errors::InvalidArgument("the value of 'dtype' in argmax and argmin "
                                   "could not be None, but received None"));
  *dtype = CastPyArg2DataType(dtype_obj, "argmax", 3, phi::DataType::INT64);
  // Check Reminding Params validity if needed
  CheckRemainingParamsValidity(args, kwargs, remaining_kwargs, nargs);

  return;
}
void ArgMaxMinMapper(PyObject* args,
                     PyObject* kwargs,
                     pir::Value* x,
                     pir::Value* axis,
                     bool* keepdims,
                     bool* flatten,
                     phi::DataType* dtype) {
  // Get Total Params count and check validity if needed
  int nargs = args ? static_cast<int>(PyTuple_Size(args)) : 0;
  int remaining_kwargs = kwargs ? static_cast<int>(PyDict_Size(kwargs)) : 0;
  const int max_args = 4;
  CheckParamsCount(nargs, remaining_kwargs, max_args);

  // Get Value from args
  PyObject* x_obj = GetItemFromArgsOrKWArgs(
      args, 0, kwargs, {"x", "input"}, nargs, &remaining_kwargs);
  *x = CastPyArg2Value(x_obj, "argmax", 0, false);

  // Parse Attributes
  PyObject* axis_obj = GetItemFromArgsOrKWArgs(
      args, 1, kwargs, {"axis", "dim"}, nargs, &remaining_kwargs);
  PyObject* keepdims_obj = GetItemFromArgsOrKWArgs(
      args, 2, kwargs, {"keepdim", "keepdims"}, nargs, &remaining_kwargs);
  PyObject* dtype_obj = GetItemFromArgsOrKWArgs(
      args, 3, kwargs, {"dtype"}, nargs, &remaining_kwargs);

  /**
      flatten = False
      if axis is None:
          flatten = True
          axis = 0
  */
  *flatten = false;
  if (axis_obj == Py_None || axis_obj == nullptr) {
    *flatten = true;
    *axis = paddle::dialect::full(
        std::vector<int64_t>{1}, 0, phi::DataType::INT64, phi::CPUPlace());
  } else if (PyObject_CheckIRValue(axis_obj)) {
    *axis = CastPyArg2Value(axis_obj, "argmax", 1);
  } else {
    int64_t axis_tmp = CastPyArg2Long(axis_obj, "argmax", 1);
    *axis = paddle::dialect::full(std::vector<int64_t>{1},
                                  axis_tmp,
                                  phi::DataType::INT64,
                                  phi::CPUPlace());
  }
  *keepdims = CastPyArg2Boolean(keepdims_obj, "argmax", 2, false);

  PADDLE_ENFORCE_NE(
      dtype_obj,
      Py_None,
      phi::errors::InvalidArgument("the value of 'dtype' in argmax and argmin "
                                   "could not be None, but received None"));
  *dtype = CastPyArg2DataType(dtype_obj, "argmax", 3, phi::DataType::INT64);

  // Check Reminding Params validity if needed
  CheckRemainingParamsValidity(args, kwargs, remaining_kwargs, nargs);
  return;
}

}  // namespace pybind
}  // namespace paddle
