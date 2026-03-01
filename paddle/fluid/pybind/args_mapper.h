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

#pragma once

#include <Python.h>
#include <vector>
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/ir_context.h"
namespace paddle {

namespace pybind {
void ArgMaxMinMapper(PyObject* args,
                     PyObject* kwargs,
                     Tensor** x_ptr_ptr,
                     paddle::experimental::Scalar* axis,
                     bool* keepdims,
                     bool* flatten,
                     DataType* dtype);
void ArgMaxMinMapper(PyObject* args,
                     PyObject* kwargs,
                     pir::Value* x,
                     pir::Value* axis,
                     bool* keepdims,
                     bool* flatten,
                     DataType* dtype);

void GeluMapper(PyObject* args,
                PyObject* kwargs,
                Tensor** x_ptr_ptr,
                bool* approximate);
void GeluMapper(PyObject* args,
                PyObject* kwargs,
                pir::Value* x,
                bool* approximate);

void ArgSumMapper(PyObject* args,
                  PyObject* kwargs,
                  Tensor** x_ptr_ptr,
                  paddle::experimental::IntArray* axis,
                  DataType* dtype,
                  bool* keepdim);
void ArgSumMapper(PyObject* args,
                  PyObject* kwargs,
                  pir::Value* x,
                  pir::Value* axis,
                  DataType* dtype,
                  bool* keepdim);

void CummaxCumminMapper(PyObject* args,
                      PyObject* kwargs,
                      Tensor** x_ptr_ptr,
                      paddle::experimental::Scalar* axis,
                      DataType* dtype);
void CummaxCumminMapper(PyObject* args,
                      PyObject* kwargs,
                      pir::Value* x,
                      pir::Value* axis,
                      DataType* dtype);

inline void CummaxCumminMapper(PyObject* args,
                               PyObject* kwargs,
                               Tensor** x_ptr_ptr,
                               int* axis,
                               DataType* dtype) {
  paddle::experimental::Scalar axis_scalar(*axis);
  CummaxCumminMapper(args, kwargs, x_ptr_ptr, &axis_scalar, dtype);
}

inline void CummaxCumminMapper(PyObject* args,
                               PyObject* kwargs,
                               pir::Value* x,
                               int* axis,
                               DataType* dtype) {
  PADDLE_THROW(common::errors::Unimplemented(
      "cummax/cummin with int axis is temporarily unsupported in static graph mode. "
      "Please use Scalar or check ops.yaml configuration."));
}

}  // namespace pybind

}  // namespace paddle
