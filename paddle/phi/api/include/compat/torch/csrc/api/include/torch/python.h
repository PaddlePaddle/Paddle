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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once
#include <ATen/Device.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty_strided.h>
#include <c10/util/Exception.h>
#include <torch/types.h>
#include <utils/scalar_type_conversion.h>

#if !defined(PADDLE_ON_INFERENCE) && !defined(PADDLE_NO_PYTHON)
// Python bindings for the C++ frontend (includes Python.h)
#include "paddle/utils/pybind.h"
#endif

namespace torch::python {
namespace detail {

inline Dtype py_object_to_dtype(py::object object) {
  PyObject* obj = object.ptr();
  return *reinterpret_cast<Dtype*>(obj);
}

inline PyObject* getTHPDtype(c10::ScalarType dtype) {
  return paddle::pybind::ToPyObject(
      compat::_PD_AtenScalarTypeToPhiDataType(dtype));
}

}  // namespace detail
}  // namespace torch::python

namespace torch {
using torch::python::detail::getTHPDtype;
}  // namespace torch
