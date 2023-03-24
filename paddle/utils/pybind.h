// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/api/include/tensor.h"
#include "paddle/utils/optional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

typedef struct {
  PyObject_HEAD paddle::Tensor tensor;
  // Weak references
  PyObject* weakrefs;
} TensorObject;

#define RETURN_PY_NONE \
  Py_INCREF(Py_None);  \
  return Py_None;

// Internal use only, to expose the Tensor type to Python.
bool PyCheckTensor(PyObject* obj);

// Internal use only, to expose the Tensor type to Python.
paddle::Tensor CastPyArg2Tensor(PyObject* obj, Py_ssize_t arg_pos);

// Internal use only, to expose the Tensor type to Python.
PyObject* ToPyObject(const paddle::Tensor& value,
                     bool return_py_none_if_not_initialize = false);

// Internal use only, switch tensor_operants_mode to phi
void EnableTensorOperantsToPhiMode();

}  // namespace pybind
}  // namespace paddle

namespace pybind11 {
namespace detail {

template <>
struct type_caster<paddle::Tensor> {
 public:
  PYBIND11_TYPE_CASTER(paddle::Tensor, _("paddle::Tensor"));

  bool load(handle src, bool) {
    paddle::pybind::EnableTensorOperantsToPhiMode();
    PyObject* obj = src.ptr();
    if (paddle::pybind::PyCheckTensor(obj)) {
      value = paddle::pybind::CastPyArg2Tensor(obj, 0);
      return true;
    }
    return false;
  }

  static handle cast(const paddle::Tensor& src,
                     return_value_policy /* policy */,
                     handle /* parent */) {
    return handle(paddle::pybind::ToPyObject(
        src, true /* return_py_none_if_not_initialize */));
  }
};

// Pybind11 bindings for optional types.
// http://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#c-17-library-containers
template <typename T>
struct type_caster<paddle::optional<T>> : optional_caster<paddle::optional<T>> {
};

}  // namespace detail
}  // namespace pybind11
