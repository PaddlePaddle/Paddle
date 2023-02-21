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

#include "paddle/utils/pybind.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace pybind {

PyTypeObject* p_tensor_type;
PyTypeObject* p_string_tensor_type;

bool PyCheckTensor(PyObject* obj) {
  if (!p_tensor_type) {
    return false;
  }
  return PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type));
}

paddle::experimental::Tensor CastPyArg2Tensor(PyObject* obj, ssize_t arg_pos) {
  if (PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type)) ||
      PyObject_IsInstance(obj,
                          reinterpret_cast<PyObject*>(p_string_tensor_type))) {
    return reinterpret_cast<TensorObject*>(obj)->tensor;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "argument (position %d) must be "
        "Tensor, but got %s",
        arg_pos + 1,
        reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name));
  }
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
    PADDLE_THROW(
        phi::errors::Fatal("tp_alloc return null, can not new a PyObject."));
  }
  return obj;
}

}  // namespace pybind
}  // namespace paddle
