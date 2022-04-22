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
// disable numpy compile error
#include <Python.h>

#include <string>
#include <vector>

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/utils/tensor_utils.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#pragma GCC diagnostic ignored "-Wwrite-strings"

namespace paddle {
namespace pybind {

extern PyTypeObject* p_tensor_type;

PyObject* tensor_properties_get_name(TensorObject* self, void* closure) {
  EAGER_TRY
  return ToPyObject(self->tensor.name());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_get_type(TensorObject* self, void* closure) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    // be same to old dygraph
    return ToPyObject(paddle::framework::proto::VarType::LOD_TENSOR);
  }
  if (self->tensor.is_dense_tensor()) {
    return ToPyObject(paddle::framework::proto::VarType::LOD_TENSOR);
  } else if (self->tensor.is_selected_rows()) {
    return ToPyObject(paddle::framework::proto::VarType::SELECTED_ROWS);
  } else {
    Py_INCREF(Py_None);
    return Py_None;
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_is_leaf(TensorObject* self, void* closure) {
  EAGER_TRY
  return ToPyObject(egr::egr_utils_api::IsLeafTensor(self->tensor));
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_name(TensorObject* self, PyObject* value,
                               void* closure) {
  EAGER_TRY
  self->tensor.set_name(CastPyArg2AttrString(value, 0));
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

PyObject* tensor_properties_get_stop_gradient(TensorObject* self,
                                              void* closure) {
  EAGER_TRY
  auto meta = egr::EagerUtils::autograd_meta(&self->tensor);
  return ToPyObject(meta->StopGradient());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_get_grad(TensorObject* self, void* closure) {
  EAGER_TRY
  VLOG(6) << "Get grad for tensor: " << self->tensor.name();
  auto meta = egr::EagerUtils::nullable_autograd_meta(self->tensor);
  if (meta && meta->Grad().initialized()) {
    return ToPyObject(meta->Grad());
  } else {
    Py_INCREF(Py_None);
    return Py_None;
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_grad(TensorObject* self, PyObject* value,
                               void* closure) {
  EAGER_TRY
  auto src = CastPyArg2Tensor(value, 0);
  PADDLE_ENFORCE(
      egr::egr_utils_api::IsLeafTensor(self->tensor),
      paddle::platform::errors::Fatal("Only leaf Tensor can be set grad."));

  paddle::experimental::Tensor* grad =
      egr::EagerUtils::mutable_grad(self->tensor);
  PADDLE_ENFORCE(grad != nullptr,
                 paddle::platform::errors::Fatal(
                     "Detected NULL grad"
                     "Please check if you have manually cleared"
                     "the grad inside autograd_meta"));
  grad->copy_(src, self->tensor.place(), true);
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

int tensor_properties_set_stop_gradient(TensorObject* self, PyObject* value,
                                        void* closure) {
  EAGER_TRY
  auto meta = egr::EagerUtils::autograd_meta(&self->tensor);
  meta->SetStopGradient(CastPyArg2AttrBoolean(value, 0));
  if (!meta->GradNode()) {
    meta->SetGradNode(std::make_shared<egr::GradNodeAccumulation>(meta));
  }
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

PyObject* tensor_properties_get_persistable(TensorObject* self, void* closure) {
  EAGER_TRY
  auto meta = egr::EagerUtils::autograd_meta(&self->tensor);
  return ToPyObject(meta->Persistable());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_persistable(TensorObject* self, PyObject* value,
                                      void* closure) {
  EAGER_TRY
  auto meta = egr::EagerUtils::autograd_meta(&self->tensor);
  meta->SetPersistable(CastPyArg2AttrBoolean(value, 0));
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

PyObject* tensor_properties_get_shape(TensorObject* self, void* closure) {
  EAGER_TRY
  std::vector<int64_t> value;
  if (!self->tensor.defined()) {
    return ToPyObject(value);
  }
  auto ddim = self->tensor.shape();
  size_t rank = static_cast<size_t>(ddim.size());
  value.resize(rank);
  for (size_t i = 0; i < rank; i++) {
    value[i] = ddim[i];
  }

  return ToPyObject(value);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_get_place(TensorObject* self, void* closure) {
  EAGER_TRY
  return ToPyObject(self->tensor.place());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_get_place_str(TensorObject* self, void* closure) {
  EAGER_TRY
  std::stringstream ostr;
  ostr << self->tensor.place();
  return ToPyObject(ostr.str());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_get_dtype(TensorObject* self, void* closure) {
  EAGER_TRY
  if (!self->tensor.defined()) {
    // be same to old dygraph
    return ToPyObject(framework::proto::VarType::FP32);
  }
  return ToPyObject(
      paddle::framework::TransToProtoVarType(self->tensor.type()));
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

struct PyGetSetDef variable_properties[] = {
    {"grad", (getter)tensor_properties_get_grad,
     (setter)tensor_properties_set_grad, nullptr, nullptr},
    {"name", (getter)tensor_properties_get_name,
     (setter)tensor_properties_set_name, nullptr, nullptr},
    {"stop_gradient", (getter)tensor_properties_get_stop_gradient,
     (setter)tensor_properties_set_stop_gradient, nullptr, nullptr},
    {"persistable", (getter)tensor_properties_get_persistable,
     (setter)tensor_properties_set_persistable, nullptr, nullptr},
    {"shape", (getter)tensor_properties_get_shape, nullptr, nullptr, nullptr},
    // {"is_leaf", (getter)tensor_properties_get_is_leaf, nullptr,
    // nullptr,
    //  nullptr},
    {"place", (getter)tensor_properties_get_place, nullptr, nullptr, nullptr},
    {"_place_str", (getter)tensor_properties_get_place_str, nullptr, nullptr,
     nullptr},
    {"dtype", (getter)tensor_properties_get_dtype, nullptr, nullptr, nullptr},
    {"type", (getter)tensor_properties_get_type, nullptr, nullptr, nullptr},
    {"is_leaf", (getter)tensor_properties_is_leaf, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

// variable_properties for core.eager.StringTensor
struct PyGetSetDef string_tensor_variable_properties[] = {
    {"name", (getter)tensor_properties_get_name,
     (setter)tensor_properties_set_name, nullptr, nullptr},
    {"shape", (getter)tensor_properties_get_shape, nullptr, nullptr, nullptr},
    {"place", (getter)tensor_properties_get_place, nullptr, nullptr, nullptr},
    {"_place_str", (getter)tensor_properties_get_place_str, nullptr, nullptr,
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

}  // namespace pybind
}  // namespace paddle
