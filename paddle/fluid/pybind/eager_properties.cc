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
  // NOTE(dev): [why not use egr::Controller::Instance::GernerateUniqueName()?]
  // Beacause Controller must holder a tracer, but 'tensor.name' maybe called
  // everywhere such as static mode in @to_static, which means tracer is None.
  static egr::UniqueNameGenerator name_generator;
  if (self->tensor.name().empty()) {
    self->tensor.set_name(name_generator.Generate());
  }
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
  } else if (egr::IsVariableCompatTensor(self->tensor)) {
    return ToPyObject(static_cast<paddle::framework::proto::VarType::Type>(
        static_cast<const egr::VariableCompatTensor*>(self->tensor.impl().get())
            ->Type()));
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_is_leaf(TensorObject* self, void* closure) {
  EAGER_TRY
  return ToPyObject(egr::egr_utils_api::IsLeafTensor(self->tensor));
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_name(TensorObject* self,
                               PyObject* value,
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
  VLOG(6) << meta << " initialized: " << meta->Grad().initialized();
  if (meta && meta->Grad().initialized()) {
    return ToPyObject(meta->Grad());
  } else {
    RETURN_PY_NONE
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_grad(TensorObject* self,
                               PyObject* value,
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

int tensor_properties_set_stop_gradient(TensorObject* self,
                                        PyObject* value,
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

int tensor_properties_set_persistable(TensorObject* self,
                                      PyObject* value,
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
  if (egr::IsVariableCompatTensor(self->tensor)) {
    auto* var_tensor = static_cast<const egr::VariableCompatTensor*>(
        self->tensor.impl().get());
    if (var_tensor->IsType<paddle::framework::Vocab>()) {
      value.emplace_back(static_cast<int64_t>(
          var_tensor->Get<paddle::framework::Vocab>().size()));
    } else if (var_tensor->IsType<paddle::framework::Strings>()) {
      value.emplace_back(static_cast<int64_t>(
          var_tensor->Get<paddle::framework::Strings>().size()));
    } else {
      PADDLE_THROW(paddle::platform::errors::Unavailable(
          "VariableCompatTensor only support get shape from Vocab or "
          "Strings."));
    }
  } else {
    auto ddim = self->tensor.shape();
    size_t rank = static_cast<size_t>(ddim.size());
    value.resize(rank);
    for (size_t i = 0; i < rank; i++) {
      value[i] = ddim[i];
    }
  }

  return ToPyObject(value);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_get_layout(TensorObject* self, void* closure) {
  EAGER_TRY
  std::string layout = "";
  if (!self->tensor.defined()) {
    return ToPyObject(layout);
  }

  if (egr::IsVariableCompatTensor(self->tensor)) {
    VLOG(3) << "VariableCompatTensor does not support `layout` method.";
    return ToPyObject(layout);
  } else {
    return ToPyObject(
        paddle::framework::DataLayoutToString(self->tensor.layout()));
  }

  return ToPyObject(layout);
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
  if (egr::IsVariableCompatTensor(self->tensor)) {
    auto* var_tensor = static_cast<const egr::VariableCompatTensor*>(
        self->tensor.impl().get());
    if (var_tensor->IsType<paddle::framework::Vocab>()) {
      return ToPyObject(framework::proto::VarType::RAW);
    } else if (var_tensor->IsType<paddle::framework::Strings>()) {
      return ToPyObject(framework::proto::VarType::STRING);
    } else {
      PADDLE_THROW(paddle::platform::errors::Unavailable(
          "VariableCompatTensor only support get shape from Vocab or "
          "Strings."));
    }
  } else {
    return ToPyObject(
        paddle::framework::TransToProtoVarType(self->tensor.type()));
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

struct PyGetSetDef variable_properties[] = {
    {"grad",
     (getter)tensor_properties_get_grad,
     (setter)tensor_properties_set_grad,
     nullptr,
     nullptr},
    {"name",
     (getter)tensor_properties_get_name,
     (setter)tensor_properties_set_name,
     nullptr,
     nullptr},
    {"stop_gradient",
     (getter)tensor_properties_get_stop_gradient,
     (setter)tensor_properties_set_stop_gradient,
     nullptr,
     nullptr},
    {"persistable",
     (getter)tensor_properties_get_persistable,
     (setter)tensor_properties_set_persistable,
     nullptr,
     nullptr},
    {"shape", (getter)tensor_properties_get_shape, nullptr, nullptr, nullptr},
    {"layout", (getter)tensor_properties_get_layout, nullptr, nullptr, nullptr},
    // {"is_leaf", (getter)tensor_properties_get_is_leaf, nullptr,
    // nullptr,
    //  nullptr},
    {"place", (getter)tensor_properties_get_place, nullptr, nullptr, nullptr},
    {"_place_str",
     (getter)tensor_properties_get_place_str,
     nullptr,
     nullptr,
     nullptr},
    {"dtype", (getter)tensor_properties_get_dtype, nullptr, nullptr, nullptr},
    {"type", (getter)tensor_properties_get_type, nullptr, nullptr, nullptr},
    {"is_leaf", (getter)tensor_properties_is_leaf, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

// variable_properties for core.eager.StringTensor
struct PyGetSetDef string_tensor_variable_properties[] = {
    {"name",
     (getter)tensor_properties_get_name,
     (setter)tensor_properties_set_name,
     nullptr,
     nullptr},
    {"shape", (getter)tensor_properties_get_shape, nullptr, nullptr, nullptr},
    {"layout", (getter)tensor_properties_get_layout, nullptr, nullptr, nullptr},
    {"place", (getter)tensor_properties_get_place, nullptr, nullptr, nullptr},
    {"_place_str",
     (getter)tensor_properties_get_place_str,
     nullptr,
     nullptr,
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

}  // namespace pybind
}  // namespace paddle
