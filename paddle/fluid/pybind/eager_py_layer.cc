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
#include <Python.h>
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

#include <set>
#include <string>
#include <vector>

#pragma GCC diagnostic ignored "-Wattributes"
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/pylayer/py_layer_node.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "pybind11/detail/internals.h"
#include "pybind11/pytypes.h"
#pragma GCC diagnostic ignored "-Wwrite-strings"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

using egr::ConvertToDistTensor;

namespace paddle::pybind {

PyTypeObject* p_pylayer_type;
extern PyTypeObject* p_tensor_type;

std::set<paddle::Tensor*> GetTensorsFromPyObject(PyObject* obj) {
  std::set<paddle::Tensor*> result;
  if (obj == nullptr) {
    return result;
  }
  if (PyCheckTensor(obj)) {
    result.insert(&reinterpret_cast<TensorObject*>(obj)->tensor);  // NOLINT
  } else if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    for (Py_ssize_t i = 0; i < len; i++) {
      if (PyCheckTensor(PyList_GetItem(obj, i))) {
        result.insert(
            &reinterpret_cast<TensorObject*>(PyList_GetItem(obj, i))  // NOLINT
                 ->tensor);
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    for (Py_ssize_t i = 0; i < len; i++) {
      if (PyCheckTensor(PyTuple_GetItem(obj, i))) {
        result.insert(
            &reinterpret_cast<TensorObject*>(PyTuple_GetItem(obj, i))  // NOLINT
                 ->tensor);
      }
    }
  }
  return result;
}

PyObject* PyLayerNew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = reinterpret_cast<PyLayerObject*>(obj);
    v->materialize_grads = true;
    v->container_be_packed = false;
    new (&v->grad_node) std::weak_ptr<egr::GradNodePyLayer>();
    new (&v->forward_input_tensor_is_duplicable) std::vector<bool>();
    new (&v->forward_output_tensor_is_duplicable) std::vector<bool>();
  }
  return obj;
}

static void PyLayerDealloc(PyLayerObject* self) {
  if (self->container) {
    Py_DECREF(self->container);
  }
  if (self->non_differentiable) {
    Py_DECREF(self->non_differentiable);
  }
  if (self->not_inplace_tensors) {
    Py_DECREF(self->not_inplace_tensors);
  }
  self->grad_node.~weak_ptr<egr::GradNodePyLayer>();
  self->unpack_hook = nullptr;
  self->forward_input_tensor_is_duplicable.~vector();
  self->forward_output_tensor_is_duplicable.~vector();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* pylayer_method_name(PyObject* self, PyObject* noargs) {
  EAGER_TRY
  return ToPyObject(
      reinterpret_cast<PyLayerObject*>(self)->grad_node.lock()->name());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* new_tensor_with_impl(paddle::Tensor* tensor) {
  PyObject* obj = p_tensor_type->tp_alloc(p_tensor_type, 0);
  if (obj) {
    auto v = reinterpret_cast<TensorObject*>(obj);
    new (&(v->tensor)) paddle::Tensor();
    v->tensor.set_impl(tensor->impl());
    v->tensor.set_name(egr::Controller::Instance().GenerateUniqueName());
    egr::EagerUtils::autograd_meta(&v->tensor)
        ->SetStopGradient(
            egr::EagerUtils::autograd_meta(tensor)->StopGradient());
  } else {
    PADDLE_THROW(
        common::errors::Fatal("tp_alloc return null, can not new a PyObject."));
  }
  return obj;
}

PyObject* pylayer_method_apply(PyObject* cls,
                               PyObject* args,
                               PyObject* kwargs) {
  EAGER_TRY
  SetPythonStack();
  VLOG(6) << "Begin run PyLayer apply...";
  PyObject* backward_function =
      PyObject_GetAttrString(cls, "_backward_function");
  if (!backward_function) {
    PADDLE_THROW(
        common::errors::InvalidArgument("Get _backward_function failed."));
  }
  PyLayerObject* ctx = reinterpret_cast<PyLayerObject*>(
      PyObject_CallFunctionObjArgs(backward_function, nullptr));
  if (!ctx) {
    PADDLE_THROW(
        common::errors::External(pybind11::detail::error_string().c_str()));
    return nullptr;
  }
  VLOG(6) << "PyLayer construct PyLayerContext finish...";

  bool require_any_grad = false;

  size_t inputs_size = 0;
  size_t args_size = 0;
  size_t kwargs_size = 0;
  PyObject* forward_args = nullptr;
  PyObject* kwargs_value_list = nullptr;
  if (kwargs) {
    kwargs_size = PyDict_Size(kwargs);
    kwargs_value_list = PyDict_Values(kwargs);
  }
  if (args) {
    args_size = PyTuple_GET_SIZE(args);
  }
  inputs_size = kwargs_size + args_size;
  forward_args = PyTuple_New(args_size + 1);  // NOLINT
  Py_INCREF(ctx);
  PyTuple_SET_ITEM(forward_args, 0, reinterpret_cast<PyObject*>(ctx));

  std::vector<std::vector<egr::AutogradMeta*>> inputs_autograd_meta;
  inputs_autograd_meta.reserve(inputs_size);
  std::vector<std::vector<paddle::Tensor*>> inputs_tensor;
  inputs_tensor.reserve(inputs_size);
  ctx->forward_input_tensor_is_duplicable.clear();
  ctx->forward_input_tensor_is_duplicable.reserve(inputs_size);
  std::set<phi::TensorBase*> input_tensorbases;

  const phi::distributed::ProcessMesh* mesh = nullptr;
  for (size_t i = 0; i < inputs_size; i++) {
    PyObject* obj = nullptr;
    if (i >= args_size) {
      obj = PyList_GetItem(kwargs_value_list, i - args_size);  // NOLINT
    } else {
      obj = PyTuple_GET_ITEM(args, i);
    }
    if (PyCheckTensor(obj)) {
      paddle::Tensor& tensor = reinterpret_cast<TensorObject*>(obj)->tensor;
      if (tensor.defined() && tensor.is_dist_tensor()) {
        mesh = &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                     tensor.impl())
                     ->dist_attr()
                     .process_mesh());
      }
    } else if (PyList_Check(obj)) {
      Py_ssize_t len = PyList_Size(obj);
      for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* o = PyList_GetItem(obj, j);
        if (PyCheckTensor(o)) {
          paddle::Tensor& tensor = reinterpret_cast<TensorObject*>(o)->tensor;
          if (tensor.defined() && tensor.is_dist_tensor()) {
            mesh = &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                         tensor.impl())
                         ->dist_attr()
                         .process_mesh());
          }
        }
      }
    } else if (PyTuple_Check(obj)) {
      Py_ssize_t len = PyTuple_Size(obj);
      for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* o = PyTuple_GetItem(obj, j);
        if (PyCheckTensor(o)) {
          paddle::Tensor& tensor = reinterpret_cast<TensorObject*>(o)->tensor;
          if (tensor.defined() && tensor.is_dist_tensor()) {
            mesh = &(std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                         tensor.impl())
                         ->dist_attr()
                         .process_mesh());
          }
        }
      }
    }
  }

  for (size_t i = 0; i < inputs_size; i++) {
    PyObject* obj = nullptr;
    if (i >= args_size) {
      obj = PyList_GetItem(kwargs_value_list, i - args_size);  // NOLINT
    } else {
      obj = PyTuple_GET_ITEM(args, i);
    }
    if (PyCheckTensor(obj)) {
      if (mesh) {
        ConvertToDistTensor(&(reinterpret_cast<TensorObject*>(obj)->tensor),
                            mesh);
      }
      input_tensorbases.insert(
          reinterpret_cast<TensorObject*>(obj)->tensor.impl().get());
      auto autograd_meta = egr::EagerUtils::nullable_autograd_meta(
          reinterpret_cast<TensorObject*>(obj)->tensor);
      inputs_autograd_meta.push_back({autograd_meta});
      inputs_tensor.push_back(
          {&(reinterpret_cast<TensorObject*>(obj)->tensor)});  // NOLINT
      bool stop_gradient =
          autograd_meta == nullptr ? true : autograd_meta->StopGradient();
      if (!stop_gradient) {
        require_any_grad = true;
      }
      ctx->forward_input_tensor_is_duplicable.push_back(false);
    } else if (PyList_Check(obj)) {
      std::vector<paddle::Tensor*> tensors;
      Py_ssize_t len = PyList_Size(obj);
      for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* o = PyList_GetItem(obj, j);
        if (PyCheckTensor(o)) {
          if (mesh) {
            ConvertToDistTensor(&(reinterpret_cast<TensorObject*>(o)->tensor),
                                mesh);
          }
          input_tensorbases.insert(
              reinterpret_cast<TensorObject*>(o)->tensor.impl().get());
          tensors.push_back(&(reinterpret_cast<TensorObject*>(o)->tensor));
        }
      }
      if (!tensors.empty()) {
        auto autograd_meta = egr::EagerUtils::nullable_autograd_meta(tensors);
        for (auto iter : autograd_meta) {
          bool stop_gradient = iter == nullptr ? true : iter->StopGradient();
          if (!stop_gradient) {
            require_any_grad = true;
          }
        }
        inputs_autograd_meta.push_back(autograd_meta);
        inputs_tensor.push_back(tensors);
        ctx->forward_input_tensor_is_duplicable.push_back(true);
      }
    } else if (PyTuple_Check(obj)) {
      std::vector<paddle::Tensor*> tensors;
      Py_ssize_t len = PyTuple_Size(obj);
      for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* o = PyTuple_GetItem(obj, j);
        if (PyCheckTensor(o)) {
          if (mesh) {
            ConvertToDistTensor(&(reinterpret_cast<TensorObject*>(o)->tensor),
                                mesh);
          }
          input_tensorbases.insert(
              reinterpret_cast<TensorObject*>(o)->tensor.impl().get());
          tensors.push_back(&(reinterpret_cast<TensorObject*>(o)->tensor));
        }
      }
      if (!tensors.empty()) {
        auto autograd_meta = egr::EagerUtils::nullable_autograd_meta(tensors);
        for (auto iter : autograd_meta) {
          bool stop_gradient = iter == nullptr ? true : iter->StopGradient();
          if (!stop_gradient) {
            require_any_grad = true;
          }
        }
        inputs_autograd_meta.push_back(autograd_meta);
        inputs_tensor.push_back(tensors);
        ctx->forward_input_tensor_is_duplicable.push_back(true);
      }
    }

    if (i < args_size) {
      Py_INCREF(obj);
      PyTuple_SET_ITEM(forward_args, i + 1, obj);
    }
  }

  VLOG(6)
      << "PyLayer forward args is ready, begin call user's forward function...";
  // call forward
  auto forward_fn = PyObject_GetAttrString(cls, "forward");
  if (!forward_fn) {
    PADDLE_THROW(
        common::errors::InvalidArgument("Get forward function failed."));
  }
  bool trace_backward = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  auto outputs = PyObject_Call(forward_fn, forward_args, kwargs);
  egr::Controller::Instance().SetHasGrad(trace_backward);
  if (!outputs) {
    Py_XDECREF(forward_args);
    Py_XDECREF(kwargs_value_list);
    Py_XDECREF(backward_function);
    Py_XDECREF(forward_fn);
    return nullptr;
  }

  PyObject* outputs_tuple = nullptr;
  if (PyTuple_Check(outputs)) {
    outputs_tuple = outputs;
  } else if (PyList_Check(outputs)) {
    outputs_tuple = PyList_AsTuple(outputs);
  } else {
    outputs_tuple = PyTuple_New(1);
    Py_INCREF(outputs);
    PyTuple_SET_ITEM(outputs_tuple, 0, outputs);
  }

  std::set<paddle::Tensor*> inplace_tensors;
  std::set<phi::TensorBase*> not_inplace_tensorbases;
  auto not_inplace_tensors = GetTensorsFromPyObject(ctx->not_inplace_tensors);
  for (auto it : not_inplace_tensors) {
    not_inplace_tensorbases.insert(it->impl().get());
  }

  auto outputs_size = PyTuple_GET_SIZE(outputs_tuple);
  std::vector<std::vector<paddle::Tensor*>> outputs_tensor;
  outputs_tensor.reserve(outputs_size);
  std::vector<std::vector<egr::AutogradMeta*>> outputs_autograd_meta;
  outputs_autograd_meta.reserve(outputs_size);
  ctx->forward_output_tensor_is_duplicable.clear();
  ctx->forward_output_tensor_is_duplicable.reserve(outputs_size);
  for (Py_ssize_t i = 0; i < outputs_size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(outputs_tuple, i);
    if (PyCheckTensor(obj)) {
      outputs_tensor.push_back(
          {&(reinterpret_cast<TensorObject*>(obj)->tensor)});
      outputs_autograd_meta.push_back({egr::EagerUtils::autograd_meta(
          &(reinterpret_cast<TensorObject*>(obj)->tensor))});
      ctx->forward_output_tensor_is_duplicable.push_back(false);
      if (input_tensorbases.count(
              reinterpret_cast<TensorObject*>(obj)->tensor.impl().get())) {
        if (not_inplace_tensorbases.count(
                reinterpret_cast<TensorObject*>(obj)->tensor.impl().get())) {
          PyTuple_SET_ITEM(outputs_tuple,
                           i,
                           new_tensor_with_impl(&(
                               reinterpret_cast<TensorObject*>(obj)->tensor)));
        } else {
          inplace_tensors.insert(
              &(reinterpret_cast<TensorObject*>(obj)->tensor));
        }
      }
    } else if (PyList_Check(obj)) {
      std::vector<paddle::Tensor*> tensors;
      Py_ssize_t len = PyList_Size(obj);
      for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* o = PyList_GetItem(obj, j);
        if (PyCheckTensor(o)) {
          tensors.push_back(&(reinterpret_cast<TensorObject*>(o)->tensor));
          if (input_tensorbases.count(
                  reinterpret_cast<TensorObject*>(o)->tensor.impl().get())) {
            if (not_inplace_tensorbases.count(
                    reinterpret_cast<TensorObject*>(o)->tensor.impl().get())) {
              PyTuple_SetItem(obj,
                              j,
                              new_tensor_with_impl(&(
                                  reinterpret_cast<TensorObject*>(o)->tensor)));
            } else {
              inplace_tensors.insert(
                  &(reinterpret_cast<TensorObject*>(o)->tensor));
            }
          }
        }
      }
      if (!tensors.empty()) {
        outputs_tensor.push_back(tensors);
        outputs_autograd_meta.push_back(
            egr::EagerUtils::autograd_meta(&tensors));
        ctx->forward_output_tensor_is_duplicable.push_back(true);
      }
    } else if (PyTuple_Check(obj)) {
      std::vector<paddle::Tensor*> tensors;
      Py_ssize_t len = PyTuple_Size(obj);
      for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* o = PyTuple_GetItem(obj, j);
        if (PyCheckTensor(o)) {
          tensors.push_back(&(reinterpret_cast<TensorObject*>(o)->tensor));
          if (input_tensorbases.count(
                  reinterpret_cast<TensorObject*>(o)->tensor.impl().get())) {
            if (not_inplace_tensorbases.count(
                    reinterpret_cast<TensorObject*>(o)->tensor.impl().get())) {
              PyTuple_SetItem(obj,
                              j,
                              new_tensor_with_impl(&(
                                  reinterpret_cast<TensorObject*>(o)->tensor)));
            } else {
              inplace_tensors.insert(
                  &(reinterpret_cast<TensorObject*>(o)->tensor));
            }
          }
        }
      }
      if (!tensors.empty()) {
        outputs_tensor.push_back(tensors);
        outputs_autograd_meta.push_back(
            egr::EagerUtils::autograd_meta(&tensors));
        ctx->forward_output_tensor_is_duplicable.push_back(true);
      }
    }
  }

  if (outputs_tensor.empty()) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "At least one output of `PyLayer.forward` is a `Tensor`."));
  }
  VLOG(6) << "PyLayer forward function finish...";

  if (require_any_grad && trace_backward) {
    auto non_differentiable = GetTensorsFromPyObject(ctx->non_differentiable);
    for (size_t i = 0; i < outputs_autograd_meta.size(); i++) {
      for (size_t j = 0; j < outputs_autograd_meta[i].size(); j++) {
        if (non_differentiable.find(outputs_tensor[i][j]) !=
            non_differentiable.end()) {
          outputs_autograd_meta[i][j]->SetStopGradient(true);
        } else {
          outputs_autograd_meta[i][j]->SetStopGradient(false);
        }
      }
    }

    for (auto inplace_tensor : inplace_tensors) {
      auto inplace_tensor_autograd_meta =
          egr::EagerUtils::autograd_meta(inplace_tensor);
      PADDLE_ENFORCE_EQ(!inplace_tensor_autograd_meta->StopGradient() &&
                            egr::EagerUtils::IsLeafTensor(*inplace_tensor),
                        false,
                        common::errors::InvalidArgument(
                            "Leaf Var (%s) that doesn't stop gradient "
                            "can't use inplace strategy.",
                            inplace_tensor->name()));
      inplace_tensor->bump_inplace_version();
      VLOG(3) << "Tensor(" << inplace_tensor->name()
              << ") uses Inplace Strategy.";
    }

    auto grad_node =
        std::make_shared<egr::GradNodePyLayer>(reinterpret_cast<PyObject*>(ctx),
                                               outputs_autograd_meta.size(),
                                               inputs_autograd_meta.size());
    ctx->grad_node = grad_node;

    if (ctx->materialize_grads) {
      grad_node->SaveForwardOutputsMeta(outputs_tensor);
    }

    for (size_t i = 0; i < inputs_autograd_meta.size(); i++) {
      if (ctx->forward_input_tensor_is_duplicable[i]) {
        std::vector<const paddle::Tensor*> tmp;
        for (auto t : inputs_tensor[i]) {
          tmp.push_back(t);
        }
        grad_node->SetGradOutMeta(tmp, i);
      } else {
        grad_node->SetGradOutMeta(*inputs_tensor[i][0], i);
      }
    }

    for (size_t i = 0; i < outputs_autograd_meta.size(); i++) {
      if (ctx->forward_output_tensor_is_duplicable[i]) {
        egr::EagerUtils::SetOutRankWithSlot(&outputs_autograd_meta[i], i);
        egr::EagerUtils::SetHistory(&outputs_autograd_meta[i], grad_node);
        grad_node->SetGradInMeta(outputs_tensor[i], i);
      } else {
        egr::EagerUtils::SetOutRankWithSlot(outputs_autograd_meta[i][0], i);
        egr::EagerUtils::SetHistory(outputs_autograd_meta[i][0], grad_node);
        grad_node->SetGradInMeta(*outputs_tensor[i][0], i);
      }
    }
    VLOG(6) << "PyLayer construct backward node finish...";
  }

  if (outputs_size == 1) {
    if (!PyTuple_Check(outputs) && !PyList_Check(outputs)) {
      Py_XDECREF(outputs);
      outputs = PyTuple_GetItem(outputs_tuple, 0);
      Py_INCREF(outputs);
      Py_XDECREF(outputs_tuple);
    }
  }

  if (PyList_Check(outputs)) {
    Py_XDECREF(outputs_tuple);
  }

  Py_XDECREF(forward_args);
  Py_XDECREF(kwargs_value_list);
  Py_XDECREF(backward_function);
  Py_XDECREF(forward_fn);
  Py_XDECREF(ctx);

  return outputs;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* call_unpack_hook(PyLayerObject* self) {
  auto unpack_hook = self->unpack_hook;
  auto packed_value = self->container;

  auto packed_value_size = PyTuple_GET_SIZE(packed_value);
  auto unpacked_value = PyTuple_New(packed_value_size);

  for (Py_ssize_t i = 0; i < packed_value_size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(packed_value, i);
    if (PyList_Check(obj)) {
      Py_ssize_t len = PyList_Size(obj);
      auto tmp_list = PyList_New(len);
      for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* o = PyList_GetItem(obj, j);
        PyTuple_SET_ITEM(tmp_list,
                         j,
                         reinterpret_cast<PyObject*>(((*unpack_hook)(
                             reinterpret_cast<void*>(o), nullptr))));
      }
      PyTuple_SET_ITEM(unpacked_value, i, tmp_list);
    } else if (PyTuple_Check(obj)) {
      Py_ssize_t len = PyTuple_Size(obj);
      auto tmp_tuple = PyTuple_New(len);
      for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* o = PyTuple_GetItem(obj, j);
        PyTuple_SET_ITEM(tmp_tuple,
                         j,
                         reinterpret_cast<PyObject*>((*unpack_hook)(
                             reinterpret_cast<void*>(o), nullptr)));
      }
      PyTuple_SET_ITEM(unpacked_value, i, tmp_tuple);
    } else {
      PyTuple_SET_ITEM(unpacked_value,
                       i,
                       reinterpret_cast<PyObject*>((*unpack_hook)(
                           reinterpret_cast<void*>(obj), nullptr)));
    }
  }

  return unpacked_value;
}

PyObject* tensor_properties_get_container(PyLayerObject* self, void* closure) {
  EAGER_TRY
  if (self->container == nullptr) {
    RETURN_PY_NONE;
  }
  if (self->container_be_packed) {
    return call_unpack_hook(self);
  } else {
    Py_INCREF(self->container);
    return self->container;
  }
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

void call_pack_hook(PyLayerObject* self, PyObject* value) {
  PyObject* saved_value = nullptr;
  if (PyTuple_Check(value)) {
    saved_value = value;
  } else if (PyList_Check(value)) {
    saved_value = PyList_AsTuple(value);
  } else {
    saved_value = PyTuple_New(1);
    Py_INCREF(value);
    PyTuple_SET_ITEM(saved_value, 0, value);
  }

  auto pack_hook = egr::SavedTensorsHooks::GetInstance().GetPackHook();
  self->unpack_hook = egr::SavedTensorsHooks::GetInstance().GetUnPackHook();

  auto saved_value_size = PyTuple_GET_SIZE(saved_value);
  PyObject* packed_value = PyTuple_New(saved_value_size);

  for (Py_ssize_t i = 0; i < saved_value_size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(saved_value, i);
    if (PyCheckTensor(obj)) {
      PyTuple_SET_ITEM(packed_value,
                       i,
                       reinterpret_cast<PyObject*>(
                           (*pack_hook)(reinterpret_cast<void*>(obj))));
    } else if (PyList_Check(obj)) {
      Py_ssize_t len = PyList_Size(obj);
      auto tmp_list = PyList_New(len);
      for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* o = PyList_GetItem(obj, j);
        if (PyCheckTensor(o)) {
          PyTuple_SET_ITEM(tmp_list,
                           j,
                           reinterpret_cast<PyObject*>(
                               (*pack_hook)(reinterpret_cast<void*>(o))));
        } else if (o == Py_None) {
          PyTuple_SET_ITEM(tmp_list,
                           j,
                           reinterpret_cast<PyObject*>(
                               (*pack_hook)(reinterpret_cast<void*>(o))));
        } else {
          PADDLE_THROW(common::errors::InvalidArgument(
              "save_for_backward only support Tensor, list of Tensor, tuple of "
              "Tensor."));
        }
      }
      PyTuple_SET_ITEM(packed_value, i, tmp_list);
    } else if (PyTuple_Check(obj)) {
      Py_ssize_t len = PyTuple_Size(obj);
      auto tmp_tuple = PyTuple_New(len);
      for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* o = PyTuple_GetItem(obj, j);
        if (PyCheckTensor(o)) {
          PyTuple_SET_ITEM(tmp_tuple,
                           j,
                           reinterpret_cast<PyObject*>(
                               (*pack_hook)(reinterpret_cast<void*>(o))));
        } else if (o == Py_None) {
          PyTuple_SET_ITEM(tmp_tuple,
                           j,
                           reinterpret_cast<PyObject*>(
                               (*pack_hook)(reinterpret_cast<void*>(o))));
        } else {
          PADDLE_THROW(common::errors::InvalidArgument(
              "save_for_backward only support Tensor, list of Tensor, tuple of "
              "Tensor."));
        }
      }
      PyTuple_SET_ITEM(packed_value, i, tmp_tuple);
    } else if (obj == Py_None) {
      PyTuple_SET_ITEM(packed_value,
                       i,
                       reinterpret_cast<PyObject*>(
                           (*pack_hook)(reinterpret_cast<void*>(obj))));
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "save_for_backward only support Tensor, list of Tensor, tuple of "
          "Tensor."));
    }
  }

  if (PyTuple_Check(value)) {
    Py_XDECREF(saved_value);
  }

  Py_XDECREF(self->container);
  self->container = packed_value;
  self->container_be_packed = true;
}

int tensor_properties_set_container(PyLayerObject* self,
                                    PyObject* value,
                                    void* closure) {
  EAGER_TRY
  if (egr::SavedTensorsHooks::GetInstance().IsEnable()) {
    call_pack_hook(self, value);
  } else {
    Py_XINCREF(value);
    Py_XDECREF(self->container);
    self->container = value;
  }
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

PyObject* tensor_properties_get_non_differentiable(PyLayerObject* self,
                                                   void* closure) {
  EAGER_TRY
  if (self->non_differentiable == nullptr) {
    RETURN_PY_NONE;
  }
  Py_INCREF(self->non_differentiable);
  return self->non_differentiable;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_non_differentiable(PyLayerObject* self,
                                             PyObject* value,
                                             void* closure) {
  EAGER_TRY
  Py_XINCREF(value);
  Py_XDECREF(self->non_differentiable);
  self->non_differentiable = value;
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

PyObject* tensor_properties_get_not_inplace_tensors(PyLayerObject* self,
                                                    void* closure) {
  EAGER_TRY
  if (self->not_inplace_tensors == nullptr) {
    RETURN_PY_NONE;
  }
  Py_INCREF(self->not_inplace_tensors);
  return self->not_inplace_tensors;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_not_inplace_tensors(PyLayerObject* self,
                                              PyObject* value,
                                              void* closure) {
  EAGER_TRY
  Py_XINCREF(value);
  Py_XDECREF(self->not_inplace_tensors);
  self->not_inplace_tensors = value;
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

int tensor_properties_set_materialize_grads(PyLayerObject* self,
                                            PyObject* value,
                                            void* closure) {
  EAGER_TRY
  self->materialize_grads = CastPyArg2AttrBoolean(value, 0);
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

PyMethodDef pylayer_methods[] = {{"name",  // NOLINT
                                  (PyCFunction)(void (*)())pylayer_method_name,
                                  METH_NOARGS,
                                  nullptr},
                                 {"apply",
                                  (PyCFunction)(void (*)())pylayer_method_apply,
                                  METH_CLASS | METH_VARARGS | METH_KEYWORDS,
                                  nullptr},
                                 {nullptr, nullptr, 0, nullptr}};

struct PyGetSetDef pylayer_properties[] {  // NOLINT
  {"container",
   (getter)tensor_properties_get_container,
   (setter)tensor_properties_set_container,
   nullptr,
   nullptr},
      {"non_differentiable",
       (getter)tensor_properties_get_non_differentiable,
       (setter)tensor_properties_set_non_differentiable,
       nullptr,
       nullptr},
      {"not_inplace_tensors",
       (getter)tensor_properties_get_not_inplace_tensors,
       (setter)tensor_properties_set_not_inplace_tensors,
       nullptr,
       nullptr},
      {"materialize_grads",
       nullptr,
       (setter)tensor_properties_set_materialize_grads,
       nullptr,
       nullptr},
  {
    nullptr, nullptr, nullptr, nullptr, nullptr
  }
};

void BindEagerPyLayer(PyObject* module) {
  auto heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  heap_type->ht_name = ToPyObject("PyLayer");
  heap_type->ht_qualname = ToPyObject("PyLayer");
  auto type = &heap_type->ht_type;
  type->tp_name = "PyLayer";
  type->tp_basicsize = sizeof(PyLayerObject);
  type->tp_dealloc = (destructor)PyLayerDealloc;
  type->tp_methods = pylayer_methods;
  type->tp_getset = pylayer_properties;
  type->tp_new = (newfunc)PyLayerNew;
  Py_INCREF(&PyBaseObject_Type);
  type->tp_base = reinterpret_cast<PyTypeObject*>(&PyBaseObject_Type);
  type->tp_flags |=
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;  // NOLINT
#if PY_VERSION_HEX >= 0x03050000
  type->tp_as_async = &heap_type->as_async;
#endif
  p_pylayer_type = type;

  if (PyType_Ready(type) < 0) {
    PADDLE_THROW(
        common::errors::Fatal("Init Paddle error in BindEager(PyType_Ready)."));
    return;
  }

  Py_INCREF(type);
  if (PyModule_AddObject(module, "PyLayer", reinterpret_cast<PyObject*>(type)) <
      0) {
    Py_DECREF(type);
    Py_DECREF(module);
    PADDLE_THROW(common::errors::Fatal(
        "Init Paddle error in BindEager(PyModule_AddObject)."));
    return;
  }
}

}  // namespace paddle::pybind
