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

#include <set>
#include <string>
#include <vector>

#pragma GCC diagnostic ignored "-Wattributes"
#include "pybind11/pytypes.h"

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/pylayer/py_layer_node.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "pybind11/detail/internals.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

PyTypeObject* p_pylayer_type;
extern PyTypeObject* p_tensor_type;

std::set<paddle::experimental::Tensor*> GetNonDifferentiableNames(
    PyObject* obj) {
  std::set<paddle::experimental::Tensor*> result;
  if (obj == nullptr) {
    return result;
  }
  if (IsEagerTensor(obj)) {
    result.insert(&reinterpret_cast<TensorObject*>(obj)->tensor);  // NOLINT
  } else if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    for (Py_ssize_t i = 0; i < len; i++) {
      if (IsEagerTensor(PyList_GetItem(obj, i))) {
        result.insert(
            &reinterpret_cast<TensorObject*>(PyList_GetItem(obj, i))  // NOLINT
                 ->tensor);
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    for (Py_ssize_t i = 0; i < len; i++) {
      if (IsEagerTensor(PyTuple_GetItem(obj, i))) {
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
  if (self->dirty_tensors) {
    Py_DECREF(self->dirty_tensors);
  }
  self->grad_node.~weak_ptr<egr::GradNodePyLayer>();
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

PyObject* pylayer_method_apply(PyObject* cls, PyObject* args,
                               PyObject* kwargs) {
  EAGER_TRY
  VLOG(6) << "Begin run PyLayer apply...";
  PyObject* backward_function =
      PyObject_GetAttrString(cls, "_backward_function");
  if (!backward_function) {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Get _backward_function faild."));
  }
  PyLayerObject* ctx = reinterpret_cast<PyLayerObject*>(
      PyObject_CallFunctionObjArgs(backward_function, nullptr));
  if (!ctx) {
    return nullptr;
  }
  VLOG(6) << "PyLayer construct PyLayerContext finish...";

  bool require_any_grad = false;

  size_t inputs_size = 0;
  PyObject* forward_args = nullptr;
  PyObject* kwargs_value_list = nullptr;
  if (kwargs) {
    inputs_size = PyDict_Size(kwargs);
    kwargs_value_list = PyDict_Values(kwargs);
    forward_args = PyTuple_New(1);
  } else {
    inputs_size = PyTuple_GET_SIZE(args);
    forward_args = PyTuple_New(inputs_size + 1);
  }
  Py_INCREF(ctx);
  PyTuple_SET_ITEM(forward_args, 0, reinterpret_cast<PyObject*>(ctx));

  std::vector<std::vector<egr::AutogradMeta*>> inputs_autograd_meta;
  inputs_autograd_meta.reserve(inputs_size);
  std::vector<std::vector<paddle::experimental::Tensor*>> inputs_tensor;
  inputs_tensor.reserve(inputs_size);
  ctx->forward_input_tensor_is_duplicable.clear();
  ctx->forward_input_tensor_is_duplicable.reserve(inputs_size);
  for (size_t i = 0; i < inputs_size; i++) {
    PyObject* obj = nullptr;
    if (kwargs) {
      obj = PyList_GetItem(kwargs_value_list, i);
    } else {
      obj = PyTuple_GET_ITEM(args, i);
    }
    if (IsEagerTensor(obj)) {
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
      std::vector<paddle::experimental::Tensor*> tensors;
      Py_ssize_t len = PyList_Size(obj);
      for (Py_ssize_t i = 0; i < len; i++) {
        if (IsEagerTensor(PyList_GetItem(obj, i))) {
          tensors.push_back(&(
              reinterpret_cast<TensorObject*>(PyList_GetItem(obj, i))->tensor));
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
      std::vector<paddle::experimental::Tensor*> tensors;
      Py_ssize_t len = PyTuple_Size(obj);
      for (Py_ssize_t i = 0; i < len; i++) {
        if (IsEagerTensor(PyTuple_GetItem(obj, i))) {
          tensors.push_back(
              &(reinterpret_cast<TensorObject*>(PyTuple_GetItem(obj, i))
                    ->tensor));
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

    if (!kwargs) {
      Py_INCREF(obj);
      PyTuple_SET_ITEM(forward_args, i + 1, obj);
    }
  }

  VLOG(6)
      << "PyLayer forward args is ready, begin call user's forward function...";
  // call forward
  auto forward_fn = PyObject_GetAttrString(cls, "forward");
  if (!forward_fn) {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Get forward function faild."));
  }
  bool trace_backward = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  auto outputs = PyObject_Call(forward_fn, forward_args, kwargs);
  egr::Controller::Instance().SetHasGrad(trace_backward);
  if (!outputs) {
    return nullptr;
  }

  PyObject* outputs_tuple = nullptr;
  if (PyTuple_Check(outputs)) {
    outputs_tuple = outputs;
  } else {
    outputs_tuple = PyTuple_New(1);
    Py_INCREF(outputs);
    PyTuple_SET_ITEM(outputs_tuple, 0, outputs);
  }

  auto outputs_size = PyTuple_GET_SIZE(outputs_tuple);
  std::vector<std::vector<paddle::experimental::Tensor*>> outputs_tensor;
  outputs_tensor.reserve(outputs_size);
  std::vector<std::vector<egr::AutogradMeta*>> outputs_autograd_meta;
  outputs_autograd_meta.reserve(outputs_size);
  ctx->forward_output_tensor_is_duplicable.clear();
  ctx->forward_output_tensor_is_duplicable.reserve(outputs_size);
  for (Py_ssize_t i = 0; i < outputs_size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(outputs_tuple, i);
    if (IsEagerTensor(obj)) {
      outputs_tensor.push_back(
          {&(reinterpret_cast<TensorObject*>(obj)->tensor)});
      outputs_autograd_meta.push_back({egr::EagerUtils::autograd_meta(
          &(reinterpret_cast<TensorObject*>(obj)->tensor))});
      ctx->forward_output_tensor_is_duplicable.push_back(false);
    } else if (PyList_Check(obj)) {
      std::vector<paddle::experimental::Tensor*> tensors;
      Py_ssize_t len = PyList_Size(obj);
      for (Py_ssize_t i = 0; i < len; i++) {
        if (IsEagerTensor(PyList_GetItem(obj, i))) {
          tensors.push_back(&(
              reinterpret_cast<TensorObject*>(PyList_GetItem(obj, i))->tensor));
        }
      }
      if (!tensors.empty()) {
        outputs_tensor.push_back(tensors);
        outputs_autograd_meta.push_back(
            egr::EagerUtils::autograd_meta(&tensors));
        ctx->forward_output_tensor_is_duplicable.push_back(true);
      }
    } else if (PyTuple_Check(obj)) {
      std::vector<paddle::experimental::Tensor*> tensors;
      Py_ssize_t len = PyTuple_Size(obj);
      for (Py_ssize_t i = 0; i < len; i++) {
        if (IsEagerTensor(PyTuple_GetItem(obj, i))) {
          tensors.push_back(
              &(reinterpret_cast<TensorObject*>(PyTuple_GetItem(obj, i))
                    ->tensor));
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

  if (outputs_tensor.size() == 0) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "At least one output of `PyLayer.forward` is a `Tensor`."));
  }
  VLOG(6) << "PyLayer forward function finish...";

  if (require_any_grad && trace_backward) {
    auto non_differentiable =
        GetNonDifferentiableNames(ctx->non_differentiable);
    for (size_t i = 0; i < outputs_autograd_meta.size(); i++) {
      for (size_t j = 0; j < outputs_autograd_meta[i].size(); j++) {
        if (non_differentiable.find(outputs_tensor[i][j]) !=
            non_differentiable.end()) {
          outputs_autograd_meta[i][j]->SetStopGradient(true);
        } else {
          outputs_autograd_meta[i][j]->WeakSetStopGradient(false);
        }
      }
    }

    // TODO(pangyoki) add inplace, inplaced tensor is ctx->dirty_tensors

    auto grad_node = std::make_shared<egr::GradNodePyLayer>(
        reinterpret_cast<PyObject*>(ctx), outputs_autograd_meta.size(),
        inputs_autograd_meta.size());
    ctx->grad_node = grad_node;

    if (ctx->materialize_grads) {
      grad_node->SaveForwardOutputsMeta(outputs_tensor);
    }

    for (size_t i = 0; i < inputs_autograd_meta.size(); i++) {
      if (ctx->forward_input_tensor_is_duplicable[i]) {
        for (auto t : inputs_tensor[i]) {
          grad_node->SetGradOutMeta(*t, i);
        }
        grad_node->AddEdges(&inputs_autograd_meta[i], i);
      } else {
        grad_node->SetGradOutMeta(*inputs_tensor[i][0], i);
        grad_node->AddEdges(inputs_autograd_meta[i][0], i);
      }
    }

    for (size_t i = 0; i < outputs_autograd_meta.size(); i++) {
      if (ctx->forward_output_tensor_is_duplicable[i]) {
        egr::EagerUtils::SetOutRankWithSlot(&outputs_autograd_meta[i], i);
        egr::EagerUtils::SetHistory(&outputs_autograd_meta[i], grad_node);
        for (auto t : outputs_tensor[i]) {
          grad_node->SetGradInMeta(*t, i);
        }
        egr::EagerUtils::CheckAndRetainGrad(outputs_tensor[i]);
      } else {
        egr::EagerUtils::SetOutRankWithSlot(outputs_autograd_meta[i][0], i);
        egr::EagerUtils::SetHistory(outputs_autograd_meta[i][0], grad_node);
        grad_node->SetGradInMeta(*outputs_tensor[i][0], i);
        egr::EagerUtils::CheckAndRetainGrad(*outputs_tensor[i][0]);
      }
    }
    VLOG(6) << "PyLayer construct backward node finish...";
  }

  return outputs;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* pylayer_method_register_hook(PyObject* _self, PyObject* hook) {
  EAGER_TRY
  return nullptr;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* tensor_properties_get_container(PyLayerObject* self, void* closure) {
  EAGER_TRY
  if (self->container == nullptr) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  Py_INCREF(self->container);
  return self->container;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_container(PyLayerObject* self, PyObject* value,
                                    void* closure) {
  EAGER_TRY
  Py_XINCREF(value);
  Py_XDECREF(self->container);
  self->container = value;
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_ZERO
}

PyObject* tensor_properties_get_non_differentiable(PyLayerObject* self,
                                                   void* closure) {
  EAGER_TRY
  if (self->non_differentiable == nullptr) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  Py_INCREF(self->non_differentiable);
  return self->non_differentiable;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_non_differentiable(PyLayerObject* self,
                                             PyObject* value, void* closure) {
  EAGER_TRY
  Py_XINCREF(value);
  Py_XDECREF(self->non_differentiable);
  self->non_differentiable = value;
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_ZERO
}

PyObject* tensor_properties_get_dirty_tensors(PyLayerObject* self,
                                              void* closure) {
  EAGER_TRY
  if (self->dirty_tensors == nullptr) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  Py_INCREF(self->dirty_tensors);
  return self->dirty_tensors;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_dirty_tensors(PyLayerObject* self, PyObject* value,
                                        void* closure) {
  EAGER_TRY
  Py_XINCREF(value);
  Py_XDECREF(self->dirty_tensors);
  self->dirty_tensors = value;
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_ZERO
}

int tensor_properties_set_materialize_grads(PyLayerObject* self,
                                            PyObject* value, void* closure) {
  EAGER_TRY
  self->materialize_grads = CastPyArg2AttrBoolean(value, 0);
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_ZERO
}

PyMethodDef pylayer_methods[] = {
    {"name", (PyCFunction)(void (*)(void))pylayer_method_name, METH_NOARGS,
     NULL},
    {"apply", (PyCFunction)(void (*)(void))pylayer_method_apply,
     METH_CLASS | METH_VARARGS | METH_KEYWORDS, NULL},
    {"register_hook", (PyCFunction)(void (*)(void))pylayer_method_register_hook,
     METH_O, NULL},
    {NULL, NULL, 0, NULL}};

struct PyGetSetDef pylayer_properties[]{
    {"container", (getter)tensor_properties_get_container,
     (setter)tensor_properties_set_container, nullptr, nullptr},
    {"non_differentiable", (getter)tensor_properties_get_non_differentiable,
     (setter)tensor_properties_set_non_differentiable, nullptr, nullptr},
    {"dirty_tensors", (getter)tensor_properties_get_dirty_tensors,
     (setter)tensor_properties_set_dirty_tensors, nullptr, nullptr},
    {"materialize_grads", nullptr,
     (setter)tensor_properties_set_materialize_grads, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

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
  type->tp_new = PyLayerNew;
  Py_INCREF(&PyBaseObject_Type);
  type->tp_base = reinterpret_cast<PyTypeObject*>(&PyBaseObject_Type);
  type->tp_flags |=
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
#if PY_VERSION_HEX >= 0x03050000
  type->tp_as_async = &heap_type->as_async;
#endif
  p_pylayer_type = type;

  if (PyType_Ready(type) < 0) {
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle error in BindEager(PyType_Ready)."));
    return;
  }

  Py_INCREF(type);
  if (PyModule_AddObject(module, "PyLayer", reinterpret_cast<PyObject*>(type)) <
      0) {
    Py_DECREF(type);
    Py_DECREF(module);
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle error in BindEager(PyModule_AddObject)."));
    return;
  }
}

}  // namespace pybind
}  // namespace paddle
