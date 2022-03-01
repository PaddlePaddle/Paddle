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

PyObject* PyLayerNew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = reinterpret_cast<PyLayerObject*>(obj);
    v->materialize_grads = true;
    // TODO(wanghuancoder) init other member
  }
  return obj;
}

static void PyLayerDealloc(PyLayerObject* self) {
  if (self->to_save) {
    Py_DECREF(self->to_save);
  }
  if (self->non_differentiable) {
    Py_DECREF(self->non_differentiable);
  }
  if (self->dirty_tensors) {
    Py_DECREF(self->dirty_tensors);
  }
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* pylayer_method_name(PyObject* self, PyObject* noargs) {
  EAGER_TRY
  return ToPyObject(
      reinterpret_cast<PyLayerObject*>(self)->grad_node.lock()->name());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyObject* pylayer_method_apply(PyObject* cls, PyObject* inputs) {
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
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Construct PyLayerContext faild."));
  }
  VLOG(6) << "PyLayer construct PyLayerContext finish...";

  bool require_any_grad = false;

  std::vector<std::vector<egr::AutogradMeta*>> inputs_autograd_meta;
  auto inputs_size = PyTuple_GET_SIZE(inputs);
  auto forward_args = PyTuple_New(inputs_size + 1);
  inputs_autograd_meta.reserve(inputs_size);
  Py_INCREF(ctx);
  PyTuple_SET_ITEM(forward_args, 0, reinterpret_cast<PyObject*>(ctx));
  for (Py_ssize_t i = 0; i < inputs_size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(inputs, i);

    if (IsEagerTensor(obj)) {
      auto autograd_meta = egr::EagerUtils::nullable_autograd_meta(
          reinterpret_cast<TensorObject*>(obj)->tensor);
      inputs_autograd_meta.push_back({autograd_meta});
      bool stop_gradient =
          autograd_meta == nullptr ? true : autograd_meta->StopGradient();
      if (!stop_gradient) {
        require_any_grad = true;
      }
    } else if (PyList_Check(obj) || PyTuple_Check(obj)) {
      auto tensors = CastPyArg2VectorOfTensor(obj, i);
      auto autograd_meta = egr::EagerUtils::nullable_autograd_meta(tensors);
      inputs_autograd_meta.push_back(autograd_meta);
    } else {
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "PyLayer forward arguements mast be Tensor or list of Tensor."));
    }
    Py_INCREF(obj);
    PyTuple_SET_ITEM(forward_args, i + 1, obj);
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
  auto outputs = PyObject_CallObject(forward_fn, forward_args);
  egr::Controller::Instance().SetHasGrad(trace_backward);
  if (!outputs) {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "forward function return a nullptr."));
  }

  VLOG(6) << "PyLayer forward function finish...";

  if (require_any_grad && trace_backward) {
    PyObject* outputs_tuple = nullptr;
    if (PyTuple_Check(outputs)) {
      outputs_tuple = outputs;
    } else {
      outputs_tuple = PyTuple_New(1);
      Py_INCREF(outputs);
      PyTuple_SET_ITEM(outputs_tuple, 0, outputs);
    }
    std::vector<std::vector<paddle::experimental::Tensor*>> outputs_tensor;
    std::vector<std::vector<egr::AutogradMeta*>> outputs_autograd_meta;
    auto outputs_size = PyTuple_GET_SIZE(outputs_tuple);
    outputs_tensor.reserve(outputs_size);
    outputs_autograd_meta.reserve(outputs_size);
    for (Py_ssize_t i = 0; i < outputs_size; i++) {
      PyObject* obj = PyTuple_GET_ITEM(outputs_tuple, i);
      if (IsEagerTensor(obj)) {
        outputs_tensor.push_back(
            {&(reinterpret_cast<TensorObject*>(obj)->tensor)});
        outputs_autograd_meta.push_back({egr::EagerUtils::autograd_meta(
            &(reinterpret_cast<TensorObject*>(obj)->tensor))});
      } else if (PyList_Check(obj) || PyTuple_Check(obj)) {
        auto tensors = GetTensorPtrListFromPyObject(obj);
        outputs_tensor.push_back(tensors);
        outputs_autograd_meta.push_back(
            egr::EagerUtils::autograd_meta(&tensors));
      } else {
        PADDLE_THROW(paddle::platform::errors::InvalidArgument(
            "PyLayer forward mast return Tensor or list of Tensor."));
      }
    }

    for (auto autograd_metas : outputs_autograd_meta) {
      for (auto autograd_meta : autograd_metas) {
        autograd_meta->WeakSetStopGradient(false);
      }
    }

    auto grad_node = std::make_shared<egr::GradNodePyLayer>(
        reinterpret_cast<PyObject*>(ctx), outputs_size, inputs_size);
    ctx->grad_node = grad_node;

    for (size_t i = 0; i < inputs_autograd_meta.size(); i++) {
      if (inputs_autograd_meta[i].size() == 1) {
        grad_node->SetGradOutMeta(inputs_autograd_meta[i][0], i);
        grad_node->AddEdges(inputs_autograd_meta[i][0], i);
      } else {
        grad_node->SetGradOutMeta(&inputs_autograd_meta[i], i);
        grad_node->AddEdges(&inputs_autograd_meta[i], i);
      }
    }

    for (size_t i = 0; i < outputs_autograd_meta.size(); i++) {
      if (outputs_autograd_meta[i].size() == 1) {
        egr::EagerUtils::SetOutRankWithSlot(outputs_autograd_meta[i][0], i);
        egr::EagerUtils::SetHistory(outputs_autograd_meta[i][0], grad_node);
        grad_node->SetGradInMeta(outputs_autograd_meta[i][0], i);
        egr::EagerUtils::CheckAndRetainGrad(*outputs_tensor[i][0]);
      } else {
        egr::EagerUtils::SetOutRankWithSlot(&outputs_autograd_meta[i], i);
        egr::EagerUtils::SetHistory(&outputs_autograd_meta[i], grad_node);
        grad_node->SetGradInMeta(&outputs_autograd_meta[i], i);
        egr::EagerUtils::CheckAndRetainGrad(outputs_tensor[i]);
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

struct PyGetSetDef pylayer_properties[1];

PyMethodDef pylayer_methods[] = {
    {"name", (PyCFunction)(void (*)(void))pylayer_method_name, METH_NOARGS,
     NULL},
    {"apply", (PyCFunction)(void (*)(void))pylayer_method_apply,
     METH_CLASS | METH_VARARGS, NULL},
    {"register_hook", (PyCFunction)(void (*)(void))pylayer_method_register_hook,
     METH_O, NULL},
    {NULL, NULL, 0, NULL}};

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
