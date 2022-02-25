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
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "pybind11/detail/internals.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

typedef struct {
  PyObject_HEAD

      PyObject* needs_input_grad;
  PyObject* to_save;
  PyObject* non_differentiable;
  PyObject* dirty_tensors;
  bool materialize_grads;
} PyLayerObject;

PyTypeObject* p_pylayer_type;

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
  if (self->needs_input_grad) {
    Py_DECREF(self->needs_input_grad);
  }
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

struct PyGetSetDef pylayer_properties[1];

PyMethodDef pylayer_methods[1];

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
