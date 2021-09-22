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

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/function_api.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/top/api/include/tensor.h"
#include "paddle/top/core/convert_utils.h"
#include "paddle/top/core/dense_tensor.h"
#include "paddle/top/core/dtype.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

PyTypeObject* pEagerTensorType;

PyObject* eagertensor_new(PyTypeObject* type, PyObject* args,
                          PyObject* kwargs) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj == nullptr) {
    PADDLE_THROW(platform::errors::Fatal(
        "tp_alloc return null, can not new a PyObject."));
  }
  return obj;
}

static void eagertensor_dealloc(EagerTensorObject* self) {
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static int eagertensor_init(EagerTensorObject* self, PyObject* args,
                            PyObject* kwargs) {
  return 0;
}

extern struct PyGetSetDef variable_properties[];

extern PyMethodDef variable_methods[];

PyTypeObject EagerTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0) "core_avx.eager.EagerTensor", /* tp_name */
    sizeof(EagerTensorObject),       /* tp_basicsize */
    0,                               /* tp_itemsize */
    (destructor)eagertensor_dealloc, /* tp_dealloc */
    0,                               /* tp_vectorcall_offset */
    0,                               /* tp_getattr */
    0,                               /* tp_setattr */
    0,                               /* tp_reserved */
    0,                               /* tp_repr */
    0,                               /* tp_as_number */
    0,                               /* tp_as_sequence */
    0,                               /* tp_as_mapping */
    0,                               /* tp_hash  */
    0,                               /* tp_call */
    0,                               /* tp_str */
    0,                               /* tp_getattro */
    0,                               /* tp_setattro */
    0,                               /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HEAPTYPE,    /* tp_flags */
    0,                          /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    variable_methods,           /* tp_methods */
    0,                          /* tp_members */
    variable_properties,        /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)eagertensor_init, /* tp_init */
    0,                          /* tp_alloc */
    eagertensor_new,            /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0,                          /* tp_del */
    0                           /* tp_version_tag */
};

void BindEager(pybind11::module* module) {
  auto m = module->def_submodule("eager");

  pEagerTensorType = &EagerTensorType;
  if (PyType_Ready(&EagerTensorType) < 0) {
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle erroe in BindEager(PyType_Ready)."));
    return;
  }

  Py_INCREF(&EagerTensorType);
  if (PyModule_AddObject(m.ptr(), "EagerTensor",
                         reinterpret_cast<PyObject*>(&EagerTensorType)) < 0) {
    Py_DECREF(&EagerTensorType);
    Py_DECREF(m.ptr());
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle erroe in BindEager(PyModule_AddObject)."));
    return;
  }

  BindFunctions(m.ptr());
}

}  // namespace pybind
}  // namespace paddle
