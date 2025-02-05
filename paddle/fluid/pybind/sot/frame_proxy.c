/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/sot/frame_proxy.h"
#include "paddle/fluid/pybind/sot/macros.h"

#if SOT_IS_SUPPORTED
#include <Python.h>

#if PY_3_11_PLUS

#define DECLARE_PROXY_PROPERTY(name)                        \
  static PyObject *PyInterpreterFrameProxy_property_##name( \
      PyInterpreterFrameProxy *self, void *closure) {       \
    Py_XINCREF(self->frame->name);                          \
    return (PyObject *)self->frame->name;                   \
  }

// clang-format off
#define REGISTER_PROXY_PROPERTY(property_name, func_name) \
  { #property_name, (getter)PyInterpreterFrameProxy_property_##func_name, NULL, NULL, NULL }
// clang-format on

#if PY_3_13_PLUS
DECLARE_PROXY_PROPERTY(f_executable)
#else
DECLARE_PROXY_PROPERTY(f_code)
#endif
#if PY_3_13_PLUS
static PyObject *PyInterpreterFrameProxy_property_f_locals(
    PyInterpreterFrameProxy *self, void *closure) {
  Py_XINCREF(self->locals);
  return self->locals;
}
#else
DECLARE_PROXY_PROPERTY(f_locals)
#endif
DECLARE_PROXY_PROPERTY(f_globals)
DECLARE_PROXY_PROPERTY(f_builtins)

// Refer to
// https://github.com/python/cpython/blob/9414ddf91898892f3f6a672ae946931ee4b3ceb7/Objects/frameobject.c#L953-L961
static PyObject *PyInterpreterFrameProxy_method_repr(
    PyInterpreterFrameProxy *self) {
#if PY_3_13_PLUS
  int lineno = Internal_PyUnstable_InterpreterFrame_GetLine(self->frame);
#else
  int lineno = Internal_PyInterpreterFrame_GetLine(self->frame);
#endif
  PyCodeObject *code = PyFrame_GET_CODE(self->frame);
  return PyUnicode_FromFormat(
      "<PyInterpreterFrameProxy at %p, file %R, line %d, code %S>",
      self,
      code->co_filename,
      lineno,
      code->co_name);
}

static PyGetSetDef PyInterpreterFrameProxy_properties[] = {
#if PY_3_13_PLUS
    REGISTER_PROXY_PROPERTY(f_code, f_executable),
#else
    REGISTER_PROXY_PROPERTY(f_code, f_code),
#endif
    REGISTER_PROXY_PROPERTY(f_locals, f_locals),
    REGISTER_PROXY_PROPERTY(f_globals, f_globals),
    REGISTER_PROXY_PROPERTY(f_builtins, f_builtins),
    {NULL} /* Sentinel */
};

// clang-format off
static PyTypeObject PyInterpreterFrameProxyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "paddle.framework.core.PyInterpreterFrameProxy",
    .tp_doc = PyDoc_STR("A proxy object for _PyInterpreterFrame, "
                        "it's only define all properties we need."),
    .tp_repr = (reprfunc)PyInterpreterFrameProxy_method_repr,
    .tp_basicsize = sizeof(PyInterpreterFrameProxy),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = PyInterpreterFrameProxy_properties,
};
// clang-format on

PyInterpreterFrameProxy *PyInterpreterFrameProxy_New(
    _PyInterpreterFrame *frame) {
  PyTypeObject *type = &PyInterpreterFrameProxyType;
  PyInterpreterFrameProxy *self =
      (PyInterpreterFrameProxy *)type->tp_alloc(type, 0);
  if (!self) {
    // VLOG(7) << "Failed to allocate PyInterpreterFrameProxy";
    return NULL;
  }
  self->frame = frame;
#if PY_3_13_PLUS
  self->locals = NULL;
#endif
  return self;
}

PyMODINIT_FUNC PyInit__frame_proxy() {
  if (PyType_Ready(&PyInterpreterFrameProxyType) < 0) {
    // VLOG(7) << "PyInterpreterFrameProxyType has not been ready!";
  }
  Py_INCREF(&PyInterpreterFrameProxyType);
  return NULL;
}

#endif

#endif
