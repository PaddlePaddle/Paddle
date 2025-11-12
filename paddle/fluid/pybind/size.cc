// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/size.h"

#include <Python.h>

#include <sstream>

#include "paddle/common/enforce.h"
#include "pybind11/pybind11.h"

namespace paddle::pybind {

extern PyTypeObject Paddle_SizeType;

PyObject* Paddle_Size_NewFromInt64Array(const int64_t* data, Py_ssize_t len) {
  PyObject* self = PyList_New(len);
  PADDLE_ENFORCE_NOT_NULL(
      self, common::errors::External("Failed to create new PyList object."));

  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PyLong_FromLongLong(data[i]);
    PADDLE_ENFORCE_NOT_NULL(
        item, common::errors::External("Failed to create PyLong object."));
    PyList_SET_ITEM(self, i, item);
  }

  reinterpret_cast<PyObject*>(self)->ob_type = &Paddle_SizeType;
  return self;
}

static PyObject* Paddle_Size_pynew(PyTypeObject* type,
                                   PyObject* args,
                                   PyObject* kwargs) {
  PyObject* self = PyList_Type.tp_new(type, args, kwargs);
  PADDLE_ENFORCE_NOT_NULL(
      self, common::errors::InvalidArgument("Failed to create new PyList."));
  Py_ssize_t n = PyList_GET_SIZE(self);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PyList_GET_ITEM(self, i);

    if (PyLong_Check(item)) continue;

    PyObject* number = PyNumber_Index(item);
    if (number && PyLong_Check(number)) {
      PyList_SetItem(self, i, number);
      continue;
    }
    Py_XDECREF(number);

    PADDLE_THROW("paddle.Size() takes an iterable of 'int' (item %zd is '%s')",
                 i,
                 Py_TYPE(item)->tp_name);

    Py_DECREF(self);
    return nullptr;
  }

  return self;
}

static PyObject* Paddle_Size_numel(PyObject* self, PyObject* Py_UNUSED(args)) {
  int64_t numel = 1;
  Py_ssize_t n = PyList_GET_SIZE(self);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PyList_GET_ITEM(self, i);
    int64_t val = PyLong_AsLongLong(item);
    if (val == -1 && PyErr_Occurred()) return nullptr;
    numel *= val;
  }
  return PyLong_FromLongLong(numel);
}

static PyMethodDef Paddle_Size_methods[] = {
    {"numel",
     Paddle_Size_numel,
     METH_NOARGS,
     "Calculates the total number of elements."},
    {nullptr, nullptr, 0, nullptr}};

static PyObject* Paddle_Size_repr(PyObject* self) {
  std::stringstream ss;
  ss << "paddle.Size([";
  Py_ssize_t n = PyList_GET_SIZE(self);
  for (Py_ssize_t i = 0; i < n; ++i) {
    if (i > 0) ss << ", ";
    PyObject* item = PyList_GET_ITEM(self, i);
    PyObject* repr = PyObject_Repr(item);
    if (!repr) return nullptr;
    ss << PyUnicode_AsUTF8(repr);
    Py_DECREF(repr);
  }
  ss << "])";
  return PyUnicode_FromString(ss.str().c_str());
}

static PyObject* Paddle_Size_subscript(PyObject* self, PyObject* key) {
  PyObject* result = PyList_Type.tp_as_mapping->mp_subscript(self, key);
  if (!result) return nullptr;
  if (PySlice_Check(key) && PyList_Check(result)) {
    result->ob_type = &Paddle_SizeType;
  }
  return result;
}

static PyObject* Paddle_Size_concat(PyObject* self, PyObject* other) {
  if (!PyList_Check(other) && !PyTuple_Check(other) &&
      !PyObject_IsInstance(other,
                           reinterpret_cast<PyObject*>(&Paddle_SizeType))) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "can only concatenate list, tuple or Size (not '%s') to Size",
        Py_TYPE(other)->tp_name));
  }

  PyObject* other_arg = other;
  bool new_list = false;
  if (PyTuple_Check(other)) {
    other_arg = PySequence_List(other);
    new_list = true;
  }

  PyObject* result = PyList_Type.tp_as_sequence->sq_concat(self, other_arg);

  if (new_list) {
    Py_DECREF(other_arg);
  }
  if (!result) return nullptr;
  result->ob_type = &Paddle_SizeType;
  return result;
}

static PyObject* Paddle_Size_repeat(PyObject* self, Py_ssize_t n) {
  PyObject* result = PyList_Type.tp_as_sequence->sq_repeat(self, n);
  if (!result) return nullptr;
  result->ob_type = &Paddle_SizeType;
  return result;
}

static PyObject* Paddle_Size_add(PyObject* left, PyObject* right) {
  if (!(PyList_Check(left) || PyTuple_Check(left)) ||
      !(PyList_Check(right) || PyTuple_Check(right))) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  // Paddle_Size_concat cannot handle tuple + Size, so convert tuple to list
  // first
  bool new_list = false;
  if (PyTuple_Check(left)) {
    left = PySequence_List(left);
    new_list = true;
  }

  PyObject* res = Paddle_Size_concat(left, right);
  if (new_list) {
    Py_DECREF(left);
  }
  return res;
}

static PyNumberMethods Paddle_Size_as_number = {
    Paddle_Size_add, /* nb_add */
    nullptr,         /* nb_subtract */
    nullptr,         /* nb_multiply */
};

static PyMappingMethods Paddle_Size_as_mapping = {
    nullptr, Paddle_Size_subscript, nullptr};

static PySequenceMethods Paddle_Size_as_sequence = {nullptr,
                                                    Paddle_Size_concat,
                                                    Paddle_Size_repeat,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr};

PyTypeObject Paddle_SizeType = {
    PyVarObject_HEAD_INIT(NULL, 0) "paddle.Size", /* tp_name */
    sizeof(PyListObject),                         /* tp_basicsize */
    0,                                            /* tp_itemsize */
    nullptr,                                      /* tp_dealloc*/
    0,                                            /* tp_vectorcall_offset */
    nullptr,                                      /* tp_getattr */
    nullptr,                                      /* tp_setattr */
    nullptr,                                      /* tp_as_async */
    Paddle_Size_repr,                             /* tp_repr */
    &Paddle_Size_as_number,                       /* tp_as_number */
    &Paddle_Size_as_sequence,                     /* tp_as_sequence */
    &Paddle_Size_as_mapping,                      /* tp_as_mapping */
    nullptr,                                      /* tp_hash */
    nullptr,                                      /* tp_call */
    nullptr,                                      /* tp_str */
    nullptr,                                      /* tp_getattro*/
    nullptr,                                      /* tp_setattro*/
    nullptr,                                      /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* tp_flags */
    nullptr,                                      /* tp_doc */
    nullptr,                                      /* tp_traverse */
    nullptr,                                      /* tp_clear */
    nullptr,                                      /* tp_richcompare */
    0,                                            /* tp_weaklistoffset */
    nullptr,                                      /* tp_iter */
    nullptr,                                      /* tp_iternext */
    Paddle_Size_methods,                          /* tp_methods */
    nullptr,                                      /* tp_members */
    nullptr,                                      /* tp_getset */
    &PyList_Type,                                 /* tp_base */
    nullptr,                                      /* tp_dict */
    nullptr,                                      /* tp_descr_get */
    nullptr,                                      /* tp_descr_set */
    0,                                            /* tp_dictoffset */
    nullptr,                                      /* tp_init */
    nullptr,                                      /* tp_alloc */
    Paddle_Size_pynew,                            /* tp_new */
};

void BindSize(pybind11::module* m) {
  if (PyType_Ready(&Paddle_SizeType) < 0) {
    PADDLE_THROW(common::errors::External("Failed to ready Paddle_SizeType"));
  }

  Py_INCREF(&Paddle_SizeType);

  m->add_object("Size", reinterpret_cast<PyObject*>(&Paddle_SizeType));
}

}  // namespace paddle::pybind
