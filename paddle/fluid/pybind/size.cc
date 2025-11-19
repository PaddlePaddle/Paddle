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

#include "pybind11/pybind11.h"

namespace paddle::pybind {

extern PyTypeObject Paddle_SizeType;

static const char* paddle_size_doc =
    R"DOC(The result type of a call to ``paddle.Tensor.size()``.
It describes the size of all dimensions of the original tensor. As a subclass of `list`,
it supports all common sequence operations like indexing, slicing, concatenation, etc.

Args:
    *args: Either a sequence of integers or multiple integer arguments representing dimensions.

Returns:
    Size: A special `list` subclass representing tensor dimensions.

Examples:
    .. code-block:: python

        >>> import paddle
        >>> size = paddle.Size([2, 3, 4])
        >>> print(size)
        paddle.Size([2, 3, 4])
)DOC";

static const char* paddle_size_numel_doc =
    R"DOC(Calculates the total number of elements in the Size.
It is the product of all dimensions.

Returns:
    int: The total number of elements.

Examples:
    .. code-block:: python

        >>> import paddle
        >>> size = paddle.Size([2, 3, 4])
        >>> size.numel()
        24
        >>> empty_size = paddle.Size([])
        >>> empty_size.numel()
        1
)DOC";

static const char* paddle_size_reduce_doc = R"DOC(Helper for pickling.)DOC";

PyObject* Paddle_Size_NewFromInt64Array(const int64_t* data, Py_ssize_t len) {
  PyObject* out = PyList_New(len);
  if (!out) {
    PyErr_SetString(PyExc_MemoryError, "Failed to create new PyList object.");
    return nullptr;
  }

  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PyLong_FromLongLong(data[i]);
    if (!item) {
      PyErr_SetString(PyExc_MemoryError, "Failed to create PyLong object.");
      Py_DECREF(out);
      return nullptr;
    }
    PyList_SET_ITEM(out, i, item);
  }

  reinterpret_cast<PyObject*>(out)->ob_type = &Paddle_SizeType;
  return out;
}

static int Paddle_Size_init(PyObject* self, PyObject* args, PyObject* kwargs) {
  if (PyList_Type.tp_init(self, args, kwargs) < 0) {
    return -1;
  }

  Py_ssize_t n = PyList_GET_SIZE(self);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PyList_GET_ITEM(self, i);

    if (PyLong_Check(item)) continue;

    PyObject* number = PyNumber_Index(item);
    if (number && PyLong_Check(number)) {
      if (PyList_SetItem(self, i, number) < 0) {
        Py_DECREF(number);
        return -1;
      }
      continue;
    }
    Py_XDECREF(number);

    PyErr_Format(PyExc_TypeError,
                 "paddle.Size() takes an iterable of 'int' (item %zd is '%s')",
                 i,
                 Py_TYPE(item)->tp_name);
    return -1;
  }

  return 0;
}

static PyObject* Paddle_Size_pynew(PyTypeObject* type,
                                   PyObject* args,
                                   PyObject* kwargs) {
  PyObject* self = PyList_Type.tp_new(type, args, kwargs);
  return self;
}

static PyObject* Paddle_Size_reduce(PyObject* self, PyObject* Py_UNUSED(args)) {
  PyObject* self_as_tuple = PyList_AsTuple(self);
  if (!self_as_tuple) {
    return nullptr;
  }

  PyObject* constructor_args = PyTuple_Pack(1, self_as_tuple);
  if (!constructor_args) {
    Py_DECREF(self_as_tuple);
    return nullptr;
  }

  Py_DECREF(self_as_tuple);

  PyObject* result = PyTuple_Pack(2, PyObject_Type(self), constructor_args);
  if (!result) {
    Py_DECREF(constructor_args);
    return nullptr;
  }

  Py_DECREF(constructor_args);

  return result;
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
    {"numel", Paddle_Size_numel, METH_NOARGS, paddle_size_numel_doc},
    {"__reduce__", Paddle_Size_reduce, METH_NOARGS, paddle_size_reduce_doc},
    {nullptr, nullptr, 0, nullptr}};

static PyObject* Paddle_Size_repr(PyObject* self) {
  PyObject* list_repr = PyList_Type.tp_repr(self);
  if (!list_repr) {
    return nullptr;
  }

  PyObject* result = PyUnicode_FromFormat("paddle.Size(%U)", list_repr);

  Py_DECREF(list_repr);

  return result;
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
    PyErr_Format(PyExc_TypeError,
                 "can only concatenate list, tuple or Size (not '%s') to Size",
                 Py_TYPE(other)->tp_name);
    return nullptr;
  }

  PyObject* other_arg = other;
  bool new_list = false;
  if (PyTuple_Check(other)) {
    other_arg = PySequence_List(other);
    if (!other_arg) {
      return nullptr;
    }
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
  // first.
  bool new_list = false;
  if (PyTuple_Check(left)) {
    left = PySequence_List(left);
    if (!left) {
      return nullptr;
    }
    new_list = true;
  }

  PyObject* res = Paddle_Size_concat(left, right);
  if (new_list) {
    Py_DECREF(left);
  }
  return res;
}

static PyObject* Paddle_Size_richcompare(PyObject* self,
                                         PyObject* other,
                                         int op) {
  if (!PyTuple_Check(other)) {
    if (Paddle_SizeType.tp_base && Paddle_SizeType.tp_base->tp_richcompare) {
      return Paddle_SizeType.tp_base->tp_richcompare(self, other, op);
    }
    Py_RETURN_NOTIMPLEMENTED;
  }

  if (op != Py_EQ && op != Py_NE) {
    Py_RETURN_NOTIMPLEMENTED;
  }

  Py_ssize_t self_len = PyList_GET_SIZE(self);
  Py_ssize_t other_len = PyTuple_GET_SIZE(other);

  if (self_len != other_len) {
    if (op == Py_EQ) {
      Py_RETURN_FALSE;
    } else {
      Py_RETURN_TRUE;
    }
  }

  for (Py_ssize_t i = 0; i < self_len; ++i) {
    PyObject* self_item = PyList_GET_ITEM(self, i);
    PyObject* other_item = PyTuple_GET_ITEM(other, i);

    int result = PyObject_RichCompareBool(self_item, other_item, Py_EQ);

    if (result == -1) {
      return nullptr;
    }

    if (result == 0) {
      if (op == Py_EQ) {
        Py_RETURN_FALSE;
      } else {
        Py_RETURN_TRUE;
      }
    }
  }

  if (op == Py_EQ) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
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
    PyVarObject_HEAD_INIT(NULL, 0) "paddle.base.libpaddle.Size", /* tp_name */
    sizeof(PyListObject),                     /* tp_basicsize */
    0,                                        /* tp_itemsize */
    nullptr,                                  /* tp_dealloc*/
    0,                                        /* tp_vectorcall_offset */
    nullptr,                                  /* tp_getattr */
    nullptr,                                  /* tp_setattr */
    nullptr,                                  /* tp_as_async */
    Paddle_Size_repr,                         /* tp_repr */
    &Paddle_Size_as_number,                   /* tp_as_number */
    &Paddle_Size_as_sequence,                 /* tp_as_sequence */
    &Paddle_Size_as_mapping,                  /* tp_as_mapping */
    nullptr,                                  /* tp_hash */
    nullptr,                                  /* tp_call */
    nullptr,                                  /* tp_str */
    nullptr,                                  /* tp_getattro*/
    nullptr,                                  /* tp_setattro*/
    nullptr,                                  /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    paddle_size_doc,                          /* tp_doc */
    nullptr,                                  /* tp_traverse */
    nullptr,                                  /* tp_clear */
    Paddle_Size_richcompare,                  /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    nullptr,                                  /* tp_iter */
    nullptr,                                  /* tp_iternext */
    Paddle_Size_methods,                      /* tp_methods */
    nullptr,                                  /* tp_members */
    nullptr,                                  /* tp_getset */
    &PyList_Type,                             /* tp_base */
    nullptr,                                  /* tp_dict */
    nullptr,                                  /* tp_descr_get */
    nullptr,                                  /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)Paddle_Size_init,               /* tp_init */
    nullptr,                                  /* tp_alloc */
    Paddle_Size_pynew,                        /* tp_new */
};

void BindSize(pybind11::module* m) {
  if (PyType_Ready(&Paddle_SizeType) < 0) {
    return;
  }

  Py_INCREF(&Paddle_SizeType);

  m->add_object("Size", reinterpret_cast<PyObject*>(&Paddle_SizeType));
}

}  // namespace paddle::pybind
