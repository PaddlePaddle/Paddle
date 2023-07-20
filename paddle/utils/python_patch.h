/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

// Define Python Macros instead of <Python.h> to avoid
// cppcoreguidelines-pro-type-cstyle-cast

/* Cast argument toc PyVarObject* type. */
#define _PyVarObject_CAST_s(op) (reinterpret_cast<PyVarObject*>(op))

#define Py_SIZE_s(ob) (_PyVarObject_CAST_s(ob)->ob_size)

// *********************
// *      PyList       *
// *********************

#define PyList_GET_ITEM_s(op, i) \
  ((reinterpret_cast<PyListObject*>(op))->ob_item[i])
#define PyList_SET_ITEM_s(op, i, v) \
  ((reinterpret_cast<PyListObject*>(op))->ob_item[i] = (v))
#define _PyList_ITEMS_s(op) ((reinterpret_cast<PyListObject*>(op))->ob_item)

#ifdef PyList_GET_ITEM
#undef PyList_GET_ITEM
#define PyList_GET_ITEM(op, i) PyList_GET_ITEM_s(op, i)
#endif

#ifdef PyList_SET_ITEM
#undef PyList_SET_ITEM
#define PyList_SET_ITEM(op, i, v) PyList_SET_ITEM_s(op, i, v)
#endif

#ifdef _PyList_ITEMS
#undef _PyList_ITEMS
#define _PyList_ITEMS(op) _PyList_ITEMS_s(op)
#endif

// *********************
// *      PyTuple       *
// *********************

/* Cast argument to PyTupleObject* type. */
#define _PyTuple_CAST_s(op) \
  (assert(PyTuple_Check(op)), reinterpret_cast<PyTupleObject*>(op))
#define PyTuple_GET_SIZE_s(op) Py_SIZE_s(_PyTuple_CAST_s(op))
#define PyTuple_GET_ITEM_s(op, i) (_PyTuple_CAST_s(op)->ob_item[i])
/* Macro, *only* to be used to fill in brand new tuples */
#define PyTuple_SET_ITEM_s(op, i, v) (_PyTuple_CAST_s(op)->ob_item[i] = v)

#ifdef PyTuple_GET_SIZE
#undef PyTuple_GET_SIZE
#define PyTuple_GET_SIZE(op) PyTuple_GET_SIZE_s(op)
#endif

#ifdef PyTuple_GET_ITEM
#undef PyTuple_GET_ITEM
#define PyTuple_GET_ITEM(op, i) PyTuple_GET_ITEM_s(op, i)
#endif

#ifdef PyTuple_SET_ITEM
#undef PyTuple_SET_ITEM
#define PyTuple_SET_ITEM(op, i, v) PyTuple_SET_ITEM_s(op, i, v)
#endif
