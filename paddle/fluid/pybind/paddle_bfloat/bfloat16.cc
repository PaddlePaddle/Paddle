// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* Modified by martin.croome@greenwaves-technologies.com - Modifications to
   allow a standalone build
   and remove requirements for pybind11 and other tensorflow dependencies
   Add support for scalar operations and python numeric types
*/
/* Further modified by jakub1.piasecki@intel.com - Modifications to allow
 * building on Windows and using with python2.
*/

#include <Python.h>

#include <array>
#include <cmath>
#include <iostream>
#include <locale>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #define DEBUG_CALLS

#include <cinttypes>
#include <vector>
#ifdef DEBUG_CALLS
#include <iostream>
#endif

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/framework/eigen.h"
#else
#include "eigen/Eigen/Core"
#endif

#include <fenv.h>
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

float signcopy(float x, float y) { return (signbit(x) != signbit(y) ? -x : x); }

namespace paddle_bfloat {
using bfloat16 = Eigen::bfloat16;
using uint8 = std::uint8_t;
using int8 = std::int8_t;
using uint16 = std::uint16_t;
using int16 = std::int16_t;
using uint64 = std::uint64_t;

struct PyDecrefDeleter {
  void operator()(PyObject *p) const { Py_DECREF(p); }
};

// Safe container for an owned PyObject. On destruction, the reference count of
// the contained object will be decremented.
using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
Safe_PyObjectPtr make_safe(PyObject *object) {
  return Safe_PyObjectPtr(object);
}

bool PyLong_CheckNoOverflow(PyObject *object) {
  if (!PyLong_Check(object)) {
    return false;
  }
  int overflow = 0;
  PyLong_AsLongAndOverflow(object, &overflow);
  return (overflow == 0);
}

// Registered numpy type ID. Global variable populated by the registration code.
// Protected by the GIL.
int npy_bfloat16 = NPY_NOTYPE;

// Forward declaration.
extern PyTypeObject bfloat16_type;
extern PyArray_Descr NPyBfloat16_Descr;

// Pointer to the bfloat16 type object we are using. This is either a pointer
// to bfloat16_type, if we choose to register it, or to the bfloat16 type
// registered by another system into NumPy.
PyTypeObject *bfloat16_type_ptr = nullptr;

// Representation of a Python bfloat16 object.
struct PyBfloat16 {
  PyObject_HEAD;  // Python object header
  bfloat16 value;
};

// Returns true if 'object' is a PyBfloat16.
bool PyBfloat16_Check(PyObject *object) {
  return PyObject_IsInstance(object,
                             reinterpret_cast<PyObject *>(&bfloat16_type));
}

// Extracts the value of a PyBfloat16 object.
bfloat16 PyBfloat16_Bfloat16(PyObject *object) {
  return reinterpret_cast<PyBfloat16 *>(object)->value;
}

// Constructs a PyBfloat16 object from a bfloat16.
PyObject *PyBfloat16_FromBfloat16(bfloat16 x) {
  return PyArray_Scalar(&x, &NPyBfloat16_Descr, NULL);
}

// Converts a Python object to a bfloat16 value. Returns true on success,
// returns false and reports a Python error on failure.
bool CastToBfloat16(PyObject *arg, bfloat16 *output) {
  if (PyBfloat16_Check(arg)) {
    *output = PyBfloat16_Bfloat16(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = bfloat16(d);
    return true;
  }
  if (PyLong_CheckNoOverflow(arg)) {
    long l = PyLong_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = bfloat16(static_cast<float>(l));
    return true;
  }
  if (PyArray_IsScalar(arg, Half)) {
    Eigen::half f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = bfloat16(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = bfloat16(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Double)) {
    double f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = bfloat16(f);
    return true;
  }
  if (PyArray_IsZeroDim(arg)) {
    Safe_PyObjectPtr ref;
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(arg);
    if (PyArray_TYPE(arr) != npy_bfloat16) {
      ref = make_safe(PyArray_Cast(arr, npy_bfloat16));
      if (PyErr_Occurred()) {
        return false;
      }
      arg = ref.get();
      arr = reinterpret_cast<PyArrayObject *>(arg);
    }
    *output = *reinterpret_cast<bfloat16 *>(PyArray_DATA(arr));
    return true;
  }
  return false;
}

// Constructs a new PyBfloat16.
PyObject *PyBfloat16_New(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "expected number as argument to bfloat16 constructor");
    return nullptr;
  }
  PyObject *arg = PyTuple_GetItem(args, 0);

  bfloat16 value;
  if (PyBfloat16_Check(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToBfloat16(arg, &value)) {
    return PyBfloat16_FromBfloat16(value);
  } else if (PyArray_Check(arg)) {
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(arg);
    if (PyArray_TYPE(arr) != npy_bfloat16) {
      return PyArray_Cast(arr, npy_bfloat16);
    } else {
      Py_INCREF(arg);
      return arg;
    }
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               arg->ob_type->tp_name);
  return nullptr;
}

// Comparisons on PyBfloat16s.
PyObject *PyBfloat16_RichCompare(PyObject *self, PyObject *other, int cmp_op) {
  PyObject *arr, *ret;

  arr = PyArray_FromScalar(self, NULL);
  if (arr == NULL) {
    return NULL;
  }
  if (PyBfloat16_Check(other)) {
    PyObject *arr_other;
    arr_other = PyArray_FromScalar(other, NULL);
    ret = Py_TYPE(arr)->tp_richcompare(arr, arr_other, cmp_op);
    Py_DECREF(arr_other);
  } else {
    ret = Py_TYPE(arr)->tp_richcompare(arr, other, cmp_op);
  }
  Py_DECREF(arr);
  return ret;
}

// Implementation of repr() for PyBfloat16.
PyObject *PyBfloat16_Repr(PyObject *self) {
  bfloat16 x = reinterpret_cast<PyBfloat16 *>(self)->value;
  std::string v = std::to_string(static_cast<float>(x));
  return PyUnicode_FromString(v.c_str());
}

// Implementation of str() for PyBfloat16.
PyObject *PyBfloat16_Str(PyObject *self) {
  bfloat16 x = reinterpret_cast<PyBfloat16 *>(self)->value;
  std::string v = std::to_string(static_cast<float>(x));
  return PyUnicode_FromString(v.c_str());
}

// Hash function for PyBfloat16. We use the identity function, which is a weak
// hash function.
#if PY_VERSION_HEX >= 0x03000000
#define HASH_RETURN_TYPE Py_hash_t
#else
#define HASH_RETURN_TYPE long
#endif
HASH_RETURN_TYPE PyBfloat16_Hash(PyObject *self) {
  bfloat16 x = reinterpret_cast<PyBfloat16 *>(self)->value;
  return x.value;
}

// Converts a PyBfloat16 into a PyFloat.
PyObject *PyBfloat16_Float(PyObject *self) {
  bfloat16 x = PyBfloat16_Bfloat16(self);
  return PyFloat_FromDouble(static_cast<double>(x));
}

// Converts a PyBfloat16 into a PyInt.
PyObject *PyBfloat16_Int(PyObject *self) {
  bfloat16 x = PyBfloat16_Bfloat16(self);
  long y = static_cast<long>(x);  // NOLINT
  return PyLong_FromLong(y);
}

#if PY_VERSION_HEX >= 0x03000000
#define nb_divide
#define nb_coerce
#define nb_oct
#define nb_hex
#define nb_inplace_multiply
#else
#define nb_divide nullptr,
#define nb_coerce nullptr,
#define nb_oct nullptr,
#define nb_hex nullptr,
#define nb_inplace_multiply nullptr,
#endif

PyNumberMethods PyBfloat16_AsNumber = {
    nullptr,                   // nb_add
    nullptr,                   // nb_subtract
    nullptr,                   // nb_multiply
    nb_divide nullptr,         // nb_remainder
    nullptr,                   // nb_divmod
    nullptr,                   // nb_power
    nullptr,                   // nb_negative
    nullptr,                   // nb_positive
    nullptr,                   // nb_absolute
    nullptr,                   // nb_nonzero / nb_bool
    nullptr,                   // nb_invert
    nullptr,                   // nb_lshift
    nullptr,                   // nb_rshift
    nullptr,                   // nb_and
    nullptr,                   // nb_xor
    nullptr,                   // nb_or
    nb_coerce PyBfloat16_Int,  // nb_int
    nullptr,                   // reserved
    PyBfloat16_Float,          // nb_float
    nb_oct nb_hex

    nullptr,                      // nb_inplace_add
    nullptr,                      // nb_inplace_subtract
    nullptr,                      // nb_inplace_multiply
    nb_inplace_multiply nullptr,  // nb_inplace_remainder
    nullptr,                      // nb_inplace_power
    nullptr,                      // nb_inplace_lshift
    nullptr,                      // nb_inplace_rshift
    nullptr,                      // nb_inplace_and
    nullptr,                      // nb_inplace_xor
    nullptr,                      // nb_inplace_or

    nullptr,  // nb_floor_divide
    nullptr,  // nb_true_divide
    nullptr,  // nb_inplace_floor_divide
    nullptr,  // nb_inplace_true_divide
    nullptr,  // nb_index
};

// format bfloat16. Convert to a float and call format on that
PyObject *PyBfloat16_Format(PyObject *self, PyObject *format) {
  bfloat16 x = reinterpret_cast<PyBfloat16 *>(self)->value;
  PyObject *f_obj = PyFloat_FromDouble(static_cast<double>(x));
  PyObject *__format__str = PyUnicode_FromString("__format__");
  PyObject *f_str =
      PyObject_CallMethodObjArgs(f_obj, __format__str, format, NULL);
  Py_DECREF(__format__str);
  Py_XDECREF(f_obj);
  return f_str;
}

static PyMethodDef PyBfloat16_methods[] = {
    {"__format__", (PyCFunction)PyBfloat16_Format, METH_O,
     "__format__ method for bfloat16"},
    {NULL} /* Sentinel */
};

#ifdef IMPLEMENT_BUFFER
int PyBfloat16_getbuffer(PyObject *exporter, Py_buffer *view, int flags) {
  view->obj = exporter;
  Py_INCREF(exporter);
  view->buf = &(reinterpret_cast<PyBfloat16 *>(exporter)->value);
  view->len = 1;
  view->itemsize = sizeof(bfloat16);
  view->readonly = 0;
  view->format = NULL;
  if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) view->format = (char *)"BB";
  view->ndim = 1;
  view->shape = NULL;
  if ((flags & PyBUF_ND) == PyBUF_ND) view->shape = &(view->len);
  view->strides = NULL;
  if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES)
    view->strides = &(view->itemsize);
  view->suboffsets = NULL;
  view->internal = NULL;
  return 0;
}

static PyBufferProcs PyBfloat16_buffer_procs = {&PyBfloat16_getbuffer, NULL};
#endif

// Python type for PyBfloat16 objects.

PyTypeObject bfloat16_type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "bfloat16",  // tp_name
    sizeof(PyBfloat16),                            // tp_basicsize
    0,                                             // tp_itemsize
    nullptr,                                       // tp_dealloc
#if PY_VERSION_HEX < 0x03080000
    nullptr,  // tp_print
#else
    0,  // tp_vectorcall_offset
#endif
    nullptr,               // tp_getattr
    nullptr,               // tp_setattr
    nullptr,               // tp_compare / tp_reserved
    PyBfloat16_Repr,       // tp_repr
    &PyBfloat16_AsNumber,  // tp_as_number
    nullptr,               // tp_as_sequence
    nullptr,               // tp_as_mapping
    PyBfloat16_Hash,       // tp_hash
    nullptr,               // tp_call
    PyBfloat16_Str,        // tp_str
    nullptr,               // tp_getattro
    nullptr,               // tp_setattro
#ifdef IMPLEMENT_BUFFER
    &PyBfloat16_buffer_procs,  // tp_as_buffer
#else
    nullptr,
#endif
    // tp_flags
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "bfloat16 floating-point values",  // tp_doc
    nullptr,                           // tp_traverse
    nullptr,                           // tp_clear
    PyBfloat16_RichCompare,            // tp_richcompare
    0,                                 // tp_weaklistoffset
    nullptr,                           // tp_iter
    nullptr,                           // tp_iternext
    PyBfloat16_methods,                // tp_methods
    nullptr,                           // tp_members
    nullptr,                           // tp_getset
    nullptr,                           // tp_base
    nullptr,                           // tp_dict
    nullptr,                           // tp_descr_get
    nullptr,                           // tp_descr_set
    0,                                 // tp_dictoffset
    nullptr,                           // tp_init
    nullptr,                           // tp_alloc
    PyBfloat16_New,                    // tp_new
    nullptr,                           // tp_free
    nullptr,                           // tp_is_gc
    nullptr,                           // tp_bases
    nullptr,                           // tp_mro
    nullptr,                           // tp_cache
    nullptr,                           // tp_subclasses
    nullptr,                           // tp_weaklist
    nullptr,                           // tp_del
    0,                                 // tp_version_tag
};

// Numpy support

PyArray_ArrFuncs NPyBfloat16_ArrFuncs;

PyArray_Descr NPyBfloat16_Descr = {
    PyObject_HEAD_INIT(nullptr)  //
                                 /*typeobj=*/
    (&bfloat16_type),
    // We must register bfloat16 with a kind other than "f", because numpy
    // considers two types with the same kind and size to be equal, but
    // float16 != bfloat16.
    // The downside of this is that NumPy scalar promotion does not work with
    // bfloat16 values.
    /*kind=*/'g',
    // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
    // character is unique.
    /*type=*/'E',
    /*byteorder=*/'=',
    /*flags=*/NPY_NEEDS_PYAPI,  // | NPY_USE_GETITEM | NPY_USE_SETITEM,
    /*type_num=*/0,
    /*elsize=*/sizeof(bfloat16),
    /*alignment=*/alignof(bfloat16),
    /*subarray=*/nullptr,
    /*fields=*/nullptr,
    /*names=*/nullptr,
    /*f=*/&NPyBfloat16_ArrFuncs,
    /*metadata=*/nullptr,
    /*c_metadata=*/nullptr,
    /*hash=*/-1,  // -1 means "not computed yet".
};

// Implementations of NumPy array methods.

PyObject *NPyBfloat16_GetItem(void *data, void *arr) {
  bfloat16 x;
  NPyBfloat16_Descr.f->copyswap(
      &x, data, !PyArray_ISNOTSWAPPED(reinterpret_cast<PyArrayObject *>(arr)),
      NULL);
  return PyBfloat16_FromBfloat16(x);
}

int NPyBfloat16_SetItem(PyObject *item, void *data, void *arr) {
  bfloat16 x;
  if (!CastToBfloat16(item, &x)) {
    PyErr_Format(PyExc_TypeError, "expected number, got %s",
                 item->ob_type->tp_name);
    return -1;
  }
  memcpy(data, &x, sizeof(bfloat16));
  return 0;
}

void ByteSwap16(void *value) {
  char *p = reinterpret_cast<char *>(value);
  std::swap(p[0], p[1]);
}

void NPyBfloat16_CopySwapN(void *dstv, npy_intp dstride, void *srcv,
                           npy_intp sstride, npy_intp n, int swap, void *arr) {
  char *dst = reinterpret_cast<char *>(dstv);
  char *src = reinterpret_cast<char *>(srcv);
  if (!src) {
    return;
  }
  if (swap) {
    for (npy_intp i = 0; i < n; i++) {
      char *r = dst + dstride * i;
      memcpy(r, src + sstride * i, sizeof(uint16_t));
      ByteSwap16(r);
    }
  } else if (dstride == sizeof(uint16_t) && sstride == sizeof(uint16_t)) {
    memcpy(dst, src, n * sizeof(uint16_t));
  } else {
    for (npy_intp i = 0; i < n; i++) {
      memcpy(dst + dstride * i, src + sstride * i, sizeof(uint16_t));
    }
  }
}

void NPyBfloat16_CopySwap(void *dst, void *src, int swap, void *arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(uint16_t));
  if (swap) {
    ByteSwap16(dst);
  }
}

npy_bool NPyBfloat16_NonZero(void *data, void *arr) {
  bfloat16 x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<bfloat16>(0);
}

int NPyBfloat16_Fill(void *buffer_raw, npy_intp length, void *ignored) {
  bfloat16 *const buffer = reinterpret_cast<bfloat16 *>(buffer_raw);
  const float start(buffer[0]);
  const float delta = static_cast<float>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<bfloat16>(start + i * delta);
  }
  return 0;
}

void NPyBfloat16_DotFunc(void *ip1, npy_intp is1, void *ip2, npy_intp is2,
                         void *op, npy_intp n, void *arr) {
  char *c1 = reinterpret_cast<char *>(ip1);
  char *c2 = reinterpret_cast<char *>(ip2);
  float acc = 0.0f;
  for (npy_intp i = 0; i < n; ++i) {
    bfloat16 *const b1 = reinterpret_cast<bfloat16 *>(c1);
    bfloat16 *const b2 = reinterpret_cast<bfloat16 *>(c2);
    acc += static_cast<float>(*b1) * static_cast<float>(*b2);
    c1 += is1;
    c2 += is2;
  }
  bfloat16 *out = reinterpret_cast<bfloat16 *>(op);
  *out = static_cast<bfloat16>(acc);
}

int NPyBfloat16_CompareFunc(const void *v1, const void *v2, void *arr) {
#ifdef DEBUG_CALLS
  std::cout << "NPyBfloat16_CompareFunc\n";
#endif
  bfloat16 b1 = *reinterpret_cast<const bfloat16 *>(v1);
  bfloat16 b2 = *reinterpret_cast<const bfloat16 *>(v2);
  if (b1 < b2) {
    return -1;
  }
  if (b1 > b2) {
    return 1;
  }
  if (!Eigen::numext::isnan(b1) && Eigen::numext::isnan(b2)) {
    return 1;
  }
  if (Eigen::numext::isnan(b2) && !Eigen::numext::isnan(b1)) {
    return -1;
  }
  return 0;
}

int NPyBfloat16_ArgMaxFunc(void *data, npy_intp n, npy_intp *max_ind,
                           void *arr) {
  const bfloat16 *bdata = reinterpret_cast<const bfloat16 *>(data);
  float max_val = -std::numeric_limits<float>::infinity();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<float>(bdata[i]) > max_val) {
      max_val = static_cast<float>(bdata[i]);
      *max_ind = i;
    }
  }
  return 0;
}

int NPyBfloat16_ArgMinFunc(void *data, npy_intp n, npy_intp *min_ind,
                           void *arr) {
  const bfloat16 *bdata = reinterpret_cast<const bfloat16 *>(data);
  float min_val = std::numeric_limits<float>::infinity();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<float>(bdata[i]) < min_val) {
      min_val = static_cast<float>(bdata[i]);
      *min_ind = i;
    }
  }
  return 0;
}

// NumPy casts
template <typename T, typename Enable = void>
struct TypeDescriptor {
  // typedef ... T;  // Representation type in memory for NumPy values of type
  // static int Dtype() { return NPY_...; }  // Numpy type number for T.
};

template <>
struct TypeDescriptor<bfloat16> {
  typedef bfloat16 T;
  static int Dtype() { return npy_bfloat16; }
};

template <>
struct TypeDescriptor<uint8> {
  typedef uint8 T;
  static int Dtype() { return NPY_UINT8; }
};

template <>
struct TypeDescriptor<uint16> {
  typedef uint16 T;
  static int Dtype() { return NPY_UINT16; }
};

// We register "int", "long", and "long long" types for portability across
// Linux, where "int" and "long" are the same type, and Windows, where "long"
// and "longlong" are the same type.
template <>
struct TypeDescriptor<unsigned int> {
  typedef unsigned int T;
  static int Dtype() { return NPY_UINT; }
};

template <>
struct TypeDescriptor<unsigned long> {  // NOLINT
  typedef unsigned long T;              // NOLINT
  static int Dtype() { return NPY_ULONG; }
};

template <>
struct TypeDescriptor<unsigned long long> {  // NOLINT
  typedef unsigned long long T;              // NOLINT
  static int Dtype() { return NPY_ULONGLONG; }
};

template <>
struct TypeDescriptor<int8> {
  typedef int8 T;
  static int Dtype() { return NPY_INT8; }
};

template <>
struct TypeDescriptor<int16> {
  typedef int16 T;
  static int Dtype() { return NPY_INT16; }
};

template <>
struct TypeDescriptor<int> {
  typedef int T;
  static int Dtype() { return NPY_INT; }
};

template <>
struct TypeDescriptor<long> {  // NOLINT
  typedef long T;              // NOLINT
  static int Dtype() { return NPY_LONG; }
};

template <>
struct TypeDescriptor<long long> {  // NOLINT
  typedef long long T;              // NOLINT
  static int Dtype() { return NPY_LONGLONG; }
};

template <>
struct TypeDescriptor<bool> {
  typedef int8 T;
  static int Dtype() { return NPY_BOOL; }
};

template <>
struct TypeDescriptor<Eigen::half> {
  typedef Eigen::half T;
  static int Dtype() { return NPY_HALF; }
};

template <>
struct TypeDescriptor<float> {
  typedef float T;
  static int Dtype() { return NPY_FLOAT; }
};

template <>
struct TypeDescriptor<double> {
  typedef double T;
  static int Dtype() { return NPY_DOUBLE; }
};

template <>
struct TypeDescriptor<std::complex<float>> {
  typedef std::complex<float> T;
  static int Dtype() { return NPY_COMPLEX64; }
};

template <>
struct TypeDescriptor<std::complex<double>> {
  typedef std::complex<double> T;
  static int Dtype() { return NPY_COMPLEX128; }
};

template <>
struct TypeDescriptor<PyObject *> {
  typedef void *T;
  static int Dtype() { return NPY_OBJECT; }
};

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyCast(void *from_void, void *to_void, npy_intp n, void *fromarr,
             void *toarr) {
  const auto *from =
      reinterpret_cast<typename TypeDescriptor<From>::T *>(from_void);
  auto *to = reinterpret_cast<typename TypeDescriptor<To>::T *>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] =
        static_cast<typename TypeDescriptor<To>::T>(static_cast<To>(from[i]));
  }
}

// Registers a cast between bfloat16 and type 'T'. 'numpy_type' is the NumPy
// type corresponding to 'T'. If 'cast_is_safe', registers that bfloat16 can be
// safely coerced to T.
template <typename T>
bool RegisterBfloat16Cast(int numpy_type, bool cast_is_safe) {
  if (PyArray_RegisterCastFunc(PyArray_DescrFromType(numpy_type), npy_bfloat16,
                               NPyCast<T, bfloat16>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(&NPyBfloat16_Descr, numpy_type,
                               NPyCast<bfloat16, T>) < 0) {
    return false;
  }
  if (cast_is_safe &&
      PyArray_RegisterCanCast(&NPyBfloat16_Descr, numpy_type, NPY_NOSCALAR) <
          0) {
    return false;
  }
  return true;
}

template <typename InType, typename OutType, typename Functor>
struct UnaryUFunc {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<OutType>::Dtype()};
  }
  static void Call(char **args, const npy_intp *dimensions,
                   const npy_intp *steps, void *data) {
    const char *i0 = args[0];
    char *o = args[1];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x =
          *reinterpret_cast<const typename TypeDescriptor<InType>::T *>(i0);
      *reinterpret_cast<typename TypeDescriptor<OutType>::T *>(o) =
          Functor()(x);
      i0 += steps[0];
      o += steps[1];
    }
  }
};

template <typename InType, typename OutType, typename OutType2,
          typename Functor>
struct UnaryUFunc2 {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<OutType>::Dtype(),
            TypeDescriptor<OutType2>::Dtype()};
  }
  static void Call(char **args, const npy_intp *dimensions,
                   const npy_intp *steps, void *data) {
    const char *i0 = args[0];
    char *o0 = args[1];
    char *o1 = args[2];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x =
          *reinterpret_cast<const typename TypeDescriptor<InType>::T *>(i0);
      std::tie(*reinterpret_cast<typename TypeDescriptor<OutType>::T *>(o0),
               *reinterpret_cast<typename TypeDescriptor<OutType2>::T *>(o1)) =
          Functor()(x);
      i0 += steps[0];
      o0 += steps[1];
      o1 += steps[2];
    }
  }
};

template <typename InType, typename OutType, typename Functor>
struct BinaryUFunc {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<InType>::Dtype(),
            TypeDescriptor<OutType>::Dtype()};
  }
  static void Call(char **args, const npy_intp *dimensions,
                   const npy_intp *steps, void *data) {
#ifdef DEBUG_CALLS
    std::cout << "BinaryUFunc->Call\n";
#endif
    const char *i0 = args[0];
    const char *i1 = args[1];
    char *o = args[2];
    fenv_t fenv;
    feholdexcept(&fenv);
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x =
          *reinterpret_cast<const typename TypeDescriptor<InType>::T *>(i0);
      auto y =
          *reinterpret_cast<const typename TypeDescriptor<InType>::T *>(i1);
      *reinterpret_cast<typename TypeDescriptor<OutType>::T *>(o) =
          Functor()(x, y);
      i0 += steps[0];
      i1 += steps[1];
      o += steps[2];
    }
    if (fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW)) {
      if (fetestexcept(FE_INVALID)) {
        PyErr_SetString(PyExc_ArithmeticError, "bfloat16 invalid");
      } else if (fetestexcept(FE_DIVBYZERO)) {
        PyErr_SetString(PyExc_ArithmeticError, "bfloat16 divide by zero");
      } else if (fetestexcept(FE_OVERFLOW)) {
        PyErr_SetString(PyExc_ArithmeticError, "bfloat16 overflow");
      } else if (fetestexcept(FE_UNDERFLOW)) {
        PyErr_SetString(PyExc_ArithmeticError, "bfloat16 underflow");
      }
    }
    fesetenv(&fenv);
  }
};

template <typename InType, typename InType2, typename OutType, typename Functor>
struct BinaryUFunc2 {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<InType2>::Dtype(),
            TypeDescriptor<OutType>::Dtype()};
  }
  static void Call(char **args, const npy_intp *dimensions,
                   const npy_intp *steps, void *data) {
#ifdef DEBUG_CALLS
    std::cout << "BinaryUFunc2->Call\n";
#endif
    const char *i0 = args[0];
    const char *i1 = args[1];
    char *o = args[2];
    fenv_t fenv;
    feholdexcept(&fenv);
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x =
          *reinterpret_cast<const typename TypeDescriptor<InType>::T *>(i0);
      auto y =
          *reinterpret_cast<const typename TypeDescriptor<InType2>::T *>(i1);
      *reinterpret_cast<typename TypeDescriptor<OutType>::T *>(o) =
          Functor()(x, y);
      i0 += steps[0];
      i1 += steps[1];
      o += steps[2];
    }
    if (fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW)) {
      if (fetestexcept(FE_INVALID)) {
        PyErr_SetString(PyExc_ArithmeticError, "bfloat16 invalid");
      } else if (fetestexcept(FE_DIVBYZERO)) {
        PyErr_SetString(PyExc_ArithmeticError, "bfloat16 divide by zero");
      } else if (fetestexcept(FE_OVERFLOW)) {
        PyErr_SetString(PyExc_ArithmeticError, "bfloat16 overflow");
      } else if (fetestexcept(FE_UNDERFLOW)) {
        PyErr_SetString(PyExc_ArithmeticError, "bfloat16 underflow");
      }
    }
    fesetenv(&fenv);
  }
};

// template <typename InType, typename OutType, typename Functor>
// struct BinaryUFuncObj
// {
// 	static std::vector<int> Types()
// 	{
// 		return {TypeDescriptor<InType>::Dtype(), NPY_OBJECT,
// 				TypeDescriptor<OutType>::Dtype()};
// 	}
// 	static void Call(char **args, const npy_intp *dimensions,
// 					 const npy_intp *steps, void *data)
// 	{
// 		const char *i0 = args[0];
// 		char *i1 = args[1];
// 		char *o = args[2];
// 		for (npy_intp k = 0; k < *dimensions; k++)
// 		{
// 			auto x = *reinterpret_cast<const typename
// TypeDescriptor<InType>::T *>(i0);
// 			bfloat16 y = *reinterpret_cast<bfloat16 *>(i1);
// 			*reinterpret_cast<typename TypeDescriptor<OutType>::T
// *>(o)
// =
// 				Functor()(x, y);
// 			i0 += steps[0];
// 			i1 += steps[1];
// 			o += steps[2];
// 		}
// 	}
// };

template <typename UFunc>
bool RegisterUFunc(PyObject *numpy, const char *name) {
  std::vector<int> types = UFunc::Types();
  PyUFuncGenericFunction fn =
      reinterpret_cast<PyUFuncGenericFunction>(UFunc::Call);
  Safe_PyObjectPtr ufunc_obj = make_safe(PyObject_GetAttrString(numpy, name));
  if (!ufunc_obj) {
    return false;
  }
  PyUFuncObject *ufunc = reinterpret_cast<PyUFuncObject *>(ufunc_obj.get());
  if (static_cast<int>(types.size()) != ufunc->nargs) {
    PyErr_Format(PyExc_AssertionError,
                 "ufunc %s takes %d arguments, loop takes %lu", name,
                 ufunc->nargs, types.size());
    return false;
  }
  if (PyUFunc_RegisterLoopForType(ufunc, npy_bfloat16, fn,
                                  const_cast<int *>(types.data()),
                                  nullptr) < 0) {
    return false;
  }
  return true;
}

namespace ufuncs {

struct Add {
  bfloat16 operator()(bfloat16 a, bfloat16 b) { return a + b; }
};
struct AddScalarFloat {
  bfloat16 operator()(bfloat16 a, float b) { return a + bfloat16(b); }
};
struct ScalarFloatAdd {
  bfloat16 operator()(float a, bfloat16 b) { return bfloat16(a) + b; }
};
struct Subtract {
  bfloat16 operator()(bfloat16 a, bfloat16 b) { return a - b; }
};
struct Multiply {
  bfloat16 operator()(bfloat16 a, bfloat16 b) { return a * b; }
};
struct TrueDivide {
  bfloat16 operator()(bfloat16 a, bfloat16 b) { return a / b; }
};

std::pair<float, float> divmod(float a, float b) {
  if (b == 0.0f) {
    float nan = std::numeric_limits<float>::quiet_NaN();
    return {nan, nan};
  }
  float mod = std::fmod(a, b);
  float div = (a - mod) / b;
  if (mod != 0.0f) {
    if ((b < 0.0f) != (mod < 0.0f)) {
      mod += b;
      div -= 1.0f;
    }
  } else {
    mod = signcopy(0.0f, b);
  }

  float floordiv;
  if (div != 0.0f) {
    floordiv = std::floor(div);
    if (div - floordiv > 0.5f) {
      floordiv += 1.0f;
    }
  } else {
    floordiv = signcopy(0.0f, a / b);
  }
  return {floordiv, mod};
}

struct FloorDivide {
  bfloat16 operator()(bfloat16 a, bfloat16 b) {
    return bfloat16(divmod(static_cast<float>(a), static_cast<float>(b)).first);
  }
};
struct Remainder {
  bfloat16 operator()(bfloat16 a, bfloat16 b) {
    return bfloat16(
        divmod(static_cast<float>(a), static_cast<float>(b)).second);
  }
};
struct DivmodUFunc {
  static std::vector<int> Types() {
    return {npy_bfloat16, npy_bfloat16, npy_bfloat16, npy_bfloat16};
  }
  static void Call(char **args, npy_intp *dimensions, npy_intp *steps,
                   void *data) {
    const char *i0 = args[0];
    const char *i1 = args[1];
    char *o0 = args[2];
    char *o1 = args[3];
    for (npy_intp k = 0; k < *dimensions; k++) {
      bfloat16 x = *reinterpret_cast<const bfloat16 *>(i0);
      bfloat16 y = *reinterpret_cast<const bfloat16 *>(i1);
      float floordiv, mod;
      std::tie(floordiv, mod) =
          divmod(static_cast<float>(x), static_cast<float>(y));
      *reinterpret_cast<bfloat16 *>(o0) = bfloat16(floordiv);
      *reinterpret_cast<bfloat16 *>(o1) = bfloat16(mod);
      i0 += steps[0];
      i1 += steps[1];
      o0 += steps[2];
      o1 += steps[3];
    }
  }
};
struct Fmod {
  bfloat16 operator()(bfloat16 a, bfloat16 b) {
    return bfloat16(std::fmod(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Negative {
  bfloat16 operator()(bfloat16 a) { return -a; }
};
struct Positive {
  bfloat16 operator()(bfloat16 a) { return a; }
};
struct Power {
  bfloat16 operator()(bfloat16 a, bfloat16 b) {
    return bfloat16(std::pow(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Abs {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::abs(static_cast<float>(a)));
  }
};
struct Cbrt {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::cbrt(static_cast<float>(a)));
  }
};
struct Ceil {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::ceil(static_cast<float>(a)));
  }
};
struct CopySign {
  bfloat16 operator()(bfloat16 a, bfloat16 b) {
    return bfloat16(signcopy(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Exp {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::exp(static_cast<float>(a)));
  }
};
struct Exp2 {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::exp2(static_cast<float>(a)));
  }
};
struct Expm1 {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::expm1(static_cast<float>(a)));
  }
};
struct Floor {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::floor(static_cast<float>(a)));
  }
};
struct Frexp {
  std::pair<bfloat16, int> operator()(bfloat16 a) {
    int exp;
    float f = std::frexp(static_cast<float>(a), &exp);
    return {bfloat16(f), exp};
  }
};
struct Heaviside {
  bfloat16 operator()(bfloat16 bx, bfloat16 h0) {
    float x = static_cast<float>(bx);
    if (Eigen::numext::isnan(x)) {
      return bx;
    }
    if (x < 0) {
      return bfloat16(0.0f);
    }
    if (x > 0) {
      return bfloat16(1.0f);
    }
    return h0;  // x == 0
  }
};
struct Conjugate {
  bfloat16 operator()(bfloat16 a) { return a; }
};
struct IsFinite {
  bool operator()(bfloat16 a) { return std::isfinite(static_cast<float>(a)); }
};
struct IsInf {
  bool operator()(bfloat16 a) { return std::isinf(static_cast<float>(a)); }
};
struct IsNan {
  bool operator()(bfloat16 a) {
    return Eigen::numext::isnan(static_cast<float>(a));
  }
};
struct Ldexp {
  bfloat16 operator()(bfloat16 a, int exp) {
    return bfloat16(std::ldexp(static_cast<float>(a), exp));
  }
};
struct Log {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::log(static_cast<float>(a)));
  }
};
struct Log2 {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::log2(static_cast<float>(a)));
  }
};
struct Log10 {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::log10(static_cast<float>(a)));
  }
};
struct Log1p {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::log1p(static_cast<float>(a)));
  }
};
struct LogAddExp {
  bfloat16 operator()(bfloat16 bx, bfloat16 by) {
    float x = static_cast<float>(bx);
    float y = static_cast<float>(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return bfloat16(x + std::log(2.0f));
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp(y - x));
    } else if (x < y) {
      out = y + std::log1p(std::exp(x - y));
    }
    return bfloat16(out);
  }
};
struct LogAddExp2 {
  bfloat16 operator()(bfloat16 bx, bfloat16 by) {
    float x = static_cast<float>(bx);
    float y = static_cast<float>(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return bfloat16(x + 1.0f);
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp2(y - x)) / std::log(2.0f);
    } else if (x < y) {
      out = y + std::log1p(std::exp2(x - y)) / std::log(2.0f);
    }
    return bfloat16(out);
  }
};
struct Modf {
  std::pair<bfloat16, bfloat16> operator()(bfloat16 a) {
    float integral;
    float f = std::modf(static_cast<float>(a), &integral);
    return {bfloat16(f), bfloat16(integral)};
  }
};

struct Reciprocal {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(1.f / static_cast<float>(a));
  }
};
struct Rint {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::rint(static_cast<float>(a)));
  }
};
struct Sign {
  bfloat16 operator()(bfloat16 a) {
    float f(a);
    if (f < 0) {
      return bfloat16(-1);
    }
    if (f > 0) {
      return bfloat16(1);
    }
    return a;
  }
};
struct SignBit {
  bool operator()(bfloat16 a) { return std::signbit(static_cast<float>(a)); }
};
struct Sqrt {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::sqrt(static_cast<float>(a)));
  }
};
struct Square {
  bfloat16 operator()(bfloat16 a) {
    float f(a);
    return bfloat16(f * f);
  }
};
struct Trunc {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::trunc(static_cast<float>(a)));
  }
};

// Trigonometric functions
struct Sin {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::sin(static_cast<float>(a)));
  }
};
struct Cos {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::cos(static_cast<float>(a)));
  }
};
struct Tan {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::tan(static_cast<float>(a)));
  }
};
struct Arcsin {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::asin(static_cast<float>(a)));
  }
};
struct Arccos {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::acos(static_cast<float>(a)));
  }
};
struct Arctan {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::atan(static_cast<float>(a)));
  }
};
struct Arctan2 {
  bfloat16 operator()(bfloat16 a, bfloat16 b) {
    return bfloat16(std::atan2(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Hypot {
  bfloat16 operator()(bfloat16 a, bfloat16 b) {
    return bfloat16(std::hypot(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Sinh {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::sinh(static_cast<float>(a)));
  }
};
struct Cosh {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::cosh(static_cast<float>(a)));
  }
};
struct Tanh {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::tanh(static_cast<float>(a)));
  }
};
struct Arcsinh {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::asinh(static_cast<float>(a)));
  }
};
struct Arccosh {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::acosh(static_cast<float>(a)));
  }
};
struct Arctanh {
  bfloat16 operator()(bfloat16 a) {
    return bfloat16(std::atanh(static_cast<float>(a)));
  }
};
struct Deg2rad {
  bfloat16 operator()(bfloat16 a) {
    static constexpr float radians_per_degree = M_PI / 180.0f;
    return bfloat16(static_cast<float>(a) * radians_per_degree);
  }
};
struct Rad2deg {
  bfloat16 operator()(bfloat16 a) {
    static constexpr float degrees_per_radian = 180.0f / M_PI;
    return bfloat16(static_cast<float>(a) * degrees_per_radian);
  }
};

struct Eq {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a == b; }
};
struct EqFloat {
  npy_bool operator()(bfloat16 a, float b) { return a == bfloat16(b); }
};
struct EqDouble {
  npy_bool operator()(bfloat16 a, double b) { return a == bfloat16(b); }
};
struct Ne {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a != b; }
};
struct NeFloat {
  npy_bool operator()(bfloat16 a, float b) { return a != bfloat16(b); }
};
struct NeDouble {
  npy_bool operator()(bfloat16 a, double b) { return a != bfloat16(b); }
};
struct Lt {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a < b; }
};
struct LtFloat {
  npy_bool operator()(bfloat16 a, float b) { return a < bfloat16(b); }
};
struct LtDouble {
  npy_bool operator()(bfloat16 a, double b) { return a < bfloat16(b); }
};
struct Gt {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a > b; }
};
struct GtFloat {
  npy_bool operator()(bfloat16 a, float b) { return a > bfloat16(b); }
};
struct GtDouble {
  npy_bool operator()(bfloat16 a, double b) { return a > bfloat16(b); }
};
struct Le {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a <= b; }
};
struct LeFloat {
  npy_bool operator()(bfloat16 a, float b) { return a <= bfloat16(b); }
};
struct LeDouble {
  npy_bool operator()(bfloat16 a, double b) { return a <= bfloat16(b); }
};
struct Ge {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a >= b; }
};
struct GeFloat {
  npy_bool operator()(bfloat16 a, float b) { return a >= bfloat16(b); }
};
struct GeDouble {
  npy_bool operator()(bfloat16 a, double b) { return a >= bfloat16(b); }
};
struct Maximum {
  bfloat16 operator()(bfloat16 a, bfloat16 b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fa) || fa > fb ? a : b;
  }
};
struct Minimum {
  bfloat16 operator()(bfloat16 a, bfloat16 b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fa) || fa < fb ? a : b;
  }
};
struct Fmax {
  bfloat16 operator()(bfloat16 a, bfloat16 b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fb) || fa > fb ? a : b;
  }
};
struct Fmin {
  bfloat16 operator()(bfloat16 a, bfloat16 b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fb) || fa < fb ? a : b;
  }
};

struct LogicalNot {
  npy_bool operator()(bfloat16 a) { return !a; }
};
struct LogicalAnd {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a && b; }
};
struct LogicalOr {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a || b; }
};
struct LogicalXor {
  npy_bool operator()(bfloat16 a, bfloat16 b) {
    return static_cast<bool>(a) ^ static_cast<bool>(b);
  }
};

struct NextAfter {
  bfloat16 operator()(bfloat16 from, bfloat16 to) {
    uint16_t from_as_int, to_as_int;
    const uint16_t sign_mask = 1 << 15;
    float from_as_float(from), to_as_float(to);
    memcpy(&from_as_int, &from, sizeof(bfloat16));
    memcpy(&to_as_int, &to, sizeof(bfloat16));
    if (Eigen::numext::isnan(from_as_float) ||
        Eigen::numext::isnan(to_as_float)) {
      return bfloat16(std::numeric_limits<float>::quiet_NaN());
    }
    if (from_as_int == to_as_int) {
      return to;
    }
    if (from_as_float == 0) {
      if (to_as_float == 0) {
        return to;
      } else {
        // Smallest subnormal signed like `to`.
        uint16_t out_int = (to_as_int & sign_mask) | 1;
        bfloat16 out;
        memcpy(&out, &out_int, sizeof(bfloat16));
        return out;
      }
    }
    uint16_t from_sign = from_as_int & sign_mask;
    uint16_t to_sign = to_as_int & sign_mask;
    uint16_t from_abs = from_as_int & ~sign_mask;
    uint16_t to_abs = to_as_int & ~sign_mask;
    uint16_t magnitude_adjustment =
        (from_abs > to_abs || from_sign != to_sign) ? 0xFFFF : 0x0001;
    uint16_t out_int = from_as_int + magnitude_adjustment;
    bfloat16 out;
    memcpy(&out, &out_int, sizeof(bfloat16));
    return out;
  }
};
}

// needed because in python < 3 import_array() returns void which causes error
// in Initialize()

#if PY_VERSION_HEX >= 0x03000000
void *wrapper_import_array() { import_array() return NULL; }
#else
void wrapper_import_array() { import_array() }
#endif

// Initializes the module.
bool Initialize() {
  wrapper_import_array();
  import_umath1(false);

  Safe_PyObjectPtr numpy_str = make_safe(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  // If another module (presumably either TF or JAX) has registered a bfloat16
  // type, use it. We don't want two bfloat16 types if we can avoid it since it
  // leads to confusion if we have two different types with the same name. This
  // assumes that the other module has a sufficiently complete bfloat16
  // implementation. The only known NumPy bfloat16 extension at the time of
  // writing is this one (distributed in TF and JAX).
  // TODO(phawkins): distribute the bfloat16 extension as its own pip package,
  // so we can unambiguously refer to a single canonical definition of bfloat16.
  int typenum = PyArray_TypeNumFromName(const_cast<char *>("bfloat16"));
  if (typenum != NPY_NOTYPE) {
    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    // The test for an argmax function here is to verify that the
    // bfloat16 implementation is sufficiently new, and, say, not from
    // an older version of TF or JAX.
    if (descr && descr->f && descr->f->argmax) {
      npy_bfloat16 = typenum;
      bfloat16_type_ptr = descr->typeobj;
      return true;
    }
  }

  bfloat16_type.tp_base = &PyGenericArrType_Type;

  if (PyType_Ready(&bfloat16_type) < 0) {
    PyErr_Print();
    PyErr_SetString(PyExc_SystemError, "could not initialize bfloat16");
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_InitArrFuncs(&NPyBfloat16_ArrFuncs);
  NPyBfloat16_ArrFuncs.getitem = NPyBfloat16_GetItem;
  NPyBfloat16_ArrFuncs.setitem = NPyBfloat16_SetItem;
  NPyBfloat16_ArrFuncs.copyswapn = NPyBfloat16_CopySwapN;
  NPyBfloat16_ArrFuncs.copyswap = NPyBfloat16_CopySwap;
  NPyBfloat16_ArrFuncs.nonzero = NPyBfloat16_NonZero;
  NPyBfloat16_ArrFuncs.fill = NPyBfloat16_Fill;
  NPyBfloat16_ArrFuncs.dotfunc = NPyBfloat16_DotFunc;
  NPyBfloat16_ArrFuncs.compare = NPyBfloat16_CompareFunc;
  NPyBfloat16_ArrFuncs.argmax = NPyBfloat16_ArgMaxFunc;
  NPyBfloat16_ArrFuncs.argmin = NPyBfloat16_ArgMinFunc;

  Py_TYPE(&NPyBfloat16_Descr) = &PyArrayDescr_Type;
  npy_bfloat16 = PyArray_RegisterDataType(&NPyBfloat16_Descr);
  bfloat16_type_ptr = &bfloat16_type;
  if (npy_bfloat16 < 0) {
    return false;
  }

  // Support dtype(bfloat16)
  if (PyDict_SetItemString(bfloat16_type.tp_dict, "dtype",
                           reinterpret_cast<PyObject *>(&NPyBfloat16_Descr)) <
      0) {
    return false;
  }

  // Register casts
  if (!RegisterBfloat16Cast<Eigen::half>(NPY_HALF, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<float>(NPY_FLOAT, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterBfloat16Cast<double>(NPY_DOUBLE, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterBfloat16Cast<bool>(NPY_BOOL, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<uint8>(NPY_UINT8, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<uint16>(NPY_UINT16, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<unsigned int>(NPY_UINT, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<unsigned long>(NPY_ULONG,  // NOLINT
                                           /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<unsigned long long>(  // NOLINT
          NPY_ULONGLONG, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<uint64>(NPY_UINT64, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<int8>(NPY_INT8, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<int16>(NPY_INT16, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<int>(NPY_INT, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<long>(NPY_LONG,  // NOLINT
                                  /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<long long>(  // NOLINT
          NPY_LONGLONG, /*cast_is_safe=*/false)) {
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterBfloat16Cast<std::complex<float>>(NPY_COMPLEX64,
                                                 /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterBfloat16Cast<std::complex<double>>(NPY_COMPLEX128,
                                                  /*cast_is_safe=*/true)) {
    return false;
  }

  bool ok =
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Add>>(numpy.get(),
                                                                  "add") &&
      RegisterUFunc<
          BinaryUFunc2<float, bfloat16, bfloat16, ufuncs::ScalarFloatAdd>>(
          numpy.get(), "add") &&
      RegisterUFunc<
          BinaryUFunc2<bfloat16, float, bfloat16, ufuncs::AddScalarFloat>>(
          numpy.get(), "add") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Subtract>>(
          numpy.get(), "subtract") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Multiply>>(
          numpy.get(), "multiply") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::TrueDivide>>(
          numpy.get(), "divide") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::LogAddExp>>(
          numpy.get(), "logaddexp") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::LogAddExp2>>(
          numpy.get(), "logaddexp2") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Negative>>(
          numpy.get(), "negative") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Positive>>(
          numpy.get(), "positive") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::TrueDivide>>(
          numpy.get(), "true_divide") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::FloorDivide>>(
          numpy.get(), "floor_divide") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Power>>(numpy.get(),
                                                                    "power") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Remainder>>(
          numpy.get(), "remainder") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Remainder>>(
          numpy.get(), "mod") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Fmod>>(numpy.get(),
                                                                   "fmod") &&
      RegisterUFunc<ufuncs::DivmodUFunc>(numpy.get(), "divmod") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Abs>>(numpy.get(),
                                                                 "absolute") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Abs>>(numpy.get(),
                                                                 "fabs") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Rint>>(numpy.get(),
                                                                  "rint") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Sign>>(numpy.get(),
                                                                  "sign") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Heaviside>>(
          numpy.get(), "heaviside") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Conjugate>>(
          numpy.get(), "conjugate") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Exp>>(numpy.get(),
                                                                 "exp") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Exp2>>(numpy.get(),
                                                                  "exp2") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Expm1>>(numpy.get(),
                                                                   "expm1") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Log>>(numpy.get(),
                                                                 "log") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Log2>>(numpy.get(),
                                                                  "log2") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Log10>>(numpy.get(),
                                                                   "log10") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Log1p>>(numpy.get(),
                                                                   "log1p") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Sqrt>>(numpy.get(),
                                                                  "sqrt") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Square>>(numpy.get(),
                                                                    "square") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Cbrt>>(numpy.get(),
                                                                  "cbrt") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Reciprocal>>(
          numpy.get(), "reciprocal") &&

      // Trigonometric functions
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Sin>>(numpy.get(),
                                                                 "sin") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Cos>>(numpy.get(),
                                                                 "cos") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Tan>>(numpy.get(),
                                                                 "tan") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Arcsin>>(numpy.get(),
                                                                    "arcsin") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Arccos>>(numpy.get(),
                                                                    "arccos") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Arctan>>(numpy.get(),
                                                                    "arctan") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Arctan2>>(
          numpy.get(), "arctan2") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Hypot>>(numpy.get(),
                                                                    "hypot") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Sinh>>(numpy.get(),
                                                                  "sinh") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Cosh>>(numpy.get(),
                                                                  "cosh") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Tanh>>(numpy.get(),
                                                                  "tanh") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Arcsinh>>(
          numpy.get(), "arcsinh") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Arccosh>>(
          numpy.get(), "arccosh") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Arctanh>>(
          numpy.get(), "arctanh") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Deg2rad>>(
          numpy.get(), "deg2rad") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Rad2deg>>(
          numpy.get(), "rad2deg") &&

      // Comparison functions
      // RegisterUFunc<BinaryUFuncObj<bfloat16, bool, ufuncs::Eq>>(numpy.get(),
      // 													   "equal")
      // &&
      RegisterUFunc<BinaryUFunc<bfloat16, bool, ufuncs::Eq>>(numpy.get(),
                                                             "equal") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, float, bool, ufuncs::EqFloat>>(
          numpy.get(), "equal") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, double, bool, ufuncs::EqDouble>>(
          numpy.get(), "equal") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bool, ufuncs::Ne>>(numpy.get(),
                                                             "not_equal") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, float, bool, ufuncs::NeFloat>>(
          numpy.get(), "not_equal") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, double, bool, ufuncs::NeDouble>>(
          numpy.get(), "not_equal") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bool, ufuncs::Lt>>(numpy.get(),
                                                             "less") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, float, bool, ufuncs::LtFloat>>(
          numpy.get(), "less") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, double, bool, ufuncs::LtDouble>>(
          numpy.get(), "less") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bool, ufuncs::Gt>>(numpy.get(),
                                                             "greater") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, float, bool, ufuncs::GtFloat>>(
          numpy.get(), "greater") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, double, bool, ufuncs::GtDouble>>(
          numpy.get(), "greater") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bool, ufuncs::Le>>(numpy.get(),
                                                             "less_equal") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, float, bool, ufuncs::LeFloat>>(
          numpy.get(), "less_equal") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, double, bool, ufuncs::LeDouble>>(
          numpy.get(), "less_equal") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bool, ufuncs::Ge>>(numpy.get(),
                                                             "greater_equal") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, float, bool, ufuncs::GeFloat>>(
          numpy.get(), "greater_equal") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, double, bool, ufuncs::GeDouble>>(
          numpy.get(), "greater_equal") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Maximum>>(
          numpy.get(), "maximum") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Minimum>>(
          numpy.get(), "minimum") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Fmax>>(numpy.get(),
                                                                   "fmax") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::Fmin>>(numpy.get(),
                                                                   "fmin") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bool, ufuncs::LogicalAnd>>(
          numpy.get(), "logical_and") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bool, ufuncs::LogicalOr>>(
          numpy.get(), "logical_or") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bool, ufuncs::LogicalXor>>(
          numpy.get(), "logical_xor") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bool, ufuncs::LogicalNot>>(
          numpy.get(), "logical_not") &&

      // Floating point functions
      RegisterUFunc<UnaryUFunc<bfloat16, bool, ufuncs::IsFinite>>(numpy.get(),
                                                                  "isfinite") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bool, ufuncs::IsInf>>(numpy.get(),
                                                               "isinf") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bool, ufuncs::IsNan>>(numpy.get(),
                                                               "isnan") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bool, ufuncs::SignBit>>(numpy.get(),
                                                                 "signbit") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::CopySign>>(
          numpy.get(), "copysign") &&
      RegisterUFunc<UnaryUFunc2<bfloat16, bfloat16, bfloat16, ufuncs::Modf>>(
          numpy.get(), "modf") &&
      RegisterUFunc<BinaryUFunc2<bfloat16, int, bfloat16, ufuncs::Ldexp>>(
          numpy.get(), "ldexp") &&
      RegisterUFunc<UnaryUFunc2<bfloat16, bfloat16, int, ufuncs::Frexp>>(
          numpy.get(), "frexp") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Floor>>(numpy.get(),
                                                                   "floor") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Ceil>>(numpy.get(),
                                                                  "ceil") &&
      RegisterUFunc<UnaryUFunc<bfloat16, bfloat16, ufuncs::Trunc>>(numpy.get(),
                                                                   "trunc") &&
      RegisterUFunc<BinaryUFunc<bfloat16, bfloat16, ufuncs::NextAfter>>(
          numpy.get(), "nextafter");

  return ok;
}

bool RegisterNumpyBfloat16() {
  if (npy_bfloat16 != NPY_NOTYPE) {
    // Already initialized.
    return true;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load bfloat16 module.");
    }
    PyErr_Print();
    return false;
  }
  return true;
}

PyObject *Bfloat16Dtype() {
  return reinterpret_cast<PyObject *>(bfloat16_type_ptr);
}

int Bfloat16NumpyType() { return npy_bfloat16; }

static PyMethodDef Bfloat16ModuleMethods[] = {{NULL, NULL, 0, NULL}};

#if PY_VERSION_HEX >= 0x03000000
#define MOD_ERROR_VAL NULL
#define MOD_SUCCESS_VAL(val) val
#define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#define MOD_DEF(ob, name, methods)                                             \
  static struct PyModuleDef moduledef = {                                      \
      PyModuleDef_HEAD_INIT, name, NULL, -1, methods, NULL, NULL, NULL, NULL}; \
  ob = PyModule_Create(&moduledef);
#else
#define MOD_ERROR_VAL
#define MOD_SUCCESS_VAL(val)
#define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
#define MOD_DEF(ob, name, methods) ob = Py_InitModule3(name, methods);
#endif

MOD_INIT(paddle_bfloat) {
  PyObject *m;
  MOD_DEF(m, "paddle_bfloat", Bfloat16ModuleMethods)

  if (m == NULL) return MOD_ERROR_VAL;
  RegisterNumpyBfloat16();
  Py_INCREF(&bfloat16_type);
  Py_XINCREF(&NPyBfloat16_Descr);
  if (PyModule_AddObject(m, "bfloat16", (PyObject *)&bfloat16_type) < 0) {
    Py_DECREF(&bfloat16_type);
    Py_DECREF(m);
    return MOD_ERROR_VAL;
  }

  return MOD_SUCCESS_VAL(m);
}
}  // namespace paddle_bfloat
