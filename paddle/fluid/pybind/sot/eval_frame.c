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

#include "paddle/fluid/pybind/sot/eval_frame.h"

#if SOT_IS_SUPPORTED

#include "paddle/fluid/pybind/sot/cpython_internals.h"
#include "paddle/fluid/pybind/sot/eval_frame_tools.h"

#include <Python.h>
#include <frameobject.h>

#if PY_VERSION_HEX >= 0x03080000 && PY_VERSION_HEX < 0x3090000
#define Py_BUILD_CORE  // internal/pycore_pymem.h need this macro
#include <internal/pycore_pystate.h>
#undef Py_BUILD_CORE
#endif
#if PY_VERSION_HEX < 0x030b0000
#include <code.h>
#endif

#include <object.h>
#include <pystate.h>

#if PY_VERSION_HEX >= 0x030b0000
// To avoid the error: undefined symbol: _PyFrame_GetFrameObject, all we need is
// to redefine this function based source code in python3.11. The advantage is
// that we don't need any modification in eval_frame functions.
typedef _PyInterpreterFrame FrameObject;
#define CALL_STAT_INC(name) ((void)0)

// clang-format off
// Define a proxy PyObject to access _PyInterpreterFrame's properties.
// It will be passed as an argument to the eval frame's callback.
typedef struct PyInterpreterFrameProxy {
  PyObject_HEAD
  _PyInterpreterFrame *frame;
} PyInterpreterFrameProxy;
// clang-format on

#define DECLARE_PROXY_PROPERTY(name)                        \
  static PyObject *PyInterpreterFrameProxy_property_##name( \
      PyInterpreterFrameProxy *self, void *closure) {       \
    Py_XINCREF(self->frame->name);                          \
    return (PyObject *)self->frame->name;                   \
  }

// clang-format off
#define REGISTER_PROXY_PROPERTY(name) \
  { #name, (getter)PyInterpreterFrameProxy_property_##name, NULL, NULL, NULL }
// clang-format on

DECLARE_PROXY_PROPERTY(f_code)
DECLARE_PROXY_PROPERTY(f_locals)
DECLARE_PROXY_PROPERTY(f_globals)
DECLARE_PROXY_PROPERTY(f_builtins)

// Refer to
// https://github.com/python/cpython/blob/9414ddf91898892f3f6a672ae946931ee4b3ceb7/Objects/frameobject.c#L953-L961
static PyObject *PyInterpreterFrameProxy_method_repr(
    PyInterpreterFrameProxy *self) {
  int lineno = Internal_PyInterpreterFrame_GetLine(self->frame);
  PyCodeObject *code = self->frame->f_code;
  return PyUnicode_FromFormat(
      "<PyInterpreterFrameProxy at %p, file %R, line %d, code %S>",
      self,
      code->co_filename,
      lineno,
      code->co_name);
}

static PyGetSetDef PyInterpreterFrameProxy_properties[] = {
    REGISTER_PROXY_PROPERTY(f_code),
    REGISTER_PROXY_PROPERTY(f_locals),
    REGISTER_PROXY_PROPERTY(f_globals),
    REGISTER_PROXY_PROPERTY(f_builtins),
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
  return self;
}

#else
typedef PyFrameObject FrameObject;
#endif

#ifdef _WIN32
#define unlikely(x) (x)
#else
#define unlikely(x) __builtin_expect((x), 0)
#endif

// Use static variable to save customed eval hook.
static Py_tss_t eval_frame_callback_key = {0, 0};

inline static PyObject *eval_frame_callback_get() {
  void *result = PyThread_tss_get(&eval_frame_callback_key);
  if (unlikely(result == NULL)) {
    Py_RETURN_NONE;
  } else {
    return (PyObject *)result;
  }
}

inline static void eval_frame_callback_set(PyObject *obj) {
  PyThread_tss_set(&eval_frame_callback_key, obj);
}

// call python default eval frame to interpret current frame.
inline static PyObject *eval_frame_default(PyThreadState *tstate,
                                           FrameObject *frame,
                                           int throw_flag) {
#if PY_VERSION_HEX >= 0x03090000
  if (tstate == NULL) {
    tstate = PyThreadState_GET();
  }
  return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
#else
  return _PyEval_EvalFrameDefault(frame, throw_flag);
#endif
}

#if PY_VERSION_HEX >= 0x030b0000

inline static PyObject *eval_custom_code_py311_plus(PyThreadState *tstate,
                                                    FrameObject *frame,
                                                    PyCodeObject *code,
                                                    int throw_flag) {
  Py_ssize_t nlocalsplus_new = code->co_nlocalsplus;
  Py_ssize_t nlocalsplus_old = frame->f_code->co_nlocalsplus;
#if PY_VERSION_HEX >= 0x030c0000
  int size = code->co_framesize;
#else
  // Create a new PyInterpreterFrame. Refer to CALL.
  // PyInterpreterFrame has a head section calls "specials". It follows
  // a contiguous section containing localplus and interpreter stack space.
  int size = nlocalsplus_new + code->co_stacksize + FRAME_SPECIALS_SIZE;
#endif
  CALL_STAT_INC(frames_pushed);
#if PY_VERSION_HEX >= 0x030c0000
  _PyInterpreterFrame *shadow = Internal_PyThreadState_PushFrame(tstate, size);
#else
  _PyInterpreterFrame *shadow =
      (_PyInterpreterFrame *)malloc(sizeof(PyObject *) * size);
#endif
  if (shadow == NULL) {
    // VLOG(7) << "Failed to allocate memory for shadow frame.";
    return NULL;
  }
  // Create a new function object from code object. Refer to MAKE_FUNCTION.
  PyFunctionObject *func =
      (PyFunctionObject *)PyFunction_New((PyObject *)code, frame->f_globals);
  Py_INCREF(func);
#if PY_VERSION_HEX >= 0x030c0000
  Py_XINCREF(((PyFunctionObject *)frame->f_funcobj)->func_closure);
  func->func_closure = ((PyFunctionObject *)frame->f_funcobj)->func_closure;
  _PyFrame_Initialize(shadow, func, NULL, code, 0);
  PyObject **fastlocals_new = shadow->localsplus;
#else
  Py_XINCREF(frame->f_func->func_closure);
  func->func_closure = frame->f_func->func_closure;
  _PyFrame_InitializeSpecials(shadow, func, NULL, code->co_nlocalsplus);
  PyObject **fastlocals_new = shadow->localsplus;

  for (Py_ssize_t i = 0; i < nlocalsplus_new; ++i) {
    fastlocals_new[i] = NULL;
  }
#endif

  PyObject **fastlocals_old = frame->localsplus;

  // The namemap to map the name to index in new frame localsplus.
  PyObject *namemap = PyDict_New();
  if (namemap == NULL) {
    // VLOG(7) << "Failed to create namemap.";
    free(shadow);
    return NULL;
  }
  for (Py_ssize_t i = 0; i < nlocalsplus_new; ++i) {
    PyObject *name = PyTuple_GET_ITEM(code->co_localsplusnames, i);
    PyObject *index = PyLong_FromSize_t(i);
    PyDict_SetItem(namemap, name, index);
  }
  for (Py_ssize_t i = 0; i < nlocalsplus_old; ++i) {
    PyObject *name = PyTuple_GET_ITEM(frame->f_code->co_localsplusnames, i);
    PyObject *index = PyDict_GetItem(namemap, name);
    if (index == NULL) {
      continue;
    }
    Py_XINCREF(fastlocals_old[i]);
    fastlocals_new[PyLong_AsSize_t(index)] = fastlocals_old[i];
  }

  PyObject *result = eval_frame_default(tstate, shadow, throw_flag);
#if PY_VERSION_HEX >= 0x030c0000
  // In Python 3.12+ believes that eval will be cleaned up, but we did not pass
  // in the frame to _PyEval_EvalFrameDefault, so we need to clean it up.
  // elaborate on see:
  // https://github.com/PaddlePaddle/Paddle/pull/61703#issuecomment-1933812625
  Internal_PyEvalFrameClearAndPop(tstate, frame);
#else
  // In Python 3.11 we to create our own isolated frame(namely shadow) and
  // release it after completion
  Internal_PyFrame_Clear(shadow);
  free(shadow);
#endif
  Py_DECREF(func);
  Py_DECREF(namemap);
  return result;
}

#else

inline static PyObject *eval_custom_code_py310_minus(PyThreadState *tstate,
                                                     FrameObject *frame,
                                                     PyCodeObject *code,
                                                     int throw_flag) {
  Py_ssize_t ncells = 0;
  Py_ssize_t nfrees = 0;
  Py_ssize_t nlocals_new = code->co_nlocals;
  Py_ssize_t nlocals_old = frame->f_code->co_nlocals;

  ncells = PyTuple_GET_SIZE(code->co_cellvars);
  nfrees = PyTuple_GET_SIZE(code->co_freevars);

  PyFrameObject *shadow = PyFrame_New(tstate, code, frame->f_globals, NULL);
  if (shadow == NULL) {
    return NULL;
  }

  PyObject **fastlocals_old = frame->f_localsplus;
  PyObject **fastlocals_new = shadow->f_localsplus;

  for (Py_ssize_t i = 0; i < nlocals_old; i++) {
    Py_XINCREF(fastlocals_old[i]);
    fastlocals_new[i] = fastlocals_old[i];
  }

  for (Py_ssize_t i = 0; i < ncells + nfrees; i++) {
    Py_XINCREF(fastlocals_old[nlocals_old + i]);
    fastlocals_new[nlocals_new + i] = fastlocals_old[nlocals_old + i];
  }

  PyObject *result = eval_frame_default(tstate, shadow, throw_flag);
  Py_DECREF(shadow);
  return result;
}

#endif

// Start a new frame and run code in this frame.
// Execute a piece of code by default frame-hook.
inline static PyObject *eval_custom_code(PyThreadState *tstate,
                                         FrameObject *frame,
                                         PyCodeObject *code,
                                         int throw_flag) {
#if PY_VERSION_HEX >= 0x030b0000
  return eval_custom_code_py311_plus(tstate, frame, code, throw_flag);
#else
  return eval_custom_code_py310_minus(tstate, frame, code, throw_flag);
#endif
}

static PyObject *_custom_eval_frame(PyThreadState *tstate,
                                    FrameObject *frame,
                                    int throw_flag,
                                    PyObject *callback) {
  PyObject *out;
  eval_frame_callback_set(Py_None);

// https://peps.python.org/pep-0558/#fast-locals-proxy-implementation-details
// https://devguide.python.org/internals/interpreter/#all-sorts-of-variables
#if PY_VERSION_HEX >= 0x030b0000
  if (frame->owner == FRAME_OWNED_BY_GENERATOR) {
    out = eval_frame_default(tstate, frame, throw_flag);
    eval_frame_callback_set(callback);
    return out;
  }
  if (PyBytes_GET_SIZE(frame->f_code->co_exceptiontable)) {
    eval_frame_callback_set(callback);
    return eval_frame_default(tstate, frame, throw_flag);
  }
  // PyFrame_FastToLocalsWithError receives a PyFrameObject, but if we created a
  // PyFrameObject from a PyInterpreterFrame, it will changes the original
  // PyInterpreterFrame and causes a Segmentation Fault when Fallback to run
  // original frame. So we pass a PyInterpreterFrame to
  // _PyFrame_FastToLocalsWithError directly. But this is an internal API, so we
  // copy many code from CPython project into our project.
  if (Internal_PyFrame_FastToLocalsWithError(frame) < 0) {
#else
  if (frame->f_code->co_flags & 0x20) {
    out = eval_frame_default(tstate, frame, throw_flag);
    eval_frame_callback_set(callback);
    return out;
  }
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
#endif
    return NULL;
  }

  // NOTE:(xiongkun): Handle GeneratorExit exception: (Spend a day)
  // In Python, gen close is also a Python function call that will enter this
  // function with GeneratorExit set, which will cause the PyObject_CallObject
  // raise SystemError. So we disable the custom behavior for GeneratorExit.
  // def func():
  //     iter = iter([1, 2, 3])
  //     for i in iter:
  //         return i # <--- Early return, cause a GeneratorExit thrown,
  //                  # <--- which Cause the PyObject_CallObject raise
  //                  SystemError.
  if (PyErr_ExceptionMatches(PyExc_GeneratorExit)) {
    out = eval_frame_default(tstate, frame, throw_flag);
    eval_frame_callback_set(callback);
    return out;
  }

  PyObject *code;
  PyObject *disable_eval_frame;

  // get code & disable_eval_frame
  if (need_skip(frame)) {
    Py_INCREF(Py_None);
    code = Py_None;
    Py_INCREF(Py_False);
    disable_eval_frame = Py_False;
  } else {
    /* should calculate guards here if we want */
#if PY_VERSION_HEX >= 0x030b0000
    PyObject *args = Py_BuildValue("(O)", PyInterpreterFrameProxy_New(frame));
#else
    PyObject *args = Py_BuildValue("(O)", frame);
#endif
    PyObject *result = PyObject_CallObject(callback, args);
    Py_DECREF(args);
    if (result == NULL) {
#if PY_VERSION_HEX >= 0x030C0000
      Internal_PyEvalFrameClearAndPop(tstate, frame);
#endif
      return NULL;
    }
    code = PyObject_GetAttrString(result, "code");
    disable_eval_frame = PyObject_GetAttrString(result, "disable_eval_frame");
    Py_DECREF(result);
  }

  // code status
  if (is_code_without_graph(code == Py_None ? frame->f_code
                                            : (PyCodeObject *)code) &&
      disable_eval_frame == Py_False) {
    out = eval_frame_default(tstate, frame, throw_flag);
    eval_frame_callback_set(callback);
    Py_DECREF(code);
    Py_DECREF(disable_eval_frame);
    return out;
  }

  // run code
  if (disable_eval_frame != Py_True) {
    // Re-enable custom behavior
    eval_frame_callback_set(callback);
    if (code != Py_None) {
      out = eval_custom_code(tstate, frame, (PyCodeObject *)code, throw_flag);
    } else {
      out = eval_frame_default(tstate, frame, throw_flag);
    }
  } else {
    if (code != Py_None) {
      out = eval_custom_code(tstate, frame, (PyCodeObject *)code, throw_flag);
    } else {
      out = eval_frame_default(tstate, frame, throw_flag);
    }
    // Re-enable custom behavior
    eval_frame_callback_set(callback);
  }

  Py_DECREF(code);
  Py_DECREF(disable_eval_frame);
  return out;
}

static PyObject *_custom_eval_frame_shim(PyThreadState *tstate,
                                         FrameObject *frame,
                                         int throw_flag) {
  PyObject *callback = eval_frame_callback_get();

  if (callback == Py_None) {
    return eval_frame_default(tstate, frame, throw_flag);
  }

  return _custom_eval_frame(tstate, frame, throw_flag, callback);
}

#if PY_VERSION_HEX >= 0x03090000
static PyObject *custom_eval_frame_shim(PyThreadState *tstate,
                                        FrameObject *frame,
                                        int throw_flag) {
  return _custom_eval_frame_shim(tstate, frame, throw_flag);
}
#else
static PyObject *custom_eval_frame_shim(FrameObject *frame, int throw_flag) {
  PyThreadState *tstate = PyThreadState_GET();
  return _custom_eval_frame_shim(tstate, frame, throw_flag);
}
#endif

static PyObject *set_eval_frame(PyObject *new_callback, PyThreadState *tstate) {
  // Change the eval frame callback and return the old one
  //  - None: disables: disable custom callback.
  //  - Python callable(): enables custom callback.
  //  NOTE: Cache is not supported now
  PyObject *old_callback = eval_frame_callback_get();

#if PY_VERSION_HEX >= 0x03090000
  _PyFrameEvalFunction old_eval_frame =
      _PyInterpreterState_GetEvalFrameFunc(tstate->interp);
#else
  // Function pointer.
  _PyFrameEvalFunction old_eval_frame = tstate->interp->eval_frame;
#endif

  // NOTE: multi-threading is not supported now
  if (old_callback != Py_None && new_callback == Py_None) {
    if (old_eval_frame != &_PyEval_EvalFrameDefault) {
      // VLOG(7) << "set _PyEval_EvalFrameDefault";
#if PY_VERSION_HEX >= 0x03090000
      _PyInterpreterState_SetEvalFrameFunc(tstate->interp,
                                           &_PyEval_EvalFrameDefault);
#else
      tstate->interp->eval_frame = &_PyEval_EvalFrameDefault;
#endif
    }
  } else if (old_callback == Py_None && new_callback != Py_None) {
    if (old_eval_frame != &custom_eval_frame_shim) {
      // VLOG(7) << "set custom_eval_frame_shim";
#if PY_VERSION_HEX >= 0x03090000
      _PyInterpreterState_SetEvalFrameFunc(tstate->interp,
                                           &custom_eval_frame_shim);
#else
      tstate->interp->eval_frame = &custom_eval_frame_shim;
#endif
    }
  }

  Py_INCREF(new_callback);
  eval_frame_callback_set(new_callback);

  return old_callback;
}

PyObject *set_eval_frame_py(PyObject *callback) {
  if (callback != Py_None && !PyCallable_Check(callback)) {
    // VLOG(7) << "callback is not a callable or none, invalid arguments.";
    Py_INCREF(Py_None);
    return Py_None;
  }
  return set_eval_frame(callback, PyThreadState_GET());
}

PyMODINIT_FUNC PyInit__eval_frame() {
  PyThread_tss_create(&eval_frame_callback_key);
  // VLOG(7) << "Set PyThread_tss_create return: " << result;

  Py_INCREF(Py_None);
  eval_frame_callback_set(Py_None);

#if PY_VERSION_HEX >= 0x030b0000
  if (PyType_Ready(&PyInterpreterFrameProxyType) < 0) {
    // VLOG(7) << "PyInterpreterFrameProxyType has not been ready!";
  }
  Py_INCREF(&PyInterpreterFrameProxyType);
#endif

  return NULL;
}

#endif
