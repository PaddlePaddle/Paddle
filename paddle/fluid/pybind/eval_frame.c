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

#include "paddle/fluid/pybind/eval_frame.h"

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
#if PY_VERSION_HEX >= 0x030b0000
#include <internal/pycore_code.h>
#include <internal/pycore_frame.h>
#define Py_BUILD_CORE       // internal/pycore_opcode.h need this macro
#define NEED_OPCODE_TABLES  // To get _PyOpcode_Caches and _PyOpcode_Deopt
#include <internal/pycore_opcode.h>
#undef NEED_OPCODE_TABLES
#undef Py_BUILD_CORE
#include <opcode.h>
#endif

#include <object.h>
#include <pystate.h>

#if PY_VERSION_HEX >= 0x030b0000
// To avoid the error: undefined symbol: _PyFrame_GetFrameObject, all we need is
// to redefine this function based source code in python3.11. The advantage is
// that we don't need any modification in eval_frame functions.
typedef _PyInterpreterFrame FrameObject;
#define CALL_STAT_INC(name) ((void)0)

int Internal_PyInterpreterFrame_GetLine(_PyInterpreterFrame *frame);
static int Internal_PyFrame_OpAlreadyRan(_PyInterpreterFrame *frame,
                                         int opcode,
                                         int oparg);
int Internal_PyFrame_FastToLocalsWithError(_PyInterpreterFrame *frame);
PyFrameObject *Internal_PyFrame_New_NoTrack(PyCodeObject *code);
PyFrameObject *Internal_PyFrame_MakeAndSetFrameObject(
    _PyInterpreterFrame *frame);
static inline PyFrameObject *Internal_PyFrame_GetFrameObject(
    _PyInterpreterFrame *frame);
static void Internal_take_ownership(PyFrameObject *f,
                                    _PyInterpreterFrame *frame);
void Internal_PyFrame_Clear(_PyInterpreterFrame *frame);

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

// We copy some cpython internal API from cpython project.
// To avoid name conflict, we use "Internal_" prefix to mark them.
int Internal_PyInterpreterFrame_GetLine(_PyInterpreterFrame *frame) {
  int addr = _PyInterpreterFrame_LASTI(frame) * sizeof(_Py_CODEUNIT);
  return PyCode_Addr2Line(frame->f_code, addr);
}

static int Internal_PyFrame_OpAlreadyRan(_PyInterpreterFrame *frame,
                                         int opcode,
                                         int oparg) {
  // This only works when opcode is a non-quickened form:
  assert(_PyOpcode_Deopt[opcode] == opcode);
  int check_oparg = 0;
  for (_Py_CODEUNIT *instruction = _PyCode_CODE(frame->f_code);
       instruction < frame->prev_instr;
       instruction++) {
    int check_opcode = _PyOpcode_Deopt[_Py_OPCODE(*instruction)];
    check_oparg |= _Py_OPARG(*instruction);
    if (check_opcode == opcode && check_oparg == oparg) {
      return 1;
    }
    if (check_opcode == EXTENDED_ARG) {
      check_oparg <<= 8;
    } else {
      check_oparg = 0;
    }
    instruction += _PyOpcode_Caches[check_opcode];
  }
  return 0;
}

int Internal_PyFrame_FastToLocalsWithError(_PyInterpreterFrame *frame) {
  /* Merge fast locals into f->f_locals */
  PyObject *locals;
  PyObject **fast;
  PyCodeObject *co;
  locals = frame->f_locals;
  if (locals == NULL) {
    locals = frame->f_locals = PyDict_New();
    if (locals == NULL) return -1;
  }
  co = frame->f_code;
  fast = _PyFrame_GetLocalsArray(frame);
  // COPY_FREE_VARS has no quickened forms, so no need to use _PyOpcode_Deopt
  // here:
  int lasti = _PyInterpreterFrame_LASTI(frame);
  if (lasti < 0 && _Py_OPCODE(_PyCode_CODE(co)[0]) == COPY_FREE_VARS) {
    /* Free vars have not been initialized -- Do that */
    PyCodeObject *co = frame->f_code;
    PyObject *closure = frame->f_func->func_closure;
    int offset = co->co_nlocals + co->co_nplaincellvars;
    for (int i = 0; i < co->co_nfreevars; ++i) {
      PyObject *o = PyTuple_GET_ITEM(closure, i);
      Py_INCREF(o);
      frame->localsplus[offset + i] = o;
    }
    // COPY_FREE_VARS doesn't have inline CACHEs, either:
    frame->prev_instr = _PyCode_CODE(frame->f_code);
  }
  for (int i = 0; i < co->co_nlocalsplus; i++) {
    _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);

    /* If the namespace is unoptimized, then one of the
       following cases applies:
       1. It does not contain free variables, because it
          uses import * or is a top-level namespace.
       2. It is a class namespace.
       We don't want to accidentally copy free variables
       into the locals dict used by the class.
    */
    if (kind & CO_FAST_FREE && !(co->co_flags & CO_OPTIMIZED)) {
      continue;
    }

    PyObject *name = PyTuple_GET_ITEM(co->co_localsplusnames, i);
    PyObject *value = fast[i];
    if (frame->stacktop) {
      if (kind & CO_FAST_FREE) {
        // The cell was set by COPY_FREE_VARS.
        assert(value != NULL && PyCell_Check(value));
        value = PyCell_GET(value);
      } else if (kind & CO_FAST_CELL) {
        // Note that no *_DEREF ops can happen before MAKE_CELL
        // executes.  So there's no need to duplicate the work
        // that MAKE_CELL would otherwise do later, if it hasn't
        // run yet.
        if (value != NULL) {
          if (PyCell_Check(value) &&
              Internal_PyFrame_OpAlreadyRan(frame, MAKE_CELL, i)) {
            // (likely) MAKE_CELL must have executed already.
            value = PyCell_GET(value);
          }
          // (likely) Otherwise it it is an arg (kind & CO_FAST_LOCAL),
          // with the initial value set when the frame was created...
          // (unlikely) ...or it was set to some initial value by
          // an earlier call to PyFrame_LocalsToFast().
        }
      }
    } else {
      assert(value == NULL);
    }
    if (value == NULL) {
      if (PyObject_DelItem(locals, name) != 0) {
        if (PyErr_ExceptionMatches(PyExc_KeyError)) {
          PyErr_Clear();
        } else {
          return -1;
        }
      }
    } else {
      if (PyObject_SetItem(locals, name, value) != 0) {
        return -1;
      }
    }
  }
  return 0;
}

PyFrameObject *Internal_PyFrame_New_NoTrack(PyCodeObject *code) {
  CALL_STAT_INC(frame_objects_created);
  int slots = code->co_nlocalsplus + code->co_stacksize;
  PyFrameObject *f = PyObject_GC_NewVar(PyFrameObject, &PyFrame_Type, slots);
  if (f == NULL) {
    return NULL;
  }
  f->f_back = NULL;
  f->f_trace = NULL;
  f->f_trace_lines = 1;
  f->f_trace_opcodes = 0;
  f->f_fast_as_locals = 0;
  f->f_lineno = 0;
  return f;
}

PyFrameObject *Internal_PyFrame_MakeAndSetFrameObject(
    _PyInterpreterFrame *frame) {
  assert(frame->frame_obj == NULL);
  PyObject *error_type, *error_value, *error_traceback;
  PyErr_Fetch(&error_type, &error_value, &error_traceback);

  PyFrameObject *f = Internal_PyFrame_New_NoTrack(frame->f_code);
  if (f == NULL) {
    Py_XDECREF(error_type);
    Py_XDECREF(error_value);
    Py_XDECREF(error_traceback);
    return NULL;
  }
  PyErr_Restore(error_type, error_value, error_traceback);
  if (frame->frame_obj) {
    // GH-97002: How did we get into this horrible situation? Most likely,
    // allocating f triggered a GC collection, which ran some code that
    // *also* created the same frame... while we were in the middle of
    // creating it! See test_sneaky_frame_object in test_frame.py for a
    // concrete example.
    //
    // Regardless, just throw f away and use that frame instead, since it's
    // already been exposed to user code. It's actually a bit tricky to do
    // this, since we aren't backed by a real _PyInterpreterFrame anymore.
    // Just pretend that we have an owned, cleared frame so frame_dealloc
    // doesn't make the situation worse:
    f->f_frame = (_PyInterpreterFrame *)f->_f_frame_data;
    f->f_frame->owner = FRAME_CLEARED;
    f->f_frame->frame_obj = f;
    Py_DECREF(f);
    return frame->frame_obj;
  }
  assert(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
  assert(frame->owner != FRAME_CLEARED);
  f->f_frame = frame;
  frame->frame_obj = f;
  return f;
}

static inline PyFrameObject *Internal_PyFrame_GetFrameObject(
    _PyInterpreterFrame *frame) {
  assert(!_PyFrame_IsIncomplete(frame));
  PyFrameObject *res = frame->frame_obj;
  if (res != NULL) {
    return res;
  }
  return Internal_PyFrame_MakeAndSetFrameObject(frame);
}

static void Internal_take_ownership(PyFrameObject *f,
                                    _PyInterpreterFrame *frame) {
  assert(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
  assert(frame->owner != FRAME_CLEARED);
  Py_ssize_t size =
      ((char *)&frame->localsplus[frame->stacktop]) - (char *)frame;
  memcpy((_PyInterpreterFrame *)f->_f_frame_data, frame, size);
  frame = (_PyInterpreterFrame *)f->_f_frame_data;
  f->f_frame = frame;
  frame->owner = FRAME_OWNED_BY_FRAME_OBJECT;
  if (_PyFrame_IsIncomplete(frame)) {
    // This may be a newly-created generator or coroutine frame. Since it's
    // dead anyways, just pretend that the first RESUME ran:
    PyCodeObject *code = frame->f_code;
    frame->prev_instr = _PyCode_CODE(code) + code->_co_firsttraceable;
  }
  assert(!_PyFrame_IsIncomplete(frame));
  assert(f->f_back == NULL);
  _PyInterpreterFrame *prev = frame->previous;
  while (prev && _PyFrame_IsIncomplete(prev)) {
    prev = prev->previous;
  }
  if (prev) {
    /* Link PyFrameObjects.f_back and remove link through
     * _PyInterpreterFrame.previous */
    PyFrameObject *back = Internal_PyFrame_GetFrameObject(prev);
    if (back == NULL) {
      /* Memory error here. */
      assert(PyErr_ExceptionMatches(PyExc_MemoryError));
      /* Nothing we can do about it */
      PyErr_Clear();
    } else {
      f->f_back = (PyFrameObject *)Py_NewRef(back);
    }
    frame->previous = NULL;
  }
  if (!PyObject_GC_IsTracked((PyObject *)f)) {
    PyObject_GC_Track((PyObject *)f);
  }
}

void Internal_PyFrame_Clear(_PyInterpreterFrame *frame) {
  /* It is the responsibility of the owning generator/coroutine
   * to have cleared the enclosing generator, if any. */
  assert(frame->owner != FRAME_OWNED_BY_GENERATOR ||
         _PyFrame_GetGenerator(frame)->gi_frame_state == FRAME_CLEARED);
  // GH-99729: Clearing this frame can expose the stack (via finalizers). It's
  // crucial that this frame has been unlinked, and is no longer visible:
  assert(_PyThreadState_GET()->cframe->current_frame != frame);
  if (frame->frame_obj) {
    PyFrameObject *f = frame->frame_obj;
    frame->frame_obj = NULL;
    if (Py_REFCNT(f) > 1) {
      Internal_take_ownership(f, frame);
      Py_DECREF(f);
      return;
    }
    Py_DECREF(f);
  }
  assert(frame->stacktop >= 0);
  for (int i = 0; i < frame->stacktop; i++) {
    Py_XDECREF(frame->localsplus[i]);
  }
  Py_XDECREF(frame->frame_obj);
  Py_XDECREF(frame->f_locals);
  Py_DECREF(frame->f_func);
  Py_DECREF(frame->f_code);
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
  // Create a new PyInterpreterFrame. Refer to CALL.
  // PyInterpreterFrame has a head section calls "specials". It follows
  // a contiguous section containing localplus and interpreter stack space.
  size_t size = nlocalsplus_new + code->co_stacksize + FRAME_SPECIALS_SIZE;
  CALL_STAT_INC(frames_pushed);
  _PyInterpreterFrame *shadow =
      (_PyInterpreterFrame *)malloc(sizeof(PyObject *) * size);
  if (shadow == NULL) {
    // VLOG(7) << "Failed to allocate memory for shadow frame.";
    return NULL;
  }
  // Create a new function object from code object. Refer to MAKE_FUNCTION.
  PyFunctionObject *func =
      (PyFunctionObject *)PyFunction_New((PyObject *)code, frame->f_globals);
  Py_XINCREF(frame->f_func->func_closure);
  func->func_closure = frame->f_func->func_closure;
  _PyFrame_InitializeSpecials(shadow, func, NULL, code->co_nlocalsplus);

  PyObject **fastlocals_old = frame->localsplus;
  PyObject **fastlocals_new = shadow->localsplus;

  for (Py_ssize_t i = 0; i < nlocalsplus_new; ++i) {
    fastlocals_new[i] = NULL;
  }

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
  Internal_PyFrame_Clear(shadow);
  free(shadow);
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
// https://peps.python.org/pep-0558/#fast-locals-proxy-implementation-details
// https://devguide.python.org/internals/interpreter/#all-sorts-of-variables
#if PY_VERSION_HEX >= 0x030b0000
  if (frame->owner == FRAME_OWNED_BY_GENERATOR) {
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
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
#endif
    return NULL;
  }

  // NOTE:(xiongkun): Handle GeneratorExit exception: (Spend a day)
  // In Python, gen close is also a Python function call that will enter this
  // function with GeneratorExit set, which will cause the PyObject_CallObject
  // raise SystemError. So we disable the custom behavior for GeneratorExit. def
  // func():
  //     iter = iter([1, 2, 3])
  //     for i in iter:
  //         return i # <--- Early return, cause a GeneratorExit thrown,
  //                  # <--- which Cause the PyObject_CallObject raise
  //                  SystemError.
  if (PyErr_ExceptionMatches(PyExc_GeneratorExit)) {
    return eval_frame_default(tstate, frame, throw_flag);
  }

  // We don't run the current custom_eval_frame behavior for guards.
  // So we temporarily set the callback to Py_None to drive the correct behavior
  // in the shim.
  eval_frame_callback_set(Py_None);

#if PY_VERSION_HEX >= 0x030b0000
  PyObject *args = Py_BuildValue("(O)", PyInterpreterFrameProxy_New(frame));
#else
  PyObject *args = Py_BuildValue("(O)", frame);
#endif
  PyObject *result = PyObject_CallObject(callback, args);
  Py_DECREF(args);
  // VLOG(7) << "After call eval_frame_function and decrease frame.";
  // class CustomCode(Protocal):
  //     code: CodeType | None
  //     disable_eval_frame: bool
  // result: CustomCode
  if (result == NULL) {
    // internal exception
    // VLOG(7) << "Error happened.";
    return NULL;
  } else {
    //  NOTE: Cache is not supported now
    PyCodeObject *code = (PyCodeObject *)PyObject_GetAttrString(result, "code");
    PyObject *disable_eval_frame =
        PyObject_GetAttrString(result, "disable_eval_frame");
    PyObject *out;
    // VLOG(7) << "Start eval new frame and code.";
    if (disable_eval_frame != Py_True) {
      // Re-enable custom behavior
      eval_frame_callback_set(callback);
      if ((PyObject *)code != Py_None) {
        out = eval_custom_code(tstate, frame, code, throw_flag);
      } else {
        out = eval_frame_default(tstate, frame, throw_flag);
      }
    } else {
      if ((PyObject *)code != Py_None) {
        out = eval_custom_code(tstate, frame, code, throw_flag);
      } else {
        out = eval_frame_default(tstate, frame, throw_flag);
      }
      // Re-enable custom behavior
      eval_frame_callback_set(callback);
    }
    Py_DECREF(result);
    Py_DECREF(code);
    return out;
  }
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
