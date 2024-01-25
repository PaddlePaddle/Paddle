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

#include "paddle/fluid/pybind/cpython_internals.h"

#include <frameobject.h>

#if PY_VERSION_HEX >= 0x030b0000
#include <internal/pycore_code.h>
#include <internal/pycore_frame.h>
#define Py_BUILD_CORE       // internal/pycore_opcode.h need this macro
#define NEED_OPCODE_TABLES  // To get _PyOpcode_Caches and _PyOpcode_Deopt
#include <internal/pycore_opcode.h>
#undef NEED_OPCODE_TABLES
#undef Py_BUILD_CORE
#include <opcode.h>

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
#if PY_VERSION_HEX >= 0x030c0000
    PyObject *closure = ((PyFunctionObject *)frame->f_funcobj)->func_closure;
    int offset = co->co_nlocals + co->co_ncellvars;
#else
    PyObject *closure = frame->f_func->func_closure;
    int offset = co->co_nlocals + co->co_nplaincellvars;
#endif
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

#if PY_VERSION_HEX < 0x030c0000

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

#endif
#endif  // Python 3.11
