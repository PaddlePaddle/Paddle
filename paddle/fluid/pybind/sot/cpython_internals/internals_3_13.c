// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/sot/cpython_internals/internals_3_13.h"
#include "paddle/fluid/pybind/sot/macros.h"

#include <internal/pycore_code.h>
#include <internal/pycore_frame.h>
#define Py_BUILD_CORE       // internal/pycore_opcode.h need this macro
#define NEED_OPCODE_TABLES  // To get _PyOpcode_Caches and _PyOpcode_Deopt

// see https://github.com/python/cpython/issues/105268#issuecomment-1678256123
#undef _PyGC_FINALIZED
#include <internal/pycore_runtime.h>
#define Internal_PyObject_Arena (_PyRuntime.allocators.obj_arena)
#define _PyGC_FINALIZED

#include <internal/pycore_opcode_metadata.h>

#undef NEED_OPCODE_TABLES
#undef Py_BUILD_CORE
#include <opcode.h>

/* Minimum size of data stack chunk */
#define DATA_STACK_CHUNK_SIZE \
  (16 * 1024)  // py314+ is _PY_DATA_STACK_CHUNK_SIZE
#define MINIMUM_OVERHEAD 1000

int Internal_PyUnstable_InterpreterFrame_GetLine(_PyInterpreterFrame *frame) {
  int addr = _PyInterpreterFrame_LASTI(frame) * sizeof(_Py_CODEUNIT);
  return PyCode_Addr2Line(_PyFrame_GetCode(frame), addr);
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
  f->f_lineno = 0;
  f->f_extra_locals = NULL;
  f->f_locals_cache = NULL;
  return f;
}

// The Internal_ prefix is not used here because it is the internal usage of the
// _PyFrame_GetFrameObject method
PyFrameObject *_PyFrame_MakeAndSetFrameObject(_PyInterpreterFrame *frame) {
  assert(frame->frame_obj == NULL);
  PyObject *exc = PyErr_GetRaisedException();

  PyFrameObject *f = Internal_PyFrame_New_NoTrack(_PyFrame_GetCode(frame));
  if (f == NULL) {
    Py_XDECREF(exc);
    return NULL;
  }
  PyErr_SetRaisedException(exc);

  // GH-97002: There was a time when a frame object could be created when we
  // are allocating the new frame object f above, so frame->frame_obj would
  // be assigned already. That path does not exist anymore. We won't call any
  // Python code in this function and garbage collection will not run.
  // Notice that _PyFrame_New_NoTrack() can potentially raise a MemoryError,
  // but it won't allocate a traceback until the frame unwinds, so we are safe
  // here.
  assert(frame->frame_obj == NULL);
  assert(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
  assert(frame->owner != FRAME_CLEARED);
  f->f_frame = frame;
  frame->frame_obj = f;
  return f;
}

static void Internal_take_ownership(PyFrameObject *f,
                                    _PyInterpreterFrame *frame) {
  assert(frame->owner != FRAME_OWNED_BY_CSTACK);
  assert(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
  assert(frame->owner != FRAME_CLEARED);
  Py_ssize_t size =
      ((char *)&frame->localsplus[frame->stacktop]) - (char *)frame;
  Py_INCREF(_PyFrame_GetCode(frame));
  memcpy((_PyInterpreterFrame *)f->_f_frame_data, frame, size);
  frame = (_PyInterpreterFrame *)f->_f_frame_data;
  f->f_frame = frame;
  frame->owner = FRAME_OWNED_BY_FRAME_OBJECT;
  if (_PyFrame_IsIncomplete(frame)) {
    // This may be a newly-created generator or coroutine frame. Since it's
    // dead anyways, just pretend that the first RESUME ran:
    PyCodeObject *code = _PyFrame_GetCode(frame);
    frame->instr_ptr = _PyCode_CODE(code) + code->_co_firsttraceable + 1;
  }
  assert(!_PyFrame_IsIncomplete(frame));
  assert(f->f_back == NULL);
  _PyInterpreterFrame *prev = _PyFrame_GetFirstComplete(frame->previous);
  frame->previous = NULL;
  if (prev) {
    assert(prev->owner != FRAME_OWNED_BY_CSTACK);
    /* Link PyFrameObjects.f_back and remove link through
     * _PyInterpreterFrame.previous */
    PyFrameObject *back = _PyFrame_GetFrameObject(prev);
    if (back == NULL) {
      /* Memory error here. */
      assert(PyErr_ExceptionMatches(PyExc_MemoryError));
      /* Nothing we can do about it */
      PyErr_Clear();
    } else {
      f->f_back = (PyFrameObject *)Py_NewRef(back);
    }
  }
  if (!_PyObject_GC_IS_TRACKED((PyObject *)f)) {
    PyObject_GC_Track((PyObject *)f);
  }
}

void Internal_PyFrame_ClearLocals(_PyInterpreterFrame *frame) {
  assert(frame->stacktop >= 0);
  int stacktop = frame->stacktop;
  frame->stacktop = 0;
  for (int i = 0; i < stacktop; i++) {
    Py_XDECREF(frame->localsplus[i]);
  }
  Py_CLEAR(frame->f_locals);
}

void Internal_PyFrame_ClearExceptCode(_PyInterpreterFrame *frame) {
  /* It is the responsibility of the owning generator/coroutine
   * to have cleared the enclosing generator, if any. */
  assert(frame->owner != FRAME_OWNED_BY_GENERATOR ||
         _PyFrame_GetGenerator(frame)->gi_frame_state == FRAME_CLEARED);
  // GH-99729: Clearing this frame can expose the stack (via finalizers). It's
  // crucial that this frame has been unlinked, and is no longer visible:
  assert(_PyThreadState_GET()->current_frame != frame);
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
  Internal_PyFrame_ClearLocals(frame);
  Py_DECREF(frame->f_funcobj);
}

static void clear_thread_frame(PyThreadState *tstate,
                               _PyInterpreterFrame *frame) {
  assert(frame->owner == FRAME_OWNED_BY_THREAD);
  // Make sure that this is, indeed, the top frame. We can't check this in
  // _PyThreadState_PopFrame, since f_code is already cleared at that point:
  assert((PyObject **)frame + _PyFrame_GetCode(frame)->co_framesize ==
         tstate->datastack_top);
  tstate->c_recursion_remaining--;
  assert(frame->frame_obj == NULL || frame->frame_obj->f_frame == frame);
  Internal_PyFrame_ClearExceptCode(frame);
  Py_DECREF(frame->f_executable);
  tstate->c_recursion_remaining++;
  _PyThreadState_PopFrame(tstate, frame);
}

static void clear_gen_frame(PyThreadState *tstate, _PyInterpreterFrame *frame) {
  assert(frame->owner == FRAME_OWNED_BY_GENERATOR);
  PyGenObject *gen = _PyFrame_GetGenerator(frame);
  gen->gi_frame_state = FRAME_CLEARED;
  assert(tstate->exc_info == &gen->gi_exc_state);
  tstate->exc_info = gen->gi_exc_state.previous_item;
  gen->gi_exc_state.previous_item = NULL;
  tstate->c_recursion_remaining--;
  assert(frame->frame_obj == NULL || frame->frame_obj->f_frame == frame);
  frame->previous = NULL;
  Internal_PyFrame_ClearExceptCode(frame);
  _PyErr_ClearExcState(&gen->gi_exc_state);
  tstate->c_recursion_remaining++;
}

void Internal_PyEval_FrameClearAndPop(PyThreadState *tstate,
                                      _PyInterpreterFrame *frame) {
  if (frame->owner == FRAME_OWNED_BY_THREAD) {
    clear_thread_frame(tstate, frame);
  } else {
    clear_gen_frame(tstate, frame);
  }
}

void *Internal_PyObject_VirtualAlloc(size_t size) {
  return Internal_PyObject_Arena.alloc(Internal_PyObject_Arena.ctx, size);
}

static _PyStackChunk *allocate_chunk(int size_in_bytes,
                                     _PyStackChunk *previous) {
  assert(size_in_bytes % sizeof(PyObject **) == 0);
  _PyStackChunk *res = Internal_PyObject_VirtualAlloc(size_in_bytes);
  if (res == NULL) {
    return NULL;
  }
  res->previous = previous;
  res->size = size_in_bytes;
  res->top = 0;
  return res;
}

static PyObject **push_chunk(PyThreadState *tstate, int size) {
  int allocate_size = DATA_STACK_CHUNK_SIZE;
  while (allocate_size < (int)sizeof(PyObject *) * (size + MINIMUM_OVERHEAD)) {
    allocate_size *= 2;
  }
  _PyStackChunk *new = allocate_chunk(allocate_size, tstate->datastack_chunk);
  if (new == NULL) {
    return NULL;
  }
  if (tstate->datastack_chunk) {
    tstate->datastack_chunk->top =
        tstate->datastack_top - &tstate->datastack_chunk->data[0];
  }
  tstate->datastack_chunk = new;
  tstate->datastack_limit = (PyObject **)(((char *)new) + allocate_size);
  // When new is the "root" chunk (i.e. new->previous == NULL), we can keep
  // _PyThreadState_PopFrame from freeing it later by "skipping" over the
  // first element:
  PyObject **res = &new->data[new->previous == NULL];
  tstate->datastack_top = res + size;
  return res;
}

_PyInterpreterFrame *Internal_PyThreadState_PushFrame(PyThreadState *tstate,
                                                      size_t size) {
  assert(size < INT_MAX / sizeof(PyObject *));
  if (_PyThreadState_HasStackSpace(tstate, (int)size)) {
    _PyInterpreterFrame *res = (_PyInterpreterFrame *)tstate->datastack_top;
    tstate->datastack_top += size;
    return res;
  }
  return (_PyInterpreterFrame *)push_chunk(tstate, (int)size);
}

// This function is used to get the locals mapping of the frame.
void update_framelocals_mapping(PyObject *mapping,
                                PyCodeObject *code,
                                int i,
                                PyObject *value) {
  _PyLocals_Kind kind = _PyLocals_GetKind(code->co_localspluskinds, i);

  if (kind & CO_FAST_FREE && !(code->co_flags & CO_OPTIMIZED)) {
    return;
  }

  if (kind & CO_FAST_HIDDEN) {
    return;
  }

  if (kind & CO_FAST_FREE) {
    assert(value != NULL && PyCell_Check(value));
    value = PyCell_GET(value);
  }

  if (value != NULL) {
    PyDict_SetItem(
        mapping, PyTuple_GET_ITEM(code->co_localsplusnames, i), value);
  }
}

// simplified version `frame_get_var`, `frame_init_get_vars` and
// `PyFrame_GetLocals`
PyObject *get_framelocals_mapping(_PyInterpreterFrame *frame) {
  PyObject *mapping = PyDict_New();

  // If the frame is not yet executed, return an empty mapping, see
  // `frame_get_var` function
  if (!frame->stacktop) {
    return mapping;
  }

  PyCodeObject *co = PyFrame_GET_CODE(frame);

  // Get local variables, see `frame_get_var` function
  int offset = co->co_nlocalsplus - co->co_nfreevars;
  for (int i = 0; i < offset; i++) {
    update_framelocals_mapping(mapping, co, i, frame->localsplus[i]);
  }

  // Get closure variables, see `frame_init_get_vars` function
  PyObject *closure = ((PyFunctionObject *)frame->f_funcobj)->func_closure;
  for (int i = 0; i < co->co_nfreevars; ++i) {
    update_framelocals_mapping(
        mapping, co, offset + i, PyTuple_GET_ITEM(closure, i));
  }

  return mapping;
}
