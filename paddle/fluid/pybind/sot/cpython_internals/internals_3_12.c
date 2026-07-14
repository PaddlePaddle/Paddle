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

#include "paddle/fluid/pybind/sot/cpython_internals/internals_3_12.h"

#if (PY_3_12_PLUS && !PY_3_13_PLUS)

#include <internal/pycore_code.h>
#include <internal/pycore_frame.h>
#define Py_BUILD_CORE       // internal/pycore_opcode.h need this macro
#define NEED_OPCODE_TABLES  // To get _PyOpcode_Caches and _PyOpcode_Deopt

// see https://github.com/python/cpython/issues/105268#issuecomment-1678256123
#undef _PyGC_FINALIZED
#include <internal/pycore_runtime.h>
#define Internal_PyObject_Arena (_PyRuntime.allocators.obj_arena)
#define _PyGC_FINALIZED

#include <internal/pycore_opcode.h>

#undef NEED_OPCODE_TABLES
#undef Py_BUILD_CORE
#include <opcode.h>

/* Minimum size of data stack chunk */
#define DATA_STACK_CHUNK_SIZE (16 * 1024)
#define MINIMUM_OVERHEAD 1000

int Internal_PyUnstable_InterpreterFrame_GetLine(_PyInterpreterFrame *frame) {
  int addr = _PyInterpreterFrame_LASTI(frame) * sizeof(_Py_CODEUNIT);
  return PyCode_Addr2Line(frame->f_code, addr);
}

static void take_ownership(PyFrameObject *f, _PyInterpreterFrame *frame) {
  assert(frame->owner != FRAME_OWNED_BY_CSTACK);
  assert(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
  assert(frame->owner != FRAME_CLEARED);
  Py_ssize_t size =
      ((char *)&frame->localsplus[frame->stacktop]) - (char *)frame;
  Py_INCREF(frame->f_code);
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

void Internal_PyFrame_ClearExceptCode(_PyInterpreterFrame *frame) {
  /* It is the responsibility of the owning generator/coroutine
   * to have cleared the enclosing generator, if any. */
  assert(frame->owner != FRAME_OWNED_BY_GENERATOR ||
         _PyFrame_GetGenerator(frame)->gi_frame_state == FRAME_CLEARED);
  // GH-99729: Clearing this frame can expose the stack (via finalizers). It's
  // crucial that this frame has been unlinked, and is no longer visible:
  assert(PyThreadState_GET()->cframe->current_frame != frame);
  if (frame->frame_obj) {
    PyFrameObject *f = frame->frame_obj;
    frame->frame_obj = NULL;
    if (Py_REFCNT(f) > 1) {
      take_ownership(f, frame);
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
  Py_DECREF(frame->f_funcobj);
}

void *Internal_PyObject_VirtualAlloc(size_t size) {
  return Internal_PyObject_Arena.alloc(Internal_PyObject_Arena.ctx, size);
}

void Internal_PyObject_VirtualFree(void *obj, size_t size) {
  Internal_PyObject_Arena.free(Internal_PyObject_Arena.ctx, obj, size);
}

void Internal_PyThreadState_PopFrame(PyThreadState *tstate,
                                     _PyInterpreterFrame *frame) {
  assert(tstate->datastack_chunk);
  PyObject **base = (PyObject **)frame;
  if (base == &tstate->datastack_chunk->data[0]) {
    _PyStackChunk *chunk = tstate->datastack_chunk;
    _PyStackChunk *previous = chunk->previous;
    // push_chunk ensures that the root chunk is never popped:
    assert(previous);
    tstate->datastack_top = &previous->data[previous->top];
    tstate->datastack_chunk = previous;
    Internal_PyObject_VirtualFree(chunk, chunk->size);
    tstate->datastack_limit =
        (PyObject **)(((char *)previous) + previous->size);
  } else {
    assert(tstate->datastack_top);
    assert(tstate->datastack_top >= base);
    tstate->datastack_top = base;
  }
}

static void clear_thread_frame(PyThreadState *tstate,
                               _PyInterpreterFrame *frame) {
  assert(frame->owner == FRAME_OWNED_BY_THREAD);
  // Make sure that this is, indeed, the top frame. We can't check this in
  // _PyThreadState_PopFrame, since f_code is already cleared at that point:
  assert((PyObject **)frame + frame->f_code->co_framesize ==
         tstate->datastack_top);
  tstate->c_recursion_remaining--;
  assert(frame->frame_obj == NULL || frame->frame_obj->f_frame == frame);
  Internal_PyFrame_ClearExceptCode(frame);
  Py_DECREF(frame->f_code);
  tstate->c_recursion_remaining++;
  Internal_PyThreadState_PopFrame(tstate, frame);
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
  Internal_PyFrame_ClearExceptCode(frame);
  tstate->c_recursion_remaining++;
  frame->previous = NULL;
}

void Internal_PyEvalFrameClearAndPop(PyThreadState *tstate,
                                     _PyInterpreterFrame *frame) {
  if (frame->owner == FRAME_OWNED_BY_THREAD) {
    clear_thread_frame(tstate, frame);
  } else {
    clear_gen_frame(tstate, frame);
  }
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

// The Internal_ prefix is not used here because it is the internal usage of the
// _PyFrame_GetFrameObject method
PyFrameObject *_PyFrame_MakeAndSetFrameObject(_PyInterpreterFrame *frame) {
  assert(frame->frame_obj == NULL);
  PyObject *exc = PyErr_GetRaisedException();

  PyFrameObject *f = Internal_PyFrame_New_NoTrack(frame->f_code);
  if (f == NULL) {
    Py_XDECREF(exc);
    return NULL;
  }
  PyErr_SetRaisedException(exc);
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

// Initialize frame free variables if needed
static void frame_init_get_vars(_PyInterpreterFrame *frame) {
  // COPY_FREE_VARS has no quickened forms, so no need to use _PyOpcode_Deopt
  // here:
  PyCodeObject *co = frame->f_code;
  int lasti = _PyInterpreterFrame_LASTI(frame);
  if (!(lasti < 0 && _PyCode_CODE(co)->op.code == COPY_FREE_VARS &&
        PyFunction_Check(frame->f_funcobj))) {
    /* Free vars are initialized */
    return;
  }

  /* Free vars have not been initialized -- Do that */
  PyObject *closure = ((PyFunctionObject *)frame->f_funcobj)->func_closure;
  int offset = PyCode_GetFirstFree(co);
  for (int i = 0; i < co->co_nfreevars; ++i) {
    PyObject *o = PyTuple_GET_ITEM(closure, i);
    frame->localsplus[offset + i] = Py_NewRef(o);
  }
  // COPY_FREE_VARS doesn't have inline caches, either:
  frame->prev_instr = _PyCode_CODE(frame->f_code);
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
    int check_opcode = _PyOpcode_Deopt[instruction->op.code];
    check_oparg |= instruction->op.arg;
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

static int frame_get_var(_PyInterpreterFrame *frame,
                         PyCodeObject *co,
                         int i,
                         PyObject **pvalue) {
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
    return 0;
  }

  PyObject *value = frame->localsplus[i];
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
  *pvalue = value;
  return 1;
}

PyObject *Internal_PyFrame_GetLocals(_PyInterpreterFrame *frame,
                                     int include_hidden) {
  /* Merge fast locals into f->f_locals */
  PyObject *locals = frame->f_locals;
  if (locals == NULL) {
    locals = frame->f_locals = PyDict_New();
    if (locals == NULL) {
      return NULL;
    }
  }
  PyObject *hidden = NULL;

  /* If include_hidden, "hidden" fast locals (from inlined comprehensions in
     module/class scopes) will be included in the returned dict, but not in
     frame->f_locals; the returned dict will be a modified copy. Non-hidden
     locals will still be updated in frame->f_locals. */
  if (include_hidden) {
    hidden = PyDict_New();
    if (hidden == NULL) {
      return NULL;
    }
  }

  frame_init_get_vars(frame);

  PyCodeObject *co = frame->f_code;
  for (int i = 0; i < co->co_nlocalsplus; i++) {
    PyObject *value;  // borrowed reference
    if (!frame_get_var(frame, co, i, &value)) {
      continue;
    }

    PyObject *name = PyTuple_GET_ITEM(co->co_localsplusnames, i);
    _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);
    if (kind & CO_FAST_HIDDEN) {
      if (include_hidden && value != NULL) {
        if (PyObject_SetItem(hidden, name, value) != 0) {
          goto error;
        }
      }
      continue;
    }
    if (value == NULL) {
      if (PyObject_DelItem(locals, name) != 0) {
        if (PyErr_ExceptionMatches(PyExc_KeyError)) {
          PyErr_Clear();
        } else {
          goto error;
        }
      }
    } else {
      if (PyObject_SetItem(locals, name, value) != 0) {
        goto error;
      }
    }
  }

  if (include_hidden && PyDict_Size(hidden)) {
    PyObject *innerlocals = PyDict_New();
    if (innerlocals == NULL) {
      goto error;
    }
    if (PyDict_Merge(innerlocals, locals, 1) != 0) {
      Py_DECREF(innerlocals);
      goto error;
    }
    if (PyDict_Merge(innerlocals, hidden, 1) != 0) {
      Py_DECREF(innerlocals);
      goto error;
    }
    locals = innerlocals;
  } else {
    Py_INCREF(locals);
  }
  Py_CLEAR(hidden);

  return locals;

error:
  Py_XDECREF(hidden);
  return NULL;
}

int Internal_PyFrame_FastToLocalsWithError(_PyInterpreterFrame *frame) {
  PyObject *locals = Internal_PyFrame_GetLocals(frame, 0);
  if (locals == NULL) {
    return -1;
  }
  Py_DECREF(locals);
  return 0;
}

#endif  // (PY_3_12_PLUS && !PY_3_13_PLUS)
