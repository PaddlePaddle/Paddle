/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <Python.h>
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

// see https://bugs.python.org/issue35886
// If py_version==3.8.*, we need to redefine _PyEvalFrameFunc and the
// related functions and structs.

#if PY_VERSION_HEX >= 0x03080000 && PY_VERSION_HEX < 0x3090000

typedef PyObject *(*_PyFrameEvalFunction)(struct _frame *, int);

struct _warnings_runtime_state {
  /* Both 'filters' and 'onceregistry' can be set in warnings.py;
     get_warnings_attr() will reset these variables accordingly. */
  PyObject *filters;        /* List */
  PyObject *once_registry;  /* Dict */
  PyObject *default_action; /* String */
  long filters_version;     // NOLINT
};

struct _is {
  struct _is *next;
  struct _ts *tstate_head;

  int64_t id;
  int64_t id_refcount;
  int requires_idref;
  PyThread_type_lock id_mutex;

  int finalizing;

  PyObject *modules;
  PyObject *modules_by_index;
  PyObject *sysdict;
  PyObject *builtins;
  PyObject *importlib;

  /* Used in Python/sysmodule.c. */
  int check_interval;

  /* Used in Modules/_threadmodule.c. */
  long num_threads;  // NOLINT
  /* Support for runtime thread stack size tuning.
     A value of 0 means using the platform's default stack size
     or the size specified by the THREAD_STACK_SIZE macro. */
  /* Used in Python/thread.c. */
  size_t pythread_stacksize;

  PyObject *codec_search_path;
  PyObject *codec_search_cache;
  PyObject *codec_error_registry;
  int codecs_initialized;

  /* fs_codec.encoding is initialized to NULL.
     Later, it is set to a non-NULL string by _PyUnicode_InitEncodings(). */
  struct {
    char *encoding; /* Filesystem encoding (encoded to UTF-8) */
    char *errors;   /* Filesystem errors (encoded to UTF-8) */
    _Py_error_handler error_handler;
  } fs_codec;

  PyConfig config;
#ifdef HAVE_DLOPEN
  int dlopenflags;
#endif

  PyObject *dict; /* Stores per-interpreter state */

  PyObject *builtins_copy;
  PyObject *import_func;
  /* Initialized to PyEval_EvalFrameDefault(). */
  _PyFrameEvalFunction eval_frame;

  Py_ssize_t co_extra_user_count;
  freefunc co_extra_freefuncs[MAX_CO_EXTRA_USERS];

#ifdef HAVE_FORK
  PyObject *before_forkers;
  PyObject *after_forkers_parent;
  PyObject *after_forkers_child;
#endif
  /* AtExit module */
  void (*pyexitfunc)(PyObject *);
  PyObject *pyexitmodule;

  uint64_t tstate_next_unique_id;

  struct _warnings_runtime_state warnings;

  PyObject *audit_hooks;
};

#endif

namespace paddle {
namespace pybind {

void BindJit(pybind11::module *m);
void BindEvalFrame(pybind11::module *m);

}  // namespace pybind
}  // namespace paddle
