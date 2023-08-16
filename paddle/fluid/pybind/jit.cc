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

#include "paddle/fluid/pybind/jit.h"

#include <Python.h>
#include <frameobject.h>

#if PY_VERSION_HEX < 0x030b0000
#include <code.h>
#endif
#if PY_VERSION_HEX >= 0x030b0000
#include <internal/pycore_frame.h>
#endif

#include <object.h>
#include <pystate.h>

#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/platform/place.h"

#include "glog/logging.h"
#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/serializer.h"
#include "paddle/utils/pybind.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

#if PY_VERSION_HEX >= 0x030b0000
// To avoid the error: undefined symbol: _PyFrame_GetFrameObject, all we need is
// to redefine this function based source code in python3.11. The advantage is
// that we don't need any modification in eval_frame functions.
typedef _PyInterpreterFrame FrameObject;
#define CALL_STAT_INC(name) ((void)0)
PyFrameObject *Paddle_PyFrame_New_NoTrack(PyCodeObject *code) {
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

static inline bool Paddle_PyFrame_IsIncomplete(_PyInterpreterFrame *frame) {
  return frame->owner != FRAME_OWNED_BY_GENERATOR &&
         frame->prev_instr <
             _PyCode_CODE(frame->f_code) + frame->f_code->_co_firsttraceable;
}

PyFrameObject *Paddle_PyFrame_MakeAndSetFrameObject(
    _PyInterpreterFrame *frame) {
  assert(frame->frame_obj == NULL);
  PyObject *error_type, *error_value, *error_traceback;
  PyErr_Fetch(&error_type, &error_value, &error_traceback);

  PyFrameObject *f = Paddle_PyFrame_New_NoTrack(frame->f_code);
  if (f == NULL) {
    Py_XDECREF(error_type);
    Py_XDECREF(error_value);
    Py_XDECREF(error_traceback);
    return NULL;  // NOLINT
  }
  PyErr_Restore(error_type, error_value, error_traceback);
  if (frame->frame_obj) {
    f->f_frame = (_PyInterpreterFrame *)f->_f_frame_data;  // NOLINT
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

static inline PyFrameObject *Paddle_PyFrame_GetFrameObject(
    _PyInterpreterFrame *frame) {
  assert(!Paddle_PyFrame_IsIncomplete(frame));
  PyFrameObject *res = frame->frame_obj;
  if (res != NULL) {
    return res;
  }
  return Paddle_PyFrame_MakeAndSetFrameObject(frame);
}

#else
typedef PyFrameObject FrameObject;
#endif

#define unlikely(x) __builtin_expect((x), 0)

// Use static variable to save customed eval hook.
static Py_tss_t eval_frame_callback_key = {0, 0};

inline static PyObject *eval_frame_callback_get() {
  void *result = PyThread_tss_get(&eval_frame_callback_key);
  if (unlikely(result == nullptr)) {
    Py_RETURN_NONE;
  } else {
    return reinterpret_cast<PyObject *>(result);
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
  if (tstate == nullptr) {
    tstate = PyThreadState_GET();
  }
  return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
#else
  return _PyEval_EvalFrameDefault(frame, throw_flag);
#endif
}

// Start a new frame and run code in this frame.
// Execute a piece of code by default frame-hook.
inline static PyObject *eval_custom_code(PyThreadState *tstate,
                                         FrameObject *frame,
                                         PyCodeObject *code,
                                         int throw_flag) {
  Py_ssize_t ncells = 0;
  Py_ssize_t nfrees = 0;
  Py_ssize_t nlocals_new = code->co_nlocals;
  Py_ssize_t nlocals_old = frame->f_code->co_nlocals;

#if PY_VERSION_HEX >= 0x030b0000
  ncells = code->co_ncellvars;
  nfrees = code->co_nfreevars;
#else
  ncells = PyTuple_GET_SIZE(code->co_cellvars);
  nfrees = PyTuple_GET_SIZE(code->co_freevars);
#endif

  PyFrameObject *shadow = PyFrame_New(tstate, code, frame->f_globals, nullptr);
  if (shadow == nullptr) {
    return nullptr;
  }

#if PY_VERSION_HEX >= 0x030b0000
  PyObject **fastlocals_old = frame->localsplus;
  PyObject **fastlocals_new = shadow->f_frame->localsplus;
#else
  PyObject **fastlocals_old = frame->f_localsplus;
  PyObject **fastlocals_new = shadow->f_localsplus;
#endif

  for (Py_ssize_t i = 0; i < nlocals_old; i++) {
    Py_XINCREF(fastlocals_old[i]);
    fastlocals_new[i] = fastlocals_old[i];
  }

  for (Py_ssize_t i = 0; i < ncells + nfrees; i++) {
    Py_XINCREF(fastlocals_old[nlocals_old + i]);
    fastlocals_new[nlocals_new + i] = fastlocals_old[nlocals_old + i];
  }

#if PY_VERSION_HEX >= 0x030b0000
  PyObject *result = eval_frame_default(tstate, shadow->f_frame, throw_flag);
#else
  PyObject *result = eval_frame_default(tstate, shadow, throw_flag);
#endif
  Py_DECREF(shadow);
  return result;
}

static PyObject *_custom_eval_frame(PyThreadState *tstate,
                                    FrameObject *frame,
                                    int throw_flag,
                                    PyObject *callback) {
// https://peps.python.org/pep-0558/#fast-locals-proxy-implementation-details
// https://devguide.python.org/internals/interpreter/#all-sorts-of-variables
#if PY_VERSION_HEX >= 0x030b0000
  // _PyFrame_GetFrameObject(frame) # this function should be the right answer,
  // but nm libpython.so | grep _PyFrame_MakeAndSetFrameObject is a `t' symbol,
  // which means it's local to library. we will get a link error if we use it.
  if (PyFrame_FastToLocalsWithError(Paddle_PyFrame_GetFrameObject(frame)) < 0) {
#else
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
#endif
    return nullptr;
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
  PyObject *args = Py_BuildValue("(O)", Paddle_PyFrame_GetFrameObject(frame));
#else
  PyObject *args = Py_BuildValue("(O)", frame);
#endif
  PyObject *result = PyObject_CallObject(callback, args);
  Py_DECREF(args);
  VLOG(7) << "After call eval_frame_function and decrease frame.";
  // result: GuardedCode
  if (result == nullptr) {
    // internal exception
    VLOG(7) << "Error happened.";
    return nullptr;
  } else if (result != Py_None) {
    //  NOTE: Cache is not supported now
    PyCodeObject *code = reinterpret_cast<PyCodeObject *>(
        PyObject_GetAttrString(result, "code"));
    PyObject *disable_eval_frame =
        PyObject_GetAttrString(result, "disable_eval_frame");
    if (disable_eval_frame != Py_True) {
      // Re-enable custom behavior
      eval_frame_callback_set(callback);
      VLOG(7) << "Start eval new frame and code.";
      auto out = eval_custom_code(tstate, frame, code, throw_flag);
      Py_DECREF(result);
      Py_DECREF(code);
      return out;
    } else {
      auto out = eval_custom_code(tstate, frame, code, throw_flag);
      // Re-enable custom behavior
      eval_frame_callback_set(callback);
      Py_DECREF(result);
      Py_DECREF(code);
      return out;
    }
  } else {
    // Re-enable custom behavior
    eval_frame_callback_set(callback);
    return eval_frame_default(tstate, frame, throw_flag);
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
  auto *old_eval_frame = _PyInterpreterState_GetEvalFrameFunc(tstate->interp);
#else
  // Function pointer.
  _PyFrameEvalFunction old_eval_frame = tstate->interp->eval_frame;
#endif

  // NOTE: multi-threading is not supported now
  if (old_callback != Py_None && new_callback == Py_None) {
    if (old_eval_frame != &_PyEval_EvalFrameDefault) {
      VLOG(7) << "set _PyEval_EvalFrameDefault";
#if PY_VERSION_HEX >= 0x03090000
      _PyInterpreterState_SetEvalFrameFunc(tstate->interp,
                                           &_PyEval_EvalFrameDefault);
#else
      tstate->interp->eval_frame = &_PyEval_EvalFrameDefault;
#endif
    }
  } else if (old_callback == Py_None && new_callback != Py_None) {
    if (old_eval_frame != &custom_eval_frame_shim) {
      VLOG(7) << "set custom_eval_frame_shim";
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

static PyObject *set_eval_frame_py(PyObject *callback) {
  if (callback != Py_None && !PyCallable_Check(callback)) {
    VLOG(7) << "callback is not a callable or none, invalid arguments.";
    RETURN_PY_NONE
  }
  return set_eval_frame(callback, PyThreadState_GET());
}

PyMODINIT_FUNC PyInit__eval_frame() {
  int result = PyThread_tss_create(&eval_frame_callback_key);
  VLOG(7) << "Set PyThread_tss_create return: " << result;

  Py_INCREF(Py_None);
  eval_frame_callback_set(Py_None);

  return nullptr;
}

PyTypeObject *g_jit_function_pytype = nullptr;
using Variable = paddle::framework::Variable;

void BindJit(pybind11::module *m) {
  py::class_<jit::Layer>(*m, "Layer", R"DOC(Layer Class.)DOC")
      .def("function_names", &jit::Layer::FunctionNames)
      .def("function", &jit::Layer::Function)
      .def("function_info", &jit::Layer::FunctionInfo);

  py::class_<jit::Function, std::shared_ptr<jit::Function>> function(
      *m, "Function", R"DOC(Function Class.)DOC");
  g_jit_function_pytype = reinterpret_cast<PyTypeObject *>(function.ptr());

  py::class_<jit::FunctionInfo, std::shared_ptr<jit::FunctionInfo>>(
      *m, "FunctionInfo", R"DOC(FunctionInfo Class.)DOC")
      .def("name", &jit::FunctionInfo::FunctionName)
      .def("input_names", &jit::FunctionInfo::InputArgNames)
      .def("output_names", &jit::FunctionInfo::OutputArgNames);

  m->def("Load",
         [](const std::string &path, const platform::CPUPlace &cpu_place) {
           return paddle::jit::Load(path, cpu_place);
         });

  m->def("Load",
         [](const std::string &path, const platform::CUDAPlace &cuda_place) {
           return paddle::jit::Load(path, cuda_place);
         });
}

void BindEvalFrame(pybind11::module *m) {
  PyInit__eval_frame();
  m->def(
      "set_eval_frame",
      [](const py::object &py_func) {
        VLOG(5) << "start call set_eval_frame_py.";
        auto ret = set_eval_frame_py(py_func.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("callback"));
}

}  // namespace pybind
}  // namespace paddle
