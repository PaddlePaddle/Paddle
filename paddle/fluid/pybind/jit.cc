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
typedef _PyInterpreterFrame FrameObject;
#else
typedef PyFrameObject FrameObject;
#endif

#define unlikely(x) __builtin_expect((x), 0)

// Use static variable to save customed eval hook.
static Py_tss_t eval_frame_callback_key = {0, 0};

inline static PyObject *eval_frame_callback_get() {
  void *result = PyThread_tss_get(&eval_frame_callback_key);
  if (unlikely(result == NULL)) {
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
  if (tstate == NULL) {
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

  PyFrameObject *shadow = PyFrame_New(tstate, code, frame->f_globals, NULL);
  if (shadow == NULL) {
    return NULL;
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
  if (PyFrame_FastToLocalsWithError(frame->frame_obj) < 0) {
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

  PyObject *args = Py_BuildValue("(O)", frame);
  PyObject *result = PyObject_CallObject(callback, args);
  // result: GuardedCode
  if (result == NULL) {
    // internal exception
    return NULL;
  } else if (result != Py_None) {
    //  NOTE: Cache is not supported now
    PyCodeObject *code = reinterpret_cast<PyCodeObject *>(
        PyObject_GetAttrString(result, "code"));
    PyObject *disable_eval_frame =
        PyObject_GetAttrString(result, "disable_eval_frame");
    if (disable_eval_frame != Py_True) {
      // Re-enable custom behavior
      eval_frame_callback_set(callback);
      auto out = eval_custom_code(tstate, frame, code, throw_flag);
      return out;
    } else {
      auto out = eval_custom_code(tstate, frame, code, throw_flag);
      // Re-enable custom behavior
      eval_frame_callback_set(callback);
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

  return NULL;
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
