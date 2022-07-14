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

#pragma once

#include <Python.h>
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/api/include/strings_api.h"
#include "paddle/phi/api/lib/dygraph_api.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "pybind11/detail/common.h"

namespace paddle {
namespace pybind {

static PyObject *eager_final_state_api_atan2(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "atan2 pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: atan2";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("atan2", "x", args, 0, false);
    auto y = GetTensorFromArgs("atan2", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::atan2_final_state_dygraph_function(x, y)) out =
        ::atan2_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_bernoulli(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "bernoulli pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: bernoulli";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("bernoulli", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::bernoulli(x)) out =
        paddle::experimental::bernoulli(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_cholesky(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "cholesky pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cholesky";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cholesky", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *upper_obj = PyTuple_GET_ITEM(args, 1);
    bool upper = CastPyArg2Boolean(upper_obj, "cholesky", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cholesky_final_state_dygraph_function(x, upper)) out =
        ::cholesky_final_state_dygraph_function(x, upper);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_cholesky_solve(PyObject *self,
                                                      PyObject *args,
                                                      PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "cholesky_solve pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cholesky_solve";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cholesky_solve", "x", args, 0, false);
    auto y = GetTensorFromArgs("cholesky_solve", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *upper_obj = PyTuple_GET_ITEM(args, 2);
    bool upper = CastPyArg2Boolean(upper_obj, "cholesky_solve", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cholesky_solve_final_state_dygraph_function(x, y, upper)) out =
        ::cholesky_solve_final_state_dygraph_function(x, y, upper);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_cross(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "cross pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cross";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cross", "x", args, 0, false);
    auto y = GetTensorFromArgs("cross", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "cross", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cross_final_state_dygraph_function(x, y, axis)) out =
        ::cross_final_state_dygraph_function(x, y, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_diagonal(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "diagonal pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: diagonal";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("diagonal", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 1);
    int offset = CastPyArg2Int(offset_obj, "diagonal", 1);
    PyObject *axis1_obj = PyTuple_GET_ITEM(args, 2);
    int axis1 = CastPyArg2Int(axis1_obj, "diagonal", 2);
    PyObject *axis2_obj = PyTuple_GET_ITEM(args, 3);
    int axis2 = CastPyArg2Int(axis2_obj, "diagonal", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::diagonal_final_state_dygraph_function(
        x, offset, axis1, axis2)) out =
        ::diagonal_final_state_dygraph_function(x, offset, axis1, axis2);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_digamma(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "digamma pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: digamma";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("digamma", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::digamma_final_state_dygraph_function(x)) out =
        ::digamma_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_dist(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "dist pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: dist";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("dist", "x", args, 0, false);
    auto y = GetTensorFromArgs("dist", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *p_obj = PyTuple_GET_ITEM(args, 2);
    float p = CastPyArg2Float(p_obj, "dist", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::dist_final_state_dygraph_function(x, y, p)) out =
        ::dist_final_state_dygraph_function(x, y, p);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_dot(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "dot pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: dot";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("dot", "x", args, 0, false);
    auto y = GetTensorFromArgs("dot", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::dot_final_state_dygraph_function(x, y)) out =
        ::dot_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_erf(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "erf pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: erf";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("erf", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::erf_final_state_dygraph_function(x)) out =
        ::erf_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_mv(PyObject *self,
                                          PyObject *args,
                                          PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "mv pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: mv";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("mv", "x", args, 0, false);
    auto vec = GetTensorFromArgs("mv", "vec", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::mv_final_state_dygraph_function(x, vec)) out =
        ::mv_final_state_dygraph_function(x, vec);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_poisson(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "poisson pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: poisson";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("poisson", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::poisson_final_state_dygraph_function(x)) out =
        ::poisson_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_trace(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "trace pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: trace";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("trace", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 1);
    int offset = CastPyArg2Int(offset_obj, "trace", 1);
    PyObject *axis1_obj = PyTuple_GET_ITEM(args, 2);
    int axis1 = CastPyArg2Int(axis1_obj, "trace", 2);
    PyObject *axis2_obj = PyTuple_GET_ITEM(args, 3);
    int axis2 = CastPyArg2Int(axis2_obj, "trace", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::trace_final_state_dygraph_function(
        x, offset, axis1, axis2)) out =
        ::trace_final_state_dygraph_function(x, offset, axis1, axis2);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_trunc(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "trunc pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: trunc";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("trunc", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::trunc_final_state_dygraph_function(x)) out =
        ::trunc_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_abs(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "abs pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: abs";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("abs", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::abs_final_state_dygraph_function(x)) out =
        ::abs_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_accuracy(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "accuracy pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: accuracy";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("accuracy", "x", args, 0, false);
    auto indices = GetTensorFromArgs("accuracy", "indices", args, 1, false);
    auto label = GetTensorFromArgs("accuracy", "label", args, 2, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::accuracy(x, indices, label)) out =
        paddle::experimental::accuracy(x, indices, label);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_acos(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "acos pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: acos";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("acos", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::acos_final_state_dygraph_function(x)) out =
        ::acos_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_acosh(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "acosh pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: acosh";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("acosh", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::acosh_final_state_dygraph_function(x)) out =
        ::acosh_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_adadelta(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "adadelta pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: adadelta";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("adadelta", "param", args, 0, false);
    auto grad = GetTensorFromArgs("adadelta", "grad", args, 1, false);
    auto avg_squared_grad =
        GetTensorFromArgs("adadelta", "avg_squared_grad", args, 2, false);
    auto avg_squared_update =
        GetTensorFromArgs("adadelta", "avg_squared_update", args, 3, false);

    // Parse Attributes if needed
    PyObject *rho_obj = PyTuple_GET_ITEM(args, 4);
    float rho = CastPyArg2Float(rho_obj, "adadelta", 4);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 5);
    float epsilon = CastPyArg2Float(epsilon_obj, "adadelta", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::adadelta(
        param, grad, avg_squared_grad, avg_squared_update, rho, epsilon)) out =
        paddle::experimental::adadelta(
            param, grad, avg_squared_grad, avg_squared_update, rho, epsilon);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_adam_(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "adam_ pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: adam_";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("adam_", "param", args, 0, false);
    auto grad = GetTensorFromArgs("adam_", "grad", args, 1, false);
    auto learning_rate =
        GetTensorFromArgs("adam_", "learning_rate", args, 2, false);
    auto moment1 = GetTensorFromArgs("adam_", "moment1", args, 3, false);
    auto moment2 = GetTensorFromArgs("adam_", "moment2", args, 4, false);
    auto beta1_pow = GetTensorFromArgs("adam_", "beta1_pow", args, 5, false);
    auto beta2_pow = GetTensorFromArgs("adam_", "beta2_pow", args, 6, false);
    auto master_param =
        GetOptionalTensorFromArgs("adam_", "master_param", args, 7, true);
    auto skip_update =
        GetOptionalTensorFromArgs("adam_", "skip_update", args, 8, true);

    // Parse Attributes if needed
    PyObject *beta1_obj = PyTuple_GET_ITEM(args, 9);
    paddle::experimental::Scalar beta1 =
        CastPyArg2Scalar(beta1_obj, "adam_", 9);
    PyObject *beta2_obj = PyTuple_GET_ITEM(args, 10);
    paddle::experimental::Scalar beta2 =
        CastPyArg2Scalar(beta2_obj, "adam_", 10);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 11);
    paddle::experimental::Scalar epsilon =
        CastPyArg2Scalar(epsilon_obj, "adam_", 11);
    PyObject *lazy_mode_obj = PyTuple_GET_ITEM(args, 12);
    bool lazy_mode = CastPyArg2Boolean(lazy_mode_obj, "adam_", 12);
    PyObject *min_row_size_to_use_multithread_obj = PyTuple_GET_ITEM(args, 13);
    int64_t min_row_size_to_use_multithread =
        CastPyArg2Long(min_row_size_to_use_multithread_obj, "adam_", 13);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 14);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "adam_", 14);
    PyObject *use_global_beta_pow_obj = PyTuple_GET_ITEM(args, 15);
    bool use_global_beta_pow =
        CastPyArg2Boolean(use_global_beta_pow_obj, "adam_", 15);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::adam_(param,
                                         grad,
                                         learning_rate,
                                         moment1,
                                         moment2,
                                         beta1_pow,
                                         beta2_pow,
                                         master_param,
                                         skip_update,
                                         beta1,
                                         beta2,
                                         epsilon,
                                         lazy_mode,
                                         min_row_size_to_use_multithread,
                                         multi_precision,
                                         use_global_beta_pow)) out =
        paddle::experimental::adam_(param,
                                    grad,
                                    learning_rate,
                                    moment1,
                                    moment2,
                                    beta1_pow,
                                    beta2_pow,
                                    master_param,
                                    skip_update,
                                    beta1,
                                    beta2,
                                    epsilon,
                                    lazy_mode,
                                    min_row_size_to_use_multithread,
                                    multi_precision,
                                    use_global_beta_pow);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;

    inplace_var_idx_map[1] = 3;

    inplace_var_idx_map[2] = 4;

    inplace_var_idx_map[3] = 5;

    inplace_var_idx_map[4] = 6;

    inplace_var_idx_map[5] = 7;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_adamax(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "adamax pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: adamax";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("adamax", "param", args, 0, false);
    auto grad = GetTensorFromArgs("adamax", "grad", args, 1, false);
    auto learning_rate =
        GetTensorFromArgs("adamax", "learning_rate", args, 2, false);
    auto moment = GetTensorFromArgs("adamax", "moment", args, 3, false);
    auto inf_norm = GetTensorFromArgs("adamax", "inf_norm", args, 4, false);
    auto beta1_pow = GetTensorFromArgs("adamax", "beta1_pow", args, 5, false);

    // Parse Attributes if needed
    PyObject *beta1_obj = PyTuple_GET_ITEM(args, 6);
    float beta1 = CastPyArg2Float(beta1_obj, "adamax", 6);
    PyObject *beta2_obj = PyTuple_GET_ITEM(args, 7);
    float beta2 = CastPyArg2Float(beta2_obj, "adamax", 7);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 8);
    float epsilon = CastPyArg2Float(epsilon_obj, "adamax", 8);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::adamax(param,
                                          grad,
                                          learning_rate,
                                          moment,
                                          inf_norm,
                                          beta1_pow,
                                          beta1,
                                          beta2,
                                          epsilon)) out =
        paddle::experimental::adamax(param,
                                     grad,
                                     learning_rate,
                                     moment,
                                     inf_norm,
                                     beta1_pow,
                                     beta1,
                                     beta2,
                                     epsilon);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_adamw(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "adamw pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: adamw";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("adamw", "param", args, 0, false);
    auto grad = GetTensorFromArgs("adamw", "grad", args, 1, false);
    auto learning_rate =
        GetTensorFromArgs("adamw", "learning_rate", args, 2, false);
    auto moment1 = GetTensorFromArgs("adamw", "moment1", args, 3, false);
    auto moment2 = GetTensorFromArgs("adamw", "moment2", args, 4, false);
    auto beta1_pow = GetTensorFromArgs("adamw", "beta1_pow", args, 5, false);
    auto beta2_pow = GetTensorFromArgs("adamw", "beta2_pow", args, 6, false);
    auto master_param =
        GetOptionalTensorFromArgs("adamw", "master_param", args, 7, true);
    auto skip_update =
        GetOptionalTensorFromArgs("adamw", "skip_update", args, 8, true);

    // Parse Attributes if needed
    PyObject *beta1_obj = PyTuple_GET_ITEM(args, 9);
    paddle::experimental::Scalar beta1 =
        CastPyArg2Scalar(beta1_obj, "adamw", 9);
    PyObject *beta2_obj = PyTuple_GET_ITEM(args, 10);
    paddle::experimental::Scalar beta2 =
        CastPyArg2Scalar(beta2_obj, "adamw", 10);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 11);
    paddle::experimental::Scalar epsilon =
        CastPyArg2Scalar(epsilon_obj, "adamw", 11);
    PyObject *lr_ratio_obj = PyTuple_GET_ITEM(args, 12);
    float lr_ratio = CastPyArg2Float(lr_ratio_obj, "adamw", 12);
    PyObject *coeff_obj = PyTuple_GET_ITEM(args, 13);
    float coeff = CastPyArg2Float(coeff_obj, "adamw", 13);
    PyObject *with_decay_obj = PyTuple_GET_ITEM(args, 14);
    bool with_decay = CastPyArg2Boolean(with_decay_obj, "adamw", 14);
    PyObject *lazy_mode_obj = PyTuple_GET_ITEM(args, 15);
    bool lazy_mode = CastPyArg2Boolean(lazy_mode_obj, "adamw", 15);
    PyObject *min_row_size_to_use_multithread_obj = PyTuple_GET_ITEM(args, 16);
    int64_t min_row_size_to_use_multithread =
        CastPyArg2Long(min_row_size_to_use_multithread_obj, "adamw", 16);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 17);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "adamw", 17);
    PyObject *use_global_beta_pow_obj = PyTuple_GET_ITEM(args, 18);
    bool use_global_beta_pow =
        CastPyArg2Boolean(use_global_beta_pow_obj, "adamw", 18);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::adamw(param,
                                         grad,
                                         learning_rate,
                                         moment1,
                                         moment2,
                                         beta1_pow,
                                         beta2_pow,
                                         master_param,
                                         skip_update,
                                         beta1,
                                         beta2,
                                         epsilon,
                                         lr_ratio,
                                         coeff,
                                         with_decay,
                                         lazy_mode,
                                         min_row_size_to_use_multithread,
                                         multi_precision,
                                         use_global_beta_pow)) out =
        paddle::experimental::adamw(param,
                                    grad,
                                    learning_rate,
                                    moment1,
                                    moment2,
                                    beta1_pow,
                                    beta2_pow,
                                    master_param,
                                    skip_update,
                                    beta1,
                                    beta2,
                                    epsilon,
                                    lr_ratio,
                                    coeff,
                                    with_decay,
                                    lazy_mode,
                                    min_row_size_to_use_multithread,
                                    multi_precision,
                                    use_global_beta_pow);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_add(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "add pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: add";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("add", "x", args, 0, false);
    auto y = GetTensorFromArgs("add", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::add_final_state_dygraph_function(x, y)) out =
        ::add_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_add_n(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "add_n pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: add_n";

    // Get EagerTensors from args
    auto x = GetTensorListFromArgs("add_n", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::add_n_final_state_dygraph_function(x)) out =
        ::add_n_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_addmm(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "addmm pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: addmm";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("addmm", "input", args, 0, false);
    auto x = GetTensorFromArgs("addmm", "x", args, 1, false);
    auto y = GetTensorFromArgs("addmm", "y", args, 2, false);

    // Parse Attributes if needed
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 3);
    float alpha = CastPyArg2Float(alpha_obj, "addmm", 3);
    PyObject *beta_obj = PyTuple_GET_ITEM(args, 4);
    float beta = CastPyArg2Float(beta_obj, "addmm", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::addmm_final_state_dygraph_function(
        input, x, y, alpha, beta)) out =
        ::addmm_final_state_dygraph_function(input, x, y, alpha, beta);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_all(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "all pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: all";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("all", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "all", 1);
    PyObject *keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "all", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::all(x, dims, keep_dim)) out =
        paddle::experimental::all(x, dims, keep_dim);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_allclose(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "allclose pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: allclose";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("allclose", "x", args, 0, false);
    auto y = GetTensorFromArgs("allclose", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *rtol_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar rtol =
        CastPyArg2Scalar(rtol_obj, "allclose", 2);
    PyObject *atol_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::Scalar atol =
        CastPyArg2Scalar(atol_obj, "allclose", 3);
    PyObject *equal_nan_obj = PyTuple_GET_ITEM(args, 4);
    bool equal_nan = CastPyArg2Boolean(equal_nan_obj, "allclose", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::allclose(x, y, rtol, atol, equal_nan)) out =
        paddle::experimental::allclose(x, y, rtol, atol, equal_nan);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_any(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "any pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: any";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("any", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "any", 1);
    PyObject *keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "any", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::any(x, dims, keep_dim)) out =
        paddle::experimental::any(x, dims, keep_dim);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_arange(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "arange pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: arange";

    // Get EagerTensors from args
    auto start = GetTensorFromArgs("arange", "start", args, 0, false);
    auto end = GetTensorFromArgs("arange", "end", args, 1, false);
    auto step = GetTensorFromArgs("arange", "step", args, 2, false);

    // Parse Attributes if needed
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "arange", 3);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 4);
    paddle::Place place = CastPyArg2Place(place_obj, "arange", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::arange(start, end, step, dtype, place)) out =
        paddle::experimental::arange(start, end, step, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_argmax(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "argmax pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: argmax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("argmax", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int64_t axis = CastPyArg2Long(axis_obj, "argmax", 1);
    PyObject *keepdims_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdims = CastPyArg2Boolean(keepdims_obj, "argmax", 2);
    PyObject *flatten_obj = PyTuple_GET_ITEM(args, 3);
    bool flatten = CastPyArg2Boolean(flatten_obj, "argmax", 3);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 4);
    int dtype = CastPyArg2Int(dtype_obj, "argmax", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::argmax(
        x, axis, keepdims, flatten, dtype)) out =
        paddle::experimental::argmax(x, axis, keepdims, flatten, dtype);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_argmin(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "argmin pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: argmin";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("argmin", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int64_t axis = CastPyArg2Long(axis_obj, "argmin", 1);
    PyObject *keepdims_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdims = CastPyArg2Boolean(keepdims_obj, "argmin", 2);
    PyObject *flatten_obj = PyTuple_GET_ITEM(args, 3);
    bool flatten = CastPyArg2Boolean(flatten_obj, "argmin", 3);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 4);
    int dtype = CastPyArg2Int(dtype_obj, "argmin", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::argmin(
        x, axis, keepdims, flatten, dtype)) out =
        paddle::experimental::argmin(x, axis, keepdims, flatten, dtype);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_argsort(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "argsort pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: argsort";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("argsort", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "argsort", 1);
    PyObject *descending_obj = PyTuple_GET_ITEM(args, 2);
    bool descending = CastPyArg2Boolean(descending_obj, "argsort", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::argsort_final_state_dygraph_function(x, axis, descending)) out =
        ::argsort_final_state_dygraph_function(x, axis, descending);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_asin(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "asin pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: asin";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("asin", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::asin_final_state_dygraph_function(x)) out =
        ::asin_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_asinh(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "asinh pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: asinh";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("asinh", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::asinh_final_state_dygraph_function(x)) out =
        ::asinh_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_assign(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "assign pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: assign";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("assign", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::assign_final_state_dygraph_function(x)) out =
        ::assign_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_assign_out_(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "assign_out_ pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: assign_out_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("assign_out_", "x", args, 0, false);
    auto output = GetTensorFromArgs("assign_out_", "output", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::assign_out__final_state_dygraph_function(x, output)) out =
        ::assign_out__final_state_dygraph_function(x, output);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 1;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_atan(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "atan pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: atan";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("atan", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::atan_final_state_dygraph_function(x)) out =
        ::atan_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_atanh(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "atanh pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: atanh";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("atanh", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::atanh_final_state_dygraph_function(x)) out =
        ::atanh_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_auc(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "auc pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: auc";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("auc", "x", args, 0, false);
    auto label = GetTensorFromArgs("auc", "label", args, 1, false);
    auto stat_pos = GetTensorFromArgs("auc", "stat_pos", args, 2, false);
    auto stat_neg = GetTensorFromArgs("auc", "stat_neg", args, 3, false);

    // Parse Attributes if needed
    PyObject *curve_obj = PyTuple_GET_ITEM(args, 4);
    std::string curve = CastPyArg2String(curve_obj, "auc", 4);
    PyObject *num_thresholds_obj = PyTuple_GET_ITEM(args, 5);
    int num_thresholds = CastPyArg2Int(num_thresholds_obj, "auc", 5);
    PyObject *slide_steps_obj = PyTuple_GET_ITEM(args, 6);
    int slide_steps = CastPyArg2Int(slide_steps_obj, "auc", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::auc(
        x, label, stat_pos, stat_neg, curve, num_thresholds, slide_steps)) out =
        paddle::experimental::auc(
            x, label, stat_pos, stat_neg, curve, num_thresholds, slide_steps);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_batch_norm(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "batch_norm pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: batch_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("batch_norm", "x", args, 0, false);
    auto scale = GetTensorFromArgs("batch_norm", "scale", args, 1, false);
    auto bias = GetTensorFromArgs("batch_norm", "bias", args, 2, false);
    auto mean = GetTensorFromArgs("batch_norm", "mean", args, 3, false);
    auto variance = GetTensorFromArgs("batch_norm", "variance", args, 4, false);

    // Parse Attributes if needed
    PyObject *momentum_obj = PyTuple_GET_ITEM(args, 5);
    float momentum = CastPyArg2Float(momentum_obj, "batch_norm", 5);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 6);
    float epsilon = CastPyArg2Float(epsilon_obj, "batch_norm", 6);
    PyObject *data_layout_obj = PyTuple_GET_ITEM(args, 7);
    std::string data_layout =
        CastPyArg2String(data_layout_obj, "batch_norm", 7);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 8);
    bool is_test = CastPyArg2Boolean(is_test_obj, "batch_norm", 8);
    PyObject *use_global_stats_obj = PyTuple_GET_ITEM(args, 9);
    bool use_global_stats =
        CastPyArg2Boolean(use_global_stats_obj, "batch_norm", 9);
    PyObject *trainable_statistics_obj = PyTuple_GET_ITEM(args, 10);
    bool trainable_statistics =
        CastPyArg2Boolean(trainable_statistics_obj, "batch_norm", 10);
    PyObject *fuse_with_relu_obj = PyTuple_GET_ITEM(args, 11);
    bool fuse_with_relu =
        CastPyArg2Boolean(fuse_with_relu_obj, "batch_norm", 11);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::batch_norm_final_state_dygraph_function(x,
                                                       scale,
                                                       bias,
                                                       mean,
                                                       variance,
                                                       momentum,
                                                       epsilon,
                                                       data_layout,
                                                       is_test,
                                                       use_global_stats,
                                                       trainable_statistics,
                                                       fuse_with_relu)) out =
        ::batch_norm_final_state_dygraph_function(x,
                                                  scale,
                                                  bias,
                                                  mean,
                                                  variance,
                                                  momentum,
                                                  epsilon,
                                                  data_layout,
                                                  is_test,
                                                  use_global_stats,
                                                  trainable_statistics,
                                                  fuse_with_relu);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_bce_loss(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "bce_loss pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: bce_loss";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("bce_loss", "input", args, 0, false);
    auto label = GetTensorFromArgs("bce_loss", "label", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::bce_loss_final_state_dygraph_function(input, label)) out =
        ::bce_loss_final_state_dygraph_function(input, label);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_bitwise_and(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "bitwise_and pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: bitwise_and";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("bitwise_and", "x", args, 0, false);
    auto y = GetTensorFromArgs("bitwise_and", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::bitwise_and(x, y)) out =
        paddle::experimental::bitwise_and(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_bitwise_not(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "bitwise_not pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: bitwise_not";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("bitwise_not", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::bitwise_not(x)) out =
        paddle::experimental::bitwise_not(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_bitwise_or(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "bitwise_or pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: bitwise_or";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("bitwise_or", "x", args, 0, false);
    auto y = GetTensorFromArgs("bitwise_or", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::bitwise_or(x, y)) out =
        paddle::experimental::bitwise_or(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_bitwise_xor(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "bitwise_xor pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: bitwise_xor";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("bitwise_xor", "x", args, 0, false);
    auto y = GetTensorFromArgs("bitwise_xor", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::bitwise_xor(x, y)) out =
        paddle::experimental::bitwise_xor(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_brelu(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "brelu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: brelu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("brelu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *t_min_obj = PyTuple_GET_ITEM(args, 1);
    float t_min = CastPyArg2Float(t_min_obj, "brelu", 1);
    PyObject *t_max_obj = PyTuple_GET_ITEM(args, 2);
    float t_max = CastPyArg2Float(t_max_obj, "brelu", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::brelu_final_state_dygraph_function(x, t_min, t_max)) out =
        ::brelu_final_state_dygraph_function(x, t_min, t_max);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_cast(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "cast pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cast";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cast", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *out_dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType out_dtype =
        CastPyArg2DataType(out_dtype_obj, "cast", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cast_final_state_dygraph_function(x, out_dtype)) out =
        ::cast_final_state_dygraph_function(x, out_dtype);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_ceil(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "ceil pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: ceil";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("ceil", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::ceil_final_state_dygraph_function(x)) out =
        ::ceil_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_celu(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "celu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: celu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("celu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 1);
    float alpha = CastPyArg2Float(alpha_obj, "celu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::celu_final_state_dygraph_function(x, alpha)) out =
        ::celu_final_state_dygraph_function(x, alpha);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_clip(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "clip pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: clip";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("clip", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *min_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar min = CastPyArg2Scalar(min_obj, "clip", 1);
    PyObject *max_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar max = CastPyArg2Scalar(max_obj, "clip", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::clip_final_state_dygraph_function(x, min, max)) out =
        ::clip_final_state_dygraph_function(x, min, max);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_clip_(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "clip pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: clip_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("clip", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *min_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar min = CastPyArg2Scalar(min_obj, "clip", 1);
    PyObject *max_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar max = CastPyArg2Scalar(max_obj, "clip", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::clip__final_state_dygraph_function(x, min, max)) out =
        ::clip__final_state_dygraph_function(x, min, max);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_concat(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "concat pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: concat";

    // Get EagerTensors from args
    auto x = GetTensorListFromArgs("concat", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar axis = CastPyArg2Scalar(axis_obj, "concat", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::concat_final_state_dygraph_function(x, axis)) out =
        ::concat_final_state_dygraph_function(x, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_conj(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "conj pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: conj";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("conj", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::conj_final_state_dygraph_function(x)) out =
        ::conj_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_conv2d(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "conv2d pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: conv2d";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("conv2d", "input", args, 0, false);
    auto filter = GetTensorFromArgs("conv2d", "filter", args, 1, false);

    // Parse Attributes if needed
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "conv2d", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "conv2d", 3);
    PyObject *paddding_algorithm_obj = PyTuple_GET_ITEM(args, 4);
    std::string paddding_algorithm =
        CastPyArg2String(paddding_algorithm_obj, "conv2d", 4);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 5);
    int groups = CastPyArg2Int(groups_obj, "conv2d", 5);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "conv2d", 6);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 7);
    std::string data_format = CastPyArg2String(data_format_obj, "conv2d", 7);
    PyObject *use_addto_obj = PyTuple_GET_ITEM(args, 8);
    bool use_addto = CastPyArg2Boolean(use_addto_obj, "conv2d", 8);
    PyObject *workspace_size_MB_obj = PyTuple_GET_ITEM(args, 9);
    int workspace_size_MB = CastPyArg2Int(workspace_size_MB_obj, "conv2d", 9);
    PyObject *exhaustive_search_obj = PyTuple_GET_ITEM(args, 10);
    bool exhaustive_search =
        CastPyArg2Boolean(exhaustive_search_obj, "conv2d", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::conv2d_final_state_dygraph_function(input,
                                                   filter,
                                                   strides,
                                                   paddings,
                                                   paddding_algorithm,
                                                   groups,
                                                   dilations,
                                                   data_format,
                                                   use_addto,
                                                   workspace_size_MB,
                                                   exhaustive_search)) out =
        ::conv2d_final_state_dygraph_function(input,
                                              filter,
                                              strides,
                                              paddings,
                                              paddding_algorithm,
                                              groups,
                                              dilations,
                                              data_format,
                                              use_addto,
                                              workspace_size_MB,
                                              exhaustive_search);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_conv2d_transpose(PyObject *self,
                                                        PyObject *args,
                                                        PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "conv2d_transpose pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: conv2d_transpose";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("conv2d_transpose", "x", args, 0, false);
    auto filter =
        GetTensorFromArgs("conv2d_transpose", "filter", args, 1, false);

    // Parse Attributes if needed
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "conv2d_transpose", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "conv2d_transpose", 3);
    PyObject *output_padding_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> output_padding =
        CastPyArg2Ints(output_padding_obj, "conv2d_transpose", 4);
    PyObject *output_size_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> output_size =
        CastPyArg2Ints(output_size_obj, "conv2d_transpose", 5);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 6);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "conv2d_transpose", 6);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 7);
    int groups = CastPyArg2Int(groups_obj, "conv2d_transpose", 7);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "conv2d_transpose", 8);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 9);
    std::string data_format =
        CastPyArg2String(data_format_obj, "conv2d_transpose", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::conv2d_transpose_final_state_dygraph_function(x,
                                                             filter,
                                                             strides,
                                                             paddings,
                                                             output_padding,
                                                             output_size,
                                                             padding_algorithm,
                                                             groups,
                                                             dilations,
                                                             data_format)) out =
        ::conv2d_transpose_final_state_dygraph_function(x,
                                                        filter,
                                                        strides,
                                                        paddings,
                                                        output_padding,
                                                        output_size,
                                                        padding_algorithm,
                                                        groups,
                                                        dilations,
                                                        data_format);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_conv3d(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "conv3d pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: conv3d";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("conv3d", "input", args, 0, false);
    auto filter = GetTensorFromArgs("conv3d", "filter", args, 1, false);

    // Parse Attributes if needed
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "conv3d", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "conv3d", 3);
    PyObject *paddding_algorithm_obj = PyTuple_GET_ITEM(args, 4);
    std::string paddding_algorithm =
        CastPyArg2String(paddding_algorithm_obj, "conv3d", 4);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 5);
    int groups = CastPyArg2Int(groups_obj, "conv3d", 5);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "conv3d", 6);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 7);
    std::string data_format = CastPyArg2String(data_format_obj, "conv3d", 7);
    PyObject *use_addto_obj = PyTuple_GET_ITEM(args, 8);
    bool use_addto = CastPyArg2Boolean(use_addto_obj, "conv3d", 8);
    PyObject *workspace_size_MB_obj = PyTuple_GET_ITEM(args, 9);
    int workspace_size_MB = CastPyArg2Int(workspace_size_MB_obj, "conv3d", 9);
    PyObject *exhaustive_search_obj = PyTuple_GET_ITEM(args, 10);
    bool exhaustive_search =
        CastPyArg2Boolean(exhaustive_search_obj, "conv3d", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::conv3d_final_state_dygraph_function(input,
                                                   filter,
                                                   strides,
                                                   paddings,
                                                   paddding_algorithm,
                                                   groups,
                                                   dilations,
                                                   data_format,
                                                   use_addto,
                                                   workspace_size_MB,
                                                   exhaustive_search)) out =
        ::conv3d_final_state_dygraph_function(input,
                                              filter,
                                              strides,
                                              paddings,
                                              paddding_algorithm,
                                              groups,
                                              dilations,
                                              data_format,
                                              use_addto,
                                              workspace_size_MB,
                                              exhaustive_search);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_conv3d_transpose(PyObject *self,
                                                        PyObject *args,
                                                        PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "conv3d_transpose pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: conv3d_transpose";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("conv3d_transpose", "x", args, 0, false);
    auto filter =
        GetTensorFromArgs("conv3d_transpose", "filter", args, 1, false);

    // Parse Attributes if needed
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "conv3d_transpose", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "conv3d_transpose", 3);
    PyObject *output_padding_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> output_padding =
        CastPyArg2Ints(output_padding_obj, "conv3d_transpose", 4);
    PyObject *output_size_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> output_size =
        CastPyArg2Ints(output_size_obj, "conv3d_transpose", 5);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 6);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "conv3d_transpose", 6);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 7);
    int groups = CastPyArg2Int(groups_obj, "conv3d_transpose", 7);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "conv3d_transpose", 8);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 9);
    std::string data_format =
        CastPyArg2String(data_format_obj, "conv3d_transpose", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::conv3d_transpose_final_state_dygraph_function(x,
                                                             filter,
                                                             strides,
                                                             paddings,
                                                             output_padding,
                                                             output_size,
                                                             padding_algorithm,
                                                             groups,
                                                             dilations,
                                                             data_format)) out =
        ::conv3d_transpose_final_state_dygraph_function(x,
                                                        filter,
                                                        strides,
                                                        paddings,
                                                        output_padding,
                                                        output_size,
                                                        padding_algorithm,
                                                        groups,
                                                        dilations,
                                                        data_format);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_copy_to(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "copy_to pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: copy_to";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("copy_to", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *place_obj = PyTuple_GET_ITEM(args, 1);
    paddle::Place place = CastPyArg2Place(place_obj, "copy_to", 1);
    PyObject *blocking_obj = PyTuple_GET_ITEM(args, 2);
    bool blocking = CastPyArg2Boolean(blocking_obj, "copy_to", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::copy_to(x, place, blocking)) out =
        paddle::experimental::copy_to(x, place, blocking);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_cos(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "cos pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cos";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cos", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cos_final_state_dygraph_function(x)) out =
        ::cos_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_cosh(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "cosh pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cosh";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cosh", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cosh_final_state_dygraph_function(x)) out =
        ::cosh_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_cross_entropy_with_softmax(
    PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "cross_entropy_with_softmax pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cross_entropy_with_softmax";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs(
        "cross_entropy_with_softmax", "input", args, 0, false);
    auto label = GetTensorFromArgs(
        "cross_entropy_with_softmax", "label", args, 1, false);

    // Parse Attributes if needed
    PyObject *soft_label_obj = PyTuple_GET_ITEM(args, 2);
    bool soft_label =
        CastPyArg2Boolean(soft_label_obj, "cross_entropy_with_softmax", 2);
    PyObject *use_softmax_obj = PyTuple_GET_ITEM(args, 3);
    bool use_softmax =
        CastPyArg2Boolean(use_softmax_obj, "cross_entropy_with_softmax", 3);
    PyObject *numeric_stable_mode_obj = PyTuple_GET_ITEM(args, 4);
    bool numeric_stable_mode = CastPyArg2Boolean(
        numeric_stable_mode_obj, "cross_entropy_with_softmax", 4);
    PyObject *ignore_index_obj = PyTuple_GET_ITEM(args, 5);
    int ignore_index =
        CastPyArg2Int(ignore_index_obj, "cross_entropy_with_softmax", 5);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 6);
    int axis = CastPyArg2Int(axis_obj, "cross_entropy_with_softmax", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cross_entropy_with_softmax_final_state_dygraph_function(
        input,
        label,
        soft_label,
        use_softmax,
        numeric_stable_mode,
        ignore_index,
        axis)) out =
        ::cross_entropy_with_softmax_final_state_dygraph_function(
            input,
            label,
            soft_label,
            use_softmax,
            numeric_stable_mode,
            ignore_index,
            axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_cumprod(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "cumprod pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cumprod";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cumprod", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *dim_obj = PyTuple_GET_ITEM(args, 1);
    int dim = CastPyArg2Int(dim_obj, "cumprod", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cumprod_final_state_dygraph_function(x, dim)) out =
        ::cumprod_final_state_dygraph_function(x, dim);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_cumsum(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "cumsum pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cumsum";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cumsum", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "cumsum", 1);
    PyObject *flatten_obj = PyTuple_GET_ITEM(args, 2);
    bool flatten = CastPyArg2Boolean(flatten_obj, "cumsum", 2);
    PyObject *exclusive_obj = PyTuple_GET_ITEM(args, 3);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "cumsum", 3);
    PyObject *reverse_obj = PyTuple_GET_ITEM(args, 4);
    bool reverse = CastPyArg2Boolean(reverse_obj, "cumsum", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cumsum_final_state_dygraph_function(
        x, axis, flatten, exclusive, reverse)) out =
        ::cumsum_final_state_dygraph_function(
            x, axis, flatten, exclusive, reverse);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_deformable_conv(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "deformable_conv pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: deformable_conv";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("deformable_conv", "x", args, 0, false);
    auto offset =
        GetTensorFromArgs("deformable_conv", "offset", args, 1, false);
    auto filter =
        GetTensorFromArgs("deformable_conv", "filter", args, 2, false);
    auto mask =
        GetOptionalTensorFromArgs("deformable_conv", "mask", args, 3, true);

    // Parse Attributes if needed
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "deformable_conv", 4);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "deformable_conv", 5);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "deformable_conv", 6);
    PyObject *deformable_groups_obj = PyTuple_GET_ITEM(args, 7);
    int deformable_groups =
        CastPyArg2Int(deformable_groups_obj, "deformable_conv", 7);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 8);
    int groups = CastPyArg2Int(groups_obj, "deformable_conv", 8);
    PyObject *im2col_step_obj = PyTuple_GET_ITEM(args, 9);
    int im2col_step = CastPyArg2Int(im2col_step_obj, "deformable_conv", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::deformable_conv_final_state_dygraph_function(x,
                                                            offset,
                                                            filter,
                                                            mask,
                                                            strides,
                                                            paddings,
                                                            dilations,
                                                            deformable_groups,
                                                            groups,
                                                            im2col_step)) out =
        ::deformable_conv_final_state_dygraph_function(x,
                                                       offset,
                                                       filter,
                                                       mask,
                                                       strides,
                                                       paddings,
                                                       dilations,
                                                       deformable_groups,
                                                       groups,
                                                       im2col_step);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_depthwise_conv2d(PyObject *self,
                                                        PyObject *args,
                                                        PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "depthwise_conv2d pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: depthwise_conv2d";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("depthwise_conv2d", "x", args, 0, false);
    auto filter =
        GetTensorFromArgs("depthwise_conv2d", "filter", args, 1, false);

    // Parse Attributes if needed
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "depthwise_conv2d", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "depthwise_conv2d", 3);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 4);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "depthwise_conv2d", 4);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 5);
    int groups = CastPyArg2Int(groups_obj, "depthwise_conv2d", 5);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "depthwise_conv2d", 6);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 7);
    std::string data_format =
        CastPyArg2String(data_format_obj, "depthwise_conv2d", 7);
    PyObject *use_addto_obj = PyTuple_GET_ITEM(args, 8);
    bool use_addto = CastPyArg2Boolean(use_addto_obj, "depthwise_conv2d", 8);
    PyObject *workspace_size_MB_obj = PyTuple_GET_ITEM(args, 9);
    int workspace_size_MB =
        CastPyArg2Int(workspace_size_MB_obj, "depthwise_conv2d", 9);
    PyObject *exhaustive_search_obj = PyTuple_GET_ITEM(args, 10);
    bool exhaustive_search =
        CastPyArg2Boolean(exhaustive_search_obj, "depthwise_conv2d", 10);
    PyObject *fuse_relu_obj = PyTuple_GET_ITEM(args, 11);
    bool fuse_relu = CastPyArg2Boolean(fuse_relu_obj, "depthwise_conv2d", 11);
    PyObject *use_gpudnn_obj = PyTuple_GET_ITEM(args, 12);
    bool use_gpudnn = CastPyArg2Boolean(use_gpudnn_obj, "depthwise_conv2d", 12);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::depthwise_conv2d_final_state_dygraph_function(x,
                                                             filter,
                                                             strides,
                                                             paddings,
                                                             padding_algorithm,
                                                             groups,
                                                             dilations,
                                                             data_format,
                                                             use_addto,
                                                             workspace_size_MB,
                                                             exhaustive_search,
                                                             fuse_relu,
                                                             use_gpudnn)) out =
        ::depthwise_conv2d_final_state_dygraph_function(x,
                                                        filter,
                                                        strides,
                                                        paddings,
                                                        padding_algorithm,
                                                        groups,
                                                        dilations,
                                                        data_format,
                                                        use_addto,
                                                        workspace_size_MB,
                                                        exhaustive_search,
                                                        fuse_relu,
                                                        use_gpudnn);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_depthwise_conv2d_transpose(
    PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "depthwise_conv2d_transpose pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: depthwise_conv2d_transpose";

    // Get EagerTensors from args
    auto x =
        GetTensorFromArgs("depthwise_conv2d_transpose", "x", args, 0, false);
    auto filter = GetTensorFromArgs(
        "depthwise_conv2d_transpose", "filter", args, 1, false);

    // Parse Attributes if needed
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "depthwise_conv2d_transpose", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "depthwise_conv2d_transpose", 3);
    PyObject *output_padding_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> output_padding =
        CastPyArg2Ints(output_padding_obj, "depthwise_conv2d_transpose", 4);
    PyObject *output_size_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> output_size =
        CastPyArg2Ints(output_size_obj, "depthwise_conv2d_transpose", 5);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 6);
    std::string padding_algorithm = CastPyArg2String(
        padding_algorithm_obj, "depthwise_conv2d_transpose", 6);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 7);
    int groups = CastPyArg2Int(groups_obj, "depthwise_conv2d_transpose", 7);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<int> dilations =
        CastPyArg2Ints(dilations_obj, "depthwise_conv2d_transpose", 8);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 9);
    std::string data_format =
        CastPyArg2String(data_format_obj, "depthwise_conv2d_transpose", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::depthwise_conv2d_transpose_final_state_dygraph_function(
        x,
        filter,
        strides,
        paddings,
        output_padding,
        output_size,
        padding_algorithm,
        groups,
        dilations,
        data_format)) out =
        ::depthwise_conv2d_transpose_final_state_dygraph_function(
            x,
            filter,
            strides,
            paddings,
            output_padding,
            output_size,
            padding_algorithm,
            groups,
            dilations,
            data_format);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_det(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "det pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: det";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("det", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::det_final_state_dygraph_function(x)) out =
        ::det_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_diag(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "diag pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: diag";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("diag", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 1);
    int offset = CastPyArg2Int(offset_obj, "diag", 1);
    PyObject *padding_value_obj = PyTuple_GET_ITEM(args, 2);
    float padding_value = CastPyArg2Float(padding_value_obj, "diag", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::diag(x, offset, padding_value)) out =
        paddle::experimental::diag(x, offset, padding_value);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_divide(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "divide pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: divide";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("divide", "x", args, 0, false);
    auto y = GetTensorFromArgs("divide", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::divide_final_state_dygraph_function(x, y)) out =
        ::divide_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_dropout(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "dropout pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: dropout";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("dropout", "x", args, 0, false);
    auto seed_tensor =
        GetOptionalTensorFromArgs("dropout", "seed_tensor", args, 1, true);

    // Parse Attributes if needed
    PyObject *p_obj = PyTuple_GET_ITEM(args, 2);
    float p = CastPyArg2Float(p_obj, "dropout", 2);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 3);
    bool is_test = CastPyArg2Boolean(is_test_obj, "dropout", 3);
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 4);
    std::string mode = CastPyArg2String(mode_obj, "dropout", 4);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 5);
    int seed = CastPyArg2Int(seed_obj, "dropout", 5);
    PyObject *fix_seed_obj = PyTuple_GET_ITEM(args, 6);
    bool fix_seed = CastPyArg2Boolean(fix_seed_obj, "dropout", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::dropout_final_state_dygraph_function(
        x, seed_tensor, p, is_test, mode, seed, fix_seed)) out =
        ::dropout_final_state_dygraph_function(
            x, seed_tensor, p, is_test, mode, seed, fix_seed);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_eigh(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "eigh pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: eigh";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("eigh", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *uplo_obj = PyTuple_GET_ITEM(args, 1);
    std::string uplo = CastPyArg2String(uplo_obj, "eigh", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::eigh_final_state_dygraph_function(x, uplo)) out =
        ::eigh_final_state_dygraph_function(x, uplo);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_einsum(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "einsum pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: einsum";

    // Get EagerTensors from args
    auto x = GetTensorListFromArgs("einsum", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *equation_obj = PyTuple_GET_ITEM(args, 1);
    std::string equation = CastPyArg2String(equation_obj, "einsum", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::einsum_final_state_dygraph_function(x, equation)) out =
        ::einsum_final_state_dygraph_function(x, equation);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_elementwise_pow(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "elementwise_pow pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: elementwise_pow";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("elementwise_pow", "x", args, 0, false);
    auto y = GetTensorFromArgs("elementwise_pow", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::elementwise_pow_final_state_dygraph_function(x, y)) out =
        ::elementwise_pow_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_elu(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "elu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: elu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("elu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 1);
    float alpha = CastPyArg2Float(alpha_obj, "elu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::elu_final_state_dygraph_function(x, alpha)) out =
        ::elu_final_state_dygraph_function(x, alpha);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_embedding(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "embedding pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: embedding";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("embedding", "x", args, 0, false);
    auto weight = GetTensorFromArgs("embedding", "weight", args, 1, false);

    // Parse Attributes if needed
    PyObject *padding_idx_obj = PyTuple_GET_ITEM(args, 2);
    int64_t padding_idx = CastPyArg2Long(padding_idx_obj, "embedding", 2);
    PyObject *sparse_obj = PyTuple_GET_ITEM(args, 3);
    bool sparse = CastPyArg2Boolean(sparse_obj, "embedding", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::embedding_final_state_dygraph_function(
        x, weight, padding_idx, sparse)) out =
        ::embedding_final_state_dygraph_function(
            x, weight, padding_idx, sparse);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_empty(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "empty pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: empty";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    paddle::experimental::IntArray shape =
        CastPyArg2IntArray(shape_obj, "empty", 0);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "empty", 1);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 2);
    paddle::Place place = CastPyArg2Place(place_obj, "empty", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::empty(shape, dtype, place)) out =
        paddle::experimental::empty(shape, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_empty_like(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "empty_like pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: empty_like";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("empty_like", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "empty_like", 1);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 2);
    paddle::Place place = CastPyArg2Place(place_obj, "empty_like", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::empty_like(x, dtype, place)) out =
        paddle::experimental::empty_like(x, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_equal(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "equal pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: equal";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("equal", "x", args, 0, false);
    auto y = GetTensorFromArgs("equal", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "equal", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::equal(x, y, axis)) out =
        paddle::experimental::equal(x, y, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_equal_all(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "equal_all pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: equal_all";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("equal_all", "x", args, 0, false);
    auto y = GetTensorFromArgs("equal_all", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::equal_all(x, y)) out =
        paddle::experimental::equal_all(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_erfinv(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "erfinv pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: erfinv";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("erfinv", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::erfinv_final_state_dygraph_function(x)) out =
        ::erfinv_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_erfinv_(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "erfinv pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: erfinv_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("erfinv", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::erfinv__final_state_dygraph_function(x)) out =
        ::erfinv__final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_exp(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "exp pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: exp";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("exp", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::exp_final_state_dygraph_function(x)) out =
        ::exp_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_expand(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "expand pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: expand";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("expand", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray shape =
        CastPyArg2IntArray(shape_obj, "expand", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::expand_final_state_dygraph_function(x, shape)) out =
        ::expand_final_state_dygraph_function(x, shape);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_expand_as(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "expand_as pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: expand_as";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("expand_as", "x", args, 0, false);
    auto y = GetOptionalTensorFromArgs("expand_as", "y", args, 1, true);

    // Parse Attributes if needed
    PyObject *target_shape_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> target_shape =
        CastPyArg2Ints(target_shape_obj, "expand_as", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::expand_as_final_state_dygraph_function(x, y, target_shape)) out =
        ::expand_as_final_state_dygraph_function(x, y, target_shape);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_expm1(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "expm1 pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: expm1";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("expm1", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::expm1_final_state_dygraph_function(x)) out =
        ::expm1_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_eye(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "eye pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: eye";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject *num_rows_obj = PyTuple_GET_ITEM(args, 0);
    int64_t num_rows = CastPyArg2Long(num_rows_obj, "eye", 0);
    PyObject *num_columns_obj = PyTuple_GET_ITEM(args, 1);
    int64_t num_columns = CastPyArg2Long(num_columns_obj, "eye", 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "eye", 2);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 3);
    paddle::Place place = CastPyArg2Place(place_obj, "eye", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::eye(
        num_rows, num_columns, dtype, place)) out =
        paddle::experimental::eye(num_rows, num_columns, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_flatten(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "flatten pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: flatten";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("flatten", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *start_axis_obj = PyTuple_GET_ITEM(args, 1);
    int start_axis = CastPyArg2Int(start_axis_obj, "flatten", 1);
    PyObject *stop_axis_obj = PyTuple_GET_ITEM(args, 2);
    int stop_axis = CastPyArg2Int(stop_axis_obj, "flatten", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::flatten_final_state_dygraph_function(
        x, start_axis, stop_axis)) out =
        ::flatten_final_state_dygraph_function(x, start_axis, stop_axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_flatten_(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "flatten pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: flatten_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("flatten", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *start_axis_obj = PyTuple_GET_ITEM(args, 1);
    int start_axis = CastPyArg2Int(start_axis_obj, "flatten", 1);
    PyObject *stop_axis_obj = PyTuple_GET_ITEM(args, 2);
    int stop_axis = CastPyArg2Int(stop_axis_obj, "flatten", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::flatten__final_state_dygraph_function(
        x, start_axis, stop_axis)) out =
        ::flatten__final_state_dygraph_function(x, start_axis, stop_axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_flip(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "flip pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: flip";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("flip", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> axis = CastPyArg2Ints(axis_obj, "flip", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::flip_final_state_dygraph_function(x, axis)) out =
        ::flip_final_state_dygraph_function(x, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_floor(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "floor pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: floor";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("floor", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::floor_final_state_dygraph_function(x)) out =
        ::floor_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_floor_divide(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "floor_divide pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: floor_divide";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("floor_divide", "x", args, 0, false);
    auto y = GetTensorFromArgs("floor_divide", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::floor_divide(x, y)) out =
        paddle::experimental::floor_divide(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_fmax(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "fmax pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fmax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fmax", "x", args, 0, false);
    auto y = GetTensorFromArgs("fmax", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "fmax", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fmax_final_state_dygraph_function(x, y, axis)) out =
        ::fmax_final_state_dygraph_function(x, y, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_fmin(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "fmin pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fmin";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fmin", "x", args, 0, false);
    auto y = GetTensorFromArgs("fmin", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "fmin", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fmin_final_state_dygraph_function(x, y, axis)) out =
        ::fmin_final_state_dygraph_function(x, y, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_frobenius_norm(PyObject *self,
                                                      PyObject *args,
                                                      PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "frobenius_norm pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: frobenius_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("frobenius_norm", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "frobenius_norm", 1);
    PyObject *keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "frobenius_norm", 2);
    PyObject *reduce_all_obj = PyTuple_GET_ITEM(args, 3);
    bool reduce_all = CastPyArg2Boolean(reduce_all_obj, "frobenius_norm", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::frobenius_norm_final_state_dygraph_function(
        x, axis, keep_dim, reduce_all)) out =
        ::frobenius_norm_final_state_dygraph_function(
            x, axis, keep_dim, reduce_all);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_full(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "full pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: full";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    paddle::experimental::IntArray shape =
        CastPyArg2IntArray(shape_obj, "full", 0);
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar value = CastPyArg2Scalar(value_obj, "full", 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "full", 2);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 3);
    paddle::Place place = CastPyArg2Place(place_obj, "full", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::full(shape, value, dtype, place)) out =
        paddle::experimental::full(shape, value, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_full_batch_size_like(PyObject *self,
                                                            PyObject *args,
                                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "full_batch_size_like pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: full_batch_size_like";

    // Get EagerTensors from args
    auto input =
        GetTensorFromArgs("full_batch_size_like", "input", args, 0, false);

    // Parse Attributes if needed
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> shape =
        CastPyArg2Ints(shape_obj, "full_batch_size_like", 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "full_batch_size_like", 2);
    PyObject *value_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::Scalar value =
        CastPyArg2Scalar(value_obj, "full_batch_size_like", 3);
    PyObject *input_dim_idx_obj = PyTuple_GET_ITEM(args, 4);
    int input_dim_idx =
        CastPyArg2Int(input_dim_idx_obj, "full_batch_size_like", 4);
    PyObject *output_dim_idx_obj = PyTuple_GET_ITEM(args, 5);
    int output_dim_idx =
        CastPyArg2Int(output_dim_idx_obj, "full_batch_size_like", 5);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 6);
    paddle::Place place = CastPyArg2Place(place_obj, "full_batch_size_like", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::full_batch_size_like(
        input, shape, dtype, value, input_dim_idx, output_dim_idx, place)) out =
        paddle::experimental::full_batch_size_like(
            input, shape, dtype, value, input_dim_idx, output_dim_idx, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_full_like(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "full_like pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: full_like";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("full_like", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar value =
        CastPyArg2Scalar(value_obj, "full_like", 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "full_like", 2);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 3);
    paddle::Place place = CastPyArg2Place(place_obj, "full_like", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::full_like(x, value, dtype, place)) out =
        paddle::experimental::full_like(x, value, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_gather(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "gather pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: gather";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("gather", "x", args, 0, false);
    auto index = GetTensorFromArgs("gather", "index", args, 1, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar axis = CastPyArg2Scalar(axis_obj, "gather", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::gather_final_state_dygraph_function(x, index, axis)) out =
        ::gather_final_state_dygraph_function(x, index, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_gather_nd(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "gather_nd pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: gather_nd";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("gather_nd", "x", args, 0, false);
    auto index = GetTensorFromArgs("gather_nd", "index", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::gather_nd_final_state_dygraph_function(x, index)) out =
        ::gather_nd_final_state_dygraph_function(x, index);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_gather_tree(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "gather_tree pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: gather_tree";

    // Get EagerTensors from args
    auto ids = GetTensorFromArgs("gather_tree", "ids", args, 0, false);
    auto parents = GetTensorFromArgs("gather_tree", "parents", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::gather_tree(ids, parents)) out =
        paddle::experimental::gather_tree(ids, parents);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_gaussian_random(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "gaussian_random pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: gaussian_random";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    paddle::experimental::IntArray shape =
        CastPyArg2IntArray(shape_obj, "gaussian_random", 0);
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 1);
    float mean = CastPyArg2Float(mean_obj, "gaussian_random", 1);
    PyObject *std_obj = PyTuple_GET_ITEM(args, 2);
    float std = CastPyArg2Float(std_obj, "gaussian_random", 2);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 3);
    int seed = CastPyArg2Int(seed_obj, "gaussian_random", 3);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 4);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "gaussian_random", 4);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 5);
    paddle::Place place = CastPyArg2Place(place_obj, "gaussian_random", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::gaussian_random(
        shape, mean, std, seed, dtype, place)) out =
        paddle::experimental::gaussian_random(
            shape, mean, std, seed, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_gelu(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "gelu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: gelu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("gelu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *approximate_obj = PyTuple_GET_ITEM(args, 1);
    bool approximate = CastPyArg2Boolean(approximate_obj, "gelu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::gelu_final_state_dygraph_function(x, approximate)) out =
        ::gelu_final_state_dygraph_function(x, approximate);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_graph_send_recv(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "graph_send_recv pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: graph_send_recv";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("graph_send_recv", "x", args, 0, false);
    auto src_index =
        GetTensorFromArgs("graph_send_recv", "src_index", args, 1, false);
    auto dst_index =
        GetTensorFromArgs("graph_send_recv", "dst_index", args, 2, false);

    // Parse Attributes if needed
    PyObject *pool_type_obj = PyTuple_GET_ITEM(args, 3);
    std::string pool_type =
        CastPyArg2String(pool_type_obj, "graph_send_recv", 3);
    PyObject *out_size_obj = PyTuple_GET_ITEM(args, 4);
    int64_t out_size = CastPyArg2Long(out_size_obj, "graph_send_recv", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::graph_send_recv_final_state_dygraph_function(
        x, src_index, dst_index, pool_type, out_size)) out =
        ::graph_send_recv_final_state_dygraph_function(
            x, src_index, dst_index, pool_type, out_size);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_greater_equal(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "greater_equal pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: greater_equal";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("greater_equal", "x", args, 0, false);
    auto y = GetTensorFromArgs("greater_equal", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "greater_equal", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::greater_equal(x, y, axis)) out =
        paddle::experimental::greater_equal(x, y, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_greater_than(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "greater_than pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: greater_than";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("greater_than", "x", args, 0, false);
    auto y = GetTensorFromArgs("greater_than", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "greater_than", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::greater_than(x, y, axis)) out =
        paddle::experimental::greater_than(x, y, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_group_norm(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "group_norm pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: group_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("group_norm", "x", args, 0, false);
    auto scale =
        GetOptionalTensorFromArgs("group_norm", "scale", args, 1, true);
    auto bias = GetOptionalTensorFromArgs("group_norm", "bias", args, 2, true);

    // Parse Attributes if needed
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 3);
    float epsilon = CastPyArg2Float(epsilon_obj, "group_norm", 3);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 4);
    int groups = CastPyArg2Int(groups_obj, "group_norm", 4);
    PyObject *data_layout_obj = PyTuple_GET_ITEM(args, 5);
    std::string data_layout =
        CastPyArg2String(data_layout_obj, "group_norm", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::group_norm_final_state_dygraph_function(
        x, scale, bias, epsilon, groups, data_layout)) out =
        ::group_norm_final_state_dygraph_function(
            x, scale, bias, epsilon, groups, data_layout);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_gumbel_softmax(PyObject *self,
                                                      PyObject *args,
                                                      PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "gumbel_softmax pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: gumbel_softmax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("gumbel_softmax", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *temperature_obj = PyTuple_GET_ITEM(args, 1);
    float temperature = CastPyArg2Float(temperature_obj, "gumbel_softmax", 1);
    PyObject *hard_obj = PyTuple_GET_ITEM(args, 2);
    bool hard = CastPyArg2Boolean(hard_obj, "gumbel_softmax", 2);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 3);
    int axis = CastPyArg2Int(axis_obj, "gumbel_softmax", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::gumbel_softmax_final_state_dygraph_function(
        x, temperature, hard, axis)) out =
        ::gumbel_softmax_final_state_dygraph_function(
            x, temperature, hard, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_hard_shrink(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "hard_shrink pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: hard_shrink";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("hard_shrink", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "hard_shrink", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::hard_shrink_final_state_dygraph_function(x, threshold)) out =
        ::hard_shrink_final_state_dygraph_function(x, threshold);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_hard_sigmoid(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "hard_sigmoid pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: hard_sigmoid";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("hard_sigmoid", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *slope_obj = PyTuple_GET_ITEM(args, 1);
    float slope = CastPyArg2Float(slope_obj, "hard_sigmoid", 1);
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 2);
    float offset = CastPyArg2Float(offset_obj, "hard_sigmoid", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::hard_sigmoid_final_state_dygraph_function(
        x, slope, offset)) out =
        ::hard_sigmoid_final_state_dygraph_function(x, slope, offset);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_hard_swish(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "hard_swish pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: hard_swish";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("hard_swish", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "hard_swish", 1);
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 2);
    float scale = CastPyArg2Float(scale_obj, "hard_swish", 2);
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 3);
    float offset = CastPyArg2Float(offset_obj, "hard_swish", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::hard_swish_final_state_dygraph_function(
        x, threshold, scale, offset)) out =
        ::hard_swish_final_state_dygraph_function(x, threshold, scale, offset);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_histogram(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "histogram pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: histogram";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("histogram", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *bins_obj = PyTuple_GET_ITEM(args, 1);
    int64_t bins = CastPyArg2Long(bins_obj, "histogram", 1);
    PyObject *min_obj = PyTuple_GET_ITEM(args, 2);
    int min = CastPyArg2Int(min_obj, "histogram", 2);
    PyObject *max_obj = PyTuple_GET_ITEM(args, 3);
    int max = CastPyArg2Int(max_obj, "histogram", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::histogram(x, bins, min, max)) out =
        paddle::experimental::histogram(x, bins, min, max);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_huber_loss(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "huber_loss pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: huber_loss";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("huber_loss", "input", args, 0, false);
    auto label = GetTensorFromArgs("huber_loss", "label", args, 1, false);

    // Parse Attributes if needed
    PyObject *delta_obj = PyTuple_GET_ITEM(args, 2);
    float delta = CastPyArg2Float(delta_obj, "huber_loss", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::huber_loss_final_state_dygraph_function(
        input, label, delta)) out =
        ::huber_loss_final_state_dygraph_function(input, label, delta);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_imag(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "imag pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: imag";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("imag", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::imag_final_state_dygraph_function(x)) out =
        ::imag_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_increment(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "increment pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: increment";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("increment", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);
    float value = CastPyArg2Float(value_obj, "increment", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::increment(x, value)) out =
        paddle::experimental::increment(x, value);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_index_sample(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "index_sample pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: index_sample";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("index_sample", "x", args, 0, false);
    auto index = GetTensorFromArgs("index_sample", "index", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::index_sample_final_state_dygraph_function(x, index)) out =
        ::index_sample_final_state_dygraph_function(x, index);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_index_select(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "index_select pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: index_select";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("index_select", "x", args, 0, false);
    auto index = GetTensorFromArgs("index_select", "index", args, 1, false);

    // Parse Attributes if needed
    PyObject *dim_obj = PyTuple_GET_ITEM(args, 2);
    int dim = CastPyArg2Int(dim_obj, "index_select", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::index_select_final_state_dygraph_function(x, index, dim)) out =
        ::index_select_final_state_dygraph_function(x, index, dim);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_instance_norm(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "instance_norm pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: instance_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("instance_norm", "x", args, 0, false);
    auto scale =
        GetOptionalTensorFromArgs("instance_norm", "scale", args, 1, true);
    auto bias =
        GetOptionalTensorFromArgs("instance_norm", "bias", args, 2, true);

    // Parse Attributes if needed
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 3);
    float epsilon = CastPyArg2Float(epsilon_obj, "instance_norm", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::instance_norm_final_state_dygraph_function(
        x, scale, bias, epsilon)) out =
        ::instance_norm_final_state_dygraph_function(x, scale, bias, epsilon);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_is_empty(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "is_empty pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: is_empty";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("is_empty", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::is_empty(x)) out =
        paddle::experimental::is_empty(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_isclose(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "isclose pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: isclose";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("isclose", "x", args, 0, false);
    auto y = GetTensorFromArgs("isclose", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *rtol_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar rtol =
        CastPyArg2Scalar(rtol_obj, "isclose", 2);
    PyObject *atol_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::Scalar atol =
        CastPyArg2Scalar(atol_obj, "isclose", 3);
    PyObject *equal_nan_obj = PyTuple_GET_ITEM(args, 4);
    bool equal_nan = CastPyArg2Boolean(equal_nan_obj, "isclose", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::isclose(x, y, rtol, atol, equal_nan)) out =
        paddle::experimental::isclose(x, y, rtol, atol, equal_nan);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_isfinite(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "isfinite pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: isfinite";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("isfinite", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::isfinite(x)) out =
        paddle::experimental::isfinite(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_isinf(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "isinf pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: isinf";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("isinf", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::isinf(x)) out =
        paddle::experimental::isinf(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_isnan(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "isnan pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: isnan";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("isnan", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::isnan(x)) out =
        paddle::experimental::isnan(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_kldiv_loss(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "kldiv_loss pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: kldiv_loss";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("kldiv_loss", "x", args, 0, false);
    auto label = GetTensorFromArgs("kldiv_loss", "label", args, 1, false);

    // Parse Attributes if needed
    PyObject *reduction_obj = PyTuple_GET_ITEM(args, 2);
    std::string reduction = CastPyArg2String(reduction_obj, "kldiv_loss", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::kldiv_loss_final_state_dygraph_function(
        x, label, reduction)) out =
        ::kldiv_loss_final_state_dygraph_function(x, label, reduction);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_kron(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "kron pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: kron";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("kron", "x", args, 0, false);
    auto y = GetTensorFromArgs("kron", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::kron_final_state_dygraph_function(x, y)) out =
        ::kron_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_kthvalue(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "kthvalue pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: kthvalue";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("kthvalue", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *k_obj = PyTuple_GET_ITEM(args, 1);
    int k = CastPyArg2Int(k_obj, "kthvalue", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "kthvalue", 2);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 3);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "kthvalue", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::kthvalue_final_state_dygraph_function(x, k, axis, keepdim)) out =
        ::kthvalue_final_state_dygraph_function(x, k, axis, keepdim);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_label_smooth(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "label_smooth pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: label_smooth";

    // Get EagerTensors from args
    auto label = GetTensorFromArgs("label_smooth", "label", args, 0, false);
    auto prior_dist =
        GetOptionalTensorFromArgs("label_smooth", "prior_dist", args, 1, true);

    // Parse Attributes if needed
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 2);
    float epsilon = CastPyArg2Float(epsilon_obj, "label_smooth", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::label_smooth_final_state_dygraph_function(
        label, prior_dist, epsilon)) out =
        ::label_smooth_final_state_dygraph_function(label, prior_dist, epsilon);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_layer_norm(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "layer_norm pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: layer_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("layer_norm", "x", args, 0, false);
    auto scale =
        GetOptionalTensorFromArgs("layer_norm", "scale", args, 1, true);
    auto bias = GetOptionalTensorFromArgs("layer_norm", "bias", args, 2, true);

    // Parse Attributes if needed
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 3);
    float epsilon = CastPyArg2Float(epsilon_obj, "layer_norm", 3);
    PyObject *begin_norm_axis_obj = PyTuple_GET_ITEM(args, 4);
    int begin_norm_axis = CastPyArg2Int(begin_norm_axis_obj, "layer_norm", 4);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 5);
    bool is_test = CastPyArg2Boolean(is_test_obj, "layer_norm", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::layer_norm_final_state_dygraph_function(
        x, scale, bias, epsilon, begin_norm_axis, is_test)) out =
        ::layer_norm_final_state_dygraph_function(
            x, scale, bias, epsilon, begin_norm_axis, is_test);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_leaky_relu(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "leaky_relu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: leaky_relu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("leaky_relu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 1);
    float alpha = CastPyArg2Float(alpha_obj, "leaky_relu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::leaky_relu_final_state_dygraph_function(x, alpha)) out =
        ::leaky_relu_final_state_dygraph_function(x, alpha);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_lerp(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "lerp pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: lerp";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("lerp", "x", args, 0, false);
    auto y = GetTensorFromArgs("lerp", "y", args, 1, false);
    auto weight = GetTensorFromArgs("lerp", "weight", args, 2, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::lerp_final_state_dygraph_function(x, y, weight)) out =
        ::lerp_final_state_dygraph_function(x, y, weight);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_less_equal(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "less_equal pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: less_equal";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("less_equal", "x", args, 0, false);
    auto y = GetTensorFromArgs("less_equal", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "less_equal", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::less_equal(x, y, axis)) out =
        paddle::experimental::less_equal(x, y, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_less_than(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "less_than pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: less_than";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("less_than", "x", args, 0, false);
    auto y = GetTensorFromArgs("less_than", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "less_than", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::less_than(x, y, axis)) out =
        paddle::experimental::less_than(x, y, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_lgamma(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "lgamma pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: lgamma";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("lgamma", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::lgamma_final_state_dygraph_function(x)) out =
        ::lgamma_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_linspace(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "linspace pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: linspace";

    // Get EagerTensors from args
    auto start = GetTensorFromArgs("linspace", "start", args, 0, false);
    auto stop = GetTensorFromArgs("linspace", "stop", args, 1, false);
    auto number = GetTensorFromArgs("linspace", "number", args, 2, false);

    // Parse Attributes if needed
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "linspace", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::linspace(start, stop, number, dtype)) out =
        paddle::experimental::linspace(start, stop, number, dtype);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_log(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "log pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: log";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("log", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::log_final_state_dygraph_function(x)) out =
        ::log_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_log10(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "log10 pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: log10";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("log10", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::log10_final_state_dygraph_function(x)) out =
        ::log10_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_log1p(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "log1p pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: log1p";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("log1p", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::log1p_final_state_dygraph_function(x)) out =
        ::log1p_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_log2(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "log2 pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: log2";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("log2", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::log2_final_state_dygraph_function(x)) out =
        ::log2_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_log_loss(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "log_loss pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: log_loss";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("log_loss", "input", args, 0, false);
    auto label = GetTensorFromArgs("log_loss", "label", args, 1, false);

    // Parse Attributes if needed
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 2);
    float epsilon = CastPyArg2Float(epsilon_obj, "log_loss", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::log_loss_final_state_dygraph_function(
        input, label, epsilon)) out =
        ::log_loss_final_state_dygraph_function(input, label, epsilon);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_log_softmax(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "log_softmax pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: log_softmax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("log_softmax", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "log_softmax", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::log_softmax_final_state_dygraph_function(x, axis)) out =
        ::log_softmax_final_state_dygraph_function(x, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_logcumsumexp(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "logcumsumexp pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: logcumsumexp";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("logcumsumexp", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "logcumsumexp", 1);
    PyObject *flatten_obj = PyTuple_GET_ITEM(args, 2);
    bool flatten = CastPyArg2Boolean(flatten_obj, "logcumsumexp", 2);
    PyObject *exclusive_obj = PyTuple_GET_ITEM(args, 3);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "logcumsumexp", 3);
    PyObject *reverse_obj = PyTuple_GET_ITEM(args, 4);
    bool reverse = CastPyArg2Boolean(reverse_obj, "logcumsumexp", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::logcumsumexp_final_state_dygraph_function(
        x, axis, flatten, exclusive, reverse)) out =
        ::logcumsumexp_final_state_dygraph_function(
            x, axis, flatten, exclusive, reverse);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_logical_and(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "logical_and pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: logical_and";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("logical_and", "x", args, 0, false);
    auto y = GetTensorFromArgs("logical_and", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::logical_and(x, y)) out =
        paddle::experimental::logical_and(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_logical_not(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "logical_not pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: logical_not";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("logical_not", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::logical_not(x)) out =
        paddle::experimental::logical_not(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_logical_or(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "logical_or pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: logical_or";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("logical_or", "x", args, 0, false);
    auto y = GetTensorFromArgs("logical_or", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::logical_or(x, y)) out =
        paddle::experimental::logical_or(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_logical_xor(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "logical_xor pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: logical_xor";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("logical_xor", "x", args, 0, false);
    auto y = GetTensorFromArgs("logical_xor", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::logical_xor(x, y)) out =
        paddle::experimental::logical_xor(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_logit(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "logit pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: logit";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("logit", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *eps_obj = PyTuple_GET_ITEM(args, 1);
    float eps = CastPyArg2Float(eps_obj, "logit", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::logit_final_state_dygraph_function(x, eps)) out =
        ::logit_final_state_dygraph_function(x, eps);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_logsigmoid(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "logsigmoid pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: logsigmoid";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("logsigmoid", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::logsigmoid_final_state_dygraph_function(x)) out =
        ::logsigmoid_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_logsumexp(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "logsumexp pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: logsumexp";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("logsumexp", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "logsumexp", 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "logsumexp", 2);
    PyObject *reduce_all_obj = PyTuple_GET_ITEM(args, 3);
    bool reduce_all = CastPyArg2Boolean(reduce_all_obj, "logsumexp", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::logsumexp_final_state_dygraph_function(
        x, axis, keepdim, reduce_all)) out =
        ::logsumexp_final_state_dygraph_function(x, axis, keepdim, reduce_all);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_masked_select(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "masked_select pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: masked_select";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("masked_select", "x", args, 0, false);
    auto mask = GetTensorFromArgs("masked_select", "mask", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::masked_select_final_state_dygraph_function(x, mask)) out =
        ::masked_select_final_state_dygraph_function(x, mask);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_matmul(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "matmul pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: matmul";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("matmul", "x", args, 0, false);
    auto y = GetTensorFromArgs("matmul", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *transpose_x_obj = PyTuple_GET_ITEM(args, 2);
    bool transpose_x = CastPyArg2Boolean(transpose_x_obj, "matmul", 2);
    PyObject *transpose_y_obj = PyTuple_GET_ITEM(args, 3);
    bool transpose_y = CastPyArg2Boolean(transpose_y_obj, "matmul", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::matmul_final_state_dygraph_function(
        x, y, transpose_x, transpose_y)) out =
        ::matmul_final_state_dygraph_function(x, y, transpose_x, transpose_y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_matrix_power(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "matrix_power pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: matrix_power";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("matrix_power", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *n_obj = PyTuple_GET_ITEM(args, 1);
    int n = CastPyArg2Int(n_obj, "matrix_power", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::matrix_power_final_state_dygraph_function(x, n)) out =
        ::matrix_power_final_state_dygraph_function(x, n);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_matrix_rank(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "matrix_rank pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: matrix_rank";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("matrix_rank", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *tol_obj = PyTuple_GET_ITEM(args, 1);
    float tol = CastPyArg2Float(tol_obj, "matrix_rank", 1);
    PyObject *use_default_tol_obj = PyTuple_GET_ITEM(args, 2);
    bool use_default_tol =
        CastPyArg2Boolean(use_default_tol_obj, "matrix_rank", 2);
    PyObject *hermitian_obj = PyTuple_GET_ITEM(args, 3);
    bool hermitian = CastPyArg2Boolean(hermitian_obj, "matrix_rank", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::matrix_rank(
        x, tol, use_default_tol, hermitian)) out =
        paddle::experimental::matrix_rank(x, tol, use_default_tol, hermitian);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_matrix_rank_tol(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "matrix_rank_tol pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: matrix_rank_tol";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("matrix_rank_tol", "x", args, 0, false);
    auto atol_tensor =
        GetTensorFromArgs("matrix_rank_tol", "atol_tensor", args, 1, false);

    // Parse Attributes if needed
    PyObject *use_default_tol_obj = PyTuple_GET_ITEM(args, 2);
    bool use_default_tol =
        CastPyArg2Boolean(use_default_tol_obj, "matrix_rank_tol", 2);
    PyObject *hermitian_obj = PyTuple_GET_ITEM(args, 3);
    bool hermitian = CastPyArg2Boolean(hermitian_obj, "matrix_rank_tol", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::matrix_rank_tol(
        x, atol_tensor, use_default_tol, hermitian)) out =
        paddle::experimental::matrix_rank_tol(
            x, atol_tensor, use_default_tol, hermitian);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_max(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "max pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: max";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("max", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "max", 1);
    PyObject *keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "max", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::max_final_state_dygraph_function(x, dims, keep_dim)) out =
        ::max_final_state_dygraph_function(x, dims, keep_dim);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_max_pool2d_with_index(PyObject *self,
                                                             PyObject *args,
                                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "max_pool2d_with_index pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: max_pool2d_with_index";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("max_pool2d_with_index", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size =
        CastPyArg2Ints(kernel_size_obj, "max_pool2d_with_index", 1);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "max_pool2d_with_index", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "max_pool2d_with_index", 3);
    PyObject *global_pooling_obj = PyTuple_GET_ITEM(args, 4);
    bool global_pooling =
        CastPyArg2Boolean(global_pooling_obj, "max_pool2d_with_index", 4);
    PyObject *adaptive_obj = PyTuple_GET_ITEM(args, 5);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "max_pool2d_with_index", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::max_pool2d_with_index_final_state_dygraph_function(
        x, kernel_size, strides, paddings, global_pooling, adaptive)) out =
        ::max_pool2d_with_index_final_state_dygraph_function(
            x, kernel_size, strides, paddings, global_pooling, adaptive);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_max_pool3d_with_index(PyObject *self,
                                                             PyObject *args,
                                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "max_pool3d_with_index pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: max_pool3d_with_index";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("max_pool3d_with_index", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size =
        CastPyArg2Ints(kernel_size_obj, "max_pool3d_with_index", 1);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "max_pool3d_with_index", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "max_pool3d_with_index", 3);
    PyObject *global_pooling_obj = PyTuple_GET_ITEM(args, 4);
    bool global_pooling =
        CastPyArg2Boolean(global_pooling_obj, "max_pool3d_with_index", 4);
    PyObject *adaptive_obj = PyTuple_GET_ITEM(args, 5);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "max_pool3d_with_index", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::max_pool3d_with_index_final_state_dygraph_function(
        x, kernel_size, strides, paddings, global_pooling, adaptive)) out =
        ::max_pool3d_with_index_final_state_dygraph_function(
            x, kernel_size, strides, paddings, global_pooling, adaptive);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_maximum(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "maximum pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: maximum";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("maximum", "x", args, 0, false);
    auto y = GetTensorFromArgs("maximum", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::maximum_final_state_dygraph_function(x, y)) out =
        ::maximum_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_maxout(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "maxout pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: maxout";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("maxout", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 1);
    int groups = CastPyArg2Int(groups_obj, "maxout", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "maxout", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::maxout_final_state_dygraph_function(x, groups, axis)) out =
        ::maxout_final_state_dygraph_function(x, groups, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_mean(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "mean pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: mean";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("mean", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "mean", 1);
    PyObject *keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "mean", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::mean_final_state_dygraph_function(x, dims, keep_dim)) out =
        ::mean_final_state_dygraph_function(x, dims, keep_dim);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_mean_all(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "mean_all pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: mean_all";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("mean_all", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::mean_all_final_state_dygraph_function(x)) out =
        ::mean_all_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_meshgrid(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "meshgrid pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: meshgrid";

    // Get EagerTensors from args
    auto inputs = GetTensorListFromArgs("meshgrid", "inputs", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::meshgrid_final_state_dygraph_function(inputs)) out =
        ::meshgrid_final_state_dygraph_function(inputs);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_min(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "min pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: min";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("min", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "min", 1);
    PyObject *keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "min", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::min_final_state_dygraph_function(x, dims, keep_dim)) out =
        ::min_final_state_dygraph_function(x, dims, keep_dim);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_minimum(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "minimum pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: minimum";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("minimum", "x", args, 0, false);
    auto y = GetTensorFromArgs("minimum", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::minimum_final_state_dygraph_function(x, y)) out =
        ::minimum_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_mish(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "mish pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: mish";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("mish", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *lambda_obj = PyTuple_GET_ITEM(args, 1);
    float lambda = CastPyArg2Float(lambda_obj, "mish", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::mish_final_state_dygraph_function(x, lambda)) out =
        ::mish_final_state_dygraph_function(x, lambda);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_mode(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "mode pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: mode";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("mode", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "mode", 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "mode", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::mode_final_state_dygraph_function(x, axis, keepdim)) out =
        ::mode_final_state_dygraph_function(x, axis, keepdim);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_modulo(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "modulo pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: modulo";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("modulo", "x", args, 0, false);
    auto y = GetTensorFromArgs("modulo", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::modulo_final_state_dygraph_function(x, y)) out =
        ::modulo_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_momentum(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "momentum pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: momentum";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("momentum", "param", args, 0, false);
    auto grad = GetTensorFromArgs("momentum", "grad", args, 1, false);
    auto velocity = GetTensorFromArgs("momentum", "velocity", args, 2, false);
    auto learning_rate =
        GetTensorFromArgs("momentum", "learning_rate", args, 3, false);
    auto master_param =
        GetOptionalTensorFromArgs("momentum", "master_param", args, 4, true);

    // Parse Attributes if needed
    PyObject *mu_obj = PyTuple_GET_ITEM(args, 5);
    float mu = CastPyArg2Float(mu_obj, "momentum", 5);
    PyObject *use_nesterov_obj = PyTuple_GET_ITEM(args, 6);
    bool use_nesterov = CastPyArg2Boolean(use_nesterov_obj, "momentum", 6);
    PyObject *regularization_method_obj = PyTuple_GET_ITEM(args, 7);
    std::string regularization_method =
        CastPyArg2String(regularization_method_obj, "momentum", 7);
    PyObject *regularization_coeff_obj = PyTuple_GET_ITEM(args, 8);
    float regularization_coeff =
        CastPyArg2Float(regularization_coeff_obj, "momentum", 8);
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 9);
    bool multi_precision =
        CastPyArg2Boolean(multi_precision_obj, "momentum", 9);
    PyObject *rescale_grad_obj = PyTuple_GET_ITEM(args, 10);
    float rescale_grad = CastPyArg2Float(rescale_grad_obj, "momentum", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::momentum(param,
                                            grad,
                                            velocity,
                                            learning_rate,
                                            master_param,
                                            mu,
                                            use_nesterov,
                                            regularization_method,
                                            regularization_coeff,
                                            multi_precision,
                                            rescale_grad)) out =
        paddle::experimental::momentum(param,
                                       grad,
                                       velocity,
                                       learning_rate,
                                       master_param,
                                       mu,
                                       use_nesterov,
                                       regularization_method,
                                       regularization_coeff,
                                       multi_precision,
                                       rescale_grad);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_multi_dot(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "multi_dot pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: multi_dot";

    // Get EagerTensors from args
    auto x = GetTensorListFromArgs("multi_dot", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::multi_dot_final_state_dygraph_function(x)) out =
        ::multi_dot_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_multinomial(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "multinomial pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: multinomial";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("multinomial", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *num_samples_obj = PyTuple_GET_ITEM(args, 1);
    int num_samples = CastPyArg2Int(num_samples_obj, "multinomial", 1);
    PyObject *replacement_obj = PyTuple_GET_ITEM(args, 2);
    bool replacement = CastPyArg2Boolean(replacement_obj, "multinomial", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::multinomial(
        x, num_samples, replacement)) out =
        paddle::experimental::multinomial(x, num_samples, replacement);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_multiplex(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "multiplex pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: multiplex";

    // Get EagerTensors from args
    auto ins = GetTensorListFromArgs("multiplex", "ins", args, 0, false);
    auto ids = GetTensorFromArgs("multiplex", "ids", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::multiplex_final_state_dygraph_function(ins, ids)) out =
        ::multiplex_final_state_dygraph_function(ins, ids);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_multiply(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "multiply pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: multiply";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("multiply", "x", args, 0, false);
    auto y = GetTensorFromArgs("multiply", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::multiply_final_state_dygraph_function(x, y)) out =
        ::multiply_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_nll_loss(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "nll_loss pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: nll_loss";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("nll_loss", "input", args, 0, false);
    auto label = GetTensorFromArgs("nll_loss", "label", args, 1, false);
    auto weight =
        GetOptionalTensorFromArgs("nll_loss", "weight", args, 2, true);

    // Parse Attributes if needed
    PyObject *ignore_index_obj = PyTuple_GET_ITEM(args, 3);
    int64_t ignore_index = CastPyArg2Long(ignore_index_obj, "nll_loss", 3);
    PyObject *reduction_obj = PyTuple_GET_ITEM(args, 4);
    std::string reduction = CastPyArg2String(reduction_obj, "nll_loss", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::nll_loss_final_state_dygraph_function(
        input, label, weight, ignore_index, reduction)) out =
        ::nll_loss_final_state_dygraph_function(
            input, label, weight, ignore_index, reduction);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_norm(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "norm pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("norm", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "norm", 1);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 2);
    float epsilon = CastPyArg2Float(epsilon_obj, "norm", 2);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 3);
    bool is_test = CastPyArg2Boolean(is_test_obj, "norm", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::norm_final_state_dygraph_function(
        x, axis, epsilon, is_test)) out =
        ::norm_final_state_dygraph_function(x, axis, epsilon, is_test);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_not_equal(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "not_equal pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: not_equal";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("not_equal", "x", args, 0, false);
    auto y = GetTensorFromArgs("not_equal", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "not_equal", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::not_equal(x, y, axis)) out =
        paddle::experimental::not_equal(x, y, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_one_hot(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "one_hot pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: one_hot";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("one_hot", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *num_classes_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar num_classes =
        CastPyArg2Scalar(num_classes_obj, "one_hot", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::one_hot(x, num_classes)) out =
        paddle::experimental::one_hot(x, num_classes);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_ones_like(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "ones_like pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: ones_like";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("ones_like", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "ones_like", 1);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 2);
    paddle::Place place = CastPyArg2Place(place_obj, "ones_like", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::ones_like(x, dtype, place)) out =
        paddle::experimental::ones_like(x, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_p_norm(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "p_norm pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: p_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("p_norm", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *porder_obj = PyTuple_GET_ITEM(args, 1);
    float porder = CastPyArg2Float(porder_obj, "p_norm", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "p_norm", 2);
    PyObject *epsilon_obj = PyTuple_GET_ITEM(args, 3);
    float epsilon = CastPyArg2Float(epsilon_obj, "p_norm", 3);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 4);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "p_norm", 4);
    PyObject *asvector_obj = PyTuple_GET_ITEM(args, 5);
    bool asvector = CastPyArg2Boolean(asvector_obj, "p_norm", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::p_norm_final_state_dygraph_function(
        x, porder, axis, epsilon, keepdim, asvector)) out =
        ::p_norm_final_state_dygraph_function(
            x, porder, axis, epsilon, keepdim, asvector);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_pad(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "pad pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pad";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pad", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "pad", 1);
    PyObject *pad_value_obj = PyTuple_GET_ITEM(args, 2);
    float pad_value = CastPyArg2Float(pad_value_obj, "pad", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pad_final_state_dygraph_function(x, paddings, pad_value)) out =
        ::pad_final_state_dygraph_function(x, paddings, pad_value);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_pad3d(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "pad3d pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pad3d";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pad3d", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray paddings =
        CastPyArg2IntArray(paddings_obj, "pad3d", 1);
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 2);
    std::string mode = CastPyArg2String(mode_obj, "pad3d", 2);
    PyObject *pad_value_obj = PyTuple_GET_ITEM(args, 3);
    float pad_value = CastPyArg2Float(pad_value_obj, "pad3d", 3);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_format = CastPyArg2String(data_format_obj, "pad3d", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pad3d_final_state_dygraph_function(
        x, paddings, mode, pad_value, data_format)) out =
        ::pad3d_final_state_dygraph_function(
            x, paddings, mode, pad_value, data_format);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_pixel_shuffle(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "pixel_shuffle pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pixel_shuffle";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pixel_shuffle", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *upscale_factor_obj = PyTuple_GET_ITEM(args, 1);
    int upscale_factor = CastPyArg2Int(upscale_factor_obj, "pixel_shuffle", 1);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 2);
    std::string data_format =
        CastPyArg2String(data_format_obj, "pixel_shuffle", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pixel_shuffle_final_state_dygraph_function(
        x, upscale_factor, data_format)) out =
        ::pixel_shuffle_final_state_dygraph_function(
            x, upscale_factor, data_format);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_pool2d(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "pool2d pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pool2d";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pool2d", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size = CastPyArg2Ints(kernel_size_obj, "pool2d", 1);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "pool2d", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "pool2d", 3);
    PyObject *ceil_mode_obj = PyTuple_GET_ITEM(args, 4);
    bool ceil_mode = CastPyArg2Boolean(ceil_mode_obj, "pool2d", 4);
    PyObject *exclusive_obj = PyTuple_GET_ITEM(args, 5);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "pool2d", 5);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 6);
    std::string data_format = CastPyArg2String(data_format_obj, "pool2d", 6);
    PyObject *pooling_type_obj = PyTuple_GET_ITEM(args, 7);
    std::string pooling_type = CastPyArg2String(pooling_type_obj, "pool2d", 7);
    PyObject *global_pooling_obj = PyTuple_GET_ITEM(args, 8);
    bool global_pooling = CastPyArg2Boolean(global_pooling_obj, "pool2d", 8);
    PyObject *adaptive_obj = PyTuple_GET_ITEM(args, 9);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "pool2d", 9);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 10);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "pool2d", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pool2d_final_state_dygraph_function(x,
                                                   kernel_size,
                                                   strides,
                                                   paddings,
                                                   ceil_mode,
                                                   exclusive,
                                                   data_format,
                                                   pooling_type,
                                                   global_pooling,
                                                   adaptive,
                                                   padding_algorithm)) out =
        ::pool2d_final_state_dygraph_function(x,
                                              kernel_size,
                                              strides,
                                              paddings,
                                              ceil_mode,
                                              exclusive,
                                              data_format,
                                              pooling_type,
                                              global_pooling,
                                              adaptive,
                                              padding_algorithm);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_pool2d_gpudnn_unused(PyObject *self,
                                                            PyObject *args,
                                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "pool2d_gpudnn_unused pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pool2d_gpudnn_unused";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pool2d_gpudnn_unused", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size =
        CastPyArg2Ints(kernel_size_obj, "pool2d_gpudnn_unused", 1);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides =
        CastPyArg2Ints(strides_obj, "pool2d_gpudnn_unused", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings =
        CastPyArg2Ints(paddings_obj, "pool2d_gpudnn_unused", 3);
    PyObject *ceil_mode_obj = PyTuple_GET_ITEM(args, 4);
    bool ceil_mode =
        CastPyArg2Boolean(ceil_mode_obj, "pool2d_gpudnn_unused", 4);
    PyObject *exclusive_obj = PyTuple_GET_ITEM(args, 5);
    bool exclusive =
        CastPyArg2Boolean(exclusive_obj, "pool2d_gpudnn_unused", 5);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 6);
    std::string data_format =
        CastPyArg2String(data_format_obj, "pool2d_gpudnn_unused", 6);
    PyObject *pooling_type_obj = PyTuple_GET_ITEM(args, 7);
    std::string pooling_type =
        CastPyArg2String(pooling_type_obj, "pool2d_gpudnn_unused", 7);
    PyObject *global_pooling_obj = PyTuple_GET_ITEM(args, 8);
    bool global_pooling =
        CastPyArg2Boolean(global_pooling_obj, "pool2d_gpudnn_unused", 8);
    PyObject *adaptive_obj = PyTuple_GET_ITEM(args, 9);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "pool2d_gpudnn_unused", 9);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 10);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "pool2d_gpudnn_unused", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pool2d_gpudnn_unused_final_state_dygraph_function(
        x,
        kernel_size,
        strides,
        paddings,
        ceil_mode,
        exclusive,
        data_format,
        pooling_type,
        global_pooling,
        adaptive,
        padding_algorithm)) out =
        ::pool2d_gpudnn_unused_final_state_dygraph_function(x,
                                                            kernel_size,
                                                            strides,
                                                            paddings,
                                                            ceil_mode,
                                                            exclusive,
                                                            data_format,
                                                            pooling_type,
                                                            global_pooling,
                                                            adaptive,
                                                            padding_algorithm);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_pool3d(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "pool3d pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pool3d";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pool3d", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size = CastPyArg2Ints(kernel_size_obj, "pool3d", 1);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "pool3d", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "pool3d", 3);
    PyObject *ceil_mode_obj = PyTuple_GET_ITEM(args, 4);
    bool ceil_mode = CastPyArg2Boolean(ceil_mode_obj, "pool3d", 4);
    PyObject *exclusive_obj = PyTuple_GET_ITEM(args, 5);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "pool3d", 5);
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 6);
    std::string data_format = CastPyArg2String(data_format_obj, "pool3d", 6);
    PyObject *pooling_type_obj = PyTuple_GET_ITEM(args, 7);
    std::string pooling_type = CastPyArg2String(pooling_type_obj, "pool3d", 7);
    PyObject *global_pooling_obj = PyTuple_GET_ITEM(args, 8);
    bool global_pooling = CastPyArg2Boolean(global_pooling_obj, "pool3d", 8);
    PyObject *adaptive_obj = PyTuple_GET_ITEM(args, 9);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "pool3d", 9);
    PyObject *padding_algorithm_obj = PyTuple_GET_ITEM(args, 10);
    std::string padding_algorithm =
        CastPyArg2String(padding_algorithm_obj, "pool3d", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pool3d_final_state_dygraph_function(x,
                                                   kernel_size,
                                                   strides,
                                                   paddings,
                                                   ceil_mode,
                                                   exclusive,
                                                   data_format,
                                                   pooling_type,
                                                   global_pooling,
                                                   adaptive,
                                                   padding_algorithm)) out =
        ::pool3d_final_state_dygraph_function(x,
                                              kernel_size,
                                              strides,
                                              paddings,
                                              ceil_mode,
                                              exclusive,
                                              data_format,
                                              pooling_type,
                                              global_pooling,
                                              adaptive,
                                              padding_algorithm);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_pow(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "pow pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pow";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pow", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *s_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar s = CastPyArg2Scalar(s_obj, "pow", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pow_final_state_dygraph_function(x, s)) out =
        ::pow_final_state_dygraph_function(x, s);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_prelu(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "prelu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: prelu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("prelu", "x", args, 0, false);
    auto alpha = GetTensorFromArgs("prelu", "alpha", args, 1, false);

    // Parse Attributes if needed
    PyObject *data_format_obj = PyTuple_GET_ITEM(args, 2);
    std::string data_format = CastPyArg2String(data_format_obj, "prelu", 2);
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 3);
    std::string mode = CastPyArg2String(mode_obj, "prelu", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::prelu_final_state_dygraph_function(
        x, alpha, data_format, mode)) out =
        ::prelu_final_state_dygraph_function(x, alpha, data_format, mode);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_psroi_pool(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "psroi_pool pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: psroi_pool";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("psroi_pool", "x", args, 0, false);
    auto boxes = GetTensorFromArgs("psroi_pool", "boxes", args, 1, false);
    auto boxes_num =
        GetOptionalTensorFromArgs("psroi_pool", "boxes_num", args, 2, true);

    // Parse Attributes if needed
    PyObject *pooled_height_obj = PyTuple_GET_ITEM(args, 3);
    int pooled_height = CastPyArg2Int(pooled_height_obj, "psroi_pool", 3);
    PyObject *pooled_width_obj = PyTuple_GET_ITEM(args, 4);
    int pooled_width = CastPyArg2Int(pooled_width_obj, "psroi_pool", 4);
    PyObject *output_channels_obj = PyTuple_GET_ITEM(args, 5);
    int output_channels = CastPyArg2Int(output_channels_obj, "psroi_pool", 5);
    PyObject *spatial_scale_obj = PyTuple_GET_ITEM(args, 6);
    float spatial_scale = CastPyArg2Float(spatial_scale_obj, "psroi_pool", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::psroi_pool_final_state_dygraph_function(x,
                                                       boxes,
                                                       boxes_num,
                                                       pooled_height,
                                                       pooled_width,
                                                       output_channels,
                                                       spatial_scale)) out =
        ::psroi_pool_final_state_dygraph_function(x,
                                                  boxes,
                                                  boxes_num,
                                                  pooled_height,
                                                  pooled_width,
                                                  output_channels,
                                                  spatial_scale);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_put_along_axis(PyObject *self,
                                                      PyObject *args,
                                                      PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "put_along_axis pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: put_along_axis";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("put_along_axis", "x", args, 0, false);
    auto index = GetTensorFromArgs("put_along_axis", "index", args, 1, false);
    auto value = GetTensorFromArgs("put_along_axis", "value", args, 2, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 3);
    int axis = CastPyArg2Int(axis_obj, "put_along_axis", 3);
    PyObject *reduce_obj = PyTuple_GET_ITEM(args, 4);
    std::string reduce = CastPyArg2String(reduce_obj, "put_along_axis", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::put_along_axis_final_state_dygraph_function(
        x, index, value, axis, reduce)) out =
        ::put_along_axis_final_state_dygraph_function(
            x, index, value, axis, reduce);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_qr(PyObject *self,
                                          PyObject *args,
                                          PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "qr pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: qr";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("qr", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *mode_obj = PyTuple_GET_ITEM(args, 1);
    std::string mode = CastPyArg2String(mode_obj, "qr", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::qr(x, mode)) out =
        paddle::experimental::qr(x, mode);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_randint(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "randint pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: randint";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject *low_obj = PyTuple_GET_ITEM(args, 0);
    int low = CastPyArg2Int(low_obj, "randint", 0);
    PyObject *high_obj = PyTuple_GET_ITEM(args, 1);
    int high = CastPyArg2Int(high_obj, "randint", 1);
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::IntArray shape =
        CastPyArg2IntArray(shape_obj, "randint", 2);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "randint", 3);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 4);
    paddle::Place place = CastPyArg2Place(place_obj, "randint", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::randint(
        low, high, shape, dtype, place)) out =
        paddle::experimental::randint(low, high, shape, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_randperm(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "randperm pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: randperm";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject *n_obj = PyTuple_GET_ITEM(args, 0);
    int n = CastPyArg2Int(n_obj, "randperm", 0);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "randperm", 1);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 2);
    paddle::Place place = CastPyArg2Place(place_obj, "randperm", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::randperm(n, dtype, place)) out =
        paddle::experimental::randperm(n, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_real(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "real pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: real";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("real", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::real_final_state_dygraph_function(x)) out =
        ::real_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_reciprocal(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "reciprocal pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: reciprocal";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("reciprocal", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::reciprocal_final_state_dygraph_function(x)) out =
        ::reciprocal_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_reduce_prod(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "reduce_prod pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: reduce_prod";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("reduce_prod", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "reduce_prod", 1);
    PyObject *keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "reduce_prod", 2);
    PyObject *reduce_all_obj = PyTuple_GET_ITEM(args, 3);
    bool reduce_all = CastPyArg2Boolean(reduce_all_obj, "reduce_prod", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::reduce_prod_final_state_dygraph_function(
        x, dims, keep_dim, reduce_all)) out =
        ::reduce_prod_final_state_dygraph_function(
            x, dims, keep_dim, reduce_all);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_relu(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "relu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: relu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("relu", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::relu_final_state_dygraph_function(x)) out =
        ::relu_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_relu_(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "relu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: relu_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("relu", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::relu__final_state_dygraph_function(x)) out =
        ::relu__final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_relu6(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "relu6 pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: relu6";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("relu6", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "relu6", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::relu6_final_state_dygraph_function(x, threshold)) out =
        ::relu6_final_state_dygraph_function(x, threshold);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_reshape(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "reshape pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: reshape";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("reshape", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray shape =
        CastPyArg2IntArray(shape_obj, "reshape", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::reshape_final_state_dygraph_function(x, shape)) out =
        ::reshape_final_state_dygraph_function(x, shape);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_reshape_(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "reshape pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: reshape_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("reshape", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray shape =
        CastPyArg2IntArray(shape_obj, "reshape", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::reshape__final_state_dygraph_function(x, shape)) out =
        ::reshape__final_state_dygraph_function(x, shape);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_roi_align(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "roi_align pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: roi_align";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("roi_align", "x", args, 0, false);
    auto boxes = GetTensorFromArgs("roi_align", "boxes", args, 1, false);
    auto boxes_num =
        GetOptionalTensorFromArgs("roi_align", "boxes_num", args, 2, true);

    // Parse Attributes if needed
    PyObject *pooled_height_obj = PyTuple_GET_ITEM(args, 3);
    int pooled_height = CastPyArg2Int(pooled_height_obj, "roi_align", 3);
    PyObject *pooled_width_obj = PyTuple_GET_ITEM(args, 4);
    int pooled_width = CastPyArg2Int(pooled_width_obj, "roi_align", 4);
    PyObject *spatial_scale_obj = PyTuple_GET_ITEM(args, 5);
    float spatial_scale = CastPyArg2Float(spatial_scale_obj, "roi_align", 5);
    PyObject *sampling_ratio_obj = PyTuple_GET_ITEM(args, 6);
    int sampling_ratio = CastPyArg2Int(sampling_ratio_obj, "roi_align", 6);
    PyObject *aligned_obj = PyTuple_GET_ITEM(args, 7);
    bool aligned = CastPyArg2Boolean(aligned_obj, "roi_align", 7);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::roi_align_final_state_dygraph_function(x,
                                                      boxes,
                                                      boxes_num,
                                                      pooled_height,
                                                      pooled_width,
                                                      spatial_scale,
                                                      sampling_ratio,
                                                      aligned)) out =
        ::roi_align_final_state_dygraph_function(x,
                                                 boxes,
                                                 boxes_num,
                                                 pooled_height,
                                                 pooled_width,
                                                 spatial_scale,
                                                 sampling_ratio,
                                                 aligned);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_roi_pool(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "roi_pool pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: roi_pool";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("roi_pool", "x", args, 0, false);
    auto boxes = GetTensorFromArgs("roi_pool", "boxes", args, 1, false);
    auto boxes_num =
        GetOptionalTensorFromArgs("roi_pool", "boxes_num", args, 2, true);

    // Parse Attributes if needed
    PyObject *pooled_height_obj = PyTuple_GET_ITEM(args, 3);
    int pooled_height = CastPyArg2Int(pooled_height_obj, "roi_pool", 3);
    PyObject *pooled_width_obj = PyTuple_GET_ITEM(args, 4);
    int pooled_width = CastPyArg2Int(pooled_width_obj, "roi_pool", 4);
    PyObject *spatial_scale_obj = PyTuple_GET_ITEM(args, 5);
    float spatial_scale = CastPyArg2Float(spatial_scale_obj, "roi_pool", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::roi_pool_final_state_dygraph_function(
        x, boxes, boxes_num, pooled_height, pooled_width, spatial_scale)) out =
        ::roi_pool_final_state_dygraph_function(
            x, boxes, boxes_num, pooled_height, pooled_width, spatial_scale);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_roll(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "roll pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: roll";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("roll", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *shifts_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray shifts =
        CastPyArg2IntArray(shifts_obj, "roll", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "roll", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::roll_final_state_dygraph_function(x, shifts, axis)) out =
        ::roll_final_state_dygraph_function(x, shifts, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_round(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "round pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: round";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("round", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::round_final_state_dygraph_function(x)) out =
        ::round_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_rsqrt(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "rsqrt pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: rsqrt";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("rsqrt", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::rsqrt_final_state_dygraph_function(x)) out =
        ::rsqrt_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_rsqrt_(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "rsqrt pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: rsqrt_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("rsqrt", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::rsqrt__final_state_dygraph_function(x)) out =
        ::rsqrt__final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_scale(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "scale pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: scale";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("scale", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar scale =
        CastPyArg2Scalar(scale_obj, "scale", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    float bias = CastPyArg2Float(bias_obj, "scale", 2);
    PyObject *bias_after_scale_obj = PyTuple_GET_ITEM(args, 3);
    bool bias_after_scale = CastPyArg2Boolean(bias_after_scale_obj, "scale", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::scale_final_state_dygraph_function(
        x, scale, bias, bias_after_scale)) out =
        ::scale_final_state_dygraph_function(x, scale, bias, bias_after_scale);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_scale_(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "scale pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: scale_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("scale", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar scale =
        CastPyArg2Scalar(scale_obj, "scale", 1);
    PyObject *bias_obj = PyTuple_GET_ITEM(args, 2);
    float bias = CastPyArg2Float(bias_obj, "scale", 2);
    PyObject *bias_after_scale_obj = PyTuple_GET_ITEM(args, 3);
    bool bias_after_scale = CastPyArg2Boolean(bias_after_scale_obj, "scale", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::scale__final_state_dygraph_function(
        x, scale, bias, bias_after_scale)) out =
        ::scale__final_state_dygraph_function(x, scale, bias, bias_after_scale);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_scatter(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "scatter pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: scatter";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("scatter", "x", args, 0, false);
    auto index = GetTensorFromArgs("scatter", "index", args, 1, false);
    auto updates = GetTensorFromArgs("scatter", "updates", args, 2, false);

    // Parse Attributes if needed
    PyObject *overwrite_obj = PyTuple_GET_ITEM(args, 3);
    bool overwrite = CastPyArg2Boolean(overwrite_obj, "scatter", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::scatter_final_state_dygraph_function(
        x, index, updates, overwrite)) out =
        ::scatter_final_state_dygraph_function(x, index, updates, overwrite);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_scatter_nd_add(PyObject *self,
                                                      PyObject *args,
                                                      PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "scatter_nd_add pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: scatter_nd_add";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("scatter_nd_add", "x", args, 0, false);
    auto index = GetTensorFromArgs("scatter_nd_add", "index", args, 1, false);
    auto updates =
        GetTensorFromArgs("scatter_nd_add", "updates", args, 2, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::scatter_nd_add_final_state_dygraph_function(
        x, index, updates)) out =
        ::scatter_nd_add_final_state_dygraph_function(x, index, updates);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_searchsorted(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "searchsorted pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: searchsorted";

    // Get EagerTensors from args
    auto sorted_sequence =
        GetTensorFromArgs("searchsorted", "sorted_sequence", args, 0, false);
    auto value = GetTensorFromArgs("searchsorted", "value", args, 1, false);

    // Parse Attributes if needed
    PyObject *out_int32_obj = PyTuple_GET_ITEM(args, 2);
    bool out_int32 = CastPyArg2Boolean(out_int32_obj, "searchsorted", 2);
    PyObject *right_obj = PyTuple_GET_ITEM(args, 3);
    bool right = CastPyArg2Boolean(right_obj, "searchsorted", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::searchsorted(
        sorted_sequence, value, out_int32, right)) out =
        paddle::experimental::searchsorted(
            sorted_sequence, value, out_int32, right);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_segment_pool(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "segment_pool pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: segment_pool";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("segment_pool", "x", args, 0, false);
    auto segment_ids =
        GetTensorFromArgs("segment_pool", "segment_ids", args, 1, false);

    // Parse Attributes if needed
    PyObject *pooltype_obj = PyTuple_GET_ITEM(args, 2);
    std::string pooltype = CastPyArg2String(pooltype_obj, "segment_pool", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::segment_pool_final_state_dygraph_function(
        x, segment_ids, pooltype)) out =
        ::segment_pool_final_state_dygraph_function(x, segment_ids, pooltype);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_selu(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "selu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: selu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("selu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *scale_obj = PyTuple_GET_ITEM(args, 1);
    float scale = CastPyArg2Float(scale_obj, "selu", 1);
    PyObject *alpha_obj = PyTuple_GET_ITEM(args, 2);
    float alpha = CastPyArg2Float(alpha_obj, "selu", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::selu_final_state_dygraph_function(x, scale, alpha)) out =
        ::selu_final_state_dygraph_function(x, scale, alpha);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_sgd_(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "sgd_ pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sgd_";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("sgd_", "param", args, 0, false);
    auto learning_rate =
        GetTensorFromArgs("sgd_", "learning_rate", args, 1, false);
    auto grad = GetTensorFromArgs("sgd_", "grad", args, 2, false);
    auto master_param =
        GetOptionalTensorFromArgs("sgd_", "master_param", args, 3, true);

    // Parse Attributes if needed
    PyObject *multi_precision_obj = PyTuple_GET_ITEM(args, 4);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "sgd_", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::sgd_(
        param, learning_rate, grad, master_param, multi_precision)) out =
        paddle::experimental::sgd_(
            param, learning_rate, grad, master_param, multi_precision);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;

    inplace_var_idx_map[1] = 3;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_shape(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "shape pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: shape";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("shape", "input", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::shape(input)) out =
        paddle::experimental::shape(input);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_shard_index(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "shard_index pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: shard_index";

    // Get EagerTensors from args
    auto in = GetTensorFromArgs("shard_index", "in", args, 0, false);

    // Parse Attributes if needed
    PyObject *index_num_obj = PyTuple_GET_ITEM(args, 1);
    int index_num = CastPyArg2Int(index_num_obj, "shard_index", 1);
    PyObject *nshards_obj = PyTuple_GET_ITEM(args, 2);
    int nshards = CastPyArg2Int(nshards_obj, "shard_index", 2);
    PyObject *shard_id_obj = PyTuple_GET_ITEM(args, 3);
    int shard_id = CastPyArg2Int(shard_id_obj, "shard_index", 3);
    PyObject *ignore_value_obj = PyTuple_GET_ITEM(args, 4);
    int ignore_value = CastPyArg2Int(ignore_value_obj, "shard_index", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::shard_index(
        in, index_num, nshards, shard_id, ignore_value)) out =
        paddle::experimental::shard_index(
            in, index_num, nshards, shard_id, ignore_value);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_sigmoid(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "sigmoid pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sigmoid";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("sigmoid", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sigmoid_final_state_dygraph_function(x)) out =
        ::sigmoid_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_sigmoid_cross_entropy_with_logits(
    PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "sigmoid_cross_entropy_with_logits pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6)
        << "Running Eager Final State API: sigmoid_cross_entropy_with_logits";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs(
        "sigmoid_cross_entropy_with_logits", "x", args, 0, false);
    auto label = GetTensorFromArgs(
        "sigmoid_cross_entropy_with_logits", "label", args, 1, false);

    // Parse Attributes if needed
    PyObject *normalize_obj = PyTuple_GET_ITEM(args, 2);
    bool normalize = CastPyArg2Boolean(
        normalize_obj, "sigmoid_cross_entropy_with_logits", 2);
    PyObject *ignore_index_obj = PyTuple_GET_ITEM(args, 3);
    int ignore_index =
        CastPyArg2Int(ignore_index_obj, "sigmoid_cross_entropy_with_logits", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sigmoid_cross_entropy_with_logits_final_state_dygraph_function(
        x, label, normalize, ignore_index)) out =
        ::sigmoid_cross_entropy_with_logits_final_state_dygraph_function(
            x, label, normalize, ignore_index);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_sign(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "sign pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sign";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("sign", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::sign(x)) out = paddle::experimental::sign(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_silu(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "silu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: silu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("silu", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::silu_final_state_dygraph_function(x)) out =
        ::silu_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_sin(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "sin pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sin";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("sin", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sin_final_state_dygraph_function(x)) out =
        ::sin_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_sinh(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "sinh pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sinh";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("sinh", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sinh_final_state_dygraph_function(x)) out =
        ::sinh_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_size(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "size pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: size";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("size", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::size(x)) out = paddle::experimental::size(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_slice(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "slice pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: slice";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("slice", "input", args, 0, false);

    // Parse Attributes if needed
    PyObject *axes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axes = CastPyArg2Longs(axes_obj, "slice", 1);
    PyObject *starts_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::IntArray starts =
        CastPyArg2IntArray(starts_obj, "slice", 2);
    PyObject *ends_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::IntArray ends =
        CastPyArg2IntArray(ends_obj, "slice", 3);
    PyObject *infer_flags_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int64_t> infer_flags =
        CastPyArg2Longs(infer_flags_obj, "slice", 4);
    PyObject *decrease_axis_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int64_t> decrease_axis =
        CastPyArg2Longs(decrease_axis_obj, "slice", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::slice_final_state_dygraph_function(
        input, axes, starts, ends, infer_flags, decrease_axis)) out =
        ::slice_final_state_dygraph_function(
            input, axes, starts, ends, infer_flags, decrease_axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_soft_shrink(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "soft_shrink pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: soft_shrink";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("soft_shrink", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *lambda_obj = PyTuple_GET_ITEM(args, 1);
    float lambda = CastPyArg2Float(lambda_obj, "soft_shrink", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::soft_shrink_final_state_dygraph_function(x, lambda)) out =
        ::soft_shrink_final_state_dygraph_function(x, lambda);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_softmax(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "softmax pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: softmax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("softmax", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "softmax", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::softmax_final_state_dygraph_function(x, axis)) out =
        ::softmax_final_state_dygraph_function(x, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_split(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "split pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: split";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("split", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *num_or_sections_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray num_or_sections =
        CastPyArg2IntArray(num_or_sections_obj, "split", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar axis = CastPyArg2Scalar(axis_obj, "split", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::split_final_state_dygraph_function(
        x, num_or_sections, axis)) out =
        ::split_final_state_dygraph_function(x, num_or_sections, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_sqrt(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "sqrt pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sqrt";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("sqrt", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sqrt_final_state_dygraph_function(x)) out =
        ::sqrt_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_square(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "square pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: square";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("square", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::square_final_state_dygraph_function(x)) out =
        ::square_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_squeeze(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "squeeze pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: squeeze";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("squeeze", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> axes = CastPyArg2Ints(axes_obj, "squeeze", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::squeeze_final_state_dygraph_function(x, axes)) out =
        ::squeeze_final_state_dygraph_function(x, axes);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_stack(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "stack pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: stack";

    // Get EagerTensors from args
    auto x = GetTensorListFromArgs("stack", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "stack", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::stack_final_state_dygraph_function(x, axis)) out =
        ::stack_final_state_dygraph_function(x, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_strided_slice(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "strided_slice pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: strided_slice";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("strided_slice", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> axes = CastPyArg2Ints(axes_obj, "strided_slice", 1);
    PyObject *starts_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::IntArray starts =
        CastPyArg2IntArray(starts_obj, "strided_slice", 2);
    PyObject *ends_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::IntArray ends =
        CastPyArg2IntArray(ends_obj, "strided_slice", 3);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 4);
    paddle::experimental::IntArray strides =
        CastPyArg2IntArray(strides_obj, "strided_slice", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::strided_slice_final_state_dygraph_function(
        x, axes, starts, ends, strides)) out =
        ::strided_slice_final_state_dygraph_function(
            x, axes, starts, ends, strides);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_subtract(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "subtract pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: subtract";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("subtract", "x", args, 0, false);
    auto y = GetTensorFromArgs("subtract", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::subtract_final_state_dygraph_function(x, y)) out =
        ::subtract_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_sum(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "sum pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sum";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("sum", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "sum", 1);
    PyObject *out_dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType out_dtype =
        CastPyArg2DataType(out_dtype_obj, "sum", 2);
    PyObject *keep_dim_obj = PyTuple_GET_ITEM(args, 3);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "sum", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sum_final_state_dygraph_function(
        x, dims, out_dtype, keep_dim)) out =
        ::sum_final_state_dygraph_function(x, dims, out_dtype, keep_dim);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_swish(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "swish pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: swish";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("swish", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *beta_obj = PyTuple_GET_ITEM(args, 1);
    float beta = CastPyArg2Float(beta_obj, "swish", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::swish_final_state_dygraph_function(x, beta)) out =
        ::swish_final_state_dygraph_function(x, beta);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_take_along_axis(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "take_along_axis pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: take_along_axis";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("take_along_axis", "x", args, 0, false);
    auto index = GetTensorFromArgs("take_along_axis", "index", args, 1, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "take_along_axis", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::take_along_axis_final_state_dygraph_function(
        x, index, axis)) out =
        ::take_along_axis_final_state_dygraph_function(x, index, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_tan(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "tan pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: tan";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("tan", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::tan_final_state_dygraph_function(x)) out =
        ::tan_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_tanh(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "tanh pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: tanh";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("tanh", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::tanh_final_state_dygraph_function(x)) out =
        ::tanh_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_tanh_shrink(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "tanh_shrink pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: tanh_shrink";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("tanh_shrink", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::tanh_shrink_final_state_dygraph_function(x)) out =
        ::tanh_shrink_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_thresholded_relu(PyObject *self,
                                                        PyObject *args,
                                                        PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "thresholded_relu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: thresholded_relu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("thresholded_relu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "thresholded_relu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::thresholded_relu_final_state_dygraph_function(x,
                                                             threshold)) out =
        ::thresholded_relu_final_state_dygraph_function(x, threshold);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_tile(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "tile pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: tile";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("tile", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *repeat_times_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray repeat_times =
        CastPyArg2IntArray(repeat_times_obj, "tile", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::tile_final_state_dygraph_function(x, repeat_times)) out =
        ::tile_final_state_dygraph_function(x, repeat_times);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_top_k(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "top_k pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: top_k";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("top_k", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *k_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar k = CastPyArg2Scalar(k_obj, "top_k", 1);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "top_k", 2);
    PyObject *largest_obj = PyTuple_GET_ITEM(args, 3);
    bool largest = CastPyArg2Boolean(largest_obj, "top_k", 3);
    PyObject *sorted_obj = PyTuple_GET_ITEM(args, 4);
    bool sorted = CastPyArg2Boolean(sorted_obj, "top_k", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::top_k_final_state_dygraph_function(
        x, k, axis, largest, sorted)) out =
        ::top_k_final_state_dygraph_function(x, k, axis, largest, sorted);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_transpose(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "transpose pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: transpose";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("transpose", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> axis = CastPyArg2Ints(axis_obj, "transpose", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::transpose_final_state_dygraph_function(x, axis)) out =
        ::transpose_final_state_dygraph_function(x, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_triangular_solve(PyObject *self,
                                                        PyObject *args,
                                                        PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "triangular_solve pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: triangular_solve";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("triangular_solve", "x", args, 0, false);
    auto y = GetTensorFromArgs("triangular_solve", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject *upper_obj = PyTuple_GET_ITEM(args, 2);
    bool upper = CastPyArg2Boolean(upper_obj, "triangular_solve", 2);
    PyObject *transpose_obj = PyTuple_GET_ITEM(args, 3);
    bool transpose = CastPyArg2Boolean(transpose_obj, "triangular_solve", 3);
    PyObject *unitriangular_obj = PyTuple_GET_ITEM(args, 4);
    bool unitriangular =
        CastPyArg2Boolean(unitriangular_obj, "triangular_solve", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::triangular_solve_final_state_dygraph_function(
        x, y, upper, transpose, unitriangular)) out =
        ::triangular_solve_final_state_dygraph_function(
            x, y, upper, transpose, unitriangular);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_tril_indices(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "tril_indices pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: tril_indices";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject *rows_obj = PyTuple_GET_ITEM(args, 0);
    int rows = CastPyArg2Int(rows_obj, "tril_indices", 0);
    PyObject *cols_obj = PyTuple_GET_ITEM(args, 1);
    int cols = CastPyArg2Int(cols_obj, "tril_indices", 1);
    PyObject *offset_obj = PyTuple_GET_ITEM(args, 2);
    int offset = CastPyArg2Int(offset_obj, "tril_indices", 2);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "tril_indices", 3);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 4);
    paddle::Place place = CastPyArg2Place(place_obj, "tril_indices", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::tril_indices(
        rows, cols, offset, dtype, place)) out =
        paddle::experimental::tril_indices(rows, cols, offset, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_tril_triu(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "tril_triu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: tril_triu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("tril_triu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *diagonal_obj = PyTuple_GET_ITEM(args, 1);
    int diagonal = CastPyArg2Int(diagonal_obj, "tril_triu", 1);
    PyObject *lower_obj = PyTuple_GET_ITEM(args, 2);
    bool lower = CastPyArg2Boolean(lower_obj, "tril_triu", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::tril_triu_final_state_dygraph_function(x, diagonal, lower)) out =
        ::tril_triu_final_state_dygraph_function(x, diagonal, lower);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_truncated_gaussian_random(
    PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "truncated_gaussian_random pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: truncated_gaussian_random";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    std::vector<int> shape =
        CastPyArg2Ints(shape_obj, "truncated_gaussian_random", 0);
    PyObject *mean_obj = PyTuple_GET_ITEM(args, 1);
    float mean = CastPyArg2Float(mean_obj, "truncated_gaussian_random", 1);
    PyObject *std_obj = PyTuple_GET_ITEM(args, 2);
    float std = CastPyArg2Float(std_obj, "truncated_gaussian_random", 2);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 3);
    int seed = CastPyArg2Int(seed_obj, "truncated_gaussian_random", 3);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 4);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "truncated_gaussian_random", 4);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 5);
    paddle::Place place =
        CastPyArg2Place(place_obj, "truncated_gaussian_random", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::truncated_gaussian_random(
        shape, mean, std, seed, dtype, place)) out =
        paddle::experimental::truncated_gaussian_random(
            shape, mean, std, seed, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_unbind(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "unbind pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unbind";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("unbind", "input", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "unbind", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::unbind_final_state_dygraph_function(input, axis)) out =
        ::unbind_final_state_dygraph_function(input, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_unfold(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "unfold pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unfold";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("unfold", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *kernel_sizes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_sizes =
        CastPyArg2Ints(kernel_sizes_obj, "unfold", 1);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "unfold", 2);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "unfold", 3);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "unfold", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::unfold_final_state_dygraph_function(
        x, kernel_sizes, strides, paddings, dilations)) out =
        ::unfold_final_state_dygraph_function(
            x, kernel_sizes, strides, paddings, dilations);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_uniform_random(PyObject *self,
                                                      PyObject *args,
                                                      PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "uniform_random pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: uniform_random";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    paddle::experimental::IntArray shape =
        CastPyArg2IntArray(shape_obj, "uniform_random", 0);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "uniform_random", 1);
    PyObject *min_obj = PyTuple_GET_ITEM(args, 2);
    float min = CastPyArg2Float(min_obj, "uniform_random", 2);
    PyObject *max_obj = PyTuple_GET_ITEM(args, 3);
    float max = CastPyArg2Float(max_obj, "uniform_random", 3);
    PyObject *seed_obj = PyTuple_GET_ITEM(args, 4);
    int seed = CastPyArg2Int(seed_obj, "uniform_random", 4);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 5);
    paddle::Place place = CastPyArg2Place(place_obj, "uniform_random", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::uniform_random(
        shape, dtype, min, max, seed, place)) out =
        paddle::experimental::uniform_random(
            shape, dtype, min, max, seed, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_unique(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "unique pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unique";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("unique", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *return_index_obj = PyTuple_GET_ITEM(args, 1);
    bool return_index = CastPyArg2Boolean(return_index_obj, "unique", 1);
    PyObject *return_inverse_obj = PyTuple_GET_ITEM(args, 2);
    bool return_inverse = CastPyArg2Boolean(return_inverse_obj, "unique", 2);
    PyObject *return_counts_obj = PyTuple_GET_ITEM(args, 3);
    bool return_counts = CastPyArg2Boolean(return_counts_obj, "unique", 3);
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> axis = CastPyArg2Ints(axis_obj, "unique", 4);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 5);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "unique", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::unique(
        x, return_index, return_inverse, return_counts, axis, dtype)) out =
        paddle::experimental::unique(
            x, return_index, return_inverse, return_counts, axis, dtype);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_unsqueeze(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "unsqueeze pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unsqueeze";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("unsqueeze", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray axis =
        CastPyArg2IntArray(axis_obj, "unsqueeze", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::unsqueeze_final_state_dygraph_function(x, axis)) out =
        ::unsqueeze_final_state_dygraph_function(x, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_viterbi_decode(PyObject *self,
                                                      PyObject *args,
                                                      PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "viterbi_decode pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: viterbi_decode";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("viterbi_decode", "input", args, 0, false);
    auto transition =
        GetTensorFromArgs("viterbi_decode", "transition", args, 1, false);
    auto length = GetTensorFromArgs("viterbi_decode", "length", args, 2, false);

    // Parse Attributes if needed
    PyObject *include_bos_eos_tag_obj = PyTuple_GET_ITEM(args, 3);
    bool include_bos_eos_tag =
        CastPyArg2Boolean(include_bos_eos_tag_obj, "viterbi_decode", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::viterbi_decode(
        input, transition, length, include_bos_eos_tag)) out =
        paddle::experimental::viterbi_decode(
            input, transition, length, include_bos_eos_tag);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_where(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "where pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: where";

    // Get EagerTensors from args
    auto condition = GetTensorFromArgs("where", "condition", args, 0, false);
    auto x = GetTensorFromArgs("where", "x", args, 1, false);
    auto y = GetTensorFromArgs("where", "y", args, 2, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::where_final_state_dygraph_function(condition, x, y)) out =
        ::where_final_state_dygraph_function(condition, x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_where_index(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "where_index pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: where_index";

    // Get EagerTensors from args
    auto condition =
        GetTensorFromArgs("where_index", "condition", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::where_index(condition)) out =
        paddle::experimental::where_index(condition);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_yolo_box(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "yolo_box pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: yolo_box";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("yolo_box", "x", args, 0, false);
    auto img_size = GetTensorFromArgs("yolo_box", "img_size", args, 1, false);

    // Parse Attributes if needed
    PyObject *anchors_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> anchors = CastPyArg2Ints(anchors_obj, "yolo_box", 2);
    PyObject *class_num_obj = PyTuple_GET_ITEM(args, 3);
    int class_num = CastPyArg2Int(class_num_obj, "yolo_box", 3);
    PyObject *conf_thresh_obj = PyTuple_GET_ITEM(args, 4);
    float conf_thresh = CastPyArg2Float(conf_thresh_obj, "yolo_box", 4);
    PyObject *downsample_ratio_obj = PyTuple_GET_ITEM(args, 5);
    int downsample_ratio = CastPyArg2Int(downsample_ratio_obj, "yolo_box", 5);
    PyObject *clip_bbox_obj = PyTuple_GET_ITEM(args, 6);
    bool clip_bbox = CastPyArg2Boolean(clip_bbox_obj, "yolo_box", 6);
    PyObject *scale_x_y_obj = PyTuple_GET_ITEM(args, 7);
    float scale_x_y = CastPyArg2Float(scale_x_y_obj, "yolo_box", 7);
    PyObject *iou_aware_obj = PyTuple_GET_ITEM(args, 8);
    bool iou_aware = CastPyArg2Boolean(iou_aware_obj, "yolo_box", 8);
    PyObject *iou_aware_factor_obj = PyTuple_GET_ITEM(args, 9);
    float iou_aware_factor =
        CastPyArg2Float(iou_aware_factor_obj, "yolo_box", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::yolo_box(x,
                                            img_size,
                                            anchors,
                                            class_num,
                                            conf_thresh,
                                            downsample_ratio,
                                            clip_bbox,
                                            scale_x_y,
                                            iou_aware,
                                            iou_aware_factor)) out =
        paddle::experimental::yolo_box(x,
                                       img_size,
                                       anchors,
                                       class_num,
                                       conf_thresh,
                                       downsample_ratio,
                                       clip_bbox,
                                       scale_x_y,
                                       iou_aware,
                                       iou_aware_factor);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_zeros_like(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "zeros_like pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: zeros_like";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("zeros_like", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "zeros_like", 1);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 2);
    paddle::Place place = CastPyArg2Place(place_obj, "zeros_like", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::zeros_like(x, dtype, place)) out =
        paddle::experimental::zeros_like(x, dtype, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

namespace sparse {

static PyObject *eager_final_state_api_add(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "add pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: add";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("add", "x", args, 0, false);
    auto y = GetTensorFromArgs("add", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::add_final_state_dygraph_function(x, y)) out =
        ::sparse::add_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_conv3d(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "conv3d pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: conv3d";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("conv3d", "x", args, 0, false);
    auto kernel = GetTensorFromArgs("conv3d", "kernel", args, 1, false);

    // Parse Attributes if needed
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "conv3d", 2);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "conv3d", 3);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "conv3d", 4);
    PyObject *groups_obj = PyTuple_GET_ITEM(args, 5);
    int groups = CastPyArg2Int(groups_obj, "conv3d", 5);
    PyObject *subm_obj = PyTuple_GET_ITEM(args, 6);
    bool subm = CastPyArg2Boolean(subm_obj, "conv3d", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::conv3d_final_state_dygraph_function(
        x, kernel, paddings, dilations, strides, groups, subm)) out =
        ::sparse::conv3d_final_state_dygraph_function(
            x, kernel, paddings, dilations, strides, groups, subm);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_coo_to_dense(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "coo_to_dense pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: coo_to_dense";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("coo_to_dense", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::coo_to_dense_final_state_dygraph_function(x)) out =
        ::sparse::coo_to_dense_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_create_sparse_coo_tensor(
    PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "create_sparse_coo_tensor pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: create_sparse_coo_tensor";

    // Get EagerTensors from args
    auto values =
        GetTensorFromArgs("create_sparse_coo_tensor", "values", args, 0, false);
    auto indices = GetTensorFromArgs(
        "create_sparse_coo_tensor", "indices", args, 1, false);

    // Parse Attributes if needed
    PyObject *dense_shape_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::IntArray dense_shape =
        CastPyArg2IntArray(dense_shape_obj, "create_sparse_coo_tensor", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::create_sparse_coo_tensor_final_state_dygraph_function(
        values, indices, dense_shape)) out =
        ::sparse::create_sparse_coo_tensor_final_state_dygraph_function(
            values, indices, dense_shape);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_dense_to_coo(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "dense_to_coo pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: dense_to_coo";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("dense_to_coo", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *sparse_dim_obj = PyTuple_GET_ITEM(args, 1);
    int64_t sparse_dim = CastPyArg2Long(sparse_dim_obj, "dense_to_coo", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::dense_to_coo_final_state_dygraph_function(
        x, sparse_dim)) out =
        ::sparse::dense_to_coo_final_state_dygraph_function(x, sparse_dim);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_divide(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "divide pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: divide";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("divide", "x", args, 0, false);
    auto y = GetTensorFromArgs("divide", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::divide_final_state_dygraph_function(x, y)) out =
        ::sparse::divide_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_multiply(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "multiply pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: multiply";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("multiply", "x", args, 0, false);
    auto y = GetTensorFromArgs("multiply", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::multiply_final_state_dygraph_function(x, y)) out =
        ::sparse::multiply_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_relu(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "relu pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: relu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("relu", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::relu_final_state_dygraph_function(x)) out =
        ::sparse::relu_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_sin(PyObject *self,
                                           PyObject *args,
                                           PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "sin pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sin";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("sin", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::sin_final_state_dygraph_function(x)) out =
        ::sparse::sin_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_softmax(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "softmax pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: softmax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("softmax", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "softmax", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::softmax_final_state_dygraph_function(x, axis)) out =
        ::sparse::softmax_final_state_dygraph_function(x, axis);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_sqrt(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "sqrt pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sqrt";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("sqrt", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::sqrt_final_state_dygraph_function(x)) out =
        ::sparse::sqrt_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_subtract(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "subtract pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: subtract";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("subtract", "x", args, 0, false);
    auto y = GetTensorFromArgs("subtract", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::subtract_final_state_dygraph_function(x, y)) out =
        ::sparse::subtract_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_tanh(PyObject *self,
                                            PyObject *args,
                                            PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "tanh pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: tanh";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("tanh", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::tanh_final_state_dygraph_function(x)) out =
        ::sparse::tanh_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_to_dense(PyObject *self,
                                                PyObject *args,
                                                PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "to_dense pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: to_dense";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("to_dense", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::sparse::to_dense(x)) out =
        paddle::experimental::sparse::to_dense(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_to_sparse_coo(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "to_sparse_coo pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: to_sparse_coo";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("to_sparse_coo", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *sparse_dim_obj = PyTuple_GET_ITEM(args, 1);
    int64_t sparse_dim = CastPyArg2Long(sparse_dim_obj, "to_sparse_coo", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::sparse::to_sparse_coo(x, sparse_dim)) out =
        paddle::experimental::sparse::to_sparse_coo(x, sparse_dim);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_to_sparse_csr(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "to_sparse_csr pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: to_sparse_csr";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("to_sparse_csr", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::sparse::to_sparse_csr(x)) out =
        paddle::experimental::sparse::to_sparse_csr(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_values(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "values pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: values";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("values", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::values_final_state_dygraph_function(x)) out =
        ::sparse::values_final_state_dygraph_function(x);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_full_like(PyObject *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "full_like pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: full_like";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("full_like", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *value_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar value =
        CastPyArg2Scalar(value_obj, "full_like", 1);
    PyObject *dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType dtype =
        CastPyArg2DataType(dtype_obj, "full_like", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::sparse::full_like(x, value, dtype)) out =
        paddle::experimental::sparse::full_like(x, value, dtype);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_fused_attention(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "fused_attention pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fused_attention";

    // Get EagerTensors from args
    auto query = GetTensorFromArgs("fused_attention", "query", args, 0, false);
    auto key = GetTensorFromArgs("fused_attention", "key", args, 1, false);
    auto value = GetTensorFromArgs("fused_attention", "value", args, 2, false);
    auto sparse_mask =
        GetTensorFromArgs("fused_attention", "sparse_mask", args, 3, false);
    auto key_padding_mask = GetOptionalTensorFromArgs(
        "fused_attention", "key_padding_mask", args, 4, true);
    auto attn_mask = GetOptionalTensorFromArgs(
        "fused_attention", "attn_mask", args, 5, true);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::fused_attention_final_state_dygraph_function(
        query, key, value, sparse_mask, key_padding_mask, attn_mask)) out =
        ::sparse::fused_attention_final_state_dygraph_function(
            query, key, value, sparse_mask, key_padding_mask, attn_mask);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_masked_matmul(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "masked_matmul pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: masked_matmul";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("masked_matmul", "x", args, 0, false);
    auto y = GetTensorFromArgs("masked_matmul", "y", args, 1, false);
    auto mask = GetTensorFromArgs("masked_matmul", "mask", args, 2, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::masked_matmul_final_state_dygraph_function(
        x, y, mask)) out =
        ::sparse::masked_matmul_final_state_dygraph_function(x, y, mask);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_matmul(PyObject *self,
                                              PyObject *args,
                                              PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "matmul pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: matmul";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("matmul", "x", args, 0, false);
    auto y = GetTensorFromArgs("matmul", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::matmul_final_state_dygraph_function(x, y)) out =
        ::sparse::matmul_final_state_dygraph_function(x, y);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_maxpool(PyObject *self,
                                               PyObject *args,
                                               PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "maxpool pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: maxpool";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("maxpool", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *kernel_sizes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_sizes =
        CastPyArg2Ints(kernel_sizes_obj, "maxpool", 1);
    PyObject *paddings_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "maxpool", 2);
    PyObject *dilations_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "maxpool", 3);
    PyObject *strides_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "maxpool", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::maxpool_final_state_dygraph_function(
        x, kernel_sizes, paddings, dilations, strides)) out =
        ::sparse::maxpool_final_state_dygraph_function(
            x, kernel_sizes, paddings, dilations, strides);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_mv(PyObject *self,
                                          PyObject *args,
                                          PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "mv pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: mv";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("mv", "x", args, 0, false);
    auto vec = GetTensorFromArgs("mv", "vec", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::mv_final_state_dygraph_function(x, vec)) out =
        ::sparse::mv_final_state_dygraph_function(x, vec);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

}  // namespace sparse

namespace strings {

static PyObject *eager_final_state_api_empty(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "empty pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: empty";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject *shape_obj = PyTuple_GET_ITEM(args, 0);
    paddle::experimental::IntArray shape =
        CastPyArg2IntArray(shape_obj, "empty", 0);
    PyObject *place_obj = PyTuple_GET_ITEM(args, 1);
    paddle::Place place = CastPyArg2Place(place_obj, "empty", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::strings::empty(shape, place)) out =
        paddle::experimental::strings::empty(shape, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_empty_like(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "empty_like pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: empty_like";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("empty_like", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *place_obj = PyTuple_GET_ITEM(args, 1);
    paddle::Place place = CastPyArg2Place(place_obj, "empty_like", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::strings::empty_like(x, place)) out =
        paddle::experimental::strings::empty_like(x, place);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_lower(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "lower pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: lower";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("lower", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *use_utf8_encoding_obj = PyTuple_GET_ITEM(args, 1);
    bool use_utf8_encoding =
        CastPyArg2Boolean(use_utf8_encoding_obj, "lower", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::strings::lower(x, use_utf8_encoding)) out =
        paddle::experimental::strings::lower(x, use_utf8_encoding);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_final_state_api_upper(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "upper pybind_imperative_func",
      paddle::platform::TracerEventType::Operator,
      1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: upper";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("upper", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject *use_utf8_encoding_obj = PyTuple_GET_ITEM(args, 1);
    bool use_utf8_encoding =
        CastPyArg2Boolean(use_utf8_encoding_obj, "upper", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }

    // Call dygraph function
    decltype(paddle::experimental::strings::upper(x, use_utf8_encoding)) out =
        paddle::experimental::strings::upper(x, use_utf8_encoding);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

}  // namespace strings

static PyObject *eager_get_final_state_core_ops_args_info(PyObject *self) {
  PyThreadState *tstate = nullptr;
  try {
    return ToPyObject(core_ops_final_state_args_info);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_get_final_state_core_ops_args_type_info(PyObject *self) {
  PyThreadState *tstate = nullptr;
  try {
    return ToPyObject(core_ops_final_state_args_type_info);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_get_final_state_core_ops_returns_info(PyObject *self) {
  PyThreadState *tstate = nullptr;
  try {
    return ToPyObject(core_ops_final_state_returns_info);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyMethodDef EagerFinalStateMethods[] = {

    {"final_state_atan2",
     (PyCFunction)(void (*)(void))eager_final_state_api_atan2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for atan2 in dygraph."}

    ,

    {"final_state_bernoulli",
     (PyCFunction)(void (*)(void))eager_final_state_api_bernoulli,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bernoulli in dygraph."}

    ,

    {"final_state_cholesky",
     (PyCFunction)(void (*)(void))eager_final_state_api_cholesky,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cholesky in dygraph."}

    ,

    {"final_state_cholesky_solve",
     (PyCFunction)(void (*)(void))eager_final_state_api_cholesky_solve,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cholesky_solve in dygraph."}

    ,

    {"final_state_cross",
     (PyCFunction)(void (*)(void))eager_final_state_api_cross,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cross in dygraph."}

    ,

    {"final_state_diagonal",
     (PyCFunction)(void (*)(void))eager_final_state_api_diagonal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for diagonal in dygraph."}

    ,

    {"final_state_digamma",
     (PyCFunction)(void (*)(void))eager_final_state_api_digamma,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for digamma in dygraph."}

    ,

    {"final_state_dist",
     (PyCFunction)(void (*)(void))eager_final_state_api_dist,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dist in dygraph."}

    ,

    {"final_state_dot",
     (PyCFunction)(void (*)(void))eager_final_state_api_dot,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dot in dygraph."}

    ,

    {"final_state_erf",
     (PyCFunction)(void (*)(void))eager_final_state_api_erf,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for erf in dygraph."}

    ,

    {"final_state_mv",
     (PyCFunction)(void (*)(void))eager_final_state_api_mv,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for mv in dygraph."}

    ,

    {"final_state_poisson",
     (PyCFunction)(void (*)(void))eager_final_state_api_poisson,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for poisson in dygraph."}

    ,

    {"final_state_trace",
     (PyCFunction)(void (*)(void))eager_final_state_api_trace,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for trace in dygraph."}

    ,

    {"final_state_trunc",
     (PyCFunction)(void (*)(void))eager_final_state_api_trunc,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for trunc in dygraph."}

    ,

    {"final_state_abs",
     (PyCFunction)(void (*)(void))eager_final_state_api_abs,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for abs in dygraph."}

    ,

    {"final_state_accuracy",
     (PyCFunction)(void (*)(void))eager_final_state_api_accuracy,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for accuracy in dygraph."}

    ,

    {"final_state_acos",
     (PyCFunction)(void (*)(void))eager_final_state_api_acos,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for acos in dygraph."}

    ,

    {"final_state_acosh",
     (PyCFunction)(void (*)(void))eager_final_state_api_acosh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for acosh in dygraph."}

    ,

    {"final_state_adadelta",
     (PyCFunction)(void (*)(void))eager_final_state_api_adadelta,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for adadelta in dygraph."}

    ,

    {"final_state_adam_",
     (PyCFunction)(void (*)(void))eager_final_state_api_adam_,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for adam_ in dygraph."}

    ,

    {"final_state_adamax",
     (PyCFunction)(void (*)(void))eager_final_state_api_adamax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for adamax in dygraph."}

    ,

    {"final_state_adamw",
     (PyCFunction)(void (*)(void))eager_final_state_api_adamw,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for adamw in dygraph."}

    ,

    {"final_state_add",
     (PyCFunction)(void (*)(void))eager_final_state_api_add,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for add in dygraph."}

    ,

    {"final_state_add_n",
     (PyCFunction)(void (*)(void))eager_final_state_api_add_n,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for add_n in dygraph."}

    ,

    {"final_state_addmm",
     (PyCFunction)(void (*)(void))eager_final_state_api_addmm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for addmm in dygraph."}

    ,

    {"final_state_all",
     (PyCFunction)(void (*)(void))eager_final_state_api_all,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for all in dygraph."}

    ,

    {"final_state_allclose",
     (PyCFunction)(void (*)(void))eager_final_state_api_allclose,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for allclose in dygraph."}

    ,

    {"final_state_any",
     (PyCFunction)(void (*)(void))eager_final_state_api_any,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for any in dygraph."}

    ,

    {"final_state_arange",
     (PyCFunction)(void (*)(void))eager_final_state_api_arange,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for arange in dygraph."}

    ,

    {"final_state_argmax",
     (PyCFunction)(void (*)(void))eager_final_state_api_argmax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for argmax in dygraph."}

    ,

    {"final_state_argmin",
     (PyCFunction)(void (*)(void))eager_final_state_api_argmin,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for argmin in dygraph."}

    ,

    {"final_state_argsort",
     (PyCFunction)(void (*)(void))eager_final_state_api_argsort,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for argsort in dygraph."}

    ,

    {"final_state_asin",
     (PyCFunction)(void (*)(void))eager_final_state_api_asin,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for asin in dygraph."}

    ,

    {"final_state_asinh",
     (PyCFunction)(void (*)(void))eager_final_state_api_asinh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for asinh in dygraph."}

    ,

    {"final_state_assign",
     (PyCFunction)(void (*)(void))eager_final_state_api_assign,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for assign in dygraph."}

    ,

    {"final_state_assign_out_",
     (PyCFunction)(void (*)(void))eager_final_state_api_assign_out_,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for assign_out_ in dygraph."}

    ,

    {"final_state_atan",
     (PyCFunction)(void (*)(void))eager_final_state_api_atan,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for atan in dygraph."}

    ,

    {"final_state_atanh",
     (PyCFunction)(void (*)(void))eager_final_state_api_atanh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for atanh in dygraph."}

    ,

    {"final_state_auc",
     (PyCFunction)(void (*)(void))eager_final_state_api_auc,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for auc in dygraph."}

    ,

    {"final_state_batch_norm",
     (PyCFunction)(void (*)(void))eager_final_state_api_batch_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for batch_norm in dygraph."}

    ,

    {"final_state_bce_loss",
     (PyCFunction)(void (*)(void))eager_final_state_api_bce_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bce_loss in dygraph."}

    ,

    {"final_state_bitwise_and",
     (PyCFunction)(void (*)(void))eager_final_state_api_bitwise_and,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bitwise_and in dygraph."}

    ,

    {"final_state_bitwise_not",
     (PyCFunction)(void (*)(void))eager_final_state_api_bitwise_not,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bitwise_not in dygraph."}

    ,

    {"final_state_bitwise_or",
     (PyCFunction)(void (*)(void))eager_final_state_api_bitwise_or,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bitwise_or in dygraph."}

    ,

    {"final_state_bitwise_xor",
     (PyCFunction)(void (*)(void))eager_final_state_api_bitwise_xor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bitwise_xor in dygraph."}

    ,

    {"final_state_brelu",
     (PyCFunction)(void (*)(void))eager_final_state_api_brelu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for brelu in dygraph."}

    ,

    {"final_state_cast",
     (PyCFunction)(void (*)(void))eager_final_state_api_cast,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cast in dygraph."}

    ,

    {"final_state_ceil",
     (PyCFunction)(void (*)(void))eager_final_state_api_ceil,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for ceil in dygraph."}

    ,

    {"final_state_celu",
     (PyCFunction)(void (*)(void))eager_final_state_api_celu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for celu in dygraph."}

    ,

    {"final_state_clip",
     (PyCFunction)(void (*)(void))eager_final_state_api_clip,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for clip in dygraph."}

    ,
    {"final_state_clip_",
     (PyCFunction)(void (*)(void))eager_final_state_api_clip_,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for clip_ in dygraph."}

    ,

    {"final_state_concat",
     (PyCFunction)(void (*)(void))eager_final_state_api_concat,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for concat in dygraph."}

    ,

    {"final_state_conj",
     (PyCFunction)(void (*)(void))eager_final_state_api_conj,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conj in dygraph."}

    ,

    {"final_state_conv2d",
     (PyCFunction)(void (*)(void))eager_final_state_api_conv2d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conv2d in dygraph."}

    ,

    {"final_state_conv2d_transpose",
     (PyCFunction)(void (*)(void))eager_final_state_api_conv2d_transpose,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conv2d_transpose in dygraph."}

    ,

    {"final_state_conv3d",
     (PyCFunction)(void (*)(void))eager_final_state_api_conv3d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conv3d in dygraph."}

    ,

    {"final_state_conv3d_transpose",
     (PyCFunction)(void (*)(void))eager_final_state_api_conv3d_transpose,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conv3d_transpose in dygraph."}

    ,

    {"final_state_copy_to",
     (PyCFunction)(void (*)(void))eager_final_state_api_copy_to,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for copy_to in dygraph."}

    ,

    {"final_state_cos",
     (PyCFunction)(void (*)(void))eager_final_state_api_cos,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cos in dygraph."}

    ,

    {"final_state_cosh",
     (PyCFunction)(void (*)(void))eager_final_state_api_cosh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cosh in dygraph."}

    ,

    {"final_state_cross_entropy_with_softmax",
     (PyCFunction)(void (*)(
         void))eager_final_state_api_cross_entropy_with_softmax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cross_entropy_with_softmax in dygraph."}

    ,

    {"final_state_cumprod",
     (PyCFunction)(void (*)(void))eager_final_state_api_cumprod,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cumprod in dygraph."}

    ,

    {"final_state_cumsum",
     (PyCFunction)(void (*)(void))eager_final_state_api_cumsum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cumsum in dygraph."}

    ,

    {"final_state_deformable_conv",
     (PyCFunction)(void (*)(void))eager_final_state_api_deformable_conv,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for deformable_conv in dygraph."}

    ,

    {"final_state_depthwise_conv2d",
     (PyCFunction)(void (*)(void))eager_final_state_api_depthwise_conv2d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for depthwise_conv2d in dygraph."}

    ,

    {"final_state_depthwise_conv2d_transpose",
     (PyCFunction)(void (*)(
         void))eager_final_state_api_depthwise_conv2d_transpose,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for depthwise_conv2d_transpose in dygraph."}

    ,

    {"final_state_det",
     (PyCFunction)(void (*)(void))eager_final_state_api_det,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for det in dygraph."}

    ,

    {"final_state_diag",
     (PyCFunction)(void (*)(void))eager_final_state_api_diag,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for diag in dygraph."}

    ,

    {"final_state_divide",
     (PyCFunction)(void (*)(void))eager_final_state_api_divide,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for divide in dygraph."}

    ,

    {"final_state_dropout",
     (PyCFunction)(void (*)(void))eager_final_state_api_dropout,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dropout in dygraph."}

    ,

    {"final_state_eigh",
     (PyCFunction)(void (*)(void))eager_final_state_api_eigh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for eigh in dygraph."}

    ,

    {"final_state_einsum",
     (PyCFunction)(void (*)(void))eager_final_state_api_einsum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for einsum in dygraph."}

    ,

    {"final_state_elementwise_pow",
     (PyCFunction)(void (*)(void))eager_final_state_api_elementwise_pow,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elementwise_pow in dygraph."}

    ,

    {"final_state_elu",
     (PyCFunction)(void (*)(void))eager_final_state_api_elu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elu in dygraph."}

    ,

    {"final_state_embedding",
     (PyCFunction)(void (*)(void))eager_final_state_api_embedding,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for embedding in dygraph."}

    ,

    {"final_state_empty",
     (PyCFunction)(void (*)(void))eager_final_state_api_empty,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for empty in dygraph."}

    ,

    {"final_state_empty_like",
     (PyCFunction)(void (*)(void))eager_final_state_api_empty_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for empty_like in dygraph."}

    ,

    {"final_state_equal",
     (PyCFunction)(void (*)(void))eager_final_state_api_equal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for equal in dygraph."}

    ,

    {"final_state_equal_all",
     (PyCFunction)(void (*)(void))eager_final_state_api_equal_all,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for equal_all in dygraph."}

    ,

    {"final_state_erfinv",
     (PyCFunction)(void (*)(void))eager_final_state_api_erfinv,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for erfinv in dygraph."}

    ,
    {"final_state_erfinv_",
     (PyCFunction)(void (*)(void))eager_final_state_api_erfinv_,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for erfinv_ in dygraph."}

    ,

    {"final_state_exp",
     (PyCFunction)(void (*)(void))eager_final_state_api_exp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for exp in dygraph."}

    ,

    {"final_state_expand",
     (PyCFunction)(void (*)(void))eager_final_state_api_expand,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for expand in dygraph."}

    ,

    {"final_state_expand_as",
     (PyCFunction)(void (*)(void))eager_final_state_api_expand_as,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for expand_as in dygraph."}

    ,

    {"final_state_expm1",
     (PyCFunction)(void (*)(void))eager_final_state_api_expm1,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for expm1 in dygraph."}

    ,

    {"final_state_eye",
     (PyCFunction)(void (*)(void))eager_final_state_api_eye,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for eye in dygraph."}

    ,

    {"final_state_flatten",
     (PyCFunction)(void (*)(void))eager_final_state_api_flatten,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for flatten in dygraph."}

    ,
    {"final_state_flatten_",
     (PyCFunction)(void (*)(void))eager_final_state_api_flatten_,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for flatten_ in dygraph."}

    ,

    {"final_state_flip",
     (PyCFunction)(void (*)(void))eager_final_state_api_flip,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for flip in dygraph."}

    ,

    {"final_state_floor",
     (PyCFunction)(void (*)(void))eager_final_state_api_floor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for floor in dygraph."}

    ,

    {"final_state_floor_divide",
     (PyCFunction)(void (*)(void))eager_final_state_api_floor_divide,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for floor_divide in dygraph."}

    ,

    {"final_state_fmax",
     (PyCFunction)(void (*)(void))eager_final_state_api_fmax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fmax in dygraph."}

    ,

    {"final_state_fmin",
     (PyCFunction)(void (*)(void))eager_final_state_api_fmin,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fmin in dygraph."}

    ,

    {"final_state_frobenius_norm",
     (PyCFunction)(void (*)(void))eager_final_state_api_frobenius_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for frobenius_norm in dygraph."}

    ,

    {"final_state_full",
     (PyCFunction)(void (*)(void))eager_final_state_api_full,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for full in dygraph."}

    ,

    {"final_state_full_batch_size_like",
     (PyCFunction)(void (*)(void))eager_final_state_api_full_batch_size_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for full_batch_size_like in dygraph."}

    ,

    {"final_state_full_like",
     (PyCFunction)(void (*)(void))eager_final_state_api_full_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for full_like in dygraph."}

    ,

    {"final_state_gather",
     (PyCFunction)(void (*)(void))eager_final_state_api_gather,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gather in dygraph."}

    ,

    {"final_state_gather_nd",
     (PyCFunction)(void (*)(void))eager_final_state_api_gather_nd,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gather_nd in dygraph."}

    ,

    {"final_state_gather_tree",
     (PyCFunction)(void (*)(void))eager_final_state_api_gather_tree,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gather_tree in dygraph."}

    ,

    {"final_state_gaussian_random",
     (PyCFunction)(void (*)(void))eager_final_state_api_gaussian_random,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gaussian_random in dygraph."}

    ,

    {"final_state_gelu",
     (PyCFunction)(void (*)(void))eager_final_state_api_gelu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gelu in dygraph."}

    ,

    {"final_state_graph_send_recv",
     (PyCFunction)(void (*)(void))eager_final_state_api_graph_send_recv,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for graph_send_recv in dygraph."}

    ,

    {"final_state_greater_equal",
     (PyCFunction)(void (*)(void))eager_final_state_api_greater_equal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for greater_equal in dygraph."}

    ,

    {"final_state_greater_than",
     (PyCFunction)(void (*)(void))eager_final_state_api_greater_than,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for greater_than in dygraph."}

    ,

    {"final_state_group_norm",
     (PyCFunction)(void (*)(void))eager_final_state_api_group_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for group_norm in dygraph."}

    ,

    {"final_state_gumbel_softmax",
     (PyCFunction)(void (*)(void))eager_final_state_api_gumbel_softmax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gumbel_softmax in dygraph."}

    ,

    {"final_state_hard_shrink",
     (PyCFunction)(void (*)(void))eager_final_state_api_hard_shrink,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for hard_shrink in dygraph."}

    ,

    {"final_state_hard_sigmoid",
     (PyCFunction)(void (*)(void))eager_final_state_api_hard_sigmoid,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for hard_sigmoid in dygraph."}

    ,

    {"final_state_hard_swish",
     (PyCFunction)(void (*)(void))eager_final_state_api_hard_swish,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for hard_swish in dygraph."}

    ,

    {"final_state_histogram",
     (PyCFunction)(void (*)(void))eager_final_state_api_histogram,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for histogram in dygraph."}

    ,

    {"final_state_huber_loss",
     (PyCFunction)(void (*)(void))eager_final_state_api_huber_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for huber_loss in dygraph."}

    ,

    {"final_state_imag",
     (PyCFunction)(void (*)(void))eager_final_state_api_imag,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for imag in dygraph."}

    ,

    {"final_state_increment",
     (PyCFunction)(void (*)(void))eager_final_state_api_increment,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for increment in dygraph."}

    ,

    {"final_state_index_sample",
     (PyCFunction)(void (*)(void))eager_final_state_api_index_sample,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for index_sample in dygraph."}

    ,

    {"final_state_index_select",
     (PyCFunction)(void (*)(void))eager_final_state_api_index_select,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for index_select in dygraph."}

    ,

    {"final_state_instance_norm",
     (PyCFunction)(void (*)(void))eager_final_state_api_instance_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for instance_norm in dygraph."}

    ,

    {"final_state_is_empty",
     (PyCFunction)(void (*)(void))eager_final_state_api_is_empty,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for is_empty in dygraph."}

    ,

    {"final_state_isclose",
     (PyCFunction)(void (*)(void))eager_final_state_api_isclose,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for isclose in dygraph."}

    ,

    {"final_state_isfinite",
     (PyCFunction)(void (*)(void))eager_final_state_api_isfinite,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for isfinite in dygraph."}

    ,

    {"final_state_isinf",
     (PyCFunction)(void (*)(void))eager_final_state_api_isinf,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for isinf in dygraph."}

    ,

    {"final_state_isnan",
     (PyCFunction)(void (*)(void))eager_final_state_api_isnan,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for isnan in dygraph."}

    ,

    {"final_state_kldiv_loss",
     (PyCFunction)(void (*)(void))eager_final_state_api_kldiv_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for kldiv_loss in dygraph."}

    ,

    {"final_state_kron",
     (PyCFunction)(void (*)(void))eager_final_state_api_kron,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for kron in dygraph."}

    ,

    {"final_state_kthvalue",
     (PyCFunction)(void (*)(void))eager_final_state_api_kthvalue,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for kthvalue in dygraph."}

    ,

    {"final_state_label_smooth",
     (PyCFunction)(void (*)(void))eager_final_state_api_label_smooth,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for label_smooth in dygraph."}

    ,

    {"final_state_layer_norm",
     (PyCFunction)(void (*)(void))eager_final_state_api_layer_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for layer_norm in dygraph."}

    ,

    {"final_state_leaky_relu",
     (PyCFunction)(void (*)(void))eager_final_state_api_leaky_relu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for leaky_relu in dygraph."}

    ,

    {"final_state_lerp",
     (PyCFunction)(void (*)(void))eager_final_state_api_lerp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lerp in dygraph."}

    ,

    {"final_state_less_equal",
     (PyCFunction)(void (*)(void))eager_final_state_api_less_equal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for less_equal in dygraph."}

    ,

    {"final_state_less_than",
     (PyCFunction)(void (*)(void))eager_final_state_api_less_than,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for less_than in dygraph."}

    ,

    {"final_state_lgamma",
     (PyCFunction)(void (*)(void))eager_final_state_api_lgamma,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lgamma in dygraph."}

    ,

    {"final_state_linspace",
     (PyCFunction)(void (*)(void))eager_final_state_api_linspace,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for linspace in dygraph."}

    ,

    {"final_state_log",
     (PyCFunction)(void (*)(void))eager_final_state_api_log,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for log in dygraph."}

    ,

    {"final_state_log10",
     (PyCFunction)(void (*)(void))eager_final_state_api_log10,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for log10 in dygraph."}

    ,

    {"final_state_log1p",
     (PyCFunction)(void (*)(void))eager_final_state_api_log1p,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for log1p in dygraph."}

    ,

    {"final_state_log2",
     (PyCFunction)(void (*)(void))eager_final_state_api_log2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for log2 in dygraph."}

    ,

    {"final_state_log_loss",
     (PyCFunction)(void (*)(void))eager_final_state_api_log_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for log_loss in dygraph."}

    ,

    {"final_state_log_softmax",
     (PyCFunction)(void (*)(void))eager_final_state_api_log_softmax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for log_softmax in dygraph."}

    ,

    {"final_state_logcumsumexp",
     (PyCFunction)(void (*)(void))eager_final_state_api_logcumsumexp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logcumsumexp in dygraph."}

    ,

    {"final_state_logical_and",
     (PyCFunction)(void (*)(void))eager_final_state_api_logical_and,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logical_and in dygraph."}

    ,

    {"final_state_logical_not",
     (PyCFunction)(void (*)(void))eager_final_state_api_logical_not,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logical_not in dygraph."}

    ,

    {"final_state_logical_or",
     (PyCFunction)(void (*)(void))eager_final_state_api_logical_or,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logical_or in dygraph."}

    ,

    {"final_state_logical_xor",
     (PyCFunction)(void (*)(void))eager_final_state_api_logical_xor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logical_xor in dygraph."}

    ,

    {"final_state_logit",
     (PyCFunction)(void (*)(void))eager_final_state_api_logit,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logit in dygraph."}

    ,

    {"final_state_logsigmoid",
     (PyCFunction)(void (*)(void))eager_final_state_api_logsigmoid,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logsigmoid in dygraph."}

    ,

    {"final_state_logsumexp",
     (PyCFunction)(void (*)(void))eager_final_state_api_logsumexp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logsumexp in dygraph."}

    ,

    {"final_state_masked_select",
     (PyCFunction)(void (*)(void))eager_final_state_api_masked_select,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for masked_select in dygraph."}

    ,

    {"final_state_matmul",
     (PyCFunction)(void (*)(void))eager_final_state_api_matmul,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for matmul in dygraph."}

    ,

    {"final_state_matrix_power",
     (PyCFunction)(void (*)(void))eager_final_state_api_matrix_power,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for matrix_power in dygraph."}

    ,

    {"final_state_matrix_rank",
     (PyCFunction)(void (*)(void))eager_final_state_api_matrix_rank,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for matrix_rank in dygraph."}

    ,

    {"final_state_matrix_rank_tol",
     (PyCFunction)(void (*)(void))eager_final_state_api_matrix_rank_tol,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for matrix_rank_tol in dygraph."}

    ,

    {"final_state_max",
     (PyCFunction)(void (*)(void))eager_final_state_api_max,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for max in dygraph."}

    ,

    {"final_state_max_pool2d_with_index",
     (PyCFunction)(void (*)(void))eager_final_state_api_max_pool2d_with_index,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for max_pool2d_with_index in dygraph."}

    ,

    {"final_state_max_pool3d_with_index",
     (PyCFunction)(void (*)(void))eager_final_state_api_max_pool3d_with_index,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for max_pool3d_with_index in dygraph."}

    ,

    {"final_state_maximum",
     (PyCFunction)(void (*)(void))eager_final_state_api_maximum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for maximum in dygraph."}

    ,

    {"final_state_maxout",
     (PyCFunction)(void (*)(void))eager_final_state_api_maxout,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for maxout in dygraph."}

    ,

    {"final_state_mean",
     (PyCFunction)(void (*)(void))eager_final_state_api_mean,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for mean in dygraph."}

    ,

    {"final_state_mean_all",
     (PyCFunction)(void (*)(void))eager_final_state_api_mean_all,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for mean_all in dygraph."}

    ,

    {"final_state_meshgrid",
     (PyCFunction)(void (*)(void))eager_final_state_api_meshgrid,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for meshgrid in dygraph."}

    ,

    {"final_state_min",
     (PyCFunction)(void (*)(void))eager_final_state_api_min,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for min in dygraph."}

    ,

    {"final_state_minimum",
     (PyCFunction)(void (*)(void))eager_final_state_api_minimum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for minimum in dygraph."}

    ,

    {"final_state_mish",
     (PyCFunction)(void (*)(void))eager_final_state_api_mish,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for mish in dygraph."}

    ,

    {"final_state_mode",
     (PyCFunction)(void (*)(void))eager_final_state_api_mode,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for mode in dygraph."}

    ,

    {"final_state_modulo",
     (PyCFunction)(void (*)(void))eager_final_state_api_modulo,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for modulo in dygraph."}

    ,

    {"final_state_momentum",
     (PyCFunction)(void (*)(void))eager_final_state_api_momentum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for momentum in dygraph."}

    ,

    {"final_state_multi_dot",
     (PyCFunction)(void (*)(void))eager_final_state_api_multi_dot,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for multi_dot in dygraph."}

    ,

    {"final_state_multinomial",
     (PyCFunction)(void (*)(void))eager_final_state_api_multinomial,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for multinomial in dygraph."}

    ,

    {"final_state_multiplex",
     (PyCFunction)(void (*)(void))eager_final_state_api_multiplex,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for multiplex in dygraph."}

    ,

    {"final_state_multiply",
     (PyCFunction)(void (*)(void))eager_final_state_api_multiply,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for multiply in dygraph."}

    ,

    {"final_state_nll_loss",
     (PyCFunction)(void (*)(void))eager_final_state_api_nll_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for nll_loss in dygraph."}

    ,

    {"final_state_norm",
     (PyCFunction)(void (*)(void))eager_final_state_api_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for norm in dygraph."}

    ,

    {"final_state_not_equal",
     (PyCFunction)(void (*)(void))eager_final_state_api_not_equal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for not_equal in dygraph."}

    ,

    {"final_state_one_hot",
     (PyCFunction)(void (*)(void))eager_final_state_api_one_hot,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for one_hot in dygraph."}

    ,

    {"final_state_ones_like",
     (PyCFunction)(void (*)(void))eager_final_state_api_ones_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for ones_like in dygraph."}

    ,

    {"final_state_p_norm",
     (PyCFunction)(void (*)(void))eager_final_state_api_p_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for p_norm in dygraph."}

    ,

    {"final_state_pad",
     (PyCFunction)(void (*)(void))eager_final_state_api_pad,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pad in dygraph."}

    ,

    {"final_state_pad3d",
     (PyCFunction)(void (*)(void))eager_final_state_api_pad3d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pad3d in dygraph."}

    ,

    {"final_state_pixel_shuffle",
     (PyCFunction)(void (*)(void))eager_final_state_api_pixel_shuffle,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pixel_shuffle in dygraph."}

    ,

    {"final_state_pool2d",
     (PyCFunction)(void (*)(void))eager_final_state_api_pool2d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pool2d in dygraph."}

    ,

    {"final_state_pool2d_gpudnn_unused",
     (PyCFunction)(void (*)(void))eager_final_state_api_pool2d_gpudnn_unused,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pool2d_gpudnn_unused in dygraph."}

    ,

    {"final_state_pool3d",
     (PyCFunction)(void (*)(void))eager_final_state_api_pool3d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pool3d in dygraph."}

    ,

    {"final_state_pow",
     (PyCFunction)(void (*)(void))eager_final_state_api_pow,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pow in dygraph."}

    ,

    {"final_state_prelu",
     (PyCFunction)(void (*)(void))eager_final_state_api_prelu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for prelu in dygraph."}

    ,

    {"final_state_psroi_pool",
     (PyCFunction)(void (*)(void))eager_final_state_api_psroi_pool,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for psroi_pool in dygraph."}

    ,

    {"final_state_put_along_axis",
     (PyCFunction)(void (*)(void))eager_final_state_api_put_along_axis,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for put_along_axis in dygraph."}

    ,

    {"final_state_qr",
     (PyCFunction)(void (*)(void))eager_final_state_api_qr,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for qr in dygraph."}

    ,

    {"final_state_randint",
     (PyCFunction)(void (*)(void))eager_final_state_api_randint,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for randint in dygraph."}

    ,

    {"final_state_randperm",
     (PyCFunction)(void (*)(void))eager_final_state_api_randperm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for randperm in dygraph."}

    ,

    {"final_state_real",
     (PyCFunction)(void (*)(void))eager_final_state_api_real,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for real in dygraph."}

    ,

    {"final_state_reciprocal",
     (PyCFunction)(void (*)(void))eager_final_state_api_reciprocal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reciprocal in dygraph."}

    ,

    {"final_state_reduce_prod",
     (PyCFunction)(void (*)(void))eager_final_state_api_reduce_prod,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reduce_prod in dygraph."}

    ,

    {"final_state_relu",
     (PyCFunction)(void (*)(void))eager_final_state_api_relu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for relu in dygraph."}

    ,
    {"final_state_relu_",
     (PyCFunction)(void (*)(void))eager_final_state_api_relu_,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for relu_ in dygraph."}

    ,

    {"final_state_relu6",
     (PyCFunction)(void (*)(void))eager_final_state_api_relu6,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for relu6 in dygraph."}

    ,

    {"final_state_reshape",
     (PyCFunction)(void (*)(void))eager_final_state_api_reshape,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reshape in dygraph."}

    ,
    {"final_state_reshape_",
     (PyCFunction)(void (*)(void))eager_final_state_api_reshape_,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reshape_ in dygraph."}

    ,

    {"final_state_roi_align",
     (PyCFunction)(void (*)(void))eager_final_state_api_roi_align,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for roi_align in dygraph."}

    ,

    {"final_state_roi_pool",
     (PyCFunction)(void (*)(void))eager_final_state_api_roi_pool,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for roi_pool in dygraph."}

    ,

    {"final_state_roll",
     (PyCFunction)(void (*)(void))eager_final_state_api_roll,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for roll in dygraph."}

    ,

    {"final_state_round",
     (PyCFunction)(void (*)(void))eager_final_state_api_round,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for round in dygraph."}

    ,

    {"final_state_rsqrt",
     (PyCFunction)(void (*)(void))eager_final_state_api_rsqrt,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for rsqrt in dygraph."}

    ,
    {"final_state_rsqrt_",
     (PyCFunction)(void (*)(void))eager_final_state_api_rsqrt_,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for rsqrt_ in dygraph."}

    ,

    {"final_state_scale",
     (PyCFunction)(void (*)(void))eager_final_state_api_scale,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for scale in dygraph."}

    ,
    {"final_state_scale_",
     (PyCFunction)(void (*)(void))eager_final_state_api_scale_,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for scale_ in dygraph."}

    ,

    {"final_state_scatter",
     (PyCFunction)(void (*)(void))eager_final_state_api_scatter,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for scatter in dygraph."}

    ,

    {"final_state_scatter_nd_add",
     (PyCFunction)(void (*)(void))eager_final_state_api_scatter_nd_add,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for scatter_nd_add in dygraph."}

    ,

    {"final_state_searchsorted",
     (PyCFunction)(void (*)(void))eager_final_state_api_searchsorted,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for searchsorted in dygraph."}

    ,

    {"final_state_segment_pool",
     (PyCFunction)(void (*)(void))eager_final_state_api_segment_pool,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for segment_pool in dygraph."}

    ,

    {"final_state_selu",
     (PyCFunction)(void (*)(void))eager_final_state_api_selu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for selu in dygraph."}

    ,

    {"final_state_sgd_",
     (PyCFunction)(void (*)(void))eager_final_state_api_sgd_,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sgd_ in dygraph."}

    ,

    {"final_state_shape",
     (PyCFunction)(void (*)(void))eager_final_state_api_shape,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for shape in dygraph."}

    ,

    {"final_state_shard_index",
     (PyCFunction)(void (*)(void))eager_final_state_api_shard_index,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for shard_index in dygraph."}

    ,

    {"final_state_sigmoid",
     (PyCFunction)(void (*)(void))eager_final_state_api_sigmoid,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sigmoid in dygraph."}

    ,

    {"final_state_sigmoid_cross_entropy_with_logits",
     (PyCFunction)(void (*)(
         void))eager_final_state_api_sigmoid_cross_entropy_with_logits,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sigmoid_cross_entropy_with_logits in dygraph."}

    ,

    {"final_state_sign",
     (PyCFunction)(void (*)(void))eager_final_state_api_sign,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sign in dygraph."}

    ,

    {"final_state_silu",
     (PyCFunction)(void (*)(void))eager_final_state_api_silu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for silu in dygraph."}

    ,

    {"final_state_sin",
     (PyCFunction)(void (*)(void))eager_final_state_api_sin,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sin in dygraph."}

    ,

    {"final_state_sinh",
     (PyCFunction)(void (*)(void))eager_final_state_api_sinh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sinh in dygraph."}

    ,

    {"final_state_size",
     (PyCFunction)(void (*)(void))eager_final_state_api_size,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for size in dygraph."}

    ,

    {"final_state_slice",
     (PyCFunction)(void (*)(void))eager_final_state_api_slice,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for slice in dygraph."}

    ,

    {"final_state_soft_shrink",
     (PyCFunction)(void (*)(void))eager_final_state_api_soft_shrink,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for soft_shrink in dygraph."}

    ,

    {"final_state_softmax",
     (PyCFunction)(void (*)(void))eager_final_state_api_softmax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for softmax in dygraph."}

    ,

    {"final_state_split",
     (PyCFunction)(void (*)(void))eager_final_state_api_split,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for split in dygraph."}

    ,

    {"final_state_sqrt",
     (PyCFunction)(void (*)(void))eager_final_state_api_sqrt,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sqrt in dygraph."}

    ,

    {"final_state_square",
     (PyCFunction)(void (*)(void))eager_final_state_api_square,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for square in dygraph."}

    ,

    {"final_state_squeeze",
     (PyCFunction)(void (*)(void))eager_final_state_api_squeeze,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for squeeze in dygraph."}

    ,

    {"final_state_stack",
     (PyCFunction)(void (*)(void))eager_final_state_api_stack,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for stack in dygraph."}

    ,

    {"final_state_strided_slice",
     (PyCFunction)(void (*)(void))eager_final_state_api_strided_slice,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for strided_slice in dygraph."}

    ,

    {"final_state_subtract",
     (PyCFunction)(void (*)(void))eager_final_state_api_subtract,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for subtract in dygraph."}

    ,

    {"final_state_sum",
     (PyCFunction)(void (*)(void))eager_final_state_api_sum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sum in dygraph."}

    ,

    {"final_state_swish",
     (PyCFunction)(void (*)(void))eager_final_state_api_swish,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for swish in dygraph."}

    ,

    {"final_state_take_along_axis",
     (PyCFunction)(void (*)(void))eager_final_state_api_take_along_axis,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for take_along_axis in dygraph."}

    ,

    {"final_state_tan",
     (PyCFunction)(void (*)(void))eager_final_state_api_tan,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tan in dygraph."}

    ,

    {"final_state_tanh",
     (PyCFunction)(void (*)(void))eager_final_state_api_tanh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tanh in dygraph."}

    ,

    {"final_state_tanh_shrink",
     (PyCFunction)(void (*)(void))eager_final_state_api_tanh_shrink,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tanh_shrink in dygraph."}

    ,

    {"final_state_thresholded_relu",
     (PyCFunction)(void (*)(void))eager_final_state_api_thresholded_relu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for thresholded_relu in dygraph."}

    ,

    {"final_state_tile",
     (PyCFunction)(void (*)(void))eager_final_state_api_tile,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tile in dygraph."}

    ,

    {"final_state_top_k",
     (PyCFunction)(void (*)(void))eager_final_state_api_top_k,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for top_k in dygraph."}

    ,

    {"final_state_transpose",
     (PyCFunction)(void (*)(void))eager_final_state_api_transpose,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for transpose in dygraph."}

    ,

    {"final_state_triangular_solve",
     (PyCFunction)(void (*)(void))eager_final_state_api_triangular_solve,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for triangular_solve in dygraph."}

    ,

    {"final_state_tril_indices",
     (PyCFunction)(void (*)(void))eager_final_state_api_tril_indices,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tril_indices in dygraph."}

    ,

    {"final_state_tril_triu",
     (PyCFunction)(void (*)(void))eager_final_state_api_tril_triu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tril_triu in dygraph."}

    ,

    {"final_state_truncated_gaussian_random",
     (PyCFunction)(void (*)(
         void))eager_final_state_api_truncated_gaussian_random,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for truncated_gaussian_random in dygraph."}

    ,

    {"final_state_unbind",
     (PyCFunction)(void (*)(void))eager_final_state_api_unbind,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unbind in dygraph."}

    ,

    {"final_state_unfold",
     (PyCFunction)(void (*)(void))eager_final_state_api_unfold,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unfold in dygraph."}

    ,

    {"final_state_uniform_random",
     (PyCFunction)(void (*)(void))eager_final_state_api_uniform_random,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for uniform_random in dygraph."}

    ,

    {"final_state_unique",
     (PyCFunction)(void (*)(void))eager_final_state_api_unique,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unique in dygraph."}

    ,

    {"final_state_unsqueeze",
     (PyCFunction)(void (*)(void))eager_final_state_api_unsqueeze,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unsqueeze in dygraph."}

    ,

    {"final_state_viterbi_decode",
     (PyCFunction)(void (*)(void))eager_final_state_api_viterbi_decode,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for viterbi_decode in dygraph."}

    ,

    {"final_state_where",
     (PyCFunction)(void (*)(void))eager_final_state_api_where,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for where in dygraph."}

    ,

    {"final_state_where_index",
     (PyCFunction)(void (*)(void))eager_final_state_api_where_index,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for where_index in dygraph."}

    ,

    {"final_state_yolo_box",
     (PyCFunction)(void (*)(void))eager_final_state_api_yolo_box,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for yolo_box in dygraph."}

    ,

    {"final_state_zeros_like",
     (PyCFunction)(void (*)(void))eager_final_state_api_zeros_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for zeros_like in dygraph."}

    ,

    {"final_state_sparse_add",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_add,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for add in dygraph."}

    ,

    {"final_state_sparse_conv3d",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_conv3d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conv3d in dygraph."}

    ,

    {"final_state_sparse_coo_to_dense",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_coo_to_dense,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for coo_to_dense in dygraph."}

    ,

    {"final_state_sparse_create_sparse_coo_tensor",
     (PyCFunction)(void (*)(
         void))sparse::eager_final_state_api_create_sparse_coo_tensor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for create_sparse_coo_tensor in dygraph."}

    ,

    {"final_state_sparse_dense_to_coo",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_dense_to_coo,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dense_to_coo in dygraph."}

    ,

    {"final_state_sparse_divide",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_divide,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for divide in dygraph."}

    ,

    {"final_state_sparse_multiply",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_multiply,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for multiply in dygraph."}

    ,

    {"final_state_sparse_relu",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_relu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for relu in dygraph."}

    ,

    {"final_state_sparse_sin",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_sin,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sin in dygraph."}

    ,

    {"final_state_sparse_softmax",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_softmax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for softmax in dygraph."}

    ,

    {"final_state_sparse_sqrt",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_sqrt,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sqrt in dygraph."}

    ,

    {"final_state_sparse_subtract",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_subtract,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for subtract in dygraph."}

    ,

    {"final_state_sparse_tanh",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_tanh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tanh in dygraph."}

    ,

    {"final_state_sparse_to_dense",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_to_dense,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for to_dense in dygraph."}

    ,

    {"final_state_sparse_to_sparse_coo",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_to_sparse_coo,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for to_sparse_coo in dygraph."}

    ,

    {"final_state_sparse_to_sparse_csr",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_to_sparse_csr,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for to_sparse_csr in dygraph."}

    ,

    {"final_state_sparse_values",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_values,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for values in dygraph."}

    ,

    {"final_state_sparse_full_like",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_full_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for full_like in dygraph."}

    ,

    {"final_state_sparse_fused_attention",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_fused_attention,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_attention in dygraph."}

    ,

    {"final_state_sparse_masked_matmul",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_masked_matmul,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for masked_matmul in dygraph."}

    ,

    {"final_state_sparse_matmul",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_matmul,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for matmul in dygraph."}

    ,

    {"final_state_sparse_maxpool",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_maxpool,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for maxpool in dygraph."}

    ,

    {"final_state_sparse_mv",
     (PyCFunction)(void (*)(void))sparse::eager_final_state_api_mv,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for mv in dygraph."}

    ,

    {"final_state_strings_empty",
     (PyCFunction)(void (*)(void))strings::eager_final_state_api_empty,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for empty in dygraph."}

    ,

    {"final_state_strings_empty_like",
     (PyCFunction)(void (*)(void))strings::eager_final_state_api_empty_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for empty_like in dygraph."}

    ,

    {"final_state_strings_lower",
     (PyCFunction)(void (*)(void))strings::eager_final_state_api_lower,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lower in dygraph."}

    ,

    {"final_state_strings_upper",
     (PyCFunction)(void (*)(void))strings::eager_final_state_api_upper,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for upper in dygraph."}

    ,

    {"get_final_state_core_ops_args_info",
     (PyCFunction)(void (*)(void))eager_get_final_state_core_ops_args_info,
     METH_NOARGS,
     "C++ interface function for eager_get_final_state_core_ops_args_info."},
    {"get_final_state_core_ops_args_type_info",
     (PyCFunction)(void (*)(void))eager_get_final_state_core_ops_args_type_info,
     METH_NOARGS,
     "C++ interface function for "
     "eager_get_final_state_core_ops_args_type_info."},
    {"get_final_state_core_ops_returns_info",
     (PyCFunction)(void (*)(void))eager_get_final_state_core_ops_returns_info,
     METH_NOARGS,
     "C++ interface function for eager_get_final_state_core_ops_returns_info."},

    {nullptr, nullptr, 0, nullptr}};

}  // namespace pybind
}  // namespace paddle
