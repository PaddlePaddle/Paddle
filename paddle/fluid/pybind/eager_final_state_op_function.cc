
#include <Python.h>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/api/include/strings_api.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/pybind/eager_final_state_custom_python_api.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/eager/amp_utils.h"
#include "paddle/fluid/eager/eager_amp_auto_cast.h"

namespace paddle {
namespace pybind {


static PyObject * eager_final_state_api_atan2(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("atan2 pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::atan2_final_state_dygraph_function(x,y)) out = ::atan2_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_bernoulli(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("bernoulli pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::bernoulli_final_state_dygraph_function(x)) out = ::bernoulli_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_cholesky(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("cholesky pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cholesky";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cholesky", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* upper_obj = PyTuple_GET_ITEM(args, 1);
    bool upper = CastPyArg2Boolean(upper_obj, "cholesky", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cholesky_final_state_dygraph_function(x,upper)) out = ::cholesky_final_state_dygraph_function(x,upper);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_cholesky_solve(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("cholesky_solve pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cholesky_solve";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cholesky_solve", "x", args, 0, false);
    auto y = GetTensorFromArgs("cholesky_solve", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* upper_obj = PyTuple_GET_ITEM(args, 2);
    bool upper = CastPyArg2Boolean(upper_obj, "cholesky_solve", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cholesky_solve_final_state_dygraph_function(x,y,upper)) out = ::cholesky_solve_final_state_dygraph_function(x,y,upper);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_cross(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("cross pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cross";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cross", "x", args, 0, false);
    auto y = GetTensorFromArgs("cross", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "cross", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cross_final_state_dygraph_function(x,y,axis)) out = ::cross_final_state_dygraph_function(x,y,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_diag(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("diag pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: diag";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("diag", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* offset_obj = PyTuple_GET_ITEM(args, 1);
    int offset = CastPyArg2Int(offset_obj, "diag", 1);
    PyObject* padding_value_obj = PyTuple_GET_ITEM(args, 2);
    float padding_value = CastPyArg2Float(padding_value_obj, "diag", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::diag_final_state_dygraph_function(x,offset,padding_value)) out = ::diag_final_state_dygraph_function(x,offset,padding_value);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_diagonal(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("diagonal pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: diagonal";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("diagonal", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* offset_obj = PyTuple_GET_ITEM(args, 1);
    int offset = CastPyArg2Int(offset_obj, "diagonal", 1);
    PyObject* axis1_obj = PyTuple_GET_ITEM(args, 2);
    int axis1 = CastPyArg2Int(axis1_obj, "diagonal", 2);
    PyObject* axis2_obj = PyTuple_GET_ITEM(args, 3);
    int axis2 = CastPyArg2Int(axis2_obj, "diagonal", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::diagonal_final_state_dygraph_function(x,offset,axis1,axis2)) out = ::diagonal_final_state_dygraph_function(x,offset,axis1,axis2);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_digamma(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("digamma pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::digamma_final_state_dygraph_function(x)) out = ::digamma_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_dist(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("dist pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: dist";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("dist", "x", args, 0, false);
    auto y = GetTensorFromArgs("dist", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* p_obj = PyTuple_GET_ITEM(args, 2);
    float p = CastPyArg2Float(p_obj, "dist", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::dist_final_state_dygraph_function(x,y,p)) out = ::dist_final_state_dygraph_function(x,y,p);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_dot(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("dot pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::dot_final_state_dygraph_function(x,y)) out = ::dot_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_erf(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("erf pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::erf_final_state_dygraph_function(x)) out = ::erf_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_erfinv(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("erfinv pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::erfinv_final_state_dygraph_function(x)) out = ::erfinv_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_erfinv_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("erfinv pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::erfinv__final_state_dygraph_function(x)) out = ::erfinv__final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_fft_c2c(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fft_c2c pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fft_c2c";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fft_c2c", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axes = CastPyArg2Longs(axes_obj, "fft_c2c", 1);
    PyObject* normalization_obj = PyTuple_GET_ITEM(args, 2);
    std::string normalization = CastPyArg2String(normalization_obj, "fft_c2c", 2);
    PyObject* forward_obj = PyTuple_GET_ITEM(args, 3);
    bool forward = CastPyArg2Boolean(forward_obj, "fft_c2c", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fft_c2c_final_state_dygraph_function(x,axes,normalization,forward)) out = ::fft_c2c_final_state_dygraph_function(x,axes,normalization,forward);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_fft_c2r(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fft_c2r pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fft_c2r";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fft_c2r", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axes = CastPyArg2Longs(axes_obj, "fft_c2r", 1);
    PyObject* normalization_obj = PyTuple_GET_ITEM(args, 2);
    std::string normalization = CastPyArg2String(normalization_obj, "fft_c2r", 2);
    PyObject* forward_obj = PyTuple_GET_ITEM(args, 3);
    bool forward = CastPyArg2Boolean(forward_obj, "fft_c2r", 3);
    PyObject* last_dim_size_obj = PyTuple_GET_ITEM(args, 4);
    int64_t last_dim_size = CastPyArg2Long(last_dim_size_obj, "fft_c2r", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fft_c2r_final_state_dygraph_function(x,axes,normalization,forward,last_dim_size)) out = ::fft_c2r_final_state_dygraph_function(x,axes,normalization,forward,last_dim_size);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_fft_r2c(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fft_r2c pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fft_r2c";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fft_r2c", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axes = CastPyArg2Longs(axes_obj, "fft_r2c", 1);
    PyObject* normalization_obj = PyTuple_GET_ITEM(args, 2);
    std::string normalization = CastPyArg2String(normalization_obj, "fft_r2c", 2);
    PyObject* forward_obj = PyTuple_GET_ITEM(args, 3);
    bool forward = CastPyArg2Boolean(forward_obj, "fft_r2c", 3);
    PyObject* onesided_obj = PyTuple_GET_ITEM(args, 4);
    bool onesided = CastPyArg2Boolean(onesided_obj, "fft_r2c", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fft_r2c_final_state_dygraph_function(x,axes,normalization,forward,onesided)) out = ::fft_r2c_final_state_dygraph_function(x,axes,normalization,forward,onesided);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_graph_send_uv(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("graph_send_uv pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: graph_send_uv";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("graph_send_uv", "x", args, 0, false);
    auto y = GetTensorFromArgs("graph_send_uv", "y", args, 1, false);
    auto src_index = GetTensorFromArgs("graph_send_uv", "src_index", args, 2, false);
    auto dst_index = GetTensorFromArgs("graph_send_uv", "dst_index", args, 3, false);

    // Parse Attributes if needed
    PyObject* message_op_obj = PyTuple_GET_ITEM(args, 4);
    std::string message_op = CastPyArg2String(message_op_obj, "graph_send_uv", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::graph_send_uv_final_state_dygraph_function(x,y,src_index,dst_index,message_op)) out = ::graph_send_uv_final_state_dygraph_function(x,y,src_index,dst_index,message_op);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_lgamma(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("lgamma pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::lgamma_final_state_dygraph_function(x)) out = ::lgamma_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_mv(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("mv pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::mv_final_state_dygraph_function(x,vec)) out = ::mv_final_state_dygraph_function(x,vec);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_poisson(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("poisson pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::poisson_final_state_dygraph_function(x)) out = ::poisson_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_solve(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("solve pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: solve";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("solve", "x", args, 0, false);
    auto y = GetTensorFromArgs("solve", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::solve_final_state_dygraph_function(x,y)) out = ::solve_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_trace(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("trace pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: trace";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("trace", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* offset_obj = PyTuple_GET_ITEM(args, 1);
    int offset = CastPyArg2Int(offset_obj, "trace", 1);
    PyObject* axis1_obj = PyTuple_GET_ITEM(args, 2);
    int axis1 = CastPyArg2Int(axis1_obj, "trace", 2);
    PyObject* axis2_obj = PyTuple_GET_ITEM(args, 3);
    int axis2 = CastPyArg2Int(axis2_obj, "trace", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::trace_final_state_dygraph_function(x,offset,axis1,axis2)) out = ::trace_final_state_dygraph_function(x,offset,axis1,axis2);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_trunc(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("trunc pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::trunc_final_state_dygraph_function(x)) out = ::trunc_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}



static PyObject * eager_final_state_api_abs(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("abs pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::abs_final_state_dygraph_function(x)) out = ::abs_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_accuracy(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("accuracy pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::accuracy_final_state_dygraph_function(x,indices,label)) out = ::accuracy_final_state_dygraph_function(x,indices,label);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_acos(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("acos pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::acos_final_state_dygraph_function(x)) out = ::acos_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_acosh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("acosh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::acosh_final_state_dygraph_function(x)) out = ::acosh_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_adadelta_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("adadelta_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: adadelta_";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("adadelta_", "param", args, 0, false);
    auto grad = GetTensorFromArgs("adadelta_", "grad", args, 1, false);
    auto avg_squared_grad = GetTensorFromArgs("adadelta_", "avg_squared_grad", args, 2, false);
    auto avg_squared_update = GetTensorFromArgs("adadelta_", "avg_squared_update", args, 3, false);

    // Parse Attributes if needed
    PyObject* rho_obj = PyTuple_GET_ITEM(args, 4);
    float rho = CastPyArg2Float(rho_obj, "adadelta_", 4);
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 5);
    float epsilon = CastPyArg2Float(epsilon_obj, "adadelta_", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::adadelta__final_state_dygraph_function(param,grad,avg_squared_grad,avg_squared_update,rho,epsilon)) out = ::adadelta__final_state_dygraph_function(param,grad,avg_squared_grad,avg_squared_update,rho,epsilon);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;

    inplace_var_idx_map[1] = 2;

    inplace_var_idx_map[2] = 3;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_adagrad_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("adagrad_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: adagrad_";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("adagrad_", "param", args, 0, false);
    auto grad = GetTensorFromArgs("adagrad_", "grad", args, 1, false);
    auto moment = GetTensorFromArgs("adagrad_", "moment", args, 2, false);
    auto learning_rate = GetTensorFromArgs("adagrad_", "learning_rate", args, 3, false);

    // Parse Attributes if needed
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 4);
    float epsilon = CastPyArg2Float(epsilon_obj, "adagrad_", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::adagrad__final_state_dygraph_function(param,grad,moment,learning_rate,epsilon)) out = ::adagrad__final_state_dygraph_function(param,grad,moment,learning_rate,epsilon);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;

    inplace_var_idx_map[1] = 2;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_adam_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("adam_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: adam_";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("adam_", "param", args, 0, false);
    auto grad = GetTensorFromArgs("adam_", "grad", args, 1, false);
    auto learning_rate = GetTensorFromArgs("adam_", "learning_rate", args, 2, false);
    auto moment1 = GetTensorFromArgs("adam_", "moment1", args, 3, false);
    auto moment2 = GetTensorFromArgs("adam_", "moment2", args, 4, false);
    auto beta1_pow = GetTensorFromArgs("adam_", "beta1_pow", args, 5, false);
    auto beta2_pow = GetTensorFromArgs("adam_", "beta2_pow", args, 6, false);
    auto master_param = GetOptionalTensorFromArgs("adam_", "master_param", args, 7, true);
    auto skip_update = GetOptionalTensorFromArgs("adam_", "skip_update", args, 8, true);

    // Parse Attributes if needed
    PyObject* beta1_obj = PyTuple_GET_ITEM(args, 9);
    paddle::experimental::Scalar beta1 = CastPyArg2Scalar(beta1_obj, "adam_", 9);
    PyObject* beta2_obj = PyTuple_GET_ITEM(args, 10);
    paddle::experimental::Scalar beta2 = CastPyArg2Scalar(beta2_obj, "adam_", 10);
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 11);
    paddle::experimental::Scalar epsilon = CastPyArg2Scalar(epsilon_obj, "adam_", 11);
    PyObject* lazy_mode_obj = PyTuple_GET_ITEM(args, 12);
    bool lazy_mode = CastPyArg2Boolean(lazy_mode_obj, "adam_", 12);
    PyObject* min_row_size_to_use_multithread_obj = PyTuple_GET_ITEM(args, 13);
    int64_t min_row_size_to_use_multithread = CastPyArg2Long(min_row_size_to_use_multithread_obj, "adam_", 13);
    PyObject* multi_precision_obj = PyTuple_GET_ITEM(args, 14);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "adam_", 14);
    PyObject* use_global_beta_pow_obj = PyTuple_GET_ITEM(args, 15);
    bool use_global_beta_pow = CastPyArg2Boolean(use_global_beta_pow_obj, "adam_", 15);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::adam__final_state_dygraph_function(param,grad,learning_rate,moment1,moment2,beta1_pow,beta2_pow,master_param,skip_update,beta1,beta2,epsilon,lazy_mode,min_row_size_to_use_multithread,multi_precision,use_global_beta_pow)) out = ::adam__final_state_dygraph_function(param,grad,learning_rate,moment1,moment2,beta1_pow,beta2_pow,master_param,skip_update,beta1,beta2,epsilon,lazy_mode,min_row_size_to_use_multithread,multi_precision,use_global_beta_pow);


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
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_adamax_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("adamax_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: adamax_";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("adamax_", "param", args, 0, false);
    auto grad = GetTensorFromArgs("adamax_", "grad", args, 1, false);
    auto learning_rate = GetTensorFromArgs("adamax_", "learning_rate", args, 2, false);
    auto moment = GetTensorFromArgs("adamax_", "moment", args, 3, false);
    auto inf_norm = GetTensorFromArgs("adamax_", "inf_norm", args, 4, false);
    auto beta1_pow = GetTensorFromArgs("adamax_", "beta1_pow", args, 5, false);

    // Parse Attributes if needed
    PyObject* beta1_obj = PyTuple_GET_ITEM(args, 6);
    float beta1 = CastPyArg2Float(beta1_obj, "adamax_", 6);
    PyObject* beta2_obj = PyTuple_GET_ITEM(args, 7);
    float beta2 = CastPyArg2Float(beta2_obj, "adamax_", 7);
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 8);
    float epsilon = CastPyArg2Float(epsilon_obj, "adamax_", 8);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::adamax__final_state_dygraph_function(param,grad,learning_rate,moment,inf_norm,beta1_pow,beta1,beta2,epsilon)) out = ::adamax__final_state_dygraph_function(param,grad,learning_rate,moment,inf_norm,beta1_pow,beta1,beta2,epsilon);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;

    inplace_var_idx_map[1] = 3;

    inplace_var_idx_map[2] = 4;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_adamw_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("adamw_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: adamw_";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("adamw_", "param", args, 0, false);
    auto grad = GetTensorFromArgs("adamw_", "grad", args, 1, false);
    auto learning_rate = GetTensorFromArgs("adamw_", "learning_rate", args, 2, false);
    auto moment1 = GetTensorFromArgs("adamw_", "moment1", args, 3, false);
    auto moment2 = GetTensorFromArgs("adamw_", "moment2", args, 4, false);
    auto beta1_pow = GetTensorFromArgs("adamw_", "beta1_pow", args, 5, false);
    auto beta2_pow = GetTensorFromArgs("adamw_", "beta2_pow", args, 6, false);
    auto master_param = GetOptionalTensorFromArgs("adamw_", "master_param", args, 7, true);
    auto skip_update = GetOptionalTensorFromArgs("adamw_", "skip_update", args, 8, true);

    // Parse Attributes if needed
    PyObject* beta1_obj = PyTuple_GET_ITEM(args, 9);
    paddle::experimental::Scalar beta1 = CastPyArg2Scalar(beta1_obj, "adamw_", 9);
    PyObject* beta2_obj = PyTuple_GET_ITEM(args, 10);
    paddle::experimental::Scalar beta2 = CastPyArg2Scalar(beta2_obj, "adamw_", 10);
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 11);
    paddle::experimental::Scalar epsilon = CastPyArg2Scalar(epsilon_obj, "adamw_", 11);
    PyObject* lr_ratio_obj = PyTuple_GET_ITEM(args, 12);
    float lr_ratio = CastPyArg2Float(lr_ratio_obj, "adamw_", 12);
    PyObject* coeff_obj = PyTuple_GET_ITEM(args, 13);
    float coeff = CastPyArg2Float(coeff_obj, "adamw_", 13);
    PyObject* with_decay_obj = PyTuple_GET_ITEM(args, 14);
    bool with_decay = CastPyArg2Boolean(with_decay_obj, "adamw_", 14);
    PyObject* lazy_mode_obj = PyTuple_GET_ITEM(args, 15);
    bool lazy_mode = CastPyArg2Boolean(lazy_mode_obj, "adamw_", 15);
    PyObject* min_row_size_to_use_multithread_obj = PyTuple_GET_ITEM(args, 16);
    int64_t min_row_size_to_use_multithread = CastPyArg2Long(min_row_size_to_use_multithread_obj, "adamw_", 16);
    PyObject* multi_precision_obj = PyTuple_GET_ITEM(args, 17);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "adamw_", 17);
    PyObject* use_global_beta_pow_obj = PyTuple_GET_ITEM(args, 18);
    bool use_global_beta_pow = CastPyArg2Boolean(use_global_beta_pow_obj, "adamw_", 18);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::adamw__final_state_dygraph_function(param,grad,learning_rate,moment1,moment2,beta1_pow,beta2_pow,master_param,skip_update,beta1,beta2,epsilon,lr_ratio,coeff,with_decay,lazy_mode,min_row_size_to_use_multithread,multi_precision,use_global_beta_pow)) out = ::adamw__final_state_dygraph_function(param,grad,learning_rate,moment1,moment2,beta1_pow,beta2_pow,master_param,skip_update,beta1,beta2,epsilon,lr_ratio,coeff,with_decay,lazy_mode,min_row_size_to_use_multithread,multi_precision,use_global_beta_pow);


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
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("add pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::add_final_state_dygraph_function(x,y)) out = ::add_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_add_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("add pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: add_";

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::add__final_state_dygraph_function(x,y)) out = ::add__final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_add_n(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("add_n pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::add_n_final_state_dygraph_function(x)) out = ::add_n_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_addmm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("addmm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: addmm";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("addmm", "input", args, 0, false);
    auto x = GetTensorFromArgs("addmm", "x", args, 1, false);
    auto y = GetTensorFromArgs("addmm", "y", args, 2, false);

    // Parse Attributes if needed
    PyObject* alpha_obj = PyTuple_GET_ITEM(args, 3);
    float alpha = CastPyArg2Float(alpha_obj, "addmm", 3);
    PyObject* beta_obj = PyTuple_GET_ITEM(args, 4);
    float beta = CastPyArg2Float(beta_obj, "addmm", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::addmm_final_state_dygraph_function(input,x,y,alpha,beta)) out = ::addmm_final_state_dygraph_function(input,x,y,alpha,beta);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_affine_grid(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("affine_grid pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: affine_grid";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("affine_grid", "input", args, 0, false);

    // Parse Attributes if needed
    PyObject* outputShape_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray outputShape = CastPyArg2IntArray(outputShape_obj, "affine_grid", 1);
    PyObject* use_cudnn_obj = PyTuple_GET_ITEM(args, 2);
    bool use_cudnn = CastPyArg2Boolean(use_cudnn_obj, "affine_grid", 2);
    PyObject* align_corners_obj = PyTuple_GET_ITEM(args, 3);
    bool align_corners = CastPyArg2Boolean(align_corners_obj, "affine_grid", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::affine_grid_final_state_dygraph_function(input,outputShape,use_cudnn,align_corners)) out = ::affine_grid_final_state_dygraph_function(input,outputShape,use_cudnn,align_corners);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_all(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("all pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: all";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("all", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "all", 1);
    PyObject* keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "all", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::all_final_state_dygraph_function(x,dims,keep_dim)) out = ::all_final_state_dygraph_function(x,dims,keep_dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_allclose(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("allclose pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: allclose";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("allclose", "x", args, 0, false);
    auto y = GetTensorFromArgs("allclose", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* rtol_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar rtol = CastPyArg2Scalar(rtol_obj, "allclose", 2);
    PyObject* atol_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::Scalar atol = CastPyArg2Scalar(atol_obj, "allclose", 3);
    PyObject* equal_nan_obj = PyTuple_GET_ITEM(args, 4);
    bool equal_nan = CastPyArg2Boolean(equal_nan_obj, "allclose", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::allclose_final_state_dygraph_function(x,y,rtol,atol,equal_nan)) out = ::allclose_final_state_dygraph_function(x,y,rtol,atol,equal_nan);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_amax(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("amax pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: amax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("amax", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "amax", 1);
    PyObject* keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "amax", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::amax_final_state_dygraph_function(x,dims,keep_dim)) out = ::amax_final_state_dygraph_function(x,dims,keep_dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_amin(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("amin pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: amin";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("amin", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "amin", 1);
    PyObject* keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "amin", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::amin_final_state_dygraph_function(x,dims,keep_dim)) out = ::amin_final_state_dygraph_function(x,dims,keep_dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_angle(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("angle pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: angle";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("angle", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::angle_final_state_dygraph_function(x)) out = ::angle_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_any(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("any pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: any";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("any", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "any", 1);
    PyObject* keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "any", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::any_final_state_dygraph_function(x,dims,keep_dim)) out = ::any_final_state_dygraph_function(x,dims,keep_dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_arange(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("arange pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: arange";

    // Get EagerTensors from args
    auto start = GetTensorFromArgs("arange", "start", args, 0, false);
    auto end = GetTensorFromArgs("arange", "end", args, 1, false);
    auto step = GetTensorFromArgs("arange", "step", args, 2, false);

    // Parse Attributes if needed
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "arange", 3);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 4);
    paddle::Place place = CastPyArg2Place(place_obj, "arange", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::arange_final_state_dygraph_function(start,end,step,dtype,place)) out = ::arange_final_state_dygraph_function(start,end,step,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_argmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("argmax pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: argmax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("argmax", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int64_t axis = CastPyArg2Long(axis_obj, "argmax", 1);
    PyObject* keepdims_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdims = CastPyArg2Boolean(keepdims_obj, "argmax", 2);
    PyObject* flatten_obj = PyTuple_GET_ITEM(args, 3);
    bool flatten = CastPyArg2Boolean(flatten_obj, "argmax", 3);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 4);
    int dtype = CastPyArg2Int(dtype_obj, "argmax", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::argmax_final_state_dygraph_function(x,axis,keepdims,flatten,dtype)) out = ::argmax_final_state_dygraph_function(x,axis,keepdims,flatten,dtype);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_argmin(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("argmin pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: argmin";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("argmin", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int64_t axis = CastPyArg2Long(axis_obj, "argmin", 1);
    PyObject* keepdims_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdims = CastPyArg2Boolean(keepdims_obj, "argmin", 2);
    PyObject* flatten_obj = PyTuple_GET_ITEM(args, 3);
    bool flatten = CastPyArg2Boolean(flatten_obj, "argmin", 3);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 4);
    int dtype = CastPyArg2Int(dtype_obj, "argmin", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::argmin_final_state_dygraph_function(x,axis,keepdims,flatten,dtype)) out = ::argmin_final_state_dygraph_function(x,axis,keepdims,flatten,dtype);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_argsort(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("argsort pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: argsort";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("argsort", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "argsort", 1);
    PyObject* descending_obj = PyTuple_GET_ITEM(args, 2);
    bool descending = CastPyArg2Boolean(descending_obj, "argsort", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::argsort_final_state_dygraph_function(x,axis,descending)) out = ::argsort_final_state_dygraph_function(x,axis,descending);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_as_complex(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("as_complex pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: as_complex";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("as_complex", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::as_complex_final_state_dygraph_function(x)) out = ::as_complex_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_as_real(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("as_real pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: as_real";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("as_real", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::as_real_final_state_dygraph_function(x)) out = ::as_real_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_asin(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("asin pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::asin_final_state_dygraph_function(x)) out = ::asin_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_asinh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("asinh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::asinh_final_state_dygraph_function(x)) out = ::asinh_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_assign(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("assign pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::assign_final_state_dygraph_function(x)) out = ::assign_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_assign_out_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("assign_out_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::assign_out__final_state_dygraph_function(x,output)) out = ::assign_out__final_state_dygraph_function(x,output);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 1;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_assign_value_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("assign_value_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: assign_value_";

    // Get EagerTensors from args
    auto output = GetTensorFromArgs("assign_value_", "output", args, 0, false);

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> shape = CastPyArg2Ints(shape_obj, "assign_value_", 1);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "assign_value_", 2);
    PyObject* values_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<phi::Scalar> values = CastPyArg2ScalarArray(values_obj, "assign_value_", 3);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 4);
    paddle::Place place = CastPyArg2Place(place_obj, "assign_value_", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::assign_value__final_state_dygraph_function(output,shape,dtype,values,place)) out = ::assign_value__final_state_dygraph_function(output,shape,dtype,values,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_atan(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("atan pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::atan_final_state_dygraph_function(x)) out = ::atan_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_atanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("atanh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::atanh_final_state_dygraph_function(x)) out = ::atanh_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_auc(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("auc pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: auc";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("auc", "x", args, 0, false);
    auto label = GetTensorFromArgs("auc", "label", args, 1, false);
    auto stat_pos = GetTensorFromArgs("auc", "stat_pos", args, 2, false);
    auto stat_neg = GetTensorFromArgs("auc", "stat_neg", args, 3, false);
    auto ins_tag_weight = GetOptionalTensorFromArgs("auc", "ins_tag_weight", args, 4, true);

    // Parse Attributes if needed
    PyObject* curve_obj = PyTuple_GET_ITEM(args, 5);
    std::string curve = CastPyArg2String(curve_obj, "auc", 5);
    PyObject* num_thresholds_obj = PyTuple_GET_ITEM(args, 6);
    int num_thresholds = CastPyArg2Int(num_thresholds_obj, "auc", 6);
    PyObject* slide_steps_obj = PyTuple_GET_ITEM(args, 7);
    int slide_steps = CastPyArg2Int(slide_steps_obj, "auc", 7);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::auc_final_state_dygraph_function(x,label,stat_pos,stat_neg,ins_tag_weight,curve,num_thresholds,slide_steps)) out = ::auc_final_state_dygraph_function(x,label,stat_pos,stat_neg,ins_tag_weight,curve,num_thresholds,slide_steps);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_average_accumulates_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("average_accumulates_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: average_accumulates_";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("average_accumulates_", "param", args, 0, false);
    auto in_sum_1 = GetTensorFromArgs("average_accumulates_", "in_sum_1", args, 1, false);
    auto in_sum_2 = GetTensorFromArgs("average_accumulates_", "in_sum_2", args, 2, false);
    auto in_sum_3 = GetTensorFromArgs("average_accumulates_", "in_sum_3", args, 3, false);
    auto in_num_accumulates = GetTensorFromArgs("average_accumulates_", "in_num_accumulates", args, 4, false);
    auto in_old_num_accumulates = GetTensorFromArgs("average_accumulates_", "in_old_num_accumulates", args, 5, false);
    auto in_num_updates = GetTensorFromArgs("average_accumulates_", "in_num_updates", args, 6, false);

    // Parse Attributes if needed
    PyObject* average_window_obj = PyTuple_GET_ITEM(args, 7);
    float average_window = CastPyArg2Float(average_window_obj, "average_accumulates_", 7);
    PyObject* max_average_window_obj = PyTuple_GET_ITEM(args, 8);
    int64_t max_average_window = CastPyArg2Long(max_average_window_obj, "average_accumulates_", 8);
    PyObject* min_average_window_obj = PyTuple_GET_ITEM(args, 9);
    int64_t min_average_window = CastPyArg2Long(min_average_window_obj, "average_accumulates_", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::average_accumulates__final_state_dygraph_function(param,in_sum_1,in_sum_2,in_sum_3,in_num_accumulates,in_old_num_accumulates,in_num_updates,average_window,max_average_window,min_average_window)) out = ::average_accumulates__final_state_dygraph_function(param,in_sum_1,in_sum_2,in_sum_3,in_num_accumulates,in_old_num_accumulates,in_num_updates,average_window,max_average_window,min_average_window);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 1;

    inplace_var_idx_map[1] = 2;

    inplace_var_idx_map[2] = 3;

    inplace_var_idx_map[3] = 4;

    inplace_var_idx_map[4] = 5;

    inplace_var_idx_map[5] = 6;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_batch_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("batch_norm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
    PyObject* momentum_obj = PyTuple_GET_ITEM(args, 5);
    float momentum = CastPyArg2Float(momentum_obj, "batch_norm", 5);
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 6);
    float epsilon = CastPyArg2Float(epsilon_obj, "batch_norm", 6);
    PyObject* data_layout_obj = PyTuple_GET_ITEM(args, 7);
    std::string data_layout = CastPyArg2String(data_layout_obj, "batch_norm", 7);
    PyObject* is_test_obj = PyTuple_GET_ITEM(args, 8);
    bool is_test = CastPyArg2Boolean(is_test_obj, "batch_norm", 8);
    PyObject* use_global_stats_obj = PyTuple_GET_ITEM(args, 9);
    bool use_global_stats = CastPyArg2Boolean(use_global_stats_obj, "batch_norm", 9);
    PyObject* trainable_statistics_obj = PyTuple_GET_ITEM(args, 10);
    bool trainable_statistics = CastPyArg2Boolean(trainable_statistics_obj, "batch_norm", 10);
    PyObject* fuse_with_relu_obj = PyTuple_GET_ITEM(args, 11);
    bool fuse_with_relu = CastPyArg2Boolean(fuse_with_relu_obj, "batch_norm", 11);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::batch_norm_final_state_dygraph_function(x,scale,bias,mean,variance,momentum,epsilon,data_layout,is_test,use_global_stats,trainable_statistics,fuse_with_relu)) out = ::batch_norm_final_state_dygraph_function(x,scale,bias,mean,variance,momentum,epsilon,data_layout,is_test,use_global_stats,trainable_statistics,fuse_with_relu);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_bce_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("bce_loss pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::bce_loss_final_state_dygraph_function(input,label)) out = ::bce_loss_final_state_dygraph_function(input,label);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_bicubic_interp(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("bicubic_interp pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: bicubic_interp";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("bicubic_interp", "x", args, 0, false);
    auto out_size = GetOptionalTensorFromArgs("bicubic_interp", "out_size", args, 1, true);
    auto size_tensor = GetOptionalTensorListFromArgs("bicubic_interp", "size_tensor", args, 2, true);
    auto scale_tensor = GetOptionalTensorFromArgs("bicubic_interp", "scale_tensor", args, 3, true);

    // Parse Attributes if needed
    PyObject* data_layout_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_layout = CastPyArg2String(data_layout_obj, "bicubic_interp", 4);
    PyObject* out_d_obj = PyTuple_GET_ITEM(args, 5);
    int out_d = CastPyArg2Int(out_d_obj, "bicubic_interp", 5);
    PyObject* out_h_obj = PyTuple_GET_ITEM(args, 6);
    int out_h = CastPyArg2Int(out_h_obj, "bicubic_interp", 6);
    PyObject* out_w_obj = PyTuple_GET_ITEM(args, 7);
    int out_w = CastPyArg2Int(out_w_obj, "bicubic_interp", 7);
    PyObject* scale_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<float> scale = CastPyArg2Floats(scale_obj, "bicubic_interp", 8);
    PyObject* interp_method_obj = PyTuple_GET_ITEM(args, 9);
    std::string interp_method = CastPyArg2String(interp_method_obj, "bicubic_interp", 9);
    PyObject* align_corners_obj = PyTuple_GET_ITEM(args, 10);
    bool align_corners = CastPyArg2Boolean(align_corners_obj, "bicubic_interp", 10);
    PyObject* align_mode_obj = PyTuple_GET_ITEM(args, 11);
    int align_mode = CastPyArg2Int(align_mode_obj, "bicubic_interp", 11);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::bicubic_interp_final_state_dygraph_function(x,out_size,size_tensor,scale_tensor,data_layout,out_d,out_h,out_w,scale,interp_method,align_corners,align_mode)) out = ::bicubic_interp_final_state_dygraph_function(x,out_size,size_tensor,scale_tensor,data_layout,out_d,out_h,out_w,scale,interp_method,align_corners,align_mode);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_bilinear_interp(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("bilinear_interp pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: bilinear_interp";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("bilinear_interp", "x", args, 0, false);
    auto out_size = GetOptionalTensorFromArgs("bilinear_interp", "out_size", args, 1, true);
    auto size_tensor = GetOptionalTensorListFromArgs("bilinear_interp", "size_tensor", args, 2, true);
    auto scale_tensor = GetOptionalTensorFromArgs("bilinear_interp", "scale_tensor", args, 3, true);

    // Parse Attributes if needed
    PyObject* data_layout_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_layout = CastPyArg2String(data_layout_obj, "bilinear_interp", 4);
    PyObject* out_d_obj = PyTuple_GET_ITEM(args, 5);
    int out_d = CastPyArg2Int(out_d_obj, "bilinear_interp", 5);
    PyObject* out_h_obj = PyTuple_GET_ITEM(args, 6);
    int out_h = CastPyArg2Int(out_h_obj, "bilinear_interp", 6);
    PyObject* out_w_obj = PyTuple_GET_ITEM(args, 7);
    int out_w = CastPyArg2Int(out_w_obj, "bilinear_interp", 7);
    PyObject* scale_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<float> scale = CastPyArg2Floats(scale_obj, "bilinear_interp", 8);
    PyObject* interp_method_obj = PyTuple_GET_ITEM(args, 9);
    std::string interp_method = CastPyArg2String(interp_method_obj, "bilinear_interp", 9);
    PyObject* align_corners_obj = PyTuple_GET_ITEM(args, 10);
    bool align_corners = CastPyArg2Boolean(align_corners_obj, "bilinear_interp", 10);
    PyObject* align_mode_obj = PyTuple_GET_ITEM(args, 11);
    int align_mode = CastPyArg2Int(align_mode_obj, "bilinear_interp", 11);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::bilinear_interp_final_state_dygraph_function(x,out_size,size_tensor,scale_tensor,data_layout,out_d,out_h,out_w,scale,interp_method,align_corners,align_mode)) out = ::bilinear_interp_final_state_dygraph_function(x,out_size,size_tensor,scale_tensor,data_layout,out_d,out_h,out_w,scale,interp_method,align_corners,align_mode);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_bilinear_tensor_product(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("bilinear_tensor_product pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: bilinear_tensor_product";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("bilinear_tensor_product", "x", args, 0, false);
    auto y = GetTensorFromArgs("bilinear_tensor_product", "y", args, 1, false);
    auto weight = GetTensorFromArgs("bilinear_tensor_product", "weight", args, 2, false);
    auto bias = GetOptionalTensorFromArgs("bilinear_tensor_product", "bias", args, 3, true);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::bilinear_tensor_product_final_state_dygraph_function(x,y,weight,bias)) out = ::bilinear_tensor_product_final_state_dygraph_function(x,y,weight,bias);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_bitwise_and(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("bitwise_and pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::bitwise_and_final_state_dygraph_function(x,y)) out = ::bitwise_and_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_bitwise_not(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("bitwise_not pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::bitwise_not_final_state_dygraph_function(x)) out = ::bitwise_not_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_bitwise_or(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("bitwise_or pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::bitwise_or_final_state_dygraph_function(x,y)) out = ::bitwise_or_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_bitwise_xor(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("bitwise_xor pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::bitwise_xor_final_state_dygraph_function(x,y)) out = ::bitwise_xor_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_bmm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("bmm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: bmm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("bmm", "x", args, 0, false);
    auto y = GetTensorFromArgs("bmm", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::bmm_final_state_dygraph_function(x,y)) out = ::bmm_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_box_coder(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("box_coder pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: box_coder";

    // Get EagerTensors from args
    auto prior_box = GetTensorFromArgs("box_coder", "prior_box", args, 0, false);
    auto prior_box_var = GetOptionalTensorFromArgs("box_coder", "prior_box_var", args, 1, true);
    auto target_box = GetTensorFromArgs("box_coder", "target_box", args, 2, false);

    // Parse Attributes if needed
    PyObject* code_type_obj = PyTuple_GET_ITEM(args, 3);
    std::string code_type = CastPyArg2String(code_type_obj, "box_coder", 3);
    PyObject* box_normalized_obj = PyTuple_GET_ITEM(args, 4);
    bool box_normalized = CastPyArg2Boolean(box_normalized_obj, "box_coder", 4);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 5);
    int axis = CastPyArg2Int(axis_obj, "box_coder", 5);
    PyObject* variance_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<float> variance = CastPyArg2Floats(variance_obj, "box_coder", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::box_coder_final_state_dygraph_function(prior_box,prior_box_var,target_box,code_type,box_normalized,axis,variance)) out = ::box_coder_final_state_dygraph_function(prior_box,prior_box_var,target_box,code_type,box_normalized,axis,variance);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_brelu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("brelu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: brelu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("brelu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* t_min_obj = PyTuple_GET_ITEM(args, 1);
    float t_min = CastPyArg2Float(t_min_obj, "brelu", 1);
    PyObject* t_max_obj = PyTuple_GET_ITEM(args, 2);
    float t_max = CastPyArg2Float(t_max_obj, "brelu", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::brelu_final_state_dygraph_function(x,t_min,t_max)) out = ::brelu_final_state_dygraph_function(x,t_min,t_max);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_cast(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("cast pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cast";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cast", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* out_dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType out_dtype = CastPyArg2DataType(out_dtype_obj, "cast", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cast_final_state_dygraph_function(x,out_dtype)) out = ::cast_final_state_dygraph_function(x,out_dtype);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_ceil(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("ceil pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::ceil_final_state_dygraph_function(x)) out = ::ceil_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_ceil_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("ceil pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: ceil_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("ceil", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::ceil__final_state_dygraph_function(x)) out = ::ceil__final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_celu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("celu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: celu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("celu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* alpha_obj = PyTuple_GET_ITEM(args, 1);
    float alpha = CastPyArg2Float(alpha_obj, "celu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::celu_final_state_dygraph_function(x,alpha)) out = ::celu_final_state_dygraph_function(x,alpha);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_class_center_sample(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("class_center_sample pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: class_center_sample";

    // Get EagerTensors from args
    auto label = GetTensorFromArgs("class_center_sample", "label", args, 0, false);

    // Parse Attributes if needed
    PyObject* num_classes_obj = PyTuple_GET_ITEM(args, 1);
    int num_classes = CastPyArg2Int(num_classes_obj, "class_center_sample", 1);
    PyObject* num_samples_obj = PyTuple_GET_ITEM(args, 2);
    int num_samples = CastPyArg2Int(num_samples_obj, "class_center_sample", 2);
    PyObject* ring_id_obj = PyTuple_GET_ITEM(args, 3);
    int ring_id = CastPyArg2Int(ring_id_obj, "class_center_sample", 3);
    PyObject* rank_obj = PyTuple_GET_ITEM(args, 4);
    int rank = CastPyArg2Int(rank_obj, "class_center_sample", 4);
    PyObject* nranks_obj = PyTuple_GET_ITEM(args, 5);
    int nranks = CastPyArg2Int(nranks_obj, "class_center_sample", 5);
    PyObject* fix_seed_obj = PyTuple_GET_ITEM(args, 6);
    bool fix_seed = CastPyArg2Boolean(fix_seed_obj, "class_center_sample", 6);
    PyObject* seed_obj = PyTuple_GET_ITEM(args, 7);
    int seed = CastPyArg2Int(seed_obj, "class_center_sample", 7);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::class_center_sample_final_state_dygraph_function(label,num_classes,num_samples,ring_id,rank,nranks,fix_seed,seed)) out = ::class_center_sample_final_state_dygraph_function(label,num_classes,num_samples,ring_id,rank,nranks,fix_seed,seed);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_clip(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("clip pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: clip";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("clip", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* min_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar min = CastPyArg2Scalar(min_obj, "clip", 1);
    PyObject* max_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar max = CastPyArg2Scalar(max_obj, "clip", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::clip_final_state_dygraph_function(x,min,max)) out = ::clip_final_state_dygraph_function(x,min,max);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_clip_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("clip pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: clip_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("clip", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* min_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar min = CastPyArg2Scalar(min_obj, "clip", 1);
    PyObject* max_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar max = CastPyArg2Scalar(max_obj, "clip", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::clip__final_state_dygraph_function(x,min,max)) out = ::clip__final_state_dygraph_function(x,min,max);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_clip_by_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("clip_by_norm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: clip_by_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("clip_by_norm", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* max_norm_obj = PyTuple_GET_ITEM(args, 1);
    float max_norm = CastPyArg2Float(max_norm_obj, "clip_by_norm", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::clip_by_norm_final_state_dygraph_function(x,max_norm)) out = ::clip_by_norm_final_state_dygraph_function(x,max_norm);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_complex(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("complex pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: complex";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("complex", "x", args, 0, false);
    auto y = GetTensorFromArgs("complex", "y", args, 1, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::complex_final_state_dygraph_function(x,y)) out = ::complex_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_concat(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("concat pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: concat";

    // Get EagerTensors from args
    auto x = GetTensorListFromArgs("concat", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar axis = CastPyArg2Scalar(axis_obj, "concat", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::concat_final_state_dygraph_function(x,axis)) out = ::concat_final_state_dygraph_function(x,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_conj(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("conj pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::conj_final_state_dygraph_function(x)) out = ::conj_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_conv2d(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("conv2d pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: conv2d";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("conv2d", "input", args, 0, false);
    auto filter = GetTensorFromArgs("conv2d", "filter", args, 1, false);

    // Parse Attributes if needed
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "conv2d", 2);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "conv2d", 3);
    PyObject* padding_algorithm_obj = PyTuple_GET_ITEM(args, 4);
    std::string padding_algorithm = CastPyArg2String(padding_algorithm_obj, "conv2d", 4);
    PyObject* groups_obj = PyTuple_GET_ITEM(args, 5);
    int groups = CastPyArg2Int(groups_obj, "conv2d", 5);
    PyObject* dilations_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "conv2d", 6);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 7);
    std::string data_format = CastPyArg2String(data_format_obj, "conv2d", 7);
    PyObject* use_addto_obj = PyTuple_GET_ITEM(args, 8);
    bool use_addto = CastPyArg2Boolean(use_addto_obj, "conv2d", 8);
    PyObject* workspace_size_MB_obj = PyTuple_GET_ITEM(args, 9);
    int workspace_size_MB = CastPyArg2Int(workspace_size_MB_obj, "conv2d", 9);
    PyObject* exhaustive_search_obj = PyTuple_GET_ITEM(args, 10);
    bool exhaustive_search = CastPyArg2Boolean(exhaustive_search_obj, "conv2d", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::conv2d_final_state_dygraph_function(input,filter,strides,paddings,padding_algorithm,groups,dilations,data_format,use_addto,workspace_size_MB,exhaustive_search)) out = ::conv2d_final_state_dygraph_function(input,filter,strides,paddings,padding_algorithm,groups,dilations,data_format,use_addto,workspace_size_MB,exhaustive_search);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_conv2d_transpose(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("conv2d_transpose pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: conv2d_transpose";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("conv2d_transpose", "x", args, 0, false);
    auto filter = GetTensorFromArgs("conv2d_transpose", "filter", args, 1, false);

    // Parse Attributes if needed
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "conv2d_transpose", 2);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "conv2d_transpose", 3);
    PyObject* output_padding_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> output_padding = CastPyArg2Ints(output_padding_obj, "conv2d_transpose", 4);
    PyObject* output_size_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> output_size = CastPyArg2Ints(output_size_obj, "conv2d_transpose", 5);
    PyObject* padding_algorithm_obj = PyTuple_GET_ITEM(args, 6);
    std::string padding_algorithm = CastPyArg2String(padding_algorithm_obj, "conv2d_transpose", 6);
    PyObject* groups_obj = PyTuple_GET_ITEM(args, 7);
    int groups = CastPyArg2Int(groups_obj, "conv2d_transpose", 7);
    PyObject* dilations_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "conv2d_transpose", 8);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 9);
    std::string data_format = CastPyArg2String(data_format_obj, "conv2d_transpose", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::conv2d_transpose_final_state_dygraph_function(x,filter,strides,paddings,output_padding,output_size,padding_algorithm,groups,dilations,data_format)) out = ::conv2d_transpose_final_state_dygraph_function(x,filter,strides,paddings,output_padding,output_size,padding_algorithm,groups,dilations,data_format);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_conv3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("conv3d pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: conv3d";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("conv3d", "input", args, 0, false);
    auto filter = GetTensorFromArgs("conv3d", "filter", args, 1, false);

    // Parse Attributes if needed
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "conv3d", 2);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "conv3d", 3);
    PyObject* paddding_algorithm_obj = PyTuple_GET_ITEM(args, 4);
    std::string paddding_algorithm = CastPyArg2String(paddding_algorithm_obj, "conv3d", 4);
    PyObject* groups_obj = PyTuple_GET_ITEM(args, 5);
    int groups = CastPyArg2Int(groups_obj, "conv3d", 5);
    PyObject* dilations_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "conv3d", 6);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 7);
    std::string data_format = CastPyArg2String(data_format_obj, "conv3d", 7);
    PyObject* use_addto_obj = PyTuple_GET_ITEM(args, 8);
    bool use_addto = CastPyArg2Boolean(use_addto_obj, "conv3d", 8);
    PyObject* workspace_size_MB_obj = PyTuple_GET_ITEM(args, 9);
    int workspace_size_MB = CastPyArg2Int(workspace_size_MB_obj, "conv3d", 9);
    PyObject* exhaustive_search_obj = PyTuple_GET_ITEM(args, 10);
    bool exhaustive_search = CastPyArg2Boolean(exhaustive_search_obj, "conv3d", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::conv3d_final_state_dygraph_function(input,filter,strides,paddings,paddding_algorithm,groups,dilations,data_format,use_addto,workspace_size_MB,exhaustive_search)) out = ::conv3d_final_state_dygraph_function(input,filter,strides,paddings,paddding_algorithm,groups,dilations,data_format,use_addto,workspace_size_MB,exhaustive_search);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_conv3d_transpose(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("conv3d_transpose pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: conv3d_transpose";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("conv3d_transpose", "x", args, 0, false);
    auto filter = GetTensorFromArgs("conv3d_transpose", "filter", args, 1, false);

    // Parse Attributes if needed
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "conv3d_transpose", 2);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "conv3d_transpose", 3);
    PyObject* output_padding_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> output_padding = CastPyArg2Ints(output_padding_obj, "conv3d_transpose", 4);
    PyObject* output_size_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> output_size = CastPyArg2Ints(output_size_obj, "conv3d_transpose", 5);
    PyObject* padding_algorithm_obj = PyTuple_GET_ITEM(args, 6);
    std::string padding_algorithm = CastPyArg2String(padding_algorithm_obj, "conv3d_transpose", 6);
    PyObject* groups_obj = PyTuple_GET_ITEM(args, 7);
    int groups = CastPyArg2Int(groups_obj, "conv3d_transpose", 7);
    PyObject* dilations_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "conv3d_transpose", 8);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 9);
    std::string data_format = CastPyArg2String(data_format_obj, "conv3d_transpose", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::conv3d_transpose_final_state_dygraph_function(x,filter,strides,paddings,output_padding,output_size,padding_algorithm,groups,dilations,data_format)) out = ::conv3d_transpose_final_state_dygraph_function(x,filter,strides,paddings,output_padding,output_size,padding_algorithm,groups,dilations,data_format);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_copy_to(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("copy_to pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: copy_to";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("copy_to", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* place_obj = PyTuple_GET_ITEM(args, 1);
    paddle::Place place = CastPyArg2Place(place_obj, "copy_to", 1);
    PyObject* blocking_obj = PyTuple_GET_ITEM(args, 2);
    bool blocking = CastPyArg2Boolean(blocking_obj, "copy_to", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::copy_to_final_state_dygraph_function(x,place,blocking)) out = ::copy_to_final_state_dygraph_function(x,place,blocking);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_cos(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("cos pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cos_final_state_dygraph_function(x)) out = ::cos_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_cosh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("cosh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cosh_final_state_dygraph_function(x)) out = ::cosh_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_crop_tensor(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("crop_tensor pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: crop_tensor";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("crop_tensor", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "crop_tensor", 1);
    PyObject* offsets_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::IntArray offsets = CastPyArg2IntArray(offsets_obj, "crop_tensor", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::crop_tensor_final_state_dygraph_function(x,shape,offsets)) out = ::crop_tensor_final_state_dygraph_function(x,shape,offsets);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_cross_entropy_with_softmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("cross_entropy_with_softmax pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cross_entropy_with_softmax";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("cross_entropy_with_softmax", "input", args, 0, false);
    auto label = GetTensorFromArgs("cross_entropy_with_softmax", "label", args, 1, false);

    // Parse Attributes if needed
    PyObject* soft_label_obj = PyTuple_GET_ITEM(args, 2);
    bool soft_label = CastPyArg2Boolean(soft_label_obj, "cross_entropy_with_softmax", 2);
    PyObject* use_softmax_obj = PyTuple_GET_ITEM(args, 3);
    bool use_softmax = CastPyArg2Boolean(use_softmax_obj, "cross_entropy_with_softmax", 3);
    PyObject* numeric_stable_mode_obj = PyTuple_GET_ITEM(args, 4);
    bool numeric_stable_mode = CastPyArg2Boolean(numeric_stable_mode_obj, "cross_entropy_with_softmax", 4);
    PyObject* ignore_index_obj = PyTuple_GET_ITEM(args, 5);
    int ignore_index = CastPyArg2Int(ignore_index_obj, "cross_entropy_with_softmax", 5);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 6);
    int axis = CastPyArg2Int(axis_obj, "cross_entropy_with_softmax", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cross_entropy_with_softmax_final_state_dygraph_function(input,label,soft_label,use_softmax,numeric_stable_mode,ignore_index,axis)) out = ::cross_entropy_with_softmax_final_state_dygraph_function(input,label,soft_label,use_softmax,numeric_stable_mode,ignore_index,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_cumprod(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("cumprod pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cumprod";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cumprod", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dim_obj = PyTuple_GET_ITEM(args, 1);
    int dim = CastPyArg2Int(dim_obj, "cumprod", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cumprod_final_state_dygraph_function(x,dim)) out = ::cumprod_final_state_dygraph_function(x,dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_cumsum(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("cumsum pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cumsum";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cumsum", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "cumsum", 1);
    PyObject* flatten_obj = PyTuple_GET_ITEM(args, 2);
    bool flatten = CastPyArg2Boolean(flatten_obj, "cumsum", 2);
    PyObject* exclusive_obj = PyTuple_GET_ITEM(args, 3);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "cumsum", 3);
    PyObject* reverse_obj = PyTuple_GET_ITEM(args, 4);
    bool reverse = CastPyArg2Boolean(reverse_obj, "cumsum", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::cumsum_final_state_dygraph_function(x,axis,flatten,exclusive,reverse)) out = ::cumsum_final_state_dygraph_function(x,axis,flatten,exclusive,reverse);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_decode_jpeg(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("decode_jpeg pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: decode_jpeg";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("decode_jpeg", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* mode_obj = PyTuple_GET_ITEM(args, 1);
    std::string mode = CastPyArg2String(mode_obj, "decode_jpeg", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::decode_jpeg_final_state_dygraph_function(x,mode)) out = ::decode_jpeg_final_state_dygraph_function(x,mode);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_deformable_conv(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("deformable_conv pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: deformable_conv";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("deformable_conv", "x", args, 0, false);
    auto offset = GetTensorFromArgs("deformable_conv", "offset", args, 1, false);
    auto filter = GetTensorFromArgs("deformable_conv", "filter", args, 2, false);
    auto mask = GetOptionalTensorFromArgs("deformable_conv", "mask", args, 3, true);

    // Parse Attributes if needed
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "deformable_conv", 4);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "deformable_conv", 5);
    PyObject* dilations_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "deformable_conv", 6);
    PyObject* deformable_groups_obj = PyTuple_GET_ITEM(args, 7);
    int deformable_groups = CastPyArg2Int(deformable_groups_obj, "deformable_conv", 7);
    PyObject* groups_obj = PyTuple_GET_ITEM(args, 8);
    int groups = CastPyArg2Int(groups_obj, "deformable_conv", 8);
    PyObject* im2col_step_obj = PyTuple_GET_ITEM(args, 9);
    int im2col_step = CastPyArg2Int(im2col_step_obj, "deformable_conv", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::deformable_conv_final_state_dygraph_function(x,offset,filter,mask,strides,paddings,dilations,deformable_groups,groups,im2col_step)) out = ::deformable_conv_final_state_dygraph_function(x,offset,filter,mask,strides,paddings,dilations,deformable_groups,groups,im2col_step);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_depthwise_conv2d(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("depthwise_conv2d pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: depthwise_conv2d";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("depthwise_conv2d", "x", args, 0, false);
    auto filter = GetTensorFromArgs("depthwise_conv2d", "filter", args, 1, false);

    // Parse Attributes if needed
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "depthwise_conv2d", 2);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "depthwise_conv2d", 3);
    PyObject* padding_algorithm_obj = PyTuple_GET_ITEM(args, 4);
    std::string padding_algorithm = CastPyArg2String(padding_algorithm_obj, "depthwise_conv2d", 4);
    PyObject* groups_obj = PyTuple_GET_ITEM(args, 5);
    int groups = CastPyArg2Int(groups_obj, "depthwise_conv2d", 5);
    PyObject* dilations_obj = PyTuple_GET_ITEM(args, 6);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "depthwise_conv2d", 6);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 7);
    std::string data_format = CastPyArg2String(data_format_obj, "depthwise_conv2d", 7);
    PyObject* use_addto_obj = PyTuple_GET_ITEM(args, 8);
    bool use_addto = CastPyArg2Boolean(use_addto_obj, "depthwise_conv2d", 8);
    PyObject* workspace_size_MB_obj = PyTuple_GET_ITEM(args, 9);
    int workspace_size_MB = CastPyArg2Int(workspace_size_MB_obj, "depthwise_conv2d", 9);
    PyObject* exhaustive_search_obj = PyTuple_GET_ITEM(args, 10);
    bool exhaustive_search = CastPyArg2Boolean(exhaustive_search_obj, "depthwise_conv2d", 10);
    PyObject* fuse_relu_obj = PyTuple_GET_ITEM(args, 11);
    bool fuse_relu = CastPyArg2Boolean(fuse_relu_obj, "depthwise_conv2d", 11);
    PyObject* use_gpudnn_obj = PyTuple_GET_ITEM(args, 12);
    bool use_gpudnn = CastPyArg2Boolean(use_gpudnn_obj, "depthwise_conv2d", 12);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::depthwise_conv2d_final_state_dygraph_function(x,filter,strides,paddings,padding_algorithm,groups,dilations,data_format,use_addto,workspace_size_MB,exhaustive_search,fuse_relu,use_gpudnn)) out = ::depthwise_conv2d_final_state_dygraph_function(x,filter,strides,paddings,padding_algorithm,groups,dilations,data_format,use_addto,workspace_size_MB,exhaustive_search,fuse_relu,use_gpudnn);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_depthwise_conv2d_transpose(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("depthwise_conv2d_transpose pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: depthwise_conv2d_transpose";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("depthwise_conv2d_transpose", "x", args, 0, false);
    auto filter = GetTensorFromArgs("depthwise_conv2d_transpose", "filter", args, 1, false);

    // Parse Attributes if needed
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "depthwise_conv2d_transpose", 2);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "depthwise_conv2d_transpose", 3);
    PyObject* output_padding_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> output_padding = CastPyArg2Ints(output_padding_obj, "depthwise_conv2d_transpose", 4);
    PyObject* output_size_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> output_size = CastPyArg2Ints(output_size_obj, "depthwise_conv2d_transpose", 5);
    PyObject* padding_algorithm_obj = PyTuple_GET_ITEM(args, 6);
    std::string padding_algorithm = CastPyArg2String(padding_algorithm_obj, "depthwise_conv2d_transpose", 6);
    PyObject* groups_obj = PyTuple_GET_ITEM(args, 7);
    int groups = CastPyArg2Int(groups_obj, "depthwise_conv2d_transpose", 7);
    PyObject* dilations_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "depthwise_conv2d_transpose", 8);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 9);
    std::string data_format = CastPyArg2String(data_format_obj, "depthwise_conv2d_transpose", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::depthwise_conv2d_transpose_final_state_dygraph_function(x,filter,strides,paddings,output_padding,output_size,padding_algorithm,groups,dilations,data_format)) out = ::depthwise_conv2d_transpose_final_state_dygraph_function(x,filter,strides,paddings,output_padding,output_size,padding_algorithm,groups,dilations,data_format);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_det(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("det pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::det_final_state_dygraph_function(x)) out = ::det_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_diag_embed(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("diag_embed pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: diag_embed";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("diag_embed", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* offset_obj = PyTuple_GET_ITEM(args, 1);
    int offset = CastPyArg2Int(offset_obj, "diag_embed", 1);
    PyObject* dim1_obj = PyTuple_GET_ITEM(args, 2);
    int dim1 = CastPyArg2Int(dim1_obj, "diag_embed", 2);
    PyObject* dim2_obj = PyTuple_GET_ITEM(args, 3);
    int dim2 = CastPyArg2Int(dim2_obj, "diag_embed", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::diag_embed_final_state_dygraph_function(x,offset,dim1,dim2)) out = ::diag_embed_final_state_dygraph_function(x,offset,dim1,dim2);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_distribute_fpn_proposals(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("distribute_fpn_proposals pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: distribute_fpn_proposals";

    // Get EagerTensors from args
    auto fpn_rois = GetTensorFromArgs("distribute_fpn_proposals", "fpn_rois", args, 0, false);
    auto rois_num = GetOptionalTensorFromArgs("distribute_fpn_proposals", "rois_num", args, 1, true);

    // Parse Attributes if needed
    PyObject* min_level_obj = PyTuple_GET_ITEM(args, 2);
    int min_level = CastPyArg2Int(min_level_obj, "distribute_fpn_proposals", 2);
    PyObject* max_level_obj = PyTuple_GET_ITEM(args, 3);
    int max_level = CastPyArg2Int(max_level_obj, "distribute_fpn_proposals", 3);
    PyObject* refer_level_obj = PyTuple_GET_ITEM(args, 4);
    int refer_level = CastPyArg2Int(refer_level_obj, "distribute_fpn_proposals", 4);
    PyObject* refer_scale_obj = PyTuple_GET_ITEM(args, 5);
    int refer_scale = CastPyArg2Int(refer_scale_obj, "distribute_fpn_proposals", 5);
    PyObject* pixel_offset_obj = PyTuple_GET_ITEM(args, 6);
    bool pixel_offset = CastPyArg2Boolean(pixel_offset_obj, "distribute_fpn_proposals", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::distribute_fpn_proposals_final_state_dygraph_function(fpn_rois,rois_num,min_level,max_level,refer_level,refer_scale,pixel_offset)) out = ::distribute_fpn_proposals_final_state_dygraph_function(fpn_rois,rois_num,min_level,max_level,refer_level,refer_scale,pixel_offset);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_divide(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("divide pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::divide_final_state_dygraph_function(x,y)) out = ::divide_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_dropout(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("dropout pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: dropout";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("dropout", "x", args, 0, false);
    auto seed_tensor = GetOptionalTensorFromArgs("dropout", "seed_tensor", args, 1, true);

    // Parse Attributes if needed
    PyObject* p_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar p = CastPyArg2Scalar(p_obj, "dropout", 2);
    PyObject* is_test_obj = PyTuple_GET_ITEM(args, 3);
    bool is_test = CastPyArg2Boolean(is_test_obj, "dropout", 3);
    PyObject* mode_obj = PyTuple_GET_ITEM(args, 4);
    std::string mode = CastPyArg2String(mode_obj, "dropout", 4);
    PyObject* seed_obj = PyTuple_GET_ITEM(args, 5);
    int seed = CastPyArg2Int(seed_obj, "dropout", 5);
    PyObject* fix_seed_obj = PyTuple_GET_ITEM(args, 6);
    bool fix_seed = CastPyArg2Boolean(fix_seed_obj, "dropout", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::dropout_final_state_dygraph_function(x,seed_tensor,p,is_test,mode,seed,fix_seed)) out = ::dropout_final_state_dygraph_function(x,seed_tensor,p,is_test,mode,seed,fix_seed);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_edit_distance(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("edit_distance pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: edit_distance";

    // Get EagerTensors from args
    auto hyps = GetTensorFromArgs("edit_distance", "hyps", args, 0, false);
    auto refs = GetTensorFromArgs("edit_distance", "refs", args, 1, false);
    auto hypslength = GetOptionalTensorFromArgs("edit_distance", "hypslength", args, 2, true);
    auto refslength = GetOptionalTensorFromArgs("edit_distance", "refslength", args, 3, true);

    // Parse Attributes if needed
    PyObject* normalized_obj = PyTuple_GET_ITEM(args, 4);
    bool normalized = CastPyArg2Boolean(normalized_obj, "edit_distance", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::edit_distance_final_state_dygraph_function(hyps,refs,hypslength,refslength,normalized)) out = ::edit_distance_final_state_dygraph_function(hyps,refs,hypslength,refslength,normalized);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_eigh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("eigh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: eigh";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("eigh", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* uplo_obj = PyTuple_GET_ITEM(args, 1);
    std::string uplo = CastPyArg2String(uplo_obj, "eigh", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::eigh_final_state_dygraph_function(x,uplo)) out = ::eigh_final_state_dygraph_function(x,uplo);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_eigvals(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("eigvals pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: eigvals";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("eigvals", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::eigvals_final_state_dygraph_function(x)) out = ::eigvals_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_eigvalsh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("eigvalsh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: eigvalsh";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("eigvalsh", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* uplo_obj = PyTuple_GET_ITEM(args, 1);
    std::string uplo = CastPyArg2String(uplo_obj, "eigvalsh", 1);
    PyObject* is_test_obj = PyTuple_GET_ITEM(args, 2);
    bool is_test = CastPyArg2Boolean(is_test_obj, "eigvalsh", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::eigvalsh_final_state_dygraph_function(x,uplo,is_test)) out = ::eigvalsh_final_state_dygraph_function(x,uplo,is_test);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_einsum(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("einsum pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: einsum";

    // Get EagerTensors from args
    auto x = GetTensorListFromArgs("einsum", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* equation_obj = PyTuple_GET_ITEM(args, 1);
    std::string equation = CastPyArg2String(equation_obj, "einsum", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::einsum_final_state_dygraph_function(x,equation)) out = ::einsum_final_state_dygraph_function(x,equation);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_elementwise_pow(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("elementwise_pow pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::elementwise_pow_final_state_dygraph_function(x,y)) out = ::elementwise_pow_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_elu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("elu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: elu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("elu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* alpha_obj = PyTuple_GET_ITEM(args, 1);
    float alpha = CastPyArg2Float(alpha_obj, "elu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::elu_final_state_dygraph_function(x,alpha)) out = ::elu_final_state_dygraph_function(x,alpha);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_elu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("elu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: elu_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("elu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* alpha_obj = PyTuple_GET_ITEM(args, 1);
    float alpha = CastPyArg2Float(alpha_obj, "elu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::elu__final_state_dygraph_function(x,alpha)) out = ::elu__final_state_dygraph_function(x,alpha);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_embedding(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("embedding pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: embedding";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("embedding", "x", args, 0, false);
    auto weight = GetTensorFromArgs("embedding", "weight", args, 1, false);

    // Parse Attributes if needed
    PyObject* padding_idx_obj = PyTuple_GET_ITEM(args, 2);
    int64_t padding_idx = CastPyArg2Long(padding_idx_obj, "embedding", 2);
    PyObject* sparse_obj = PyTuple_GET_ITEM(args, 3);
    bool sparse = CastPyArg2Boolean(sparse_obj, "embedding", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::embedding_final_state_dygraph_function(x,weight,padding_idx,sparse)) out = ::embedding_final_state_dygraph_function(x,weight,padding_idx,sparse);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_empty(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("empty pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: empty";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 0);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "empty", 0);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "empty", 1);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 2);
    paddle::Place place = CastPyArg2Place(place_obj, "empty", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::empty_final_state_dygraph_function(shape,dtype,place)) out = ::empty_final_state_dygraph_function(shape,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_empty_like(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("empty_like pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: empty_like";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("empty_like", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "empty_like", 1);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 2);
    paddle::Place place = CastPyArg2Place(place_obj, "empty_like", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::empty_like_final_state_dygraph_function(x,dtype,place)) out = ::empty_like_final_state_dygraph_function(x,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_equal(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("equal pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: equal";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("equal", "x", args, 0, false);
    auto y = GetTensorFromArgs("equal", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "equal", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::equal_final_state_dygraph_function(x,y,axis)) out = ::equal_final_state_dygraph_function(x,y,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_equal_all(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("equal_all pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::equal_all_final_state_dygraph_function(x,y)) out = ::equal_all_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_exp(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("exp pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::exp_final_state_dygraph_function(x)) out = ::exp_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_exp_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("exp pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: exp_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("exp", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::exp__final_state_dygraph_function(x)) out = ::exp__final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_expand(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("expand pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: expand";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("expand", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "expand", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::expand_final_state_dygraph_function(x,shape)) out = ::expand_final_state_dygraph_function(x,shape);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_expand_as(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("expand_as pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: expand_as";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("expand_as", "x", args, 0, false);
    auto y = GetOptionalTensorFromArgs("expand_as", "y", args, 1, true);

    // Parse Attributes if needed
    PyObject* target_shape_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> target_shape = CastPyArg2Ints(target_shape_obj, "expand_as", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::expand_as_final_state_dygraph_function(x,y,target_shape)) out = ::expand_as_final_state_dygraph_function(x,y,target_shape);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_expm1(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("expm1 pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::expm1_final_state_dygraph_function(x)) out = ::expm1_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_exponential_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("exponential_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: exponential_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("exponential_", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* lambda_obj = PyTuple_GET_ITEM(args, 1);
    float lambda = CastPyArg2Float(lambda_obj, "exponential_", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::exponential__final_state_dygraph_function(x,lambda)) out = ::exponential__final_state_dygraph_function(x,lambda);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_eye(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("eye pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: eye";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject* num_rows_obj = PyTuple_GET_ITEM(args, 0);
    int64_t num_rows = CastPyArg2Long(num_rows_obj, "eye", 0);
    PyObject* num_columns_obj = PyTuple_GET_ITEM(args, 1);
    int64_t num_columns = CastPyArg2Long(num_columns_obj, "eye", 1);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "eye", 2);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 3);
    paddle::Place place = CastPyArg2Place(place_obj, "eye", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::eye_final_state_dygraph_function(num_rows,num_columns,dtype,place)) out = ::eye_final_state_dygraph_function(num_rows,num_columns,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_fill(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fill pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fill";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fill", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* value_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar value = CastPyArg2Scalar(value_obj, "fill", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fill_final_state_dygraph_function(x,value)) out = ::fill_final_state_dygraph_function(x,value);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_fill_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fill pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fill_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fill", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* value_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar value = CastPyArg2Scalar(value_obj, "fill", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fill__final_state_dygraph_function(x,value)) out = ::fill__final_state_dygraph_function(x,value);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_fill_diagonal(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fill_diagonal pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fill_diagonal";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fill_diagonal", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* value_obj = PyTuple_GET_ITEM(args, 1);
    float value = CastPyArg2Float(value_obj, "fill_diagonal", 1);
    PyObject* offset_obj = PyTuple_GET_ITEM(args, 2);
    int offset = CastPyArg2Int(offset_obj, "fill_diagonal", 2);
    PyObject* wrap_obj = PyTuple_GET_ITEM(args, 3);
    bool wrap = CastPyArg2Boolean(wrap_obj, "fill_diagonal", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fill_diagonal_final_state_dygraph_function(x,value,offset,wrap)) out = ::fill_diagonal_final_state_dygraph_function(x,value,offset,wrap);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_fill_diagonal_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fill_diagonal pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fill_diagonal_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fill_diagonal", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* value_obj = PyTuple_GET_ITEM(args, 1);
    float value = CastPyArg2Float(value_obj, "fill_diagonal", 1);
    PyObject* offset_obj = PyTuple_GET_ITEM(args, 2);
    int offset = CastPyArg2Int(offset_obj, "fill_diagonal", 2);
    PyObject* wrap_obj = PyTuple_GET_ITEM(args, 3);
    bool wrap = CastPyArg2Boolean(wrap_obj, "fill_diagonal", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fill_diagonal__final_state_dygraph_function(x,value,offset,wrap)) out = ::fill_diagonal__final_state_dygraph_function(x,value,offset,wrap);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_fill_diagonal_tensor(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fill_diagonal_tensor pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fill_diagonal_tensor";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fill_diagonal_tensor", "x", args, 0, false);
    auto y = GetTensorFromArgs("fill_diagonal_tensor", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* offset_obj = PyTuple_GET_ITEM(args, 2);
    int64_t offset = CastPyArg2Long(offset_obj, "fill_diagonal_tensor", 2);
    PyObject* dim1_obj = PyTuple_GET_ITEM(args, 3);
    int dim1 = CastPyArg2Int(dim1_obj, "fill_diagonal_tensor", 3);
    PyObject* dim2_obj = PyTuple_GET_ITEM(args, 4);
    int dim2 = CastPyArg2Int(dim2_obj, "fill_diagonal_tensor", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fill_diagonal_tensor_final_state_dygraph_function(x,y,offset,dim1,dim2)) out = ::fill_diagonal_tensor_final_state_dygraph_function(x,y,offset,dim1,dim2);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_fill_diagonal_tensor_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fill_diagonal_tensor pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fill_diagonal_tensor_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fill_diagonal_tensor", "x", args, 0, false);
    auto y = GetTensorFromArgs("fill_diagonal_tensor", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* offset_obj = PyTuple_GET_ITEM(args, 2);
    int64_t offset = CastPyArg2Long(offset_obj, "fill_diagonal_tensor", 2);
    PyObject* dim1_obj = PyTuple_GET_ITEM(args, 3);
    int dim1 = CastPyArg2Int(dim1_obj, "fill_diagonal_tensor", 3);
    PyObject* dim2_obj = PyTuple_GET_ITEM(args, 4);
    int dim2 = CastPyArg2Int(dim2_obj, "fill_diagonal_tensor", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fill_diagonal_tensor__final_state_dygraph_function(x,y,offset,dim1,dim2)) out = ::fill_diagonal_tensor__final_state_dygraph_function(x,y,offset,dim1,dim2);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_flatten(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("flatten pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: flatten";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("flatten", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* start_axis_obj = PyTuple_GET_ITEM(args, 1);
    int start_axis = CastPyArg2Int(start_axis_obj, "flatten", 1);
    PyObject* stop_axis_obj = PyTuple_GET_ITEM(args, 2);
    int stop_axis = CastPyArg2Int(stop_axis_obj, "flatten", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::flatten_final_state_dygraph_function(x,start_axis,stop_axis)) out = ::flatten_final_state_dygraph_function(x,start_axis,stop_axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_flatten_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("flatten pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: flatten_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("flatten", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* start_axis_obj = PyTuple_GET_ITEM(args, 1);
    int start_axis = CastPyArg2Int(start_axis_obj, "flatten", 1);
    PyObject* stop_axis_obj = PyTuple_GET_ITEM(args, 2);
    int stop_axis = CastPyArg2Int(stop_axis_obj, "flatten", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::flatten__final_state_dygraph_function(x,start_axis,stop_axis)) out = ::flatten__final_state_dygraph_function(x,start_axis,stop_axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_flip(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("flip pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: flip";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("flip", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> axis = CastPyArg2Ints(axis_obj, "flip", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::flip_final_state_dygraph_function(x,axis)) out = ::flip_final_state_dygraph_function(x,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_floor(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("floor pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::floor_final_state_dygraph_function(x)) out = ::floor_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_floor_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("floor pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: floor_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("floor", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::floor__final_state_dygraph_function(x)) out = ::floor__final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_floor_divide(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("floor_divide pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::floor_divide_final_state_dygraph_function(x,y)) out = ::floor_divide_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_fmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fmax pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fmax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fmax", "x", args, 0, false);
    auto y = GetTensorFromArgs("fmax", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "fmax", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fmax_final_state_dygraph_function(x,y,axis)) out = ::fmax_final_state_dygraph_function(x,y,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_fmin(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fmin pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fmin";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fmin", "x", args, 0, false);
    auto y = GetTensorFromArgs("fmin", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "fmin", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fmin_final_state_dygraph_function(x,y,axis)) out = ::fmin_final_state_dygraph_function(x,y,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_frame(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("frame pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: frame";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("frame", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* frame_length_obj = PyTuple_GET_ITEM(args, 1);
    int frame_length = CastPyArg2Int(frame_length_obj, "frame", 1);
    PyObject* hop_length_obj = PyTuple_GET_ITEM(args, 2);
    int hop_length = CastPyArg2Int(hop_length_obj, "frame", 2);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 3);
    int axis = CastPyArg2Int(axis_obj, "frame", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::frame_final_state_dygraph_function(x,frame_length,hop_length,axis)) out = ::frame_final_state_dygraph_function(x,frame_length,hop_length,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_frobenius_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("frobenius_norm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: frobenius_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("frobenius_norm", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "frobenius_norm", 1);
    PyObject* keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "frobenius_norm", 2);
    PyObject* reduce_all_obj = PyTuple_GET_ITEM(args, 3);
    bool reduce_all = CastPyArg2Boolean(reduce_all_obj, "frobenius_norm", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::frobenius_norm_final_state_dygraph_function(x,axis,keep_dim,reduce_all)) out = ::frobenius_norm_final_state_dygraph_function(x,axis,keep_dim,reduce_all);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_full(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("full pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: full";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 0);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "full", 0);
    PyObject* value_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar value = CastPyArg2Scalar(value_obj, "full", 1);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "full", 2);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 3);
    paddle::Place place = CastPyArg2Place(place_obj, "full", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::full_final_state_dygraph_function(shape,value,dtype,place)) out = ::full_final_state_dygraph_function(shape,value,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_full_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("full_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: full_";

    // Get EagerTensors from args
    auto output = GetTensorFromArgs("full_", "output", args, 0, false);

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "full_", 1);
    PyObject* value_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar value = CastPyArg2Scalar(value_obj, "full_", 2);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "full_", 3);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 4);
    paddle::Place place = CastPyArg2Place(place_obj, "full_", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::full__final_state_dygraph_function(output,shape,value,dtype,place)) out = ::full__final_state_dygraph_function(output,shape,value,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_full_batch_size_like(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("full_batch_size_like pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: full_batch_size_like";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("full_batch_size_like", "input", args, 0, false);

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> shape = CastPyArg2Ints(shape_obj, "full_batch_size_like", 1);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "full_batch_size_like", 2);
    PyObject* value_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::Scalar value = CastPyArg2Scalar(value_obj, "full_batch_size_like", 3);
    PyObject* input_dim_idx_obj = PyTuple_GET_ITEM(args, 4);
    int input_dim_idx = CastPyArg2Int(input_dim_idx_obj, "full_batch_size_like", 4);
    PyObject* output_dim_idx_obj = PyTuple_GET_ITEM(args, 5);
    int output_dim_idx = CastPyArg2Int(output_dim_idx_obj, "full_batch_size_like", 5);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 6);
    paddle::Place place = CastPyArg2Place(place_obj, "full_batch_size_like", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::full_batch_size_like_final_state_dygraph_function(input,shape,dtype,value,input_dim_idx,output_dim_idx,place)) out = ::full_batch_size_like_final_state_dygraph_function(input,shape,dtype,value,input_dim_idx,output_dim_idx,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_full_like(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("full_like pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: full_like";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("full_like", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* value_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar value = CastPyArg2Scalar(value_obj, "full_like", 1);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "full_like", 2);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 3);
    paddle::Place place = CastPyArg2Place(place_obj, "full_like", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::full_like_final_state_dygraph_function(x,value,dtype,place)) out = ::full_like_final_state_dygraph_function(x,value,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_gather(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("gather pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: gather";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("gather", "x", args, 0, false);
    auto index = GetTensorFromArgs("gather", "index", args, 1, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar axis = CastPyArg2Scalar(axis_obj, "gather", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::gather_final_state_dygraph_function(x,index,axis)) out = ::gather_final_state_dygraph_function(x,index,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_gather_nd(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("gather_nd pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::gather_nd_final_state_dygraph_function(x,index)) out = ::gather_nd_final_state_dygraph_function(x,index);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_gather_tree(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("gather_tree pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::gather_tree_final_state_dygraph_function(ids,parents)) out = ::gather_tree_final_state_dygraph_function(ids,parents);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_gaussian_random(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("gaussian_random pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: gaussian_random";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 0);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "gaussian_random", 0);
    PyObject* mean_obj = PyTuple_GET_ITEM(args, 1);
    float mean = CastPyArg2Float(mean_obj, "gaussian_random", 1);
    PyObject* std_obj = PyTuple_GET_ITEM(args, 2);
    float std = CastPyArg2Float(std_obj, "gaussian_random", 2);
    PyObject* seed_obj = PyTuple_GET_ITEM(args, 3);
    int seed = CastPyArg2Int(seed_obj, "gaussian_random", 3);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 4);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "gaussian_random", 4);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 5);
    paddle::Place place = CastPyArg2Place(place_obj, "gaussian_random", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::gaussian_random_final_state_dygraph_function(shape,mean,std,seed,dtype,place)) out = ::gaussian_random_final_state_dygraph_function(shape,mean,std,seed,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_gelu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("gelu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: gelu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("gelu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* approximate_obj = PyTuple_GET_ITEM(args, 1);
    bool approximate = CastPyArg2Boolean(approximate_obj, "gelu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::gelu_final_state_dygraph_function(x,approximate)) out = ::gelu_final_state_dygraph_function(x,approximate);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_generate_proposals_v2(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("generate_proposals_v2 pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: generate_proposals_v2";

    // Get EagerTensors from args
    auto scores = GetTensorFromArgs("generate_proposals_v2", "scores", args, 0, false);
    auto bbox_deltas = GetTensorFromArgs("generate_proposals_v2", "bbox_deltas", args, 1, false);
    auto im_shape = GetTensorFromArgs("generate_proposals_v2", "im_shape", args, 2, false);
    auto anchors = GetTensorFromArgs("generate_proposals_v2", "anchors", args, 3, false);
    auto variances = GetTensorFromArgs("generate_proposals_v2", "variances", args, 4, false);

    // Parse Attributes if needed
    PyObject* pre_nms_top_n_obj = PyTuple_GET_ITEM(args, 5);
    int pre_nms_top_n = CastPyArg2Int(pre_nms_top_n_obj, "generate_proposals_v2", 5);
    PyObject* post_nms_top_n_obj = PyTuple_GET_ITEM(args, 6);
    int post_nms_top_n = CastPyArg2Int(post_nms_top_n_obj, "generate_proposals_v2", 6);
    PyObject* nms_thresh_obj = PyTuple_GET_ITEM(args, 7);
    float nms_thresh = CastPyArg2Float(nms_thresh_obj, "generate_proposals_v2", 7);
    PyObject* min_size_obj = PyTuple_GET_ITEM(args, 8);
    float min_size = CastPyArg2Float(min_size_obj, "generate_proposals_v2", 8);
    PyObject* eta_obj = PyTuple_GET_ITEM(args, 9);
    float eta = CastPyArg2Float(eta_obj, "generate_proposals_v2", 9);
    PyObject* pixel_offset_obj = PyTuple_GET_ITEM(args, 10);
    bool pixel_offset = CastPyArg2Boolean(pixel_offset_obj, "generate_proposals_v2", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::generate_proposals_v2_final_state_dygraph_function(scores,bbox_deltas,im_shape,anchors,variances,pre_nms_top_n,post_nms_top_n,nms_thresh,min_size,eta,pixel_offset)) out = ::generate_proposals_v2_final_state_dygraph_function(scores,bbox_deltas,im_shape,anchors,variances,pre_nms_top_n,post_nms_top_n,nms_thresh,min_size,eta,pixel_offset);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_graph_send_recv(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("graph_send_recv pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: graph_send_recv";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("graph_send_recv", "x", args, 0, false);
    auto src_index = GetTensorFromArgs("graph_send_recv", "src_index", args, 1, false);
    auto dst_index = GetTensorFromArgs("graph_send_recv", "dst_index", args, 2, false);

    // Parse Attributes if needed
    PyObject* reduce_op_obj = PyTuple_GET_ITEM(args, 3);
    std::string reduce_op = CastPyArg2String(reduce_op_obj, "graph_send_recv", 3);
    PyObject* out_size_obj = PyTuple_GET_ITEM(args, 4);
    paddle::experimental::IntArray out_size = CastPyArg2IntArray(out_size_obj, "graph_send_recv", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::graph_send_recv_final_state_dygraph_function(x,src_index,dst_index,reduce_op,out_size)) out = ::graph_send_recv_final_state_dygraph_function(x,src_index,dst_index,reduce_op,out_size);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_graph_send_ue_recv(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("graph_send_ue_recv pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: graph_send_ue_recv";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("graph_send_ue_recv", "x", args, 0, false);
    auto y = GetTensorFromArgs("graph_send_ue_recv", "y", args, 1, false);
    auto src_index = GetTensorFromArgs("graph_send_ue_recv", "src_index", args, 2, false);
    auto dst_index = GetTensorFromArgs("graph_send_ue_recv", "dst_index", args, 3, false);

    // Parse Attributes if needed
    PyObject* message_op_obj = PyTuple_GET_ITEM(args, 4);
    std::string message_op = CastPyArg2String(message_op_obj, "graph_send_ue_recv", 4);
    PyObject* reduce_op_obj = PyTuple_GET_ITEM(args, 5);
    std::string reduce_op = CastPyArg2String(reduce_op_obj, "graph_send_ue_recv", 5);
    PyObject* out_size_obj = PyTuple_GET_ITEM(args, 6);
    paddle::experimental::IntArray out_size = CastPyArg2IntArray(out_size_obj, "graph_send_ue_recv", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::graph_send_ue_recv_final_state_dygraph_function(x,y,src_index,dst_index,message_op,reduce_op,out_size)) out = ::graph_send_ue_recv_final_state_dygraph_function(x,y,src_index,dst_index,message_op,reduce_op,out_size);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_greater_equal(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("greater_equal pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: greater_equal";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("greater_equal", "x", args, 0, false);
    auto y = GetTensorFromArgs("greater_equal", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "greater_equal", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::greater_equal_final_state_dygraph_function(x,y,axis)) out = ::greater_equal_final_state_dygraph_function(x,y,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_greater_than(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("greater_than pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: greater_than";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("greater_than", "x", args, 0, false);
    auto y = GetTensorFromArgs("greater_than", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "greater_than", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::greater_than_final_state_dygraph_function(x,y,axis)) out = ::greater_than_final_state_dygraph_function(x,y,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_grid_sample(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("grid_sample pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: grid_sample";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("grid_sample", "x", args, 0, false);
    auto grid = GetTensorFromArgs("grid_sample", "grid", args, 1, false);

    // Parse Attributes if needed
    PyObject* mode_obj = PyTuple_GET_ITEM(args, 2);
    std::string mode = CastPyArg2String(mode_obj, "grid_sample", 2);
    PyObject* padding_mode_obj = PyTuple_GET_ITEM(args, 3);
    std::string padding_mode = CastPyArg2String(padding_mode_obj, "grid_sample", 3);
    PyObject* align_corners_obj = PyTuple_GET_ITEM(args, 4);
    bool align_corners = CastPyArg2Boolean(align_corners_obj, "grid_sample", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::grid_sample_final_state_dygraph_function(x,grid,mode,padding_mode,align_corners)) out = ::grid_sample_final_state_dygraph_function(x,grid,mode,padding_mode,align_corners);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_group_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("group_norm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: group_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("group_norm", "x", args, 0, false);
    auto scale = GetOptionalTensorFromArgs("group_norm", "scale", args, 1, true);
    auto bias = GetOptionalTensorFromArgs("group_norm", "bias", args, 2, true);

    // Parse Attributes if needed
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 3);
    float epsilon = CastPyArg2Float(epsilon_obj, "group_norm", 3);
    PyObject* groups_obj = PyTuple_GET_ITEM(args, 4);
    int groups = CastPyArg2Int(groups_obj, "group_norm", 4);
    PyObject* data_layout_obj = PyTuple_GET_ITEM(args, 5);
    std::string data_layout = CastPyArg2String(data_layout_obj, "group_norm", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::group_norm_final_state_dygraph_function(x,scale,bias,epsilon,groups,data_layout)) out = ::group_norm_final_state_dygraph_function(x,scale,bias,epsilon,groups,data_layout);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_gumbel_softmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("gumbel_softmax pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: gumbel_softmax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("gumbel_softmax", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* temperature_obj = PyTuple_GET_ITEM(args, 1);
    float temperature = CastPyArg2Float(temperature_obj, "gumbel_softmax", 1);
    PyObject* hard_obj = PyTuple_GET_ITEM(args, 2);
    bool hard = CastPyArg2Boolean(hard_obj, "gumbel_softmax", 2);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 3);
    int axis = CastPyArg2Int(axis_obj, "gumbel_softmax", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::gumbel_softmax_final_state_dygraph_function(x,temperature,hard,axis)) out = ::gumbel_softmax_final_state_dygraph_function(x,temperature,hard,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_hard_shrink(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("hard_shrink pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: hard_shrink";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("hard_shrink", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "hard_shrink", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::hard_shrink_final_state_dygraph_function(x,threshold)) out = ::hard_shrink_final_state_dygraph_function(x,threshold);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_hard_sigmoid(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("hard_sigmoid pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: hard_sigmoid";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("hard_sigmoid", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* slope_obj = PyTuple_GET_ITEM(args, 1);
    float slope = CastPyArg2Float(slope_obj, "hard_sigmoid", 1);
    PyObject* offset_obj = PyTuple_GET_ITEM(args, 2);
    float offset = CastPyArg2Float(offset_obj, "hard_sigmoid", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::hard_sigmoid_final_state_dygraph_function(x,slope,offset)) out = ::hard_sigmoid_final_state_dygraph_function(x,slope,offset);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_hard_swish(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("hard_swish pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: hard_swish";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("hard_swish", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "hard_swish", 1);
    PyObject* scale_obj = PyTuple_GET_ITEM(args, 2);
    float scale = CastPyArg2Float(scale_obj, "hard_swish", 2);
    PyObject* offset_obj = PyTuple_GET_ITEM(args, 3);
    float offset = CastPyArg2Float(offset_obj, "hard_swish", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::hard_swish_final_state_dygraph_function(x,threshold,scale,offset)) out = ::hard_swish_final_state_dygraph_function(x,threshold,scale,offset);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_hierarchical_sigmoid(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("hierarchical_sigmoid pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: hierarchical_sigmoid";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("hierarchical_sigmoid", "x", args, 0, false);
    auto w = GetTensorFromArgs("hierarchical_sigmoid", "w", args, 1, false);
    auto label = GetTensorFromArgs("hierarchical_sigmoid", "label", args, 2, false);
    auto path = GetOptionalTensorFromArgs("hierarchical_sigmoid", "path", args, 3, true);
    auto code = GetOptionalTensorFromArgs("hierarchical_sigmoid", "code", args, 4, true);
    auto bias = GetOptionalTensorFromArgs("hierarchical_sigmoid", "bias", args, 5, true);

    // Parse Attributes if needed
    PyObject* num_classes_obj = PyTuple_GET_ITEM(args, 6);
    int num_classes = CastPyArg2Int(num_classes_obj, "hierarchical_sigmoid", 6);
    PyObject* remote_prefetch_obj = PyTuple_GET_ITEM(args, 7);
    bool remote_prefetch = CastPyArg2Boolean(remote_prefetch_obj, "hierarchical_sigmoid", 7);
    PyObject* trainer_id_obj = PyTuple_GET_ITEM(args, 8);
    int trainer_id = CastPyArg2Int(trainer_id_obj, "hierarchical_sigmoid", 8);
    PyObject* height_sections_obj = PyTuple_GET_ITEM(args, 9);
    std::vector<int64_t> height_sections = CastPyArg2Longs(height_sections_obj, "hierarchical_sigmoid", 9);
    PyObject* epmap_obj = PyTuple_GET_ITEM(args, 10);
    std::vector<std::string> epmap = CastPyArg2Strings(epmap_obj, "hierarchical_sigmoid", 10);
    PyObject* table_names_obj = PyTuple_GET_ITEM(args, 11);
    std::vector<std::string> table_names = CastPyArg2Strings(table_names_obj, "hierarchical_sigmoid", 11);
    PyObject* is_sparse_obj = PyTuple_GET_ITEM(args, 12);
    bool is_sparse = CastPyArg2Boolean(is_sparse_obj, "hierarchical_sigmoid", 12);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::hierarchical_sigmoid_final_state_dygraph_function(x,w,label,path,code,bias,num_classes,remote_prefetch,trainer_id,height_sections,epmap,table_names,is_sparse)) out = ::hierarchical_sigmoid_final_state_dygraph_function(x,w,label,path,code,bias,num_classes,remote_prefetch,trainer_id,height_sections,epmap,table_names,is_sparse);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_histogram(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("histogram pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: histogram";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("histogram", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* bins_obj = PyTuple_GET_ITEM(args, 1);
    int64_t bins = CastPyArg2Long(bins_obj, "histogram", 1);
    PyObject* min_obj = PyTuple_GET_ITEM(args, 2);
    int min = CastPyArg2Int(min_obj, "histogram", 2);
    PyObject* max_obj = PyTuple_GET_ITEM(args, 3);
    int max = CastPyArg2Int(max_obj, "histogram", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::histogram_final_state_dygraph_function(x,bins,min,max)) out = ::histogram_final_state_dygraph_function(x,bins,min,max);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_huber_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("huber_loss pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: huber_loss";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("huber_loss", "input", args, 0, false);
    auto label = GetTensorFromArgs("huber_loss", "label", args, 1, false);

    // Parse Attributes if needed
    PyObject* delta_obj = PyTuple_GET_ITEM(args, 2);
    float delta = CastPyArg2Float(delta_obj, "huber_loss", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::huber_loss_final_state_dygraph_function(input,label,delta)) out = ::huber_loss_final_state_dygraph_function(input,label,delta);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_imag(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("imag pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::imag_final_state_dygraph_function(x)) out = ::imag_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_increment(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("increment pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: increment";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("increment", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* value_obj = PyTuple_GET_ITEM(args, 1);
    float value = CastPyArg2Float(value_obj, "increment", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::increment_final_state_dygraph_function(x,value)) out = ::increment_final_state_dygraph_function(x,value);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_index_sample(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("index_sample pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::index_sample_final_state_dygraph_function(x,index)) out = ::index_sample_final_state_dygraph_function(x,index);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_index_select(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("index_select pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: index_select";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("index_select", "x", args, 0, false);
    auto index = GetTensorFromArgs("index_select", "index", args, 1, false);

    // Parse Attributes if needed
    PyObject* dim_obj = PyTuple_GET_ITEM(args, 2);
    int dim = CastPyArg2Int(dim_obj, "index_select", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::index_select_final_state_dygraph_function(x,index,dim)) out = ::index_select_final_state_dygraph_function(x,index,dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_instance_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("instance_norm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: instance_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("instance_norm", "x", args, 0, false);
    auto scale = GetOptionalTensorFromArgs("instance_norm", "scale", args, 1, true);
    auto bias = GetOptionalTensorFromArgs("instance_norm", "bias", args, 2, true);

    // Parse Attributes if needed
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 3);
    float epsilon = CastPyArg2Float(epsilon_obj, "instance_norm", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::instance_norm_final_state_dygraph_function(x,scale,bias,epsilon)) out = ::instance_norm_final_state_dygraph_function(x,scale,bias,epsilon);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_inverse(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("inverse pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: inverse";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("inverse", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::inverse_final_state_dygraph_function(x)) out = ::inverse_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_is_empty(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("is_empty pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::is_empty_final_state_dygraph_function(x)) out = ::is_empty_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_isclose(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("isclose pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: isclose";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("isclose", "x", args, 0, false);
    auto y = GetTensorFromArgs("isclose", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* rtol_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar rtol = CastPyArg2Scalar(rtol_obj, "isclose", 2);
    PyObject* atol_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::Scalar atol = CastPyArg2Scalar(atol_obj, "isclose", 3);
    PyObject* equal_nan_obj = PyTuple_GET_ITEM(args, 4);
    bool equal_nan = CastPyArg2Boolean(equal_nan_obj, "isclose", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::isclose_final_state_dygraph_function(x,y,rtol,atol,equal_nan)) out = ::isclose_final_state_dygraph_function(x,y,rtol,atol,equal_nan);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_isfinite(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("isfinite pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::isfinite_final_state_dygraph_function(x)) out = ::isfinite_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_isinf(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("isinf pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::isinf_final_state_dygraph_function(x)) out = ::isinf_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_isnan(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("isnan pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::isnan_final_state_dygraph_function(x)) out = ::isnan_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_kldiv_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("kldiv_loss pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: kldiv_loss";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("kldiv_loss", "x", args, 0, false);
    auto label = GetTensorFromArgs("kldiv_loss", "label", args, 1, false);

    // Parse Attributes if needed
    PyObject* reduction_obj = PyTuple_GET_ITEM(args, 2);
    std::string reduction = CastPyArg2String(reduction_obj, "kldiv_loss", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::kldiv_loss_final_state_dygraph_function(x,label,reduction)) out = ::kldiv_loss_final_state_dygraph_function(x,label,reduction);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_kron(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("kron pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::kron_final_state_dygraph_function(x,y)) out = ::kron_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_kthvalue(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("kthvalue pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: kthvalue";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("kthvalue", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* k_obj = PyTuple_GET_ITEM(args, 1);
    int k = CastPyArg2Int(k_obj, "kthvalue", 1);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "kthvalue", 2);
    PyObject* keepdim_obj = PyTuple_GET_ITEM(args, 3);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "kthvalue", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::kthvalue_final_state_dygraph_function(x,k,axis,keepdim)) out = ::kthvalue_final_state_dygraph_function(x,k,axis,keepdim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_label_smooth(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("label_smooth pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: label_smooth";

    // Get EagerTensors from args
    auto label = GetTensorFromArgs("label_smooth", "label", args, 0, false);
    auto prior_dist = GetOptionalTensorFromArgs("label_smooth", "prior_dist", args, 1, true);

    // Parse Attributes if needed
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 2);
    float epsilon = CastPyArg2Float(epsilon_obj, "label_smooth", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::label_smooth_final_state_dygraph_function(label,prior_dist,epsilon)) out = ::label_smooth_final_state_dygraph_function(label,prior_dist,epsilon);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_lamb_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("lamb_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: lamb_";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("lamb_", "param", args, 0, false);
    auto grad = GetTensorFromArgs("lamb_", "grad", args, 1, false);
    auto learning_rate = GetTensorFromArgs("lamb_", "learning_rate", args, 2, false);
    auto moment1 = GetTensorFromArgs("lamb_", "moment1", args, 3, false);
    auto moment2 = GetTensorFromArgs("lamb_", "moment2", args, 4, false);
    auto beta1_pow = GetTensorFromArgs("lamb_", "beta1_pow", args, 5, false);
    auto beta2_pow = GetTensorFromArgs("lamb_", "beta2_pow", args, 6, false);
    auto master_param = GetOptionalTensorFromArgs("lamb_", "master_param", args, 7, true);
    auto skip_update = GetOptionalTensorFromArgs("lamb_", "skip_update", args, 8, true);

    // Parse Attributes if needed
    PyObject* weight_decay_obj = PyTuple_GET_ITEM(args, 9);
    float weight_decay = CastPyArg2Float(weight_decay_obj, "lamb_", 9);
    PyObject* beta1_obj = PyTuple_GET_ITEM(args, 10);
    float beta1 = CastPyArg2Float(beta1_obj, "lamb_", 10);
    PyObject* beta2_obj = PyTuple_GET_ITEM(args, 11);
    float beta2 = CastPyArg2Float(beta2_obj, "lamb_", 11);
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 12);
    float epsilon = CastPyArg2Float(epsilon_obj, "lamb_", 12);
    PyObject* multi_precision_obj = PyTuple_GET_ITEM(args, 13);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "lamb_", 13);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::lamb__final_state_dygraph_function(param,grad,learning_rate,moment1,moment2,beta1_pow,beta2_pow,master_param,skip_update,weight_decay,beta1,beta2,epsilon,multi_precision)) out = ::lamb__final_state_dygraph_function(param,grad,learning_rate,moment1,moment2,beta1_pow,beta2_pow,master_param,skip_update,weight_decay,beta1,beta2,epsilon,multi_precision);


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
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_layer_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("layer_norm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: layer_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("layer_norm", "x", args, 0, false);
    auto scale = GetOptionalTensorFromArgs("layer_norm", "scale", args, 1, true);
    auto bias = GetOptionalTensorFromArgs("layer_norm", "bias", args, 2, true);

    // Parse Attributes if needed
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 3);
    float epsilon = CastPyArg2Float(epsilon_obj, "layer_norm", 3);
    PyObject* begin_norm_axis_obj = PyTuple_GET_ITEM(args, 4);
    int begin_norm_axis = CastPyArg2Int(begin_norm_axis_obj, "layer_norm", 4);
    PyObject* is_test_obj = PyTuple_GET_ITEM(args, 5);
    bool is_test = CastPyArg2Boolean(is_test_obj, "layer_norm", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::layer_norm_final_state_dygraph_function(x,scale,bias,epsilon,begin_norm_axis,is_test)) out = ::layer_norm_final_state_dygraph_function(x,scale,bias,epsilon,begin_norm_axis,is_test);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_leaky_relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("leaky_relu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: leaky_relu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("leaky_relu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* alpha_obj = PyTuple_GET_ITEM(args, 1);
    float alpha = CastPyArg2Float(alpha_obj, "leaky_relu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::leaky_relu_final_state_dygraph_function(x,alpha)) out = ::leaky_relu_final_state_dygraph_function(x,alpha);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_lerp(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("lerp pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::lerp_final_state_dygraph_function(x,y,weight)) out = ::lerp_final_state_dygraph_function(x,y,weight);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_lerp_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("lerp pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: lerp_";

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::lerp__final_state_dygraph_function(x,y,weight)) out = ::lerp__final_state_dygraph_function(x,y,weight);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_less_equal(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("less_equal pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: less_equal";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("less_equal", "x", args, 0, false);
    auto y = GetTensorFromArgs("less_equal", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "less_equal", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::less_equal_final_state_dygraph_function(x,y,axis)) out = ::less_equal_final_state_dygraph_function(x,y,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_less_than(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("less_than pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: less_than";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("less_than", "x", args, 0, false);
    auto y = GetTensorFromArgs("less_than", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "less_than", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::less_than_final_state_dygraph_function(x,y,axis)) out = ::less_than_final_state_dygraph_function(x,y,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_linear_interp(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("linear_interp pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: linear_interp";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("linear_interp", "x", args, 0, false);
    auto out_size = GetOptionalTensorFromArgs("linear_interp", "out_size", args, 1, true);
    auto size_tensor = GetOptionalTensorListFromArgs("linear_interp", "size_tensor", args, 2, true);
    auto scale_tensor = GetOptionalTensorFromArgs("linear_interp", "scale_tensor", args, 3, true);

    // Parse Attributes if needed
    PyObject* data_layout_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_layout = CastPyArg2String(data_layout_obj, "linear_interp", 4);
    PyObject* out_d_obj = PyTuple_GET_ITEM(args, 5);
    int out_d = CastPyArg2Int(out_d_obj, "linear_interp", 5);
    PyObject* out_h_obj = PyTuple_GET_ITEM(args, 6);
    int out_h = CastPyArg2Int(out_h_obj, "linear_interp", 6);
    PyObject* out_w_obj = PyTuple_GET_ITEM(args, 7);
    int out_w = CastPyArg2Int(out_w_obj, "linear_interp", 7);
    PyObject* scale_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<float> scale = CastPyArg2Floats(scale_obj, "linear_interp", 8);
    PyObject* interp_method_obj = PyTuple_GET_ITEM(args, 9);
    std::string interp_method = CastPyArg2String(interp_method_obj, "linear_interp", 9);
    PyObject* align_corners_obj = PyTuple_GET_ITEM(args, 10);
    bool align_corners = CastPyArg2Boolean(align_corners_obj, "linear_interp", 10);
    PyObject* align_mode_obj = PyTuple_GET_ITEM(args, 11);
    int align_mode = CastPyArg2Int(align_mode_obj, "linear_interp", 11);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::linear_interp_final_state_dygraph_function(x,out_size,size_tensor,scale_tensor,data_layout,out_d,out_h,out_w,scale,interp_method,align_corners,align_mode)) out = ::linear_interp_final_state_dygraph_function(x,out_size,size_tensor,scale_tensor,data_layout,out_d,out_h,out_w,scale,interp_method,align_corners,align_mode);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_linspace(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("linspace pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: linspace";

    // Get EagerTensors from args
    auto start = GetTensorFromArgs("linspace", "start", args, 0, false);
    auto stop = GetTensorFromArgs("linspace", "stop", args, 1, false);
    auto number = GetTensorFromArgs("linspace", "number", args, 2, false);

    // Parse Attributes if needed
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "linspace", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::linspace_final_state_dygraph_function(start,stop,number,dtype)) out = ::linspace_final_state_dygraph_function(start,stop,number,dtype);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_log(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("log pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::log_final_state_dygraph_function(x)) out = ::log_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_log10(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("log10 pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::log10_final_state_dygraph_function(x)) out = ::log10_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_log1p(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("log1p pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::log1p_final_state_dygraph_function(x)) out = ::log1p_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_log2(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("log2 pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::log2_final_state_dygraph_function(x)) out = ::log2_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_log_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("log_loss pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: log_loss";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("log_loss", "input", args, 0, false);
    auto label = GetTensorFromArgs("log_loss", "label", args, 1, false);

    // Parse Attributes if needed
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 2);
    float epsilon = CastPyArg2Float(epsilon_obj, "log_loss", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::log_loss_final_state_dygraph_function(input,label,epsilon)) out = ::log_loss_final_state_dygraph_function(input,label,epsilon);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_log_softmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("log_softmax pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: log_softmax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("log_softmax", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "log_softmax", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::log_softmax_final_state_dygraph_function(x,axis)) out = ::log_softmax_final_state_dygraph_function(x,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_logcumsumexp(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("logcumsumexp pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: logcumsumexp";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("logcumsumexp", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "logcumsumexp", 1);
    PyObject* flatten_obj = PyTuple_GET_ITEM(args, 2);
    bool flatten = CastPyArg2Boolean(flatten_obj, "logcumsumexp", 2);
    PyObject* exclusive_obj = PyTuple_GET_ITEM(args, 3);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "logcumsumexp", 3);
    PyObject* reverse_obj = PyTuple_GET_ITEM(args, 4);
    bool reverse = CastPyArg2Boolean(reverse_obj, "logcumsumexp", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::logcumsumexp_final_state_dygraph_function(x,axis,flatten,exclusive,reverse)) out = ::logcumsumexp_final_state_dygraph_function(x,axis,flatten,exclusive,reverse);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_logical_and(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("logical_and pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::logical_and_final_state_dygraph_function(x,y)) out = ::logical_and_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_logical_not(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("logical_not pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::logical_not_final_state_dygraph_function(x)) out = ::logical_not_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_logical_or(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("logical_or pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::logical_or_final_state_dygraph_function(x,y)) out = ::logical_or_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_logical_xor(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("logical_xor pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::logical_xor_final_state_dygraph_function(x,y)) out = ::logical_xor_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_logit(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("logit pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: logit";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("logit", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* eps_obj = PyTuple_GET_ITEM(args, 1);
    float eps = CastPyArg2Float(eps_obj, "logit", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::logit_final_state_dygraph_function(x,eps)) out = ::logit_final_state_dygraph_function(x,eps);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_logsigmoid(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("logsigmoid pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::logsigmoid_final_state_dygraph_function(x)) out = ::logsigmoid_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_logsumexp(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("logsumexp pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: logsumexp";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("logsumexp", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "logsumexp", 1);
    PyObject* keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "logsumexp", 2);
    PyObject* reduce_all_obj = PyTuple_GET_ITEM(args, 3);
    bool reduce_all = CastPyArg2Boolean(reduce_all_obj, "logsumexp", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::logsumexp_final_state_dygraph_function(x,axis,keepdim,reduce_all)) out = ::logsumexp_final_state_dygraph_function(x,axis,keepdim,reduce_all);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_lstsq(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("lstsq pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: lstsq";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("lstsq", "x", args, 0, false);
    auto y = GetTensorFromArgs("lstsq", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* rcond_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar rcond = CastPyArg2Scalar(rcond_obj, "lstsq", 2);
    PyObject* driver_obj = PyTuple_GET_ITEM(args, 3);
    std::string driver = CastPyArg2String(driver_obj, "lstsq", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::lstsq_final_state_dygraph_function(x,y,rcond,driver)) out = ::lstsq_final_state_dygraph_function(x,y,rcond,driver);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_lu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("lu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: lu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("lu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* pivot_obj = PyTuple_GET_ITEM(args, 1);
    bool pivot = CastPyArg2Boolean(pivot_obj, "lu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::lu_final_state_dygraph_function(x,pivot)) out = ::lu_final_state_dygraph_function(x,pivot);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_lu_unpack(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("lu_unpack pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: lu_unpack";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("lu_unpack", "x", args, 0, false);
    auto pivots = GetTensorFromArgs("lu_unpack", "pivots", args, 1, false);

    // Parse Attributes if needed
    PyObject* unpack_ludata_obj = PyTuple_GET_ITEM(args, 2);
    bool unpack_ludata = CastPyArg2Boolean(unpack_ludata_obj, "lu_unpack", 2);
    PyObject* unpack_pivots_obj = PyTuple_GET_ITEM(args, 3);
    bool unpack_pivots = CastPyArg2Boolean(unpack_pivots_obj, "lu_unpack", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::lu_unpack_final_state_dygraph_function(x,pivots,unpack_ludata,unpack_pivots)) out = ::lu_unpack_final_state_dygraph_function(x,pivots,unpack_ludata,unpack_pivots);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_margin_cross_entropy(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("margin_cross_entropy pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: margin_cross_entropy";

    // Get EagerTensors from args
    auto logits = GetTensorFromArgs("margin_cross_entropy", "logits", args, 0, false);
    auto label = GetTensorFromArgs("margin_cross_entropy", "label", args, 1, false);

    // Parse Attributes if needed
    PyObject* return_softmax_obj = PyTuple_GET_ITEM(args, 2);
    bool return_softmax = CastPyArg2Boolean(return_softmax_obj, "margin_cross_entropy", 2);
    PyObject* ring_id_obj = PyTuple_GET_ITEM(args, 3);
    int ring_id = CastPyArg2Int(ring_id_obj, "margin_cross_entropy", 3);
    PyObject* rank_obj = PyTuple_GET_ITEM(args, 4);
    int rank = CastPyArg2Int(rank_obj, "margin_cross_entropy", 4);
    PyObject* nranks_obj = PyTuple_GET_ITEM(args, 5);
    int nranks = CastPyArg2Int(nranks_obj, "margin_cross_entropy", 5);
    PyObject* margin1_obj = PyTuple_GET_ITEM(args, 6);
    float margin1 = CastPyArg2Float(margin1_obj, "margin_cross_entropy", 6);
    PyObject* margin2_obj = PyTuple_GET_ITEM(args, 7);
    float margin2 = CastPyArg2Float(margin2_obj, "margin_cross_entropy", 7);
    PyObject* margin3_obj = PyTuple_GET_ITEM(args, 8);
    float margin3 = CastPyArg2Float(margin3_obj, "margin_cross_entropy", 8);
    PyObject* scale_obj = PyTuple_GET_ITEM(args, 9);
    float scale = CastPyArg2Float(scale_obj, "margin_cross_entropy", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::margin_cross_entropy_final_state_dygraph_function(logits,label,return_softmax,ring_id,rank,nranks,margin1,margin2,margin3,scale)) out = ::margin_cross_entropy_final_state_dygraph_function(logits,label,return_softmax,ring_id,rank,nranks,margin1,margin2,margin3,scale);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_masked_select(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("masked_select pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::masked_select_final_state_dygraph_function(x,mask)) out = ::masked_select_final_state_dygraph_function(x,mask);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_matmul(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("matmul pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: matmul";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("matmul", "x", args, 0, false);
    auto y = GetTensorFromArgs("matmul", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* transpose_x_obj = PyTuple_GET_ITEM(args, 2);
    bool transpose_x = CastPyArg2Boolean(transpose_x_obj, "matmul", 2);
    PyObject* transpose_y_obj = PyTuple_GET_ITEM(args, 3);
    bool transpose_y = CastPyArg2Boolean(transpose_y_obj, "matmul", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::matmul_final_state_dygraph_function(x,y,transpose_x,transpose_y)) out = ::matmul_final_state_dygraph_function(x,y,transpose_x,transpose_y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_matrix_nms(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("matrix_nms pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: matrix_nms";

    // Get EagerTensors from args
    auto bboxes = GetTensorFromArgs("matrix_nms", "bboxes", args, 0, false);
    auto scores = GetTensorFromArgs("matrix_nms", "scores", args, 1, false);

    // Parse Attributes if needed
    PyObject* score_threshold_obj = PyTuple_GET_ITEM(args, 2);
    float score_threshold = CastPyArg2Float(score_threshold_obj, "matrix_nms", 2);
    PyObject* nms_top_k_obj = PyTuple_GET_ITEM(args, 3);
    int nms_top_k = CastPyArg2Int(nms_top_k_obj, "matrix_nms", 3);
    PyObject* keep_top_k_obj = PyTuple_GET_ITEM(args, 4);
    int keep_top_k = CastPyArg2Int(keep_top_k_obj, "matrix_nms", 4);
    PyObject* post_threshold_obj = PyTuple_GET_ITEM(args, 5);
    float post_threshold = CastPyArg2Float(post_threshold_obj, "matrix_nms", 5);
    PyObject* use_gaussian_obj = PyTuple_GET_ITEM(args, 6);
    bool use_gaussian = CastPyArg2Boolean(use_gaussian_obj, "matrix_nms", 6);
    PyObject* gaussian_sigma_obj = PyTuple_GET_ITEM(args, 7);
    float gaussian_sigma = CastPyArg2Float(gaussian_sigma_obj, "matrix_nms", 7);
    PyObject* background_label_obj = PyTuple_GET_ITEM(args, 8);
    int background_label = CastPyArg2Int(background_label_obj, "matrix_nms", 8);
    PyObject* normalized_obj = PyTuple_GET_ITEM(args, 9);
    bool normalized = CastPyArg2Boolean(normalized_obj, "matrix_nms", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::matrix_nms_final_state_dygraph_function(bboxes,scores,score_threshold,nms_top_k,keep_top_k,post_threshold,use_gaussian,gaussian_sigma,background_label,normalized)) out = ::matrix_nms_final_state_dygraph_function(bboxes,scores,score_threshold,nms_top_k,keep_top_k,post_threshold,use_gaussian,gaussian_sigma,background_label,normalized);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_matrix_power(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("matrix_power pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: matrix_power";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("matrix_power", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* n_obj = PyTuple_GET_ITEM(args, 1);
    int n = CastPyArg2Int(n_obj, "matrix_power", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::matrix_power_final_state_dygraph_function(x,n)) out = ::matrix_power_final_state_dygraph_function(x,n);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_matrix_rank(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("matrix_rank pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: matrix_rank";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("matrix_rank", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* tol_obj = PyTuple_GET_ITEM(args, 1);
    float tol = CastPyArg2Float(tol_obj, "matrix_rank", 1);
    PyObject* use_default_tol_obj = PyTuple_GET_ITEM(args, 2);
    bool use_default_tol = CastPyArg2Boolean(use_default_tol_obj, "matrix_rank", 2);
    PyObject* hermitian_obj = PyTuple_GET_ITEM(args, 3);
    bool hermitian = CastPyArg2Boolean(hermitian_obj, "matrix_rank", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::matrix_rank_final_state_dygraph_function(x,tol,use_default_tol,hermitian)) out = ::matrix_rank_final_state_dygraph_function(x,tol,use_default_tol,hermitian);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_matrix_rank_tol(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("matrix_rank_tol pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: matrix_rank_tol";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("matrix_rank_tol", "x", args, 0, false);
    auto atol_tensor = GetTensorFromArgs("matrix_rank_tol", "atol_tensor", args, 1, false);

    // Parse Attributes if needed
    PyObject* use_default_tol_obj = PyTuple_GET_ITEM(args, 2);
    bool use_default_tol = CastPyArg2Boolean(use_default_tol_obj, "matrix_rank_tol", 2);
    PyObject* hermitian_obj = PyTuple_GET_ITEM(args, 3);
    bool hermitian = CastPyArg2Boolean(hermitian_obj, "matrix_rank_tol", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::matrix_rank_tol_final_state_dygraph_function(x,atol_tensor,use_default_tol,hermitian)) out = ::matrix_rank_tol_final_state_dygraph_function(x,atol_tensor,use_default_tol,hermitian);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_max(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("max pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: max";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("max", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "max", 1);
    PyObject* keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "max", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::max_final_state_dygraph_function(x,dims,keep_dim)) out = ::max_final_state_dygraph_function(x,dims,keep_dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_max_pool2d_with_index(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("max_pool2d_with_index pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: max_pool2d_with_index";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("max_pool2d_with_index", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size = CastPyArg2Ints(kernel_size_obj, "max_pool2d_with_index", 1);
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "max_pool2d_with_index", 2);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "max_pool2d_with_index", 3);
    PyObject* global_pooling_obj = PyTuple_GET_ITEM(args, 4);
    bool global_pooling = CastPyArg2Boolean(global_pooling_obj, "max_pool2d_with_index", 4);
    PyObject* adaptive_obj = PyTuple_GET_ITEM(args, 5);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "max_pool2d_with_index", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::max_pool2d_with_index_final_state_dygraph_function(x,kernel_size,strides,paddings,global_pooling,adaptive)) out = ::max_pool2d_with_index_final_state_dygraph_function(x,kernel_size,strides,paddings,global_pooling,adaptive);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_max_pool3d_with_index(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("max_pool3d_with_index pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: max_pool3d_with_index";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("max_pool3d_with_index", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size = CastPyArg2Ints(kernel_size_obj, "max_pool3d_with_index", 1);
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "max_pool3d_with_index", 2);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "max_pool3d_with_index", 3);
    PyObject* global_pooling_obj = PyTuple_GET_ITEM(args, 4);
    bool global_pooling = CastPyArg2Boolean(global_pooling_obj, "max_pool3d_with_index", 4);
    PyObject* adaptive_obj = PyTuple_GET_ITEM(args, 5);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "max_pool3d_with_index", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::max_pool3d_with_index_final_state_dygraph_function(x,kernel_size,strides,paddings,global_pooling,adaptive)) out = ::max_pool3d_with_index_final_state_dygraph_function(x,kernel_size,strides,paddings,global_pooling,adaptive);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_maximum(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("maximum pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::maximum_final_state_dygraph_function(x,y)) out = ::maximum_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_maxout(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("maxout pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: maxout";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("maxout", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* groups_obj = PyTuple_GET_ITEM(args, 1);
    int groups = CastPyArg2Int(groups_obj, "maxout", 1);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "maxout", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::maxout_final_state_dygraph_function(x,groups,axis)) out = ::maxout_final_state_dygraph_function(x,groups,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_mean(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("mean pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: mean";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("mean", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "mean", 1);
    PyObject* keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "mean", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::mean_final_state_dygraph_function(x,dims,keep_dim)) out = ::mean_final_state_dygraph_function(x,dims,keep_dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_mean_all(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("mean_all pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::mean_all_final_state_dygraph_function(x)) out = ::mean_all_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_meshgrid(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("meshgrid pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::meshgrid_final_state_dygraph_function(inputs)) out = ::meshgrid_final_state_dygraph_function(inputs);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_min(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("min pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: min";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("min", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "min", 1);
    PyObject* keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "min", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::min_final_state_dygraph_function(x,dims,keep_dim)) out = ::min_final_state_dygraph_function(x,dims,keep_dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_minimum(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("minimum pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::minimum_final_state_dygraph_function(x,y)) out = ::minimum_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_mish(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("mish pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: mish";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("mish", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* lambda_obj = PyTuple_GET_ITEM(args, 1);
    float lambda = CastPyArg2Float(lambda_obj, "mish", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::mish_final_state_dygraph_function(x,lambda)) out = ::mish_final_state_dygraph_function(x,lambda);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_mode(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("mode pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: mode";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("mode", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "mode", 1);
    PyObject* keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "mode", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::mode_final_state_dygraph_function(x,axis,keepdim)) out = ::mode_final_state_dygraph_function(x,axis,keepdim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_modulo(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("modulo pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::modulo_final_state_dygraph_function(x,y)) out = ::modulo_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_momentum_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("momentum_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: momentum_";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("momentum_", "param", args, 0, false);
    auto grad = GetTensorFromArgs("momentum_", "grad", args, 1, false);
    auto velocity = GetTensorFromArgs("momentum_", "velocity", args, 2, false);
    auto learning_rate = GetTensorFromArgs("momentum_", "learning_rate", args, 3, false);
    auto master_param = GetOptionalTensorFromArgs("momentum_", "master_param", args, 4, true);

    // Parse Attributes if needed
    PyObject* mu_obj = PyTuple_GET_ITEM(args, 5);
    float mu = CastPyArg2Float(mu_obj, "momentum_", 5);
    PyObject* use_nesterov_obj = PyTuple_GET_ITEM(args, 6);
    bool use_nesterov = CastPyArg2Boolean(use_nesterov_obj, "momentum_", 6);
    PyObject* regularization_method_obj = PyTuple_GET_ITEM(args, 7);
    std::string regularization_method = CastPyArg2String(regularization_method_obj, "momentum_", 7);
    PyObject* regularization_coeff_obj = PyTuple_GET_ITEM(args, 8);
    float regularization_coeff = CastPyArg2Float(regularization_coeff_obj, "momentum_", 8);
    PyObject* multi_precision_obj = PyTuple_GET_ITEM(args, 9);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "momentum_", 9);
    PyObject* rescale_grad_obj = PyTuple_GET_ITEM(args, 10);
    float rescale_grad = CastPyArg2Float(rescale_grad_obj, "momentum_", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::momentum__final_state_dygraph_function(param,grad,velocity,learning_rate,master_param,mu,use_nesterov,regularization_method,regularization_coeff,multi_precision,rescale_grad)) out = ::momentum__final_state_dygraph_function(param,grad,velocity,learning_rate,master_param,mu,use_nesterov,regularization_method,regularization_coeff,multi_precision,rescale_grad);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;

    inplace_var_idx_map[1] = 2;

    inplace_var_idx_map[2] = 4;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_multi_dot(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("multi_dot pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::multi_dot_final_state_dygraph_function(x)) out = ::multi_dot_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_multiclass_nms3(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("multiclass_nms3 pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: multiclass_nms3";

    // Get EagerTensors from args
    auto bboxes = GetTensorFromArgs("multiclass_nms3", "bboxes", args, 0, false);
    auto scores = GetTensorFromArgs("multiclass_nms3", "scores", args, 1, false);
    auto rois_num = GetOptionalTensorFromArgs("multiclass_nms3", "rois_num", args, 2, true);

    // Parse Attributes if needed
    PyObject* score_threshold_obj = PyTuple_GET_ITEM(args, 3);
    float score_threshold = CastPyArg2Float(score_threshold_obj, "multiclass_nms3", 3);
    PyObject* nms_top_k_obj = PyTuple_GET_ITEM(args, 4);
    int nms_top_k = CastPyArg2Int(nms_top_k_obj, "multiclass_nms3", 4);
    PyObject* keep_top_k_obj = PyTuple_GET_ITEM(args, 5);
    int keep_top_k = CastPyArg2Int(keep_top_k_obj, "multiclass_nms3", 5);
    PyObject* nms_threshold_obj = PyTuple_GET_ITEM(args, 6);
    float nms_threshold = CastPyArg2Float(nms_threshold_obj, "multiclass_nms3", 6);
    PyObject* normalized_obj = PyTuple_GET_ITEM(args, 7);
    bool normalized = CastPyArg2Boolean(normalized_obj, "multiclass_nms3", 7);
    PyObject* nms_eta_obj = PyTuple_GET_ITEM(args, 8);
    float nms_eta = CastPyArg2Float(nms_eta_obj, "multiclass_nms3", 8);
    PyObject* background_label_obj = PyTuple_GET_ITEM(args, 9);
    int background_label = CastPyArg2Int(background_label_obj, "multiclass_nms3", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::multiclass_nms3_final_state_dygraph_function(bboxes,scores,rois_num,score_threshold,nms_top_k,keep_top_k,nms_threshold,normalized,nms_eta,background_label)) out = ::multiclass_nms3_final_state_dygraph_function(bboxes,scores,rois_num,score_threshold,nms_top_k,keep_top_k,nms_threshold,normalized,nms_eta,background_label);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_multinomial(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("multinomial pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: multinomial";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("multinomial", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* num_samples_obj = PyTuple_GET_ITEM(args, 1);
    int num_samples = CastPyArg2Int(num_samples_obj, "multinomial", 1);
    PyObject* replacement_obj = PyTuple_GET_ITEM(args, 2);
    bool replacement = CastPyArg2Boolean(replacement_obj, "multinomial", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::multinomial_final_state_dygraph_function(x,num_samples,replacement)) out = ::multinomial_final_state_dygraph_function(x,num_samples,replacement);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_multiplex(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("multiplex pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::multiplex_final_state_dygraph_function(ins,ids)) out = ::multiplex_final_state_dygraph_function(ins,ids);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_multiply(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("multiply pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::multiply_final_state_dygraph_function(x,y)) out = ::multiply_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_nearest_interp(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("nearest_interp pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: nearest_interp";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("nearest_interp", "x", args, 0, false);
    auto out_size = GetOptionalTensorFromArgs("nearest_interp", "out_size", args, 1, true);
    auto size_tensor = GetOptionalTensorListFromArgs("nearest_interp", "size_tensor", args, 2, true);
    auto scale_tensor = GetOptionalTensorFromArgs("nearest_interp", "scale_tensor", args, 3, true);

    // Parse Attributes if needed
    PyObject* data_layout_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_layout = CastPyArg2String(data_layout_obj, "nearest_interp", 4);
    PyObject* out_d_obj = PyTuple_GET_ITEM(args, 5);
    int out_d = CastPyArg2Int(out_d_obj, "nearest_interp", 5);
    PyObject* out_h_obj = PyTuple_GET_ITEM(args, 6);
    int out_h = CastPyArg2Int(out_h_obj, "nearest_interp", 6);
    PyObject* out_w_obj = PyTuple_GET_ITEM(args, 7);
    int out_w = CastPyArg2Int(out_w_obj, "nearest_interp", 7);
    PyObject* scale_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<float> scale = CastPyArg2Floats(scale_obj, "nearest_interp", 8);
    PyObject* interp_method_obj = PyTuple_GET_ITEM(args, 9);
    std::string interp_method = CastPyArg2String(interp_method_obj, "nearest_interp", 9);
    PyObject* align_corners_obj = PyTuple_GET_ITEM(args, 10);
    bool align_corners = CastPyArg2Boolean(align_corners_obj, "nearest_interp", 10);
    PyObject* align_mode_obj = PyTuple_GET_ITEM(args, 11);
    int align_mode = CastPyArg2Int(align_mode_obj, "nearest_interp", 11);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::nearest_interp_final_state_dygraph_function(x,out_size,size_tensor,scale_tensor,data_layout,out_d,out_h,out_w,scale,interp_method,align_corners,align_mode)) out = ::nearest_interp_final_state_dygraph_function(x,out_size,size_tensor,scale_tensor,data_layout,out_d,out_h,out_w,scale,interp_method,align_corners,align_mode);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_nll_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("nll_loss pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: nll_loss";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("nll_loss", "input", args, 0, false);
    auto label = GetTensorFromArgs("nll_loss", "label", args, 1, false);
    auto weight = GetOptionalTensorFromArgs("nll_loss", "weight", args, 2, true);

    // Parse Attributes if needed
    PyObject* ignore_index_obj = PyTuple_GET_ITEM(args, 3);
    int64_t ignore_index = CastPyArg2Long(ignore_index_obj, "nll_loss", 3);
    PyObject* reduction_obj = PyTuple_GET_ITEM(args, 4);
    std::string reduction = CastPyArg2String(reduction_obj, "nll_loss", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::nll_loss_final_state_dygraph_function(input,label,weight,ignore_index,reduction)) out = ::nll_loss_final_state_dygraph_function(input,label,weight,ignore_index,reduction);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_nms(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("nms pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: nms";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("nms", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "nms", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::nms_final_state_dygraph_function(x,threshold)) out = ::nms_final_state_dygraph_function(x,threshold);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("norm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("norm", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "norm", 1);
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 2);
    float epsilon = CastPyArg2Float(epsilon_obj, "norm", 2);
    PyObject* is_test_obj = PyTuple_GET_ITEM(args, 3);
    bool is_test = CastPyArg2Boolean(is_test_obj, "norm", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::norm_final_state_dygraph_function(x,axis,epsilon,is_test)) out = ::norm_final_state_dygraph_function(x,axis,epsilon,is_test);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_not_equal(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("not_equal pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: not_equal";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("not_equal", "x", args, 0, false);
    auto y = GetTensorFromArgs("not_equal", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "not_equal", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::not_equal_final_state_dygraph_function(x,y,axis)) out = ::not_equal_final_state_dygraph_function(x,y,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_one_hot(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("one_hot pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: one_hot";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("one_hot", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* num_classes_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar num_classes = CastPyArg2Scalar(num_classes_obj, "one_hot", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::one_hot_final_state_dygraph_function(x,num_classes)) out = ::one_hot_final_state_dygraph_function(x,num_classes);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_ones(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("ones pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: ones";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 0);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "ones", 0);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "ones", 1);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 2);
    paddle::Place place = CastPyArg2Place(place_obj, "ones", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::ones_final_state_dygraph_function(shape,dtype,place)) out = ::ones_final_state_dygraph_function(shape,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_ones_like(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("ones_like pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: ones_like";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("ones_like", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "ones_like", 1);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 2);
    paddle::Place place = CastPyArg2Place(place_obj, "ones_like", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::ones_like_final_state_dygraph_function(x,dtype,place)) out = ::ones_like_final_state_dygraph_function(x,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_p_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("p_norm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: p_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("p_norm", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* porder_obj = PyTuple_GET_ITEM(args, 1);
    float porder = CastPyArg2Float(porder_obj, "p_norm", 1);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "p_norm", 2);
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 3);
    float epsilon = CastPyArg2Float(epsilon_obj, "p_norm", 3);
    PyObject* keepdim_obj = PyTuple_GET_ITEM(args, 4);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "p_norm", 4);
    PyObject* asvector_obj = PyTuple_GET_ITEM(args, 5);
    bool asvector = CastPyArg2Boolean(asvector_obj, "p_norm", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::p_norm_final_state_dygraph_function(x,porder,axis,epsilon,keepdim,asvector)) out = ::p_norm_final_state_dygraph_function(x,porder,axis,epsilon,keepdim,asvector);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_pad(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("pad pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pad";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pad", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "pad", 1);
    PyObject* pad_value_obj = PyTuple_GET_ITEM(args, 2);
    float pad_value = CastPyArg2Float(pad_value_obj, "pad", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pad_final_state_dygraph_function(x,paddings,pad_value)) out = ::pad_final_state_dygraph_function(x,paddings,pad_value);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_pad3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("pad3d pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pad3d";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pad3d", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray paddings = CastPyArg2IntArray(paddings_obj, "pad3d", 1);
    PyObject* mode_obj = PyTuple_GET_ITEM(args, 2);
    std::string mode = CastPyArg2String(mode_obj, "pad3d", 2);
    PyObject* pad_value_obj = PyTuple_GET_ITEM(args, 3);
    float pad_value = CastPyArg2Float(pad_value_obj, "pad3d", 3);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_format = CastPyArg2String(data_format_obj, "pad3d", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pad3d_final_state_dygraph_function(x,paddings,mode,pad_value,data_format)) out = ::pad3d_final_state_dygraph_function(x,paddings,mode,pad_value,data_format);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_pixel_shuffle(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("pixel_shuffle pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pixel_shuffle";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pixel_shuffle", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* upscale_factor_obj = PyTuple_GET_ITEM(args, 1);
    int upscale_factor = CastPyArg2Int(upscale_factor_obj, "pixel_shuffle", 1);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 2);
    std::string data_format = CastPyArg2String(data_format_obj, "pixel_shuffle", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pixel_shuffle_final_state_dygraph_function(x,upscale_factor,data_format)) out = ::pixel_shuffle_final_state_dygraph_function(x,upscale_factor,data_format);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_pool2d(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("pool2d pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pool2d";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pool2d", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size = CastPyArg2Ints(kernel_size_obj, "pool2d", 1);
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "pool2d", 2);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "pool2d", 3);
    PyObject* ceil_mode_obj = PyTuple_GET_ITEM(args, 4);
    bool ceil_mode = CastPyArg2Boolean(ceil_mode_obj, "pool2d", 4);
    PyObject* exclusive_obj = PyTuple_GET_ITEM(args, 5);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "pool2d", 5);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 6);
    std::string data_format = CastPyArg2String(data_format_obj, "pool2d", 6);
    PyObject* pooling_type_obj = PyTuple_GET_ITEM(args, 7);
    std::string pooling_type = CastPyArg2String(pooling_type_obj, "pool2d", 7);
    PyObject* global_pooling_obj = PyTuple_GET_ITEM(args, 8);
    bool global_pooling = CastPyArg2Boolean(global_pooling_obj, "pool2d", 8);
    PyObject* adaptive_obj = PyTuple_GET_ITEM(args, 9);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "pool2d", 9);
    PyObject* padding_algorithm_obj = PyTuple_GET_ITEM(args, 10);
    std::string padding_algorithm = CastPyArg2String(padding_algorithm_obj, "pool2d", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pool2d_final_state_dygraph_function(x,kernel_size,strides,paddings,ceil_mode,exclusive,data_format,pooling_type,global_pooling,adaptive,padding_algorithm)) out = ::pool2d_final_state_dygraph_function(x,kernel_size,strides,paddings,ceil_mode,exclusive,data_format,pooling_type,global_pooling,adaptive,padding_algorithm);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_pool2d_gpudnn_unused(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("pool2d_gpudnn_unused pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pool2d_gpudnn_unused";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pool2d_gpudnn_unused", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size = CastPyArg2Ints(kernel_size_obj, "pool2d_gpudnn_unused", 1);
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "pool2d_gpudnn_unused", 2);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "pool2d_gpudnn_unused", 3);
    PyObject* ceil_mode_obj = PyTuple_GET_ITEM(args, 4);
    bool ceil_mode = CastPyArg2Boolean(ceil_mode_obj, "pool2d_gpudnn_unused", 4);
    PyObject* exclusive_obj = PyTuple_GET_ITEM(args, 5);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "pool2d_gpudnn_unused", 5);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 6);
    std::string data_format = CastPyArg2String(data_format_obj, "pool2d_gpudnn_unused", 6);
    PyObject* pooling_type_obj = PyTuple_GET_ITEM(args, 7);
    std::string pooling_type = CastPyArg2String(pooling_type_obj, "pool2d_gpudnn_unused", 7);
    PyObject* global_pooling_obj = PyTuple_GET_ITEM(args, 8);
    bool global_pooling = CastPyArg2Boolean(global_pooling_obj, "pool2d_gpudnn_unused", 8);
    PyObject* adaptive_obj = PyTuple_GET_ITEM(args, 9);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "pool2d_gpudnn_unused", 9);
    PyObject* padding_algorithm_obj = PyTuple_GET_ITEM(args, 10);
    std::string padding_algorithm = CastPyArg2String(padding_algorithm_obj, "pool2d_gpudnn_unused", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pool2d_gpudnn_unused_final_state_dygraph_function(x,kernel_size,strides,paddings,ceil_mode,exclusive,data_format,pooling_type,global_pooling,adaptive,padding_algorithm)) out = ::pool2d_gpudnn_unused_final_state_dygraph_function(x,kernel_size,strides,paddings,ceil_mode,exclusive,data_format,pooling_type,global_pooling,adaptive,padding_algorithm);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_pool3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("pool3d pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pool3d";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pool3d", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* kernel_size_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_size = CastPyArg2Ints(kernel_size_obj, "pool3d", 1);
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "pool3d", 2);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "pool3d", 3);
    PyObject* ceil_mode_obj = PyTuple_GET_ITEM(args, 4);
    bool ceil_mode = CastPyArg2Boolean(ceil_mode_obj, "pool3d", 4);
    PyObject* exclusive_obj = PyTuple_GET_ITEM(args, 5);
    bool exclusive = CastPyArg2Boolean(exclusive_obj, "pool3d", 5);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 6);
    std::string data_format = CastPyArg2String(data_format_obj, "pool3d", 6);
    PyObject* pooling_type_obj = PyTuple_GET_ITEM(args, 7);
    std::string pooling_type = CastPyArg2String(pooling_type_obj, "pool3d", 7);
    PyObject* global_pooling_obj = PyTuple_GET_ITEM(args, 8);
    bool global_pooling = CastPyArg2Boolean(global_pooling_obj, "pool3d", 8);
    PyObject* adaptive_obj = PyTuple_GET_ITEM(args, 9);
    bool adaptive = CastPyArg2Boolean(adaptive_obj, "pool3d", 9);
    PyObject* padding_algorithm_obj = PyTuple_GET_ITEM(args, 10);
    std::string padding_algorithm = CastPyArg2String(padding_algorithm_obj, "pool3d", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pool3d_final_state_dygraph_function(x,kernel_size,strides,paddings,ceil_mode,exclusive,data_format,pooling_type,global_pooling,adaptive,padding_algorithm)) out = ::pool3d_final_state_dygraph_function(x,kernel_size,strides,paddings,ceil_mode,exclusive,data_format,pooling_type,global_pooling,adaptive,padding_algorithm);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_pow(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("pow pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pow";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pow", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* s_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar s = CastPyArg2Scalar(s_obj, "pow", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::pow_final_state_dygraph_function(x,s)) out = ::pow_final_state_dygraph_function(x,s);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_prelu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("prelu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: prelu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("prelu", "x", args, 0, false);
    auto alpha = GetTensorFromArgs("prelu", "alpha", args, 1, false);

    // Parse Attributes if needed
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 2);
    std::string data_format = CastPyArg2String(data_format_obj, "prelu", 2);
    PyObject* mode_obj = PyTuple_GET_ITEM(args, 3);
    std::string mode = CastPyArg2String(mode_obj, "prelu", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::prelu_final_state_dygraph_function(x,alpha,data_format,mode)) out = ::prelu_final_state_dygraph_function(x,alpha,data_format,mode);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_prior_box(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("prior_box pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: prior_box";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("prior_box", "input", args, 0, false);
    auto image = GetTensorFromArgs("prior_box", "image", args, 1, false);

    // Parse Attributes if needed
    PyObject* min_sizes_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<float> min_sizes = CastPyArg2Floats(min_sizes_obj, "prior_box", 2);
    PyObject* aspect_ratios_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<float> aspect_ratios = CastPyArg2Floats(aspect_ratios_obj, "prior_box", 3);
    PyObject* variances_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<float> variances = CastPyArg2Floats(variances_obj, "prior_box", 4);
    PyObject* max_sizes_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<float> max_sizes = CastPyArg2Floats(max_sizes_obj, "prior_box", 5);
    PyObject* flip_obj = PyTuple_GET_ITEM(args, 6);
    bool flip = CastPyArg2Boolean(flip_obj, "prior_box", 6);
    PyObject* clip_obj = PyTuple_GET_ITEM(args, 7);
    bool clip = CastPyArg2Boolean(clip_obj, "prior_box", 7);
    PyObject* step_w_obj = PyTuple_GET_ITEM(args, 8);
    float step_w = CastPyArg2Float(step_w_obj, "prior_box", 8);
    PyObject* step_h_obj = PyTuple_GET_ITEM(args, 9);
    float step_h = CastPyArg2Float(step_h_obj, "prior_box", 9);
    PyObject* offset_obj = PyTuple_GET_ITEM(args, 10);
    float offset = CastPyArg2Float(offset_obj, "prior_box", 10);
    PyObject* min_max_aspect_ratios_order_obj = PyTuple_GET_ITEM(args, 11);
    bool min_max_aspect_ratios_order = CastPyArg2Boolean(min_max_aspect_ratios_order_obj, "prior_box", 11);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::prior_box_final_state_dygraph_function(input,image,min_sizes,aspect_ratios,variances,max_sizes,flip,clip,step_w,step_h,offset,min_max_aspect_ratios_order)) out = ::prior_box_final_state_dygraph_function(input,image,min_sizes,aspect_ratios,variances,max_sizes,flip,clip,step_w,step_h,offset,min_max_aspect_ratios_order);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_psroi_pool(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("psroi_pool pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: psroi_pool";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("psroi_pool", "x", args, 0, false);
    auto boxes = GetTensorFromArgs("psroi_pool", "boxes", args, 1, false);
    auto boxes_num = GetOptionalTensorFromArgs("psroi_pool", "boxes_num", args, 2, true);

    // Parse Attributes if needed
    PyObject* pooled_height_obj = PyTuple_GET_ITEM(args, 3);
    int pooled_height = CastPyArg2Int(pooled_height_obj, "psroi_pool", 3);
    PyObject* pooled_width_obj = PyTuple_GET_ITEM(args, 4);
    int pooled_width = CastPyArg2Int(pooled_width_obj, "psroi_pool", 4);
    PyObject* output_channels_obj = PyTuple_GET_ITEM(args, 5);
    int output_channels = CastPyArg2Int(output_channels_obj, "psroi_pool", 5);
    PyObject* spatial_scale_obj = PyTuple_GET_ITEM(args, 6);
    float spatial_scale = CastPyArg2Float(spatial_scale_obj, "psroi_pool", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::psroi_pool_final_state_dygraph_function(x,boxes,boxes_num,pooled_height,pooled_width,output_channels,spatial_scale)) out = ::psroi_pool_final_state_dygraph_function(x,boxes,boxes_num,pooled_height,pooled_width,output_channels,spatial_scale);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_put_along_axis(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("put_along_axis pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: put_along_axis";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("put_along_axis", "x", args, 0, false);
    auto index = GetTensorFromArgs("put_along_axis", "index", args, 1, false);
    auto value = GetTensorFromArgs("put_along_axis", "value", args, 2, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 3);
    int axis = CastPyArg2Int(axis_obj, "put_along_axis", 3);
    PyObject* reduce_obj = PyTuple_GET_ITEM(args, 4);
    std::string reduce = CastPyArg2String(reduce_obj, "put_along_axis", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::put_along_axis_final_state_dygraph_function(x,index,value,axis,reduce)) out = ::put_along_axis_final_state_dygraph_function(x,index,value,axis,reduce);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_put_along_axis_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("put_along_axis pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: put_along_axis_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("put_along_axis", "x", args, 0, false);
    auto index = GetTensorFromArgs("put_along_axis", "index", args, 1, false);
    auto value = GetTensorFromArgs("put_along_axis", "value", args, 2, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 3);
    int axis = CastPyArg2Int(axis_obj, "put_along_axis", 3);
    PyObject* reduce_obj = PyTuple_GET_ITEM(args, 4);
    std::string reduce = CastPyArg2String(reduce_obj, "put_along_axis", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::put_along_axis__final_state_dygraph_function(x,index,value,axis,reduce)) out = ::put_along_axis__final_state_dygraph_function(x,index,value,axis,reduce);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_qr(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("qr pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: qr";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("qr", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* mode_obj = PyTuple_GET_ITEM(args, 1);
    std::string mode = CastPyArg2String(mode_obj, "qr", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::qr_final_state_dygraph_function(x,mode)) out = ::qr_final_state_dygraph_function(x,mode);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_randint(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("randint pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: randint";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject* low_obj = PyTuple_GET_ITEM(args, 0);
    int low = CastPyArg2Int(low_obj, "randint", 0);
    PyObject* high_obj = PyTuple_GET_ITEM(args, 1);
    int high = CastPyArg2Int(high_obj, "randint", 1);
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "randint", 2);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "randint", 3);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 4);
    paddle::Place place = CastPyArg2Place(place_obj, "randint", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::randint_final_state_dygraph_function(low,high,shape,dtype,place)) out = ::randint_final_state_dygraph_function(low,high,shape,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_randperm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("randperm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: randperm";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject* n_obj = PyTuple_GET_ITEM(args, 0);
    int n = CastPyArg2Int(n_obj, "randperm", 0);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "randperm", 1);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 2);
    paddle::Place place = CastPyArg2Place(place_obj, "randperm", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::randperm_final_state_dygraph_function(n,dtype,place)) out = ::randperm_final_state_dygraph_function(n,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_real(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("real pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::real_final_state_dygraph_function(x)) out = ::real_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_reciprocal(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("reciprocal pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::reciprocal_final_state_dygraph_function(x)) out = ::reciprocal_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_reciprocal_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("reciprocal pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: reciprocal_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("reciprocal", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::reciprocal__final_state_dygraph_function(x)) out = ::reciprocal__final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_reduce_prod(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("reduce_prod pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: reduce_prod";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("reduce_prod", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "reduce_prod", 1);
    PyObject* keep_dim_obj = PyTuple_GET_ITEM(args, 2);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "reduce_prod", 2);
    PyObject* reduce_all_obj = PyTuple_GET_ITEM(args, 3);
    bool reduce_all = CastPyArg2Boolean(reduce_all_obj, "reduce_prod", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::reduce_prod_final_state_dygraph_function(x,dims,keep_dim,reduce_all)) out = ::reduce_prod_final_state_dygraph_function(x,dims,keep_dim,reduce_all);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("relu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::relu_final_state_dygraph_function(x)) out = ::relu_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_relu_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("relu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::relu__final_state_dygraph_function(x)) out = ::relu__final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_relu6(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("relu6 pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: relu6";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("relu6", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "relu6", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::relu6_final_state_dygraph_function(x,threshold)) out = ::relu6_final_state_dygraph_function(x,threshold);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_renorm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("renorm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: renorm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("renorm", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* p_obj = PyTuple_GET_ITEM(args, 1);
    float p = CastPyArg2Float(p_obj, "renorm", 1);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "renorm", 2);
    PyObject* max_norm_obj = PyTuple_GET_ITEM(args, 3);
    float max_norm = CastPyArg2Float(max_norm_obj, "renorm", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::renorm_final_state_dygraph_function(x,p,axis,max_norm)) out = ::renorm_final_state_dygraph_function(x,p,axis,max_norm);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_repeat_interleave(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("repeat_interleave pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: repeat_interleave";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("repeat_interleave", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* repeats_obj = PyTuple_GET_ITEM(args, 1);
    int repeats = CastPyArg2Int(repeats_obj, "repeat_interleave", 1);
    PyObject* dim_obj = PyTuple_GET_ITEM(args, 2);
    int dim = CastPyArg2Int(dim_obj, "repeat_interleave", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::repeat_interleave_final_state_dygraph_function(x,repeats,dim)) out = ::repeat_interleave_final_state_dygraph_function(x,repeats,dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_repeat_interleave_with_tensor_index(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("repeat_interleave_with_tensor_index pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: repeat_interleave_with_tensor_index";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("repeat_interleave_with_tensor_index", "x", args, 0, false);
    auto repeats = GetTensorFromArgs("repeat_interleave_with_tensor_index", "repeats", args, 1, false);

    // Parse Attributes if needed
    PyObject* dim_obj = PyTuple_GET_ITEM(args, 2);
    int dim = CastPyArg2Int(dim_obj, "repeat_interleave_with_tensor_index", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::repeat_interleave_with_tensor_index_final_state_dygraph_function(x,repeats,dim)) out = ::repeat_interleave_with_tensor_index_final_state_dygraph_function(x,repeats,dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_reshape(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("reshape pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: reshape";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("reshape", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "reshape", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::reshape_final_state_dygraph_function(x,shape)) out = ::reshape_final_state_dygraph_function(x,shape);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_reshape_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("reshape pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: reshape_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("reshape", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "reshape", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::reshape__final_state_dygraph_function(x,shape)) out = ::reshape__final_state_dygraph_function(x,shape);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_reverse(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("reverse pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: reverse";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("reverse", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> axis = CastPyArg2Ints(axis_obj, "reverse", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::reverse_final_state_dygraph_function(x,axis)) out = ::reverse_final_state_dygraph_function(x,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_reverse_array(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("reverse_array pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: reverse_array";

    // Get EagerTensors from args
    auto x = GetTensorListFromArgs("reverse_array", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> axis = CastPyArg2Ints(axis_obj, "reverse_array", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::reverse_array_final_state_dygraph_function(x,axis)) out = ::reverse_array_final_state_dygraph_function(x,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_rmsprop_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("rmsprop_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: rmsprop_";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("rmsprop_", "param", args, 0, false);
    auto mean_square = GetTensorFromArgs("rmsprop_", "mean_square", args, 1, false);
    auto grad = GetTensorFromArgs("rmsprop_", "grad", args, 2, false);
    auto moment = GetTensorFromArgs("rmsprop_", "moment", args, 3, false);
    auto learning_rate = GetTensorFromArgs("rmsprop_", "learning_rate", args, 4, false);
    auto mean_grad = GetTensorFromArgs("rmsprop_", "mean_grad", args, 5, false);

    // Parse Attributes if needed
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 6);
    float epsilon = CastPyArg2Float(epsilon_obj, "rmsprop_", 6);
    PyObject* decay_obj = PyTuple_GET_ITEM(args, 7);
    float decay = CastPyArg2Float(decay_obj, "rmsprop_", 7);
    PyObject* momentum_obj = PyTuple_GET_ITEM(args, 8);
    float momentum = CastPyArg2Float(momentum_obj, "rmsprop_", 8);
    PyObject* centered_obj = PyTuple_GET_ITEM(args, 9);
    bool centered = CastPyArg2Boolean(centered_obj, "rmsprop_", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::rmsprop__final_state_dygraph_function(param,mean_square,grad,moment,learning_rate,mean_grad,epsilon,decay,momentum,centered)) out = ::rmsprop__final_state_dygraph_function(param,mean_square,grad,moment,learning_rate,mean_grad,epsilon,decay,momentum,centered);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;

    inplace_var_idx_map[1] = 3;

    inplace_var_idx_map[2] = 1;

    inplace_var_idx_map[3] = 5;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_roi_align(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("roi_align pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: roi_align";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("roi_align", "x", args, 0, false);
    auto boxes = GetTensorFromArgs("roi_align", "boxes", args, 1, false);
    auto boxes_num = GetOptionalTensorFromArgs("roi_align", "boxes_num", args, 2, true);

    // Parse Attributes if needed
    PyObject* pooled_height_obj = PyTuple_GET_ITEM(args, 3);
    int pooled_height = CastPyArg2Int(pooled_height_obj, "roi_align", 3);
    PyObject* pooled_width_obj = PyTuple_GET_ITEM(args, 4);
    int pooled_width = CastPyArg2Int(pooled_width_obj, "roi_align", 4);
    PyObject* spatial_scale_obj = PyTuple_GET_ITEM(args, 5);
    float spatial_scale = CastPyArg2Float(spatial_scale_obj, "roi_align", 5);
    PyObject* sampling_ratio_obj = PyTuple_GET_ITEM(args, 6);
    int sampling_ratio = CastPyArg2Int(sampling_ratio_obj, "roi_align", 6);
    PyObject* aligned_obj = PyTuple_GET_ITEM(args, 7);
    bool aligned = CastPyArg2Boolean(aligned_obj, "roi_align", 7);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::roi_align_final_state_dygraph_function(x,boxes,boxes_num,pooled_height,pooled_width,spatial_scale,sampling_ratio,aligned)) out = ::roi_align_final_state_dygraph_function(x,boxes,boxes_num,pooled_height,pooled_width,spatial_scale,sampling_ratio,aligned);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_roi_pool(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("roi_pool pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: roi_pool";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("roi_pool", "x", args, 0, false);
    auto boxes = GetTensorFromArgs("roi_pool", "boxes", args, 1, false);
    auto boxes_num = GetOptionalTensorFromArgs("roi_pool", "boxes_num", args, 2, true);

    // Parse Attributes if needed
    PyObject* pooled_height_obj = PyTuple_GET_ITEM(args, 3);
    int pooled_height = CastPyArg2Int(pooled_height_obj, "roi_pool", 3);
    PyObject* pooled_width_obj = PyTuple_GET_ITEM(args, 4);
    int pooled_width = CastPyArg2Int(pooled_width_obj, "roi_pool", 4);
    PyObject* spatial_scale_obj = PyTuple_GET_ITEM(args, 5);
    float spatial_scale = CastPyArg2Float(spatial_scale_obj, "roi_pool", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::roi_pool_final_state_dygraph_function(x,boxes,boxes_num,pooled_height,pooled_width,spatial_scale)) out = ::roi_pool_final_state_dygraph_function(x,boxes,boxes_num,pooled_height,pooled_width,spatial_scale);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_roll(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("roll pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: roll";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("roll", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* shifts_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray shifts = CastPyArg2IntArray(shifts_obj, "roll", 1);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int64_t> axis = CastPyArg2Longs(axis_obj, "roll", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::roll_final_state_dygraph_function(x,shifts,axis)) out = ::roll_final_state_dygraph_function(x,shifts,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_round(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("round pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::round_final_state_dygraph_function(x)) out = ::round_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_round_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("round pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: round_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("round", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::round__final_state_dygraph_function(x)) out = ::round__final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_rsqrt(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("rsqrt pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::rsqrt_final_state_dygraph_function(x)) out = ::rsqrt_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_rsqrt_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("rsqrt pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::rsqrt__final_state_dygraph_function(x)) out = ::rsqrt__final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_scale(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("scale pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: scale";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("scale", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* scale_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar scale = CastPyArg2Scalar(scale_obj, "scale", 1);
    PyObject* bias_obj = PyTuple_GET_ITEM(args, 2);
    float bias = CastPyArg2Float(bias_obj, "scale", 2);
    PyObject* bias_after_scale_obj = PyTuple_GET_ITEM(args, 3);
    bool bias_after_scale = CastPyArg2Boolean(bias_after_scale_obj, "scale", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::scale_final_state_dygraph_function(x,scale,bias,bias_after_scale)) out = ::scale_final_state_dygraph_function(x,scale,bias,bias_after_scale);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_scale_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("scale pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: scale_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("scale", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* scale_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar scale = CastPyArg2Scalar(scale_obj, "scale", 1);
    PyObject* bias_obj = PyTuple_GET_ITEM(args, 2);
    float bias = CastPyArg2Float(bias_obj, "scale", 2);
    PyObject* bias_after_scale_obj = PyTuple_GET_ITEM(args, 3);
    bool bias_after_scale = CastPyArg2Boolean(bias_after_scale_obj, "scale", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::scale__final_state_dygraph_function(x,scale,bias,bias_after_scale)) out = ::scale__final_state_dygraph_function(x,scale,bias,bias_after_scale);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_scatter(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("scatter pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: scatter";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("scatter", "x", args, 0, false);
    auto index = GetTensorFromArgs("scatter", "index", args, 1, false);
    auto updates = GetTensorFromArgs("scatter", "updates", args, 2, false);

    // Parse Attributes if needed
    PyObject* overwrite_obj = PyTuple_GET_ITEM(args, 3);
    bool overwrite = CastPyArg2Boolean(overwrite_obj, "scatter", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::scatter_final_state_dygraph_function(x,index,updates,overwrite)) out = ::scatter_final_state_dygraph_function(x,index,updates,overwrite);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_scatter_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("scatter pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: scatter_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("scatter", "x", args, 0, false);
    auto index = GetTensorFromArgs("scatter", "index", args, 1, false);
    auto updates = GetTensorFromArgs("scatter", "updates", args, 2, false);

    // Parse Attributes if needed
    PyObject* overwrite_obj = PyTuple_GET_ITEM(args, 3);
    bool overwrite = CastPyArg2Boolean(overwrite_obj, "scatter", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::scatter__final_state_dygraph_function(x,index,updates,overwrite)) out = ::scatter__final_state_dygraph_function(x,index,updates,overwrite);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_scatter_nd_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("scatter_nd_add pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: scatter_nd_add";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("scatter_nd_add", "x", args, 0, false);
    auto index = GetTensorFromArgs("scatter_nd_add", "index", args, 1, false);
    auto updates = GetTensorFromArgs("scatter_nd_add", "updates", args, 2, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::scatter_nd_add_final_state_dygraph_function(x,index,updates)) out = ::scatter_nd_add_final_state_dygraph_function(x,index,updates);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_searchsorted(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("searchsorted pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: searchsorted";

    // Get EagerTensors from args
    auto sorted_sequence = GetTensorFromArgs("searchsorted", "sorted_sequence", args, 0, false);
    auto value = GetTensorFromArgs("searchsorted", "value", args, 1, false);

    // Parse Attributes if needed
    PyObject* out_int32_obj = PyTuple_GET_ITEM(args, 2);
    bool out_int32 = CastPyArg2Boolean(out_int32_obj, "searchsorted", 2);
    PyObject* right_obj = PyTuple_GET_ITEM(args, 3);
    bool right = CastPyArg2Boolean(right_obj, "searchsorted", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::searchsorted_final_state_dygraph_function(sorted_sequence,value,out_int32,right)) out = ::searchsorted_final_state_dygraph_function(sorted_sequence,value,out_int32,right);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_segment_pool(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("segment_pool pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: segment_pool";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("segment_pool", "x", args, 0, false);
    auto segment_ids = GetTensorFromArgs("segment_pool", "segment_ids", args, 1, false);

    // Parse Attributes if needed
    PyObject* pooltype_obj = PyTuple_GET_ITEM(args, 2);
    std::string pooltype = CastPyArg2String(pooltype_obj, "segment_pool", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::segment_pool_final_state_dygraph_function(x,segment_ids,pooltype)) out = ::segment_pool_final_state_dygraph_function(x,segment_ids,pooltype);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_selu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("selu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: selu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("selu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* scale_obj = PyTuple_GET_ITEM(args, 1);
    float scale = CastPyArg2Float(scale_obj, "selu", 1);
    PyObject* alpha_obj = PyTuple_GET_ITEM(args, 2);
    float alpha = CastPyArg2Float(alpha_obj, "selu", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::selu_final_state_dygraph_function(x,scale,alpha)) out = ::selu_final_state_dygraph_function(x,scale,alpha);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_sgd_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sgd_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sgd_";

    // Get EagerTensors from args
    auto param = GetTensorFromArgs("sgd_", "param", args, 0, false);
    auto learning_rate = GetTensorFromArgs("sgd_", "learning_rate", args, 1, false);
    auto grad = GetTensorFromArgs("sgd_", "grad", args, 2, false);
    auto master_param = GetOptionalTensorFromArgs("sgd_", "master_param", args, 3, true);

    // Parse Attributes if needed
    PyObject* multi_precision_obj = PyTuple_GET_ITEM(args, 4);
    bool multi_precision = CastPyArg2Boolean(multi_precision_obj, "sgd_", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sgd__final_state_dygraph_function(param,learning_rate,grad,master_param,multi_precision)) out = ::sgd__final_state_dygraph_function(param,learning_rate,grad,master_param,multi_precision);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;

    inplace_var_idx_map[1] = 3;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_shape(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("shape pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::shape_final_state_dygraph_function(input)) out = ::shape_final_state_dygraph_function(input);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_shard_index(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("shard_index pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: shard_index";

    // Get EagerTensors from args
    auto in = GetTensorFromArgs("shard_index", "in", args, 0, false);

    // Parse Attributes if needed
    PyObject* index_num_obj = PyTuple_GET_ITEM(args, 1);
    int index_num = CastPyArg2Int(index_num_obj, "shard_index", 1);
    PyObject* nshards_obj = PyTuple_GET_ITEM(args, 2);
    int nshards = CastPyArg2Int(nshards_obj, "shard_index", 2);
    PyObject* shard_id_obj = PyTuple_GET_ITEM(args, 3);
    int shard_id = CastPyArg2Int(shard_id_obj, "shard_index", 3);
    PyObject* ignore_value_obj = PyTuple_GET_ITEM(args, 4);
    int ignore_value = CastPyArg2Int(ignore_value_obj, "shard_index", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::shard_index_final_state_dygraph_function(in,index_num,nshards,shard_id,ignore_value)) out = ::shard_index_final_state_dygraph_function(in,index_num,nshards,shard_id,ignore_value);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_sigmoid(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sigmoid pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sigmoid_final_state_dygraph_function(x)) out = ::sigmoid_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_sigmoid_cross_entropy_with_logits(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sigmoid_cross_entropy_with_logits pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sigmoid_cross_entropy_with_logits";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("sigmoid_cross_entropy_with_logits", "x", args, 0, false);
    auto label = GetTensorFromArgs("sigmoid_cross_entropy_with_logits", "label", args, 1, false);

    // Parse Attributes if needed
    PyObject* normalize_obj = PyTuple_GET_ITEM(args, 2);
    bool normalize = CastPyArg2Boolean(normalize_obj, "sigmoid_cross_entropy_with_logits", 2);
    PyObject* ignore_index_obj = PyTuple_GET_ITEM(args, 3);
    int ignore_index = CastPyArg2Int(ignore_index_obj, "sigmoid_cross_entropy_with_logits", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sigmoid_cross_entropy_with_logits_final_state_dygraph_function(x,label,normalize,ignore_index)) out = ::sigmoid_cross_entropy_with_logits_final_state_dygraph_function(x,label,normalize,ignore_index);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_sign(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sign pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sign_final_state_dygraph_function(x)) out = ::sign_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_silu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("silu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::silu_final_state_dygraph_function(x)) out = ::silu_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_sin(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sin pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sin_final_state_dygraph_function(x)) out = ::sin_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_sinh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sinh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sinh_final_state_dygraph_function(x)) out = ::sinh_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_size(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("size pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::size_final_state_dygraph_function(x)) out = ::size_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_slice(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("slice pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: slice";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("slice", "input", args, 0, false);

    // Parse Attributes if needed
    PyObject* axes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> axes = CastPyArg2Longs(axes_obj, "slice", 1);
    PyObject* starts_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::IntArray starts = CastPyArg2IntArray(starts_obj, "slice", 2);
    PyObject* ends_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::IntArray ends = CastPyArg2IntArray(ends_obj, "slice", 3);
    PyObject* infer_flags_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int64_t> infer_flags = CastPyArg2Longs(infer_flags_obj, "slice", 4);
    PyObject* decrease_axis_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int64_t> decrease_axis = CastPyArg2Longs(decrease_axis_obj, "slice", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::slice_final_state_dygraph_function(input,axes,starts,ends,infer_flags,decrease_axis)) out = ::slice_final_state_dygraph_function(input,axes,starts,ends,infer_flags,decrease_axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_slogdet(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("slogdet pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: slogdet";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("slogdet", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::slogdet_final_state_dygraph_function(x)) out = ::slogdet_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_soft_shrink(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("soft_shrink pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: soft_shrink";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("soft_shrink", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* lambda_obj = PyTuple_GET_ITEM(args, 1);
    float lambda = CastPyArg2Float(lambda_obj, "soft_shrink", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::soft_shrink_final_state_dygraph_function(x,lambda)) out = ::soft_shrink_final_state_dygraph_function(x,lambda);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_softmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("softmax pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: softmax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("softmax", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "softmax", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::softmax_final_state_dygraph_function(x,axis)) out = ::softmax_final_state_dygraph_function(x,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_softmax_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("softmax pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: softmax_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("softmax", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "softmax", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::softmax__final_state_dygraph_function(x,axis)) out = ::softmax__final_state_dygraph_function(x,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_softplus(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("softplus pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: softplus";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("softplus", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* beta_obj = PyTuple_GET_ITEM(args, 1);
    float beta = CastPyArg2Float(beta_obj, "softplus", 1);
    PyObject* threshold_obj = PyTuple_GET_ITEM(args, 2);
    float threshold = CastPyArg2Float(threshold_obj, "softplus", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::softplus_final_state_dygraph_function(x,beta,threshold)) out = ::softplus_final_state_dygraph_function(x,beta,threshold);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_softsign(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("softsign pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: softsign";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("softsign", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::softsign_final_state_dygraph_function(x)) out = ::softsign_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_spectral_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("spectral_norm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: spectral_norm";

    // Get EagerTensors from args
    auto weight = GetTensorFromArgs("spectral_norm", "weight", args, 0, false);
    auto u = GetTensorFromArgs("spectral_norm", "u", args, 1, false);
    auto v = GetTensorFromArgs("spectral_norm", "v", args, 2, false);

    // Parse Attributes if needed
    PyObject* dim_obj = PyTuple_GET_ITEM(args, 3);
    int dim = CastPyArg2Int(dim_obj, "spectral_norm", 3);
    PyObject* power_iters_obj = PyTuple_GET_ITEM(args, 4);
    int power_iters = CastPyArg2Int(power_iters_obj, "spectral_norm", 4);
    PyObject* eps_obj = PyTuple_GET_ITEM(args, 5);
    float eps = CastPyArg2Float(eps_obj, "spectral_norm", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::spectral_norm_final_state_dygraph_function(weight,u,v,dim,power_iters,eps)) out = ::spectral_norm_final_state_dygraph_function(weight,u,v,dim,power_iters,eps);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_split(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("split pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: split";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("split", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* num_or_sections_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray num_or_sections = CastPyArg2IntArray(num_or_sections_obj, "split", 1);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::Scalar axis = CastPyArg2Scalar(axis_obj, "split", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::split_final_state_dygraph_function(x,num_or_sections,axis)) out = ::split_final_state_dygraph_function(x,num_or_sections,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_sqrt(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sqrt pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sqrt_final_state_dygraph_function(x)) out = ::sqrt_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_sqrt_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sqrt pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sqrt_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("sqrt", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sqrt__final_state_dygraph_function(x)) out = ::sqrt__final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_square(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("square pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::square_final_state_dygraph_function(x)) out = ::square_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_squared_l2_norm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("squared_l2_norm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: squared_l2_norm";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("squared_l2_norm", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::squared_l2_norm_final_state_dygraph_function(x)) out = ::squared_l2_norm_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_squeeze(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("squeeze pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: squeeze";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("squeeze", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axes_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray axes = CastPyArg2IntArray(axes_obj, "squeeze", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::squeeze_final_state_dygraph_function(x,axes)) out = ::squeeze_final_state_dygraph_function(x,axes);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_squeeze_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("squeeze pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: squeeze_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("squeeze", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axes_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray axes = CastPyArg2IntArray(axes_obj, "squeeze", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::squeeze__final_state_dygraph_function(x,axes)) out = ::squeeze__final_state_dygraph_function(x,axes);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_stack(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("stack pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: stack";

    // Get EagerTensors from args
    auto x = GetTensorListFromArgs("stack", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "stack", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::stack_final_state_dygraph_function(x,axis)) out = ::stack_final_state_dygraph_function(x,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_strided_slice(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("strided_slice pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: strided_slice";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("strided_slice", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> axes = CastPyArg2Ints(axes_obj, "strided_slice", 1);
    PyObject* starts_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::IntArray starts = CastPyArg2IntArray(starts_obj, "strided_slice", 2);
    PyObject* ends_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::IntArray ends = CastPyArg2IntArray(ends_obj, "strided_slice", 3);
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 4);
    paddle::experimental::IntArray strides = CastPyArg2IntArray(strides_obj, "strided_slice", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::strided_slice_final_state_dygraph_function(x,axes,starts,ends,strides)) out = ::strided_slice_final_state_dygraph_function(x,axes,starts,ends,strides);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_subtract(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("subtract pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::subtract_final_state_dygraph_function(x,y)) out = ::subtract_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_subtract_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("subtract pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: subtract_";

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::subtract__final_state_dygraph_function(x,y)) out = ::subtract__final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_sum(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sum pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sum";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("sum", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dims_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int64_t> dims = CastPyArg2Longs(dims_obj, "sum", 1);
    PyObject* out_dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType out_dtype = CastPyArg2DataType(out_dtype_obj, "sum", 2);
    PyObject* keep_dim_obj = PyTuple_GET_ITEM(args, 3);
    bool keep_dim = CastPyArg2Boolean(keep_dim_obj, "sum", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sum_final_state_dygraph_function(x,dims,out_dtype,keep_dim)) out = ::sum_final_state_dygraph_function(x,dims,out_dtype,keep_dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_svd(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("svd pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: svd";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("svd", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* full_metrices_obj = PyTuple_GET_ITEM(args, 1);
    bool full_metrices = CastPyArg2Boolean(full_metrices_obj, "svd", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::svd_final_state_dygraph_function(x,full_metrices)) out = ::svd_final_state_dygraph_function(x,full_metrices);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_swish(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("swish pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: swish";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("swish", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* beta_obj = PyTuple_GET_ITEM(args, 1);
    float beta = CastPyArg2Float(beta_obj, "swish", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::swish_final_state_dygraph_function(x,beta)) out = ::swish_final_state_dygraph_function(x,beta);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_sync_batch_norm_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sync_batch_norm_ pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: sync_batch_norm_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("sync_batch_norm_", "x", args, 0, false);
    auto scale = GetTensorFromArgs("sync_batch_norm_", "scale", args, 1, false);
    auto bias = GetTensorFromArgs("sync_batch_norm_", "bias", args, 2, false);
    auto mean = GetTensorFromArgs("sync_batch_norm_", "mean", args, 3, false);
    auto variance = GetTensorFromArgs("sync_batch_norm_", "variance", args, 4, false);

    // Parse Attributes if needed
    PyObject* momentum_obj = PyTuple_GET_ITEM(args, 5);
    float momentum = CastPyArg2Float(momentum_obj, "sync_batch_norm_", 5);
    PyObject* epsilon_obj = PyTuple_GET_ITEM(args, 6);
    float epsilon = CastPyArg2Float(epsilon_obj, "sync_batch_norm_", 6);
    PyObject* data_layout_obj = PyTuple_GET_ITEM(args, 7);
    std::string data_layout = CastPyArg2String(data_layout_obj, "sync_batch_norm_", 7);
    PyObject* is_test_obj = PyTuple_GET_ITEM(args, 8);
    bool is_test = CastPyArg2Boolean(is_test_obj, "sync_batch_norm_", 8);
    PyObject* use_global_stats_obj = PyTuple_GET_ITEM(args, 9);
    bool use_global_stats = CastPyArg2Boolean(use_global_stats_obj, "sync_batch_norm_", 9);
    PyObject* trainable_statistics_obj = PyTuple_GET_ITEM(args, 10);
    bool trainable_statistics = CastPyArg2Boolean(trainable_statistics_obj, "sync_batch_norm_", 10);
    PyObject* fuse_with_relu_obj = PyTuple_GET_ITEM(args, 11);
    bool fuse_with_relu = CastPyArg2Boolean(fuse_with_relu_obj, "sync_batch_norm_", 11);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sync_batch_norm__final_state_dygraph_function(x,scale,bias,mean,variance,momentum,epsilon,data_layout,is_test,use_global_stats,trainable_statistics,fuse_with_relu)) out = ::sync_batch_norm__final_state_dygraph_function(x,scale,bias,mean,variance,momentum,epsilon,data_layout,is_test,use_global_stats,trainable_statistics,fuse_with_relu);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[1] = 3;

    inplace_var_idx_map[2] = 4;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_take_along_axis(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("take_along_axis pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: take_along_axis";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("take_along_axis", "x", args, 0, false);
    auto index = GetTensorFromArgs("take_along_axis", "index", args, 1, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "take_along_axis", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::take_along_axis_final_state_dygraph_function(x,index,axis)) out = ::take_along_axis_final_state_dygraph_function(x,index,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_tan(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("tan pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::tan_final_state_dygraph_function(x)) out = ::tan_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_tanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("tanh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::tanh_final_state_dygraph_function(x)) out = ::tanh_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_tanh_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("tanh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: tanh_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("tanh", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::tanh__final_state_dygraph_function(x)) out = ::tanh__final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_tanh_shrink(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("tanh_shrink pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::tanh_shrink_final_state_dygraph_function(x)) out = ::tanh_shrink_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_temporal_shift(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("temporal_shift pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: temporal_shift";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("temporal_shift", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* seg_num_obj = PyTuple_GET_ITEM(args, 1);
    int seg_num = CastPyArg2Int(seg_num_obj, "temporal_shift", 1);
    PyObject* shift_ratio_obj = PyTuple_GET_ITEM(args, 2);
    float shift_ratio = CastPyArg2Float(shift_ratio_obj, "temporal_shift", 2);
    PyObject* data_format_str_obj = PyTuple_GET_ITEM(args, 3);
    std::string data_format_str = CastPyArg2String(data_format_str_obj, "temporal_shift", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::temporal_shift_final_state_dygraph_function(x,seg_num,shift_ratio,data_format_str)) out = ::temporal_shift_final_state_dygraph_function(x,seg_num,shift_ratio,data_format_str);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_thresholded_relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("thresholded_relu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: thresholded_relu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("thresholded_relu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "thresholded_relu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::thresholded_relu_final_state_dygraph_function(x,threshold)) out = ::thresholded_relu_final_state_dygraph_function(x,threshold);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_tile(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("tile pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: tile";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("tile", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* repeat_times_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray repeat_times = CastPyArg2IntArray(repeat_times_obj, "tile", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::tile_final_state_dygraph_function(x,repeat_times)) out = ::tile_final_state_dygraph_function(x,repeat_times);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_top_k(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("top_k pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: top_k";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("top_k", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* k_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar k = CastPyArg2Scalar(k_obj, "top_k", 1);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "top_k", 2);
    PyObject* largest_obj = PyTuple_GET_ITEM(args, 3);
    bool largest = CastPyArg2Boolean(largest_obj, "top_k", 3);
    PyObject* sorted_obj = PyTuple_GET_ITEM(args, 4);
    bool sorted = CastPyArg2Boolean(sorted_obj, "top_k", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::top_k_final_state_dygraph_function(x,k,axis,largest,sorted)) out = ::top_k_final_state_dygraph_function(x,k,axis,largest,sorted);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_transpose(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("transpose pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: transpose";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("transpose", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> axis = CastPyArg2Ints(axis_obj, "transpose", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::transpose_final_state_dygraph_function(x,axis)) out = ::transpose_final_state_dygraph_function(x,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_triangular_solve(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("triangular_solve pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: triangular_solve";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("triangular_solve", "x", args, 0, false);
    auto y = GetTensorFromArgs("triangular_solve", "y", args, 1, false);

    // Parse Attributes if needed
    PyObject* upper_obj = PyTuple_GET_ITEM(args, 2);
    bool upper = CastPyArg2Boolean(upper_obj, "triangular_solve", 2);
    PyObject* transpose_obj = PyTuple_GET_ITEM(args, 3);
    bool transpose = CastPyArg2Boolean(transpose_obj, "triangular_solve", 3);
    PyObject* unitriangular_obj = PyTuple_GET_ITEM(args, 4);
    bool unitriangular = CastPyArg2Boolean(unitriangular_obj, "triangular_solve", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::triangular_solve_final_state_dygraph_function(x,y,upper,transpose,unitriangular)) out = ::triangular_solve_final_state_dygraph_function(x,y,upper,transpose,unitriangular);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_tril_indices(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("tril_indices pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: tril_indices";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject* rows_obj = PyTuple_GET_ITEM(args, 0);
    int rows = CastPyArg2Int(rows_obj, "tril_indices", 0);
    PyObject* cols_obj = PyTuple_GET_ITEM(args, 1);
    int cols = CastPyArg2Int(cols_obj, "tril_indices", 1);
    PyObject* offset_obj = PyTuple_GET_ITEM(args, 2);
    int offset = CastPyArg2Int(offset_obj, "tril_indices", 2);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 3);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "tril_indices", 3);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 4);
    paddle::Place place = CastPyArg2Place(place_obj, "tril_indices", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::tril_indices_final_state_dygraph_function(rows,cols,offset,dtype,place)) out = ::tril_indices_final_state_dygraph_function(rows,cols,offset,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_tril_triu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("tril_triu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: tril_triu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("tril_triu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* diagonal_obj = PyTuple_GET_ITEM(args, 1);
    int diagonal = CastPyArg2Int(diagonal_obj, "tril_triu", 1);
    PyObject* lower_obj = PyTuple_GET_ITEM(args, 2);
    bool lower = CastPyArg2Boolean(lower_obj, "tril_triu", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::tril_triu_final_state_dygraph_function(x,diagonal,lower)) out = ::tril_triu_final_state_dygraph_function(x,diagonal,lower);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_trilinear_interp(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("trilinear_interp pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: trilinear_interp";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("trilinear_interp", "x", args, 0, false);
    auto out_size = GetOptionalTensorFromArgs("trilinear_interp", "out_size", args, 1, true);
    auto size_tensor = GetOptionalTensorListFromArgs("trilinear_interp", "size_tensor", args, 2, true);
    auto scale_tensor = GetOptionalTensorFromArgs("trilinear_interp", "scale_tensor", args, 3, true);

    // Parse Attributes if needed
    PyObject* data_layout_obj = PyTuple_GET_ITEM(args, 4);
    std::string data_layout = CastPyArg2String(data_layout_obj, "trilinear_interp", 4);
    PyObject* out_d_obj = PyTuple_GET_ITEM(args, 5);
    int out_d = CastPyArg2Int(out_d_obj, "trilinear_interp", 5);
    PyObject* out_h_obj = PyTuple_GET_ITEM(args, 6);
    int out_h = CastPyArg2Int(out_h_obj, "trilinear_interp", 6);
    PyObject* out_w_obj = PyTuple_GET_ITEM(args, 7);
    int out_w = CastPyArg2Int(out_w_obj, "trilinear_interp", 7);
    PyObject* scale_obj = PyTuple_GET_ITEM(args, 8);
    std::vector<float> scale = CastPyArg2Floats(scale_obj, "trilinear_interp", 8);
    PyObject* interp_method_obj = PyTuple_GET_ITEM(args, 9);
    std::string interp_method = CastPyArg2String(interp_method_obj, "trilinear_interp", 9);
    PyObject* align_corners_obj = PyTuple_GET_ITEM(args, 10);
    bool align_corners = CastPyArg2Boolean(align_corners_obj, "trilinear_interp", 10);
    PyObject* align_mode_obj = PyTuple_GET_ITEM(args, 11);
    int align_mode = CastPyArg2Int(align_mode_obj, "trilinear_interp", 11);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::trilinear_interp_final_state_dygraph_function(x,out_size,size_tensor,scale_tensor,data_layout,out_d,out_h,out_w,scale,interp_method,align_corners,align_mode)) out = ::trilinear_interp_final_state_dygraph_function(x,out_size,size_tensor,scale_tensor,data_layout,out_d,out_h,out_w,scale,interp_method,align_corners,align_mode);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_truncated_gaussian_random(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("truncated_gaussian_random pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: truncated_gaussian_random";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 0);
    std::vector<int> shape = CastPyArg2Ints(shape_obj, "truncated_gaussian_random", 0);
    PyObject* mean_obj = PyTuple_GET_ITEM(args, 1);
    float mean = CastPyArg2Float(mean_obj, "truncated_gaussian_random", 1);
    PyObject* std_obj = PyTuple_GET_ITEM(args, 2);
    float std = CastPyArg2Float(std_obj, "truncated_gaussian_random", 2);
    PyObject* seed_obj = PyTuple_GET_ITEM(args, 3);
    int seed = CastPyArg2Int(seed_obj, "truncated_gaussian_random", 3);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 4);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "truncated_gaussian_random", 4);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 5);
    paddle::Place place = CastPyArg2Place(place_obj, "truncated_gaussian_random", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::truncated_gaussian_random_final_state_dygraph_function(shape,mean,std,seed,dtype,place)) out = ::truncated_gaussian_random_final_state_dygraph_function(shape,mean,std,seed,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_unbind(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("unbind pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unbind";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("unbind", "input", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "unbind", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::unbind_final_state_dygraph_function(input,axis)) out = ::unbind_final_state_dygraph_function(input,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_unfold(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("unfold pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unfold";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("unfold", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* kernel_sizes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_sizes = CastPyArg2Ints(kernel_sizes_obj, "unfold", 1);
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "unfold", 2);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "unfold", 3);
    PyObject* dilations_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "unfold", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::unfold_final_state_dygraph_function(x,kernel_sizes,strides,paddings,dilations)) out = ::unfold_final_state_dygraph_function(x,kernel_sizes,strides,paddings,dilations);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_uniform_random(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("uniform_random pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: uniform_random";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 0);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "uniform_random", 0);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "uniform_random", 1);
    PyObject* min_obj = PyTuple_GET_ITEM(args, 2);
    float min = CastPyArg2Float(min_obj, "uniform_random", 2);
    PyObject* max_obj = PyTuple_GET_ITEM(args, 3);
    float max = CastPyArg2Float(max_obj, "uniform_random", 3);
    PyObject* seed_obj = PyTuple_GET_ITEM(args, 4);
    int seed = CastPyArg2Int(seed_obj, "uniform_random", 4);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 5);
    paddle::Place place = CastPyArg2Place(place_obj, "uniform_random", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::uniform_random_final_state_dygraph_function(shape,dtype,min,max,seed,place)) out = ::uniform_random_final_state_dygraph_function(shape,dtype,min,max,seed,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_unique(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("unique pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unique";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("unique", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* return_index_obj = PyTuple_GET_ITEM(args, 1);
    bool return_index = CastPyArg2Boolean(return_index_obj, "unique", 1);
    PyObject* return_inverse_obj = PyTuple_GET_ITEM(args, 2);
    bool return_inverse = CastPyArg2Boolean(return_inverse_obj, "unique", 2);
    PyObject* return_counts_obj = PyTuple_GET_ITEM(args, 3);
    bool return_counts = CastPyArg2Boolean(return_counts_obj, "unique", 3);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> axis = CastPyArg2Ints(axis_obj, "unique", 4);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 5);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "unique", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::unique_final_state_dygraph_function(x,return_index,return_inverse,return_counts,axis,dtype)) out = ::unique_final_state_dygraph_function(x,return_index,return_inverse,return_counts,axis,dtype);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_unique_consecutive(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("unique_consecutive pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unique_consecutive";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("unique_consecutive", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* return_inverse_obj = PyTuple_GET_ITEM(args, 1);
    bool return_inverse = CastPyArg2Boolean(return_inverse_obj, "unique_consecutive", 1);
    PyObject* return_counts_obj = PyTuple_GET_ITEM(args, 2);
    bool return_counts = CastPyArg2Boolean(return_counts_obj, "unique_consecutive", 2);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> axis = CastPyArg2Ints(axis_obj, "unique_consecutive", 3);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 4);
    int dtype = CastPyArg2Int(dtype_obj, "unique_consecutive", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::unique_consecutive_final_state_dygraph_function(x,return_inverse,return_counts,axis,dtype)) out = ::unique_consecutive_final_state_dygraph_function(x,return_inverse,return_counts,axis,dtype);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_unsqueeze(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("unsqueeze pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unsqueeze";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("unsqueeze", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray axis = CastPyArg2IntArray(axis_obj, "unsqueeze", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::unsqueeze_final_state_dygraph_function(x,axis)) out = ::unsqueeze_final_state_dygraph_function(x,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_unsqueeze_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("unsqueeze pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unsqueeze_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("unsqueeze", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray axis = CastPyArg2IntArray(axis_obj, "unsqueeze", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::unsqueeze__final_state_dygraph_function(x,axis)) out = ::unsqueeze__final_state_dygraph_function(x,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_unstack(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("unstack pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unstack";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("unstack", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "unstack", 1);
    PyObject* num_obj = PyTuple_GET_ITEM(args, 2);
    int num = CastPyArg2Int(num_obj, "unstack", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::unstack_final_state_dygraph_function(x,axis,num)) out = ::unstack_final_state_dygraph_function(x,axis,num);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_viterbi_decode(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("viterbi_decode pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: viterbi_decode";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("viterbi_decode", "input", args, 0, false);
    auto transition = GetTensorFromArgs("viterbi_decode", "transition", args, 1, false);
    auto length = GetTensorFromArgs("viterbi_decode", "length", args, 2, false);

    // Parse Attributes if needed
    PyObject* include_bos_eos_tag_obj = PyTuple_GET_ITEM(args, 3);
    bool include_bos_eos_tag = CastPyArg2Boolean(include_bos_eos_tag_obj, "viterbi_decode", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::viterbi_decode_final_state_dygraph_function(input,transition,length,include_bos_eos_tag)) out = ::viterbi_decode_final_state_dygraph_function(input,transition,length,include_bos_eos_tag);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_warpctc(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("warpctc pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: warpctc";

    // Get EagerTensors from args
    auto logits = GetTensorFromArgs("warpctc", "logits", args, 0, false);
    auto label = GetTensorFromArgs("warpctc", "label", args, 1, false);
    auto logits_length = GetOptionalTensorFromArgs("warpctc", "logits_length", args, 2, true);
    auto labels_length = GetOptionalTensorFromArgs("warpctc", "labels_length", args, 3, true);

    // Parse Attributes if needed
    PyObject* blank_obj = PyTuple_GET_ITEM(args, 4);
    int blank = CastPyArg2Int(blank_obj, "warpctc", 4);
    PyObject* norm_by_times_obj = PyTuple_GET_ITEM(args, 5);
    bool norm_by_times = CastPyArg2Boolean(norm_by_times_obj, "warpctc", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::warpctc_final_state_dygraph_function(logits,label,logits_length,labels_length,blank,norm_by_times)) out = ::warpctc_final_state_dygraph_function(logits,label,logits_length,labels_length,blank,norm_by_times);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_where(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("where pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::where_final_state_dygraph_function(condition,x,y)) out = ::where_final_state_dygraph_function(condition,x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_where_index(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("where_index pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: where_index";

    // Get EagerTensors from args
    auto condition = GetTensorFromArgs("where_index", "condition", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::where_index_final_state_dygraph_function(condition)) out = ::where_index_final_state_dygraph_function(condition);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_yolo_box(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("yolo_box pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: yolo_box";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("yolo_box", "x", args, 0, false);
    auto img_size = GetTensorFromArgs("yolo_box", "img_size", args, 1, false);

    // Parse Attributes if needed
    PyObject* anchors_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> anchors = CastPyArg2Ints(anchors_obj, "yolo_box", 2);
    PyObject* class_num_obj = PyTuple_GET_ITEM(args, 3);
    int class_num = CastPyArg2Int(class_num_obj, "yolo_box", 3);
    PyObject* conf_thresh_obj = PyTuple_GET_ITEM(args, 4);
    float conf_thresh = CastPyArg2Float(conf_thresh_obj, "yolo_box", 4);
    PyObject* downsample_ratio_obj = PyTuple_GET_ITEM(args, 5);
    int downsample_ratio = CastPyArg2Int(downsample_ratio_obj, "yolo_box", 5);
    PyObject* clip_bbox_obj = PyTuple_GET_ITEM(args, 6);
    bool clip_bbox = CastPyArg2Boolean(clip_bbox_obj, "yolo_box", 6);
    PyObject* scale_x_y_obj = PyTuple_GET_ITEM(args, 7);
    float scale_x_y = CastPyArg2Float(scale_x_y_obj, "yolo_box", 7);
    PyObject* iou_aware_obj = PyTuple_GET_ITEM(args, 8);
    bool iou_aware = CastPyArg2Boolean(iou_aware_obj, "yolo_box", 8);
    PyObject* iou_aware_factor_obj = PyTuple_GET_ITEM(args, 9);
    float iou_aware_factor = CastPyArg2Float(iou_aware_factor_obj, "yolo_box", 9);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::yolo_box_final_state_dygraph_function(x,img_size,anchors,class_num,conf_thresh,downsample_ratio,clip_bbox,scale_x_y,iou_aware,iou_aware_factor)) out = ::yolo_box_final_state_dygraph_function(x,img_size,anchors,class_num,conf_thresh,downsample_ratio,clip_bbox,scale_x_y,iou_aware,iou_aware_factor);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_yolov3_loss(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("yolov3_loss pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: yolov3_loss";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("yolov3_loss", "x", args, 0, false);
    auto gt_box = GetTensorFromArgs("yolov3_loss", "gt_box", args, 1, false);
    auto gt_label = GetTensorFromArgs("yolov3_loss", "gt_label", args, 2, false);
    auto gt_score = GetOptionalTensorFromArgs("yolov3_loss", "gt_score", args, 3, true);

    // Parse Attributes if needed
    PyObject* anchors_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> anchors = CastPyArg2Ints(anchors_obj, "yolov3_loss", 4);
    PyObject* anchor_mask_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> anchor_mask = CastPyArg2Ints(anchor_mask_obj, "yolov3_loss", 5);
    PyObject* class_num_obj = PyTuple_GET_ITEM(args, 6);
    int class_num = CastPyArg2Int(class_num_obj, "yolov3_loss", 6);
    PyObject* ignore_thresh_obj = PyTuple_GET_ITEM(args, 7);
    float ignore_thresh = CastPyArg2Float(ignore_thresh_obj, "yolov3_loss", 7);
    PyObject* downsample_ratio_obj = PyTuple_GET_ITEM(args, 8);
    int downsample_ratio = CastPyArg2Int(downsample_ratio_obj, "yolov3_loss", 8);
    PyObject* use_label_smooth_obj = PyTuple_GET_ITEM(args, 9);
    bool use_label_smooth = CastPyArg2Boolean(use_label_smooth_obj, "yolov3_loss", 9);
    PyObject* scale_x_y_obj = PyTuple_GET_ITEM(args, 10);
    float scale_x_y = CastPyArg2Float(scale_x_y_obj, "yolov3_loss", 10);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::yolov3_loss_final_state_dygraph_function(x,gt_box,gt_label,gt_score,anchors,anchor_mask,class_num,ignore_thresh,downsample_ratio,use_label_smooth,scale_x_y)) out = ::yolov3_loss_final_state_dygraph_function(x,gt_box,gt_label,gt_score,anchors,anchor_mask,class_num,ignore_thresh,downsample_ratio,use_label_smooth,scale_x_y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_zeros(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("zeros pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: zeros";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 0);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "zeros", 0);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "zeros", 1);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 2);
    paddle::Place place = CastPyArg2Place(place_obj, "zeros", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::zeros_final_state_dygraph_function(shape,dtype,place)) out = ::zeros_final_state_dygraph_function(shape,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_zeros_like(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("zeros_like pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: zeros_like";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("zeros_like", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "zeros_like", 1);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 2);
    paddle::Place place = CastPyArg2Place(place_obj, "zeros_like", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::zeros_like_final_state_dygraph_function(x,dtype,place)) out = ::zeros_like_final_state_dygraph_function(x,dtype,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_broadcast_tensors(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("broadcast_tensors pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: broadcast_tensors";

    // Get EagerTensors from args
    auto x = GetTensorListFromArgs("broadcast_tensors", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::broadcast_tensors_final_state_dygraph_function(x)) out = ::broadcast_tensors_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_dirichlet(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("dirichlet pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: dirichlet";

    // Get EagerTensors from args
    auto alpha = GetTensorFromArgs("dirichlet", "alpha", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::dirichlet_final_state_dygraph_function(alpha)) out = ::dirichlet_final_state_dygraph_function(alpha);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_eig(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("eig pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: eig";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("eig", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::eig_final_state_dygraph_function(x)) out = ::eig_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_fold(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fold pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fold";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("fold", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* output_sizes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> output_sizes = CastPyArg2Ints(output_sizes_obj, "fold", 1);
    PyObject* kernel_sizes_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> kernel_sizes = CastPyArg2Ints(kernel_sizes_obj, "fold", 2);
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "fold", 3);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "fold", 4);
    PyObject* dilations_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "fold", 5);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::fold_final_state_dygraph_function(x,output_sizes,kernel_sizes,strides,paddings,dilations)) out = ::fold_final_state_dygraph_function(x,output_sizes,kernel_sizes,strides,paddings,dilations);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_overlap_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("overlap_add pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: overlap_add";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("overlap_add", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* hop_length_obj = PyTuple_GET_ITEM(args, 1);
    int hop_length = CastPyArg2Int(hop_length_obj, "overlap_add", 1);
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 2);
    int axis = CastPyArg2Int(axis_obj, "overlap_add", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::overlap_add_final_state_dygraph_function(x,hop_length,axis)) out = ::overlap_add_final_state_dygraph_function(x,hop_length,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_uniform_random_inplace(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("uniform_random_inplace pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: uniform_random_inplace";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("uniform_random_inplace", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* min_obj = PyTuple_GET_ITEM(args, 1);
    float min = CastPyArg2Float(min_obj, "uniform_random_inplace", 1);
    PyObject* max_obj = PyTuple_GET_ITEM(args, 2);
    float max = CastPyArg2Float(max_obj, "uniform_random_inplace", 2);
    PyObject* seed_obj = PyTuple_GET_ITEM(args, 3);
    int seed = CastPyArg2Int(seed_obj, "uniform_random_inplace", 3);
    PyObject* diag_num_obj = PyTuple_GET_ITEM(args, 4);
    int diag_num = CastPyArg2Int(diag_num_obj, "uniform_random_inplace", 4);
    PyObject* diag_step_obj = PyTuple_GET_ITEM(args, 5);
    int diag_step = CastPyArg2Int(diag_step_obj, "uniform_random_inplace", 5);
    PyObject* diag_val_obj = PyTuple_GET_ITEM(args, 6);
    float diag_val = CastPyArg2Float(diag_val_obj, "uniform_random_inplace", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::uniform_random_inplace_final_state_dygraph_function(x,min,max,seed,diag_num,diag_step,diag_val)) out = ::uniform_random_inplace_final_state_dygraph_function(x,min,max,seed,diag_num,diag_step,diag_val);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * eager_final_state_api_uniform_random_inplace_(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("uniform_random_inplace pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: uniform_random_inplace_";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("uniform_random_inplace", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* min_obj = PyTuple_GET_ITEM(args, 1);
    float min = CastPyArg2Float(min_obj, "uniform_random_inplace", 1);
    PyObject* max_obj = PyTuple_GET_ITEM(args, 2);
    float max = CastPyArg2Float(max_obj, "uniform_random_inplace", 2);
    PyObject* seed_obj = PyTuple_GET_ITEM(args, 3);
    int seed = CastPyArg2Int(seed_obj, "uniform_random_inplace", 3);
    PyObject* diag_num_obj = PyTuple_GET_ITEM(args, 4);
    int diag_num = CastPyArg2Int(diag_num_obj, "uniform_random_inplace", 4);
    PyObject* diag_step_obj = PyTuple_GET_ITEM(args, 5);
    int diag_step = CastPyArg2Int(diag_step_obj, "uniform_random_inplace", 5);
    PyObject* diag_val_obj = PyTuple_GET_ITEM(args, 6);
    float diag_val = CastPyArg2Float(diag_val_obj, "uniform_random_inplace", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::uniform_random_inplace__final_state_dygraph_function(x,min,max,seed,diag_num,diag_step,diag_val)) out = ::uniform_random_inplace__final_state_dygraph_function(x,min,max,seed,diag_num,diag_step,diag_val);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    std::map<ssize_t, ssize_t> inplace_var_idx_map;
    inplace_var_idx_map[0] = 0;
    return ToPyObject(out, args, inplace_var_idx_map);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_unpool(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("unpool pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unpool";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("unpool", "x", args, 0, false);
    auto indices = GetTensorFromArgs("unpool", "indices", args, 1, false);

    // Parse Attributes if needed
    PyObject* ksize_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> ksize = CastPyArg2Ints(ksize_obj, "unpool", 2);
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "unpool", 3);
    PyObject* padding_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> padding = CastPyArg2Ints(padding_obj, "unpool", 4);
    PyObject* output_size_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> output_size = CastPyArg2Ints(output_size_obj, "unpool", 5);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 6);
    std::string data_format = CastPyArg2String(data_format_obj, "unpool", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::unpool_final_state_dygraph_function(x,indices,ksize,strides,padding,output_size,data_format)) out = ::unpool_final_state_dygraph_function(x,indices,ksize,strides,padding,output_size,data_format);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_unpool3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("unpool3d pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: unpool3d";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("unpool3d", "x", args, 0, false);
    auto indices = GetTensorFromArgs("unpool3d", "indices", args, 1, false);

    // Parse Attributes if needed
    PyObject* ksize_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> ksize = CastPyArg2Ints(ksize_obj, "unpool3d", 2);
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "unpool3d", 3);
    PyObject* padding_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> padding = CastPyArg2Ints(padding_obj, "unpool3d", 4);
    PyObject* output_size_obj = PyTuple_GET_ITEM(args, 5);
    std::vector<int> output_size = CastPyArg2Ints(output_size_obj, "unpool3d", 5);
    PyObject* data_format_obj = PyTuple_GET_ITEM(args, 6);
    std::string data_format = CastPyArg2String(data_format_obj, "unpool3d", 6);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::unpool3d_final_state_dygraph_function(x,indices,ksize,strides,padding,output_size,data_format)) out = ::unpool3d_final_state_dygraph_function(x,indices,ksize,strides,padding,output_size,data_format);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


namespace sparse {
    
static PyObject * eager_final_state_api_abs(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("abs pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::abs_final_state_dygraph_function(x)) out = ::sparse::abs_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_acos(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("acos pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::acos_final_state_dygraph_function(x)) out = ::sparse::acos_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_acosh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("acosh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::acosh_final_state_dygraph_function(x)) out = ::sparse::acosh_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("add pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::add_final_state_dygraph_function(x,y)) out = ::sparse::add_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_asin(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("asin pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::asin_final_state_dygraph_function(x)) out = ::sparse::asin_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_asinh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("asinh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::asinh_final_state_dygraph_function(x)) out = ::sparse::asinh_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_atan(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("atan pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::atan_final_state_dygraph_function(x)) out = ::sparse::atan_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_atanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("atanh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::atanh_final_state_dygraph_function(x)) out = ::sparse::atanh_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_cast(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("cast pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: cast";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("cast", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* index_dtype_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::DataType index_dtype = CastPyArg2DataType(index_dtype_obj, "cast", 1);
    PyObject* value_dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType value_dtype = CastPyArg2DataType(value_dtype_obj, "cast", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::cast_final_state_dygraph_function(x,index_dtype,value_dtype)) out = ::sparse::cast_final_state_dygraph_function(x,index_dtype,value_dtype);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_conv3d(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("conv3d pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: conv3d";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("conv3d", "x", args, 0, false);
    auto kernel = GetTensorFromArgs("conv3d", "kernel", args, 1, false);

    // Parse Attributes if needed
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "conv3d", 2);
    PyObject* dilations_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "conv3d", 3);
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "conv3d", 4);
    PyObject* groups_obj = PyTuple_GET_ITEM(args, 5);
    int groups = CastPyArg2Int(groups_obj, "conv3d", 5);
    PyObject* subm_obj = PyTuple_GET_ITEM(args, 6);
    bool subm = CastPyArg2Boolean(subm_obj, "conv3d", 6);
    PyObject* key_obj = PyTuple_GET_ITEM(args, 7);
    std::string key = CastPyArg2String(key_obj, "conv3d", 7);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::conv3d_final_state_dygraph_function(x,kernel,paddings,dilations,strides,groups,subm,key)) out = ::sparse::conv3d_final_state_dygraph_function(x,kernel,paddings,dilations,strides,groups,subm,key);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_coo_to_dense(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("coo_to_dense pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::coo_to_dense_final_state_dygraph_function(x)) out = ::sparse::coo_to_dense_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_create_sparse_coo_tensor(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("create_sparse_coo_tensor pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: create_sparse_coo_tensor";

    // Get EagerTensors from args
    auto values = GetTensorFromArgs("create_sparse_coo_tensor", "values", args, 0, false);
    auto indices = GetTensorFromArgs("create_sparse_coo_tensor", "indices", args, 1, false);

    // Parse Attributes if needed
    PyObject* dense_shape_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::IntArray dense_shape = CastPyArg2IntArray(dense_shape_obj, "create_sparse_coo_tensor", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::create_sparse_coo_tensor_final_state_dygraph_function(values,indices,dense_shape)) out = ::sparse::create_sparse_coo_tensor_final_state_dygraph_function(values,indices,dense_shape);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_dense_to_coo(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("dense_to_coo pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: dense_to_coo";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("dense_to_coo", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* sparse_dim_obj = PyTuple_GET_ITEM(args, 1);
    int64_t sparse_dim = CastPyArg2Long(sparse_dim_obj, "dense_to_coo", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::dense_to_coo_final_state_dygraph_function(x,sparse_dim)) out = ::sparse::dense_to_coo_final_state_dygraph_function(x,sparse_dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_divide(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("divide pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::divide_final_state_dygraph_function(x,y)) out = ::sparse::divide_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_divide_scalar(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("divide_scalar pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: divide_scalar";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("divide_scalar", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* scalar_obj = PyTuple_GET_ITEM(args, 1);
    float scalar = CastPyArg2Float(scalar_obj, "divide_scalar", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::divide_scalar_final_state_dygraph_function(x,scalar)) out = ::sparse::divide_scalar_final_state_dygraph_function(x,scalar);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_expm1(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("expm1 pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::expm1_final_state_dygraph_function(x)) out = ::sparse::expm1_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_leaky_relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("leaky_relu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: leaky_relu";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("leaky_relu", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* alpha_obj = PyTuple_GET_ITEM(args, 1);
    float alpha = CastPyArg2Float(alpha_obj, "leaky_relu", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::leaky_relu_final_state_dygraph_function(x,alpha)) out = ::sparse::leaky_relu_final_state_dygraph_function(x,alpha);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_log1p(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("log1p pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::log1p_final_state_dygraph_function(x)) out = ::sparse::log1p_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_multiply(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("multiply pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::multiply_final_state_dygraph_function(x,y)) out = ::sparse::multiply_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_pow(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("pow pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: pow";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("pow", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* factor_obj = PyTuple_GET_ITEM(args, 1);
    float factor = CastPyArg2Float(factor_obj, "pow", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::pow_final_state_dygraph_function(x,factor)) out = ::sparse::pow_final_state_dygraph_function(x,factor);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_relu(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("relu pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::relu_final_state_dygraph_function(x)) out = ::sparse::relu_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_relu6(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("relu6 pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: relu6";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("relu6", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* threshold_obj = PyTuple_GET_ITEM(args, 1);
    float threshold = CastPyArg2Float(threshold_obj, "relu6", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::relu6_final_state_dygraph_function(x,threshold)) out = ::sparse::relu6_final_state_dygraph_function(x,threshold);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_scale(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("scale pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: scale";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("scale", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* scale_obj = PyTuple_GET_ITEM(args, 1);
    float scale = CastPyArg2Float(scale_obj, "scale", 1);
    PyObject* bias_obj = PyTuple_GET_ITEM(args, 2);
    float bias = CastPyArg2Float(bias_obj, "scale", 2);
    PyObject* bias_after_scale_obj = PyTuple_GET_ITEM(args, 3);
    bool bias_after_scale = CastPyArg2Boolean(bias_after_scale_obj, "scale", 3);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::scale_final_state_dygraph_function(x,scale,bias,bias_after_scale)) out = ::sparse::scale_final_state_dygraph_function(x,scale,bias,bias_after_scale);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_sin(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sin pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::sin_final_state_dygraph_function(x)) out = ::sparse::sin_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_sinh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sinh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::sinh_final_state_dygraph_function(x)) out = ::sparse::sinh_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_softmax(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("softmax pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: softmax";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("softmax", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* axis_obj = PyTuple_GET_ITEM(args, 1);
    int axis = CastPyArg2Int(axis_obj, "softmax", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::softmax_final_state_dygraph_function(x,axis)) out = ::sparse::softmax_final_state_dygraph_function(x,axis);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_sqrt(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("sqrt pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::sqrt_final_state_dygraph_function(x)) out = ::sparse::sqrt_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_square(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("square pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::square_final_state_dygraph_function(x)) out = ::sparse::square_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_subtract(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("subtract pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::subtract_final_state_dygraph_function(x,y)) out = ::sparse::subtract_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_tan(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("tan pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::tan_final_state_dygraph_function(x)) out = ::sparse::tan_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_tanh(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("tanh pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::tanh_final_state_dygraph_function(x)) out = ::sparse::tanh_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_to_dense(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("to_dense pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::to_dense_final_state_dygraph_function(x)) out = ::sparse::to_dense_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_to_sparse_coo(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("to_sparse_coo pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: to_sparse_coo";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("to_sparse_coo", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* sparse_dim_obj = PyTuple_GET_ITEM(args, 1);
    int64_t sparse_dim = CastPyArg2Long(sparse_dim_obj, "to_sparse_coo", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::to_sparse_coo_final_state_dygraph_function(x,sparse_dim)) out = ::sparse::to_sparse_coo_final_state_dygraph_function(x,sparse_dim);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_to_sparse_csr(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("to_sparse_csr pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::to_sparse_csr_final_state_dygraph_function(x)) out = ::sparse::to_sparse_csr_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_values(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("values pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::values_final_state_dygraph_function(x)) out = ::sparse::values_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_addmm(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("addmm pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: addmm";

    // Get EagerTensors from args
    auto input = GetTensorFromArgs("addmm", "input", args, 0, false);
    auto x = GetTensorFromArgs("addmm", "x", args, 1, false);
    auto y = GetTensorFromArgs("addmm", "y", args, 2, false);

    // Parse Attributes if needed
    PyObject* alpha_obj = PyTuple_GET_ITEM(args, 3);
    float alpha = CastPyArg2Float(alpha_obj, "addmm", 3);
    PyObject* beta_obj = PyTuple_GET_ITEM(args, 4);
    float beta = CastPyArg2Float(beta_obj, "addmm", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::addmm_final_state_dygraph_function(input,x,y,alpha,beta)) out = ::sparse::addmm_final_state_dygraph_function(input,x,y,alpha,beta);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_coalesce(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("coalesce pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: coalesce";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("coalesce", "x", args, 0, false);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::coalesce_final_state_dygraph_function(x)) out = ::sparse::coalesce_final_state_dygraph_function(x);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_full_like(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("full_like pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: full_like";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("full_like", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* value_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::Scalar value = CastPyArg2Scalar(value_obj, "full_like", 1);
    PyObject* dtype_obj = PyTuple_GET_ITEM(args, 2);
    paddle::experimental::DataType dtype = CastPyArg2DataType(dtype_obj, "full_like", 2);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::full_like_final_state_dygraph_function(x,value,dtype)) out = ::sparse::full_like_final_state_dygraph_function(x,value,dtype);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_fused_attention(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("fused_attention pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: fused_attention";

    // Get EagerTensors from args
    auto query = GetTensorFromArgs("fused_attention", "query", args, 0, false);
    auto key = GetTensorFromArgs("fused_attention", "key", args, 1, false);
    auto value = GetTensorFromArgs("fused_attention", "value", args, 2, false);
    auto sparse_mask = GetTensorFromArgs("fused_attention", "sparse_mask", args, 3, false);
    auto key_padding_mask = GetOptionalTensorFromArgs("fused_attention", "key_padding_mask", args, 4, true);
    auto attn_mask = GetOptionalTensorFromArgs("fused_attention", "attn_mask", args, 5, true);

    // Parse Attributes if needed

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::fused_attention_final_state_dygraph_function(query,key,value,sparse_mask,key_padding_mask,attn_mask)) out = ::sparse::fused_attention_final_state_dygraph_function(query,key,value,sparse_mask,key_padding_mask,attn_mask);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_masked_matmul(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("masked_matmul pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::masked_matmul_final_state_dygraph_function(x,y,mask)) out = ::sparse::masked_matmul_final_state_dygraph_function(x,y,mask);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_matmul(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("matmul pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::matmul_final_state_dygraph_function(x,y)) out = ::sparse::matmul_final_state_dygraph_function(x,y);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_maxpool(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("maxpool pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: maxpool";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("maxpool", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* kernel_sizes_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<int> kernel_sizes = CastPyArg2Ints(kernel_sizes_obj, "maxpool", 1);
    PyObject* paddings_obj = PyTuple_GET_ITEM(args, 2);
    std::vector<int> paddings = CastPyArg2Ints(paddings_obj, "maxpool", 2);
    PyObject* dilations_obj = PyTuple_GET_ITEM(args, 3);
    std::vector<int> dilations = CastPyArg2Ints(dilations_obj, "maxpool", 3);
    PyObject* strides_obj = PyTuple_GET_ITEM(args, 4);
    std::vector<int> strides = CastPyArg2Ints(strides_obj, "maxpool", 4);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::maxpool_final_state_dygraph_function(x,kernel_sizes,paddings,dilations,strides)) out = ::sparse::maxpool_final_state_dygraph_function(x,kernel_sizes,paddings,dilations,strides);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_mv(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("mv pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

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
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::sparse::mv_final_state_dygraph_function(x,vec)) out = ::sparse::mv_final_state_dygraph_function(x,vec);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


}

namespace strings {
    
static PyObject * eager_final_state_api_empty(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("empty pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: empty";

    // Get EagerTensors from args

    // Parse Attributes if needed
    PyObject* shape_obj = PyTuple_GET_ITEM(args, 0);
    paddle::experimental::IntArray shape = CastPyArg2IntArray(shape_obj, "empty", 0);
    PyObject* place_obj = PyTuple_GET_ITEM(args, 1);
    paddle::Place place = CastPyArg2Place(place_obj, "empty", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::strings::empty_final_state_dygraph_function(shape,place)) out = ::strings::empty_final_state_dygraph_function(shape,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_empty_like(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("empty_like pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: empty_like";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("empty_like", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* place_obj = PyTuple_GET_ITEM(args, 1);
    paddle::Place place = CastPyArg2Place(place_obj, "empty_like", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::strings::empty_like_final_state_dygraph_function(x,place)) out = ::strings::empty_like_final_state_dygraph_function(x,place);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_lower(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("lower pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: lower";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("lower", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* use_utf8_encoding_obj = PyTuple_GET_ITEM(args, 1);
    bool use_utf8_encoding = CastPyArg2Boolean(use_utf8_encoding_obj, "lower", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::strings::lower_final_state_dygraph_function(x,use_utf8_encoding)) out = ::strings::lower_final_state_dygraph_function(x,use_utf8_encoding);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


static PyObject * eager_final_state_api_upper(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("upper pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);

  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: upper";

    // Get EagerTensors from args
    auto x = GetTensorFromArgs("upper", "x", args, 0, false);

    // Parse Attributes if needed
    PyObject* use_utf8_encoding_obj = PyTuple_GET_ITEM(args, 1);
    bool use_utf8_encoding = CastPyArg2Boolean(use_utf8_encoding_obj, "upper", 1);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(1) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(1) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }

    // Call dygraph function
    decltype(::strings::upper_final_state_dygraph_function(x,use_utf8_encoding)) out = ::strings::upper_final_state_dygraph_function(x,use_utf8_encoding);


    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}


}


static PyObject * eager_get_final_state_core_ops_args_info(PyObject *self) {
    PyThreadState *tstate = nullptr;
    try
    {
      return ToPyObject(core_ops_final_state_args_info);
    }
    catch(...) {
      if (tstate) {
        PyEval_RestoreThread(tstate);
      }
      ThrowExceptionToPython(std::current_exception());
      return nullptr;
    }
}

static PyObject * eager_get_final_state_core_ops_args_type_info(PyObject *self) {
    PyThreadState *tstate = nullptr;
    try
    {
      return ToPyObject(core_ops_final_state_args_type_info);
    }
    catch(...) {
      if (tstate) {
        PyEval_RestoreThread(tstate);
      }
      ThrowExceptionToPython(std::current_exception());
      return nullptr;
    }
}

static PyObject * eager_get_final_state_core_ops_returns_info(PyObject *self) {
    PyThreadState *tstate = nullptr;
    try
    {
      return ToPyObject(core_ops_final_state_returns_info);
    }
    catch(...) {
      if (tstate) {
        PyEval_RestoreThread(tstate);
      }
      ThrowExceptionToPython(std::current_exception());
      return nullptr;
    }
}


static PyMethodDef EagerFinalStateMethods[] = {
    
{"final_state_atan2", (PyCFunction)(void(*)(void)) eager_final_state_api_atan2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan2 in dygraph."}

,

{"final_state_bernoulli", (PyCFunction)(void(*)(void)) eager_final_state_api_bernoulli, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bernoulli in dygraph."}

,

{"final_state_cholesky", (PyCFunction)(void(*)(void)) eager_final_state_api_cholesky, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cholesky in dygraph."}

,

{"final_state_cholesky_solve", (PyCFunction)(void(*)(void)) eager_final_state_api_cholesky_solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cholesky_solve in dygraph."}

,

{"final_state_cross", (PyCFunction)(void(*)(void)) eager_final_state_api_cross, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross in dygraph."}

,

{"final_state_diag", (PyCFunction)(void(*)(void)) eager_final_state_api_diag, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag in dygraph."}

,

{"final_state_diagonal", (PyCFunction)(void(*)(void)) eager_final_state_api_diagonal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diagonal in dygraph."}

,

{"final_state_digamma", (PyCFunction)(void(*)(void)) eager_final_state_api_digamma, METH_VARARGS | METH_KEYWORDS, "C++ interface function for digamma in dygraph."}

,

{"final_state_dist", (PyCFunction)(void(*)(void)) eager_final_state_api_dist, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dist in dygraph."}

,

{"final_state_dot", (PyCFunction)(void(*)(void)) eager_final_state_api_dot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dot in dygraph."}

,

{"final_state_erf", (PyCFunction)(void(*)(void)) eager_final_state_api_erf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erf in dygraph."}

,

{"final_state_erfinv", (PyCFunction)(void(*)(void)) eager_final_state_api_erfinv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erfinv in dygraph."}


,
{"final_state_erfinv_", (PyCFunction)(void(*)(void)) eager_final_state_api_erfinv_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erfinv_ in dygraph."}

,

{"final_state_fft_c2c", (PyCFunction)(void(*)(void)) eager_final_state_api_fft_c2c, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_c2c in dygraph."}

,

{"final_state_fft_c2r", (PyCFunction)(void(*)(void)) eager_final_state_api_fft_c2r, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_c2r in dygraph."}

,

{"final_state_fft_r2c", (PyCFunction)(void(*)(void)) eager_final_state_api_fft_r2c, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_r2c in dygraph."}

,

{"final_state_graph_send_uv", (PyCFunction)(void(*)(void)) eager_final_state_api_graph_send_uv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_send_uv in dygraph."}

,

{"final_state_lgamma", (PyCFunction)(void(*)(void)) eager_final_state_api_lgamma, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lgamma in dygraph."}

,

{"final_state_mv", (PyCFunction)(void(*)(void)) eager_final_state_api_mv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mv in dygraph."}

,

{"final_state_poisson", (PyCFunction)(void(*)(void)) eager_final_state_api_poisson, METH_VARARGS | METH_KEYWORDS, "C++ interface function for poisson in dygraph."}

,

{"final_state_solve", (PyCFunction)(void(*)(void)) eager_final_state_api_solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for solve in dygraph."}

,

{"final_state_trace", (PyCFunction)(void(*)(void)) eager_final_state_api_trace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trace in dygraph."}

,

{"final_state_trunc", (PyCFunction)(void(*)(void)) eager_final_state_api_trunc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trunc in dygraph."}

,


{"final_state_abs", (PyCFunction)(void(*)(void)) eager_final_state_api_abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for abs in dygraph."}

,

{"final_state_accuracy", (PyCFunction)(void(*)(void)) eager_final_state_api_accuracy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for accuracy in dygraph."}

,

{"final_state_acos", (PyCFunction)(void(*)(void)) eager_final_state_api_acos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acos in dygraph."}

,

{"final_state_acosh", (PyCFunction)(void(*)(void)) eager_final_state_api_acosh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acosh in dygraph."}

,

{"final_state_adadelta_", (PyCFunction)(void(*)(void)) eager_final_state_api_adadelta_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adadelta_ in dygraph."}

,

{"final_state_adagrad_", (PyCFunction)(void(*)(void)) eager_final_state_api_adagrad_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adagrad_ in dygraph."}

,

{"final_state_adam_", (PyCFunction)(void(*)(void)) eager_final_state_api_adam_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adam_ in dygraph."}

,

{"final_state_adamax_", (PyCFunction)(void(*)(void)) eager_final_state_api_adamax_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adamax_ in dygraph."}

,

{"final_state_adamw_", (PyCFunction)(void(*)(void)) eager_final_state_api_adamw_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adamw_ in dygraph."}

,

{"final_state_add", (PyCFunction)(void(*)(void)) eager_final_state_api_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add in dygraph."}


,
{"final_state_add_", (PyCFunction)(void(*)(void)) eager_final_state_api_add_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_ in dygraph."}

,

{"final_state_add_n", (PyCFunction)(void(*)(void)) eager_final_state_api_add_n, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_n in dygraph."}

,

{"final_state_addmm", (PyCFunction)(void(*)(void)) eager_final_state_api_addmm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for addmm in dygraph."}

,

{"final_state_affine_grid", (PyCFunction)(void(*)(void)) eager_final_state_api_affine_grid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for affine_grid in dygraph."}

,

{"final_state_all", (PyCFunction)(void(*)(void)) eager_final_state_api_all, METH_VARARGS | METH_KEYWORDS, "C++ interface function for all in dygraph."}

,

{"final_state_allclose", (PyCFunction)(void(*)(void)) eager_final_state_api_allclose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for allclose in dygraph."}

,

{"final_state_amax", (PyCFunction)(void(*)(void)) eager_final_state_api_amax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for amax in dygraph."}

,

{"final_state_amin", (PyCFunction)(void(*)(void)) eager_final_state_api_amin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for amin in dygraph."}

,

{"final_state_angle", (PyCFunction)(void(*)(void)) eager_final_state_api_angle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for angle in dygraph."}

,

{"final_state_any", (PyCFunction)(void(*)(void)) eager_final_state_api_any, METH_VARARGS | METH_KEYWORDS, "C++ interface function for any in dygraph."}

,

{"final_state_arange", (PyCFunction)(void(*)(void)) eager_final_state_api_arange, METH_VARARGS | METH_KEYWORDS, "C++ interface function for arange in dygraph."}

,

{"final_state_argmax", (PyCFunction)(void(*)(void)) eager_final_state_api_argmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for argmax in dygraph."}

,

{"final_state_argmin", (PyCFunction)(void(*)(void)) eager_final_state_api_argmin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for argmin in dygraph."}

,

{"final_state_argsort", (PyCFunction)(void(*)(void)) eager_final_state_api_argsort, METH_VARARGS | METH_KEYWORDS, "C++ interface function for argsort in dygraph."}

,

{"final_state_as_complex", (PyCFunction)(void(*)(void)) eager_final_state_api_as_complex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for as_complex in dygraph."}

,

{"final_state_as_real", (PyCFunction)(void(*)(void)) eager_final_state_api_as_real, METH_VARARGS | METH_KEYWORDS, "C++ interface function for as_real in dygraph."}

,

{"final_state_asin", (PyCFunction)(void(*)(void)) eager_final_state_api_asin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asin in dygraph."}

,

{"final_state_asinh", (PyCFunction)(void(*)(void)) eager_final_state_api_asinh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asinh in dygraph."}

,

{"final_state_assign", (PyCFunction)(void(*)(void)) eager_final_state_api_assign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign in dygraph."}

,

{"final_state_assign_out_", (PyCFunction)(void(*)(void)) eager_final_state_api_assign_out_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_out_ in dygraph."}

,

{"final_state_assign_value_", (PyCFunction)(void(*)(void)) eager_final_state_api_assign_value_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_value_ in dygraph."}

,

{"final_state_atan", (PyCFunction)(void(*)(void)) eager_final_state_api_atan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan in dygraph."}

,

{"final_state_atanh", (PyCFunction)(void(*)(void)) eager_final_state_api_atanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atanh in dygraph."}

,

{"final_state_auc", (PyCFunction)(void(*)(void)) eager_final_state_api_auc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for auc in dygraph."}

,

{"final_state_average_accumulates_", (PyCFunction)(void(*)(void)) eager_final_state_api_average_accumulates_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for average_accumulates_ in dygraph."}

,

{"final_state_batch_norm", (PyCFunction)(void(*)(void)) eager_final_state_api_batch_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for batch_norm in dygraph."}

,

{"final_state_bce_loss", (PyCFunction)(void(*)(void)) eager_final_state_api_bce_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bce_loss in dygraph."}

,

{"final_state_bicubic_interp", (PyCFunction)(void(*)(void)) eager_final_state_api_bicubic_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bicubic_interp in dygraph."}

,

{"final_state_bilinear_interp", (PyCFunction)(void(*)(void)) eager_final_state_api_bilinear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear_interp in dygraph."}

,

{"final_state_bilinear_tensor_product", (PyCFunction)(void(*)(void)) eager_final_state_api_bilinear_tensor_product, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear_tensor_product in dygraph."}

,

{"final_state_bitwise_and", (PyCFunction)(void(*)(void)) eager_final_state_api_bitwise_and, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_and in dygraph."}

,

{"final_state_bitwise_not", (PyCFunction)(void(*)(void)) eager_final_state_api_bitwise_not, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_not in dygraph."}

,

{"final_state_bitwise_or", (PyCFunction)(void(*)(void)) eager_final_state_api_bitwise_or, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_or in dygraph."}

,

{"final_state_bitwise_xor", (PyCFunction)(void(*)(void)) eager_final_state_api_bitwise_xor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_xor in dygraph."}

,

{"final_state_bmm", (PyCFunction)(void(*)(void)) eager_final_state_api_bmm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bmm in dygraph."}

,

{"final_state_box_coder", (PyCFunction)(void(*)(void)) eager_final_state_api_box_coder, METH_VARARGS | METH_KEYWORDS, "C++ interface function for box_coder in dygraph."}

,

{"final_state_brelu", (PyCFunction)(void(*)(void)) eager_final_state_api_brelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for brelu in dygraph."}

,

{"final_state_cast", (PyCFunction)(void(*)(void)) eager_final_state_api_cast, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cast in dygraph."}

,

{"final_state_ceil", (PyCFunction)(void(*)(void)) eager_final_state_api_ceil, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ceil in dygraph."}


,
{"final_state_ceil_", (PyCFunction)(void(*)(void)) eager_final_state_api_ceil_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ceil_ in dygraph."}

,

{"final_state_celu", (PyCFunction)(void(*)(void)) eager_final_state_api_celu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for celu in dygraph."}

,

{"final_state_class_center_sample", (PyCFunction)(void(*)(void)) eager_final_state_api_class_center_sample, METH_VARARGS | METH_KEYWORDS, "C++ interface function for class_center_sample in dygraph."}

,

{"final_state_clip", (PyCFunction)(void(*)(void)) eager_final_state_api_clip, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip in dygraph."}


,
{"final_state_clip_", (PyCFunction)(void(*)(void)) eager_final_state_api_clip_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip_ in dygraph."}

,

{"final_state_clip_by_norm", (PyCFunction)(void(*)(void)) eager_final_state_api_clip_by_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip_by_norm in dygraph."}

,

{"final_state_complex", (PyCFunction)(void(*)(void)) eager_final_state_api_complex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for complex in dygraph."}

,

{"final_state_concat", (PyCFunction)(void(*)(void)) eager_final_state_api_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for concat in dygraph."}

,

{"final_state_conj", (PyCFunction)(void(*)(void)) eager_final_state_api_conj, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conj in dygraph."}

,

{"final_state_conv2d", (PyCFunction)(void(*)(void)) eager_final_state_api_conv2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d in dygraph."}

,

{"final_state_conv2d_transpose", (PyCFunction)(void(*)(void)) eager_final_state_api_conv2d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d_transpose in dygraph."}

,

{"final_state_conv3d", (PyCFunction)(void(*)(void)) eager_final_state_api_conv3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d in dygraph."}

,

{"final_state_conv3d_transpose", (PyCFunction)(void(*)(void)) eager_final_state_api_conv3d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d_transpose in dygraph."}

,

{"final_state_copy_to", (PyCFunction)(void(*)(void)) eager_final_state_api_copy_to, METH_VARARGS | METH_KEYWORDS, "C++ interface function for copy_to in dygraph."}

,

{"final_state_cos", (PyCFunction)(void(*)(void)) eager_final_state_api_cos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cos in dygraph."}

,

{"final_state_cosh", (PyCFunction)(void(*)(void)) eager_final_state_api_cosh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cosh in dygraph."}

,

{"final_state_crop_tensor", (PyCFunction)(void(*)(void)) eager_final_state_api_crop_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for crop_tensor in dygraph."}

,

{"final_state_cross_entropy_with_softmax", (PyCFunction)(void(*)(void)) eager_final_state_api_cross_entropy_with_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross_entropy_with_softmax in dygraph."}

,

{"final_state_cumprod", (PyCFunction)(void(*)(void)) eager_final_state_api_cumprod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumprod in dygraph."}

,

{"final_state_cumsum", (PyCFunction)(void(*)(void)) eager_final_state_api_cumsum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumsum in dygraph."}

,

{"final_state_decode_jpeg", (PyCFunction)(void(*)(void)) eager_final_state_api_decode_jpeg, METH_VARARGS | METH_KEYWORDS, "C++ interface function for decode_jpeg in dygraph."}

,

{"final_state_deformable_conv", (PyCFunction)(void(*)(void)) eager_final_state_api_deformable_conv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for deformable_conv in dygraph."}

,

{"final_state_depthwise_conv2d", (PyCFunction)(void(*)(void)) eager_final_state_api_depthwise_conv2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for depthwise_conv2d in dygraph."}

,

{"final_state_depthwise_conv2d_transpose", (PyCFunction)(void(*)(void)) eager_final_state_api_depthwise_conv2d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for depthwise_conv2d_transpose in dygraph."}

,

{"final_state_det", (PyCFunction)(void(*)(void)) eager_final_state_api_det, METH_VARARGS | METH_KEYWORDS, "C++ interface function for det in dygraph."}

,

{"final_state_diag_embed", (PyCFunction)(void(*)(void)) eager_final_state_api_diag_embed, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag_embed in dygraph."}

,

{"final_state_distribute_fpn_proposals", (PyCFunction)(void(*)(void)) eager_final_state_api_distribute_fpn_proposals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distribute_fpn_proposals in dygraph."}

,

{"final_state_divide", (PyCFunction)(void(*)(void)) eager_final_state_api_divide, METH_VARARGS | METH_KEYWORDS, "C++ interface function for divide in dygraph."}

,

{"final_state_dropout", (PyCFunction)(void(*)(void)) eager_final_state_api_dropout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dropout in dygraph."}

,

{"final_state_edit_distance", (PyCFunction)(void(*)(void)) eager_final_state_api_edit_distance, METH_VARARGS | METH_KEYWORDS, "C++ interface function for edit_distance in dygraph."}

,

{"final_state_eigh", (PyCFunction)(void(*)(void)) eager_final_state_api_eigh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigh in dygraph."}

,

{"final_state_eigvals", (PyCFunction)(void(*)(void)) eager_final_state_api_eigvals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigvals in dygraph."}

,

{"final_state_eigvalsh", (PyCFunction)(void(*)(void)) eager_final_state_api_eigvalsh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigvalsh in dygraph."}

,

{"final_state_einsum", (PyCFunction)(void(*)(void)) eager_final_state_api_einsum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for einsum in dygraph."}

,

{"final_state_elementwise_pow", (PyCFunction)(void(*)(void)) eager_final_state_api_elementwise_pow, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_pow in dygraph."}

,

{"final_state_elu", (PyCFunction)(void(*)(void)) eager_final_state_api_elu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elu in dygraph."}


,
{"final_state_elu_", (PyCFunction)(void(*)(void)) eager_final_state_api_elu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elu_ in dygraph."}

,

{"final_state_embedding", (PyCFunction)(void(*)(void)) eager_final_state_api_embedding, METH_VARARGS | METH_KEYWORDS, "C++ interface function for embedding in dygraph."}

,

{"final_state_empty", (PyCFunction)(void(*)(void)) eager_final_state_api_empty, METH_VARARGS | METH_KEYWORDS, "C++ interface function for empty in dygraph."}

,

{"final_state_empty_like", (PyCFunction)(void(*)(void)) eager_final_state_api_empty_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for empty_like in dygraph."}

,

{"final_state_equal", (PyCFunction)(void(*)(void)) eager_final_state_api_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for equal in dygraph."}

,

{"final_state_equal_all", (PyCFunction)(void(*)(void)) eager_final_state_api_equal_all, METH_VARARGS | METH_KEYWORDS, "C++ interface function for equal_all in dygraph."}

,

{"final_state_exp", (PyCFunction)(void(*)(void)) eager_final_state_api_exp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exp in dygraph."}


,
{"final_state_exp_", (PyCFunction)(void(*)(void)) eager_final_state_api_exp_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exp_ in dygraph."}

,

{"final_state_expand", (PyCFunction)(void(*)(void)) eager_final_state_api_expand, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand in dygraph."}

,

{"final_state_expand_as", (PyCFunction)(void(*)(void)) eager_final_state_api_expand_as, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand_as in dygraph."}

,

{"final_state_expm1", (PyCFunction)(void(*)(void)) eager_final_state_api_expm1, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expm1 in dygraph."}

,

{"final_state_exponential_", (PyCFunction)(void(*)(void)) eager_final_state_api_exponential_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exponential_ in dygraph."}

,

{"final_state_eye", (PyCFunction)(void(*)(void)) eager_final_state_api_eye, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eye in dygraph."}

,

{"final_state_fill", (PyCFunction)(void(*)(void)) eager_final_state_api_fill, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill in dygraph."}


,
{"final_state_fill_", (PyCFunction)(void(*)(void)) eager_final_state_api_fill_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_ in dygraph."}

,

{"final_state_fill_diagonal", (PyCFunction)(void(*)(void)) eager_final_state_api_fill_diagonal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal in dygraph."}


,
{"final_state_fill_diagonal_", (PyCFunction)(void(*)(void)) eager_final_state_api_fill_diagonal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_ in dygraph."}

,

{"final_state_fill_diagonal_tensor", (PyCFunction)(void(*)(void)) eager_final_state_api_fill_diagonal_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_tensor in dygraph."}


,
{"final_state_fill_diagonal_tensor_", (PyCFunction)(void(*)(void)) eager_final_state_api_fill_diagonal_tensor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_tensor_ in dygraph."}

,

{"final_state_flatten", (PyCFunction)(void(*)(void)) eager_final_state_api_flatten, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten in dygraph."}


,
{"final_state_flatten_", (PyCFunction)(void(*)(void)) eager_final_state_api_flatten_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten_ in dygraph."}

,

{"final_state_flip", (PyCFunction)(void(*)(void)) eager_final_state_api_flip, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flip in dygraph."}

,

{"final_state_floor", (PyCFunction)(void(*)(void)) eager_final_state_api_floor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor in dygraph."}


,
{"final_state_floor_", (PyCFunction)(void(*)(void)) eager_final_state_api_floor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor_ in dygraph."}

,

{"final_state_floor_divide", (PyCFunction)(void(*)(void)) eager_final_state_api_floor_divide, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor_divide in dygraph."}

,

{"final_state_fmax", (PyCFunction)(void(*)(void)) eager_final_state_api_fmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fmax in dygraph."}

,

{"final_state_fmin", (PyCFunction)(void(*)(void)) eager_final_state_api_fmin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fmin in dygraph."}

,

{"final_state_frame", (PyCFunction)(void(*)(void)) eager_final_state_api_frame, METH_VARARGS | METH_KEYWORDS, "C++ interface function for frame in dygraph."}

,

{"final_state_frobenius_norm", (PyCFunction)(void(*)(void)) eager_final_state_api_frobenius_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for frobenius_norm in dygraph."}

,

{"final_state_full", (PyCFunction)(void(*)(void)) eager_final_state_api_full, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full in dygraph."}

,

{"final_state_full_", (PyCFunction)(void(*)(void)) eager_final_state_api_full_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_ in dygraph."}

,

{"final_state_full_batch_size_like", (PyCFunction)(void(*)(void)) eager_final_state_api_full_batch_size_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_batch_size_like in dygraph."}

,

{"final_state_full_like", (PyCFunction)(void(*)(void)) eager_final_state_api_full_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_like in dygraph."}

,

{"final_state_gather", (PyCFunction)(void(*)(void)) eager_final_state_api_gather, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather in dygraph."}

,

{"final_state_gather_nd", (PyCFunction)(void(*)(void)) eager_final_state_api_gather_nd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather_nd in dygraph."}

,

{"final_state_gather_tree", (PyCFunction)(void(*)(void)) eager_final_state_api_gather_tree, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather_tree in dygraph."}

,

{"final_state_gaussian_random", (PyCFunction)(void(*)(void)) eager_final_state_api_gaussian_random, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian_random in dygraph."}

,

{"final_state_gelu", (PyCFunction)(void(*)(void)) eager_final_state_api_gelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gelu in dygraph."}

,

{"final_state_generate_proposals_v2", (PyCFunction)(void(*)(void)) eager_final_state_api_generate_proposals_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for generate_proposals_v2 in dygraph."}

,

{"final_state_graph_send_recv", (PyCFunction)(void(*)(void)) eager_final_state_api_graph_send_recv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_send_recv in dygraph."}

,

{"final_state_graph_send_ue_recv", (PyCFunction)(void(*)(void)) eager_final_state_api_graph_send_ue_recv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_send_ue_recv in dygraph."}

,

{"final_state_greater_equal", (PyCFunction)(void(*)(void)) eager_final_state_api_greater_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_equal in dygraph."}

,

{"final_state_greater_than", (PyCFunction)(void(*)(void)) eager_final_state_api_greater_than, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_than in dygraph."}

,

{"final_state_grid_sample", (PyCFunction)(void(*)(void)) eager_final_state_api_grid_sample, METH_VARARGS | METH_KEYWORDS, "C++ interface function for grid_sample in dygraph."}

,

{"final_state_group_norm", (PyCFunction)(void(*)(void)) eager_final_state_api_group_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for group_norm in dygraph."}

,

{"final_state_gumbel_softmax", (PyCFunction)(void(*)(void)) eager_final_state_api_gumbel_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gumbel_softmax in dygraph."}

,

{"final_state_hard_shrink", (PyCFunction)(void(*)(void)) eager_final_state_api_hard_shrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hard_shrink in dygraph."}

,

{"final_state_hard_sigmoid", (PyCFunction)(void(*)(void)) eager_final_state_api_hard_sigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hard_sigmoid in dygraph."}

,

{"final_state_hard_swish", (PyCFunction)(void(*)(void)) eager_final_state_api_hard_swish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hard_swish in dygraph."}

,

{"final_state_hierarchical_sigmoid", (PyCFunction)(void(*)(void)) eager_final_state_api_hierarchical_sigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hierarchical_sigmoid in dygraph."}

,

{"final_state_histogram", (PyCFunction)(void(*)(void)) eager_final_state_api_histogram, METH_VARARGS | METH_KEYWORDS, "C++ interface function for histogram in dygraph."}

,

{"final_state_huber_loss", (PyCFunction)(void(*)(void)) eager_final_state_api_huber_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for huber_loss in dygraph."}

,

{"final_state_imag", (PyCFunction)(void(*)(void)) eager_final_state_api_imag, METH_VARARGS | METH_KEYWORDS, "C++ interface function for imag in dygraph."}

,

{"final_state_increment", (PyCFunction)(void(*)(void)) eager_final_state_api_increment, METH_VARARGS | METH_KEYWORDS, "C++ interface function for increment in dygraph."}

,

{"final_state_index_sample", (PyCFunction)(void(*)(void)) eager_final_state_api_index_sample, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_sample in dygraph."}

,

{"final_state_index_select", (PyCFunction)(void(*)(void)) eager_final_state_api_index_select, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_select in dygraph."}

,

{"final_state_instance_norm", (PyCFunction)(void(*)(void)) eager_final_state_api_instance_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for instance_norm in dygraph."}

,

{"final_state_inverse", (PyCFunction)(void(*)(void)) eager_final_state_api_inverse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for inverse in dygraph."}

,

{"final_state_is_empty", (PyCFunction)(void(*)(void)) eager_final_state_api_is_empty, METH_VARARGS | METH_KEYWORDS, "C++ interface function for is_empty in dygraph."}

,

{"final_state_isclose", (PyCFunction)(void(*)(void)) eager_final_state_api_isclose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isclose in dygraph."}

,

{"final_state_isfinite", (PyCFunction)(void(*)(void)) eager_final_state_api_isfinite, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isfinite in dygraph."}

,

{"final_state_isinf", (PyCFunction)(void(*)(void)) eager_final_state_api_isinf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isinf in dygraph."}

,

{"final_state_isnan", (PyCFunction)(void(*)(void)) eager_final_state_api_isnan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isnan in dygraph."}

,

{"final_state_kldiv_loss", (PyCFunction)(void(*)(void)) eager_final_state_api_kldiv_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kldiv_loss in dygraph."}

,

{"final_state_kron", (PyCFunction)(void(*)(void)) eager_final_state_api_kron, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kron in dygraph."}

,

{"final_state_kthvalue", (PyCFunction)(void(*)(void)) eager_final_state_api_kthvalue, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kthvalue in dygraph."}

,

{"final_state_label_smooth", (PyCFunction)(void(*)(void)) eager_final_state_api_label_smooth, METH_VARARGS | METH_KEYWORDS, "C++ interface function for label_smooth in dygraph."}

,

{"final_state_lamb_", (PyCFunction)(void(*)(void)) eager_final_state_api_lamb_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lamb_ in dygraph."}

,

{"final_state_layer_norm", (PyCFunction)(void(*)(void)) eager_final_state_api_layer_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for layer_norm in dygraph."}

,

{"final_state_leaky_relu", (PyCFunction)(void(*)(void)) eager_final_state_api_leaky_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for leaky_relu in dygraph."}

,

{"final_state_lerp", (PyCFunction)(void(*)(void)) eager_final_state_api_lerp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lerp in dygraph."}


,
{"final_state_lerp_", (PyCFunction)(void(*)(void)) eager_final_state_api_lerp_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lerp_ in dygraph."}

,

{"final_state_less_equal", (PyCFunction)(void(*)(void)) eager_final_state_api_less_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_equal in dygraph."}

,

{"final_state_less_than", (PyCFunction)(void(*)(void)) eager_final_state_api_less_than, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_than in dygraph."}

,

{"final_state_linear_interp", (PyCFunction)(void(*)(void)) eager_final_state_api_linear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linear_interp in dygraph."}

,

{"final_state_linspace", (PyCFunction)(void(*)(void)) eager_final_state_api_linspace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linspace in dygraph."}

,

{"final_state_log", (PyCFunction)(void(*)(void)) eager_final_state_api_log, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log in dygraph."}

,

{"final_state_log10", (PyCFunction)(void(*)(void)) eager_final_state_api_log10, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log10 in dygraph."}

,

{"final_state_log1p", (PyCFunction)(void(*)(void)) eager_final_state_api_log1p, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log1p in dygraph."}

,

{"final_state_log2", (PyCFunction)(void(*)(void)) eager_final_state_api_log2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log2 in dygraph."}

,

{"final_state_log_loss", (PyCFunction)(void(*)(void)) eager_final_state_api_log_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_loss in dygraph."}

,

{"final_state_log_softmax", (PyCFunction)(void(*)(void)) eager_final_state_api_log_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_softmax in dygraph."}

,

{"final_state_logcumsumexp", (PyCFunction)(void(*)(void)) eager_final_state_api_logcumsumexp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logcumsumexp in dygraph."}

,

{"final_state_logical_and", (PyCFunction)(void(*)(void)) eager_final_state_api_logical_and, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_and in dygraph."}

,

{"final_state_logical_not", (PyCFunction)(void(*)(void)) eager_final_state_api_logical_not, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_not in dygraph."}

,

{"final_state_logical_or", (PyCFunction)(void(*)(void)) eager_final_state_api_logical_or, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_or in dygraph."}

,

{"final_state_logical_xor", (PyCFunction)(void(*)(void)) eager_final_state_api_logical_xor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_xor in dygraph."}

,

{"final_state_logit", (PyCFunction)(void(*)(void)) eager_final_state_api_logit, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logit in dygraph."}

,

{"final_state_logsigmoid", (PyCFunction)(void(*)(void)) eager_final_state_api_logsigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logsigmoid in dygraph."}

,

{"final_state_logsumexp", (PyCFunction)(void(*)(void)) eager_final_state_api_logsumexp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logsumexp in dygraph."}

,

{"final_state_lstsq", (PyCFunction)(void(*)(void)) eager_final_state_api_lstsq, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lstsq in dygraph."}

,

{"final_state_lu", (PyCFunction)(void(*)(void)) eager_final_state_api_lu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu in dygraph."}

,

{"final_state_lu_unpack", (PyCFunction)(void(*)(void)) eager_final_state_api_lu_unpack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_unpack in dygraph."}

,

{"final_state_margin_cross_entropy", (PyCFunction)(void(*)(void)) eager_final_state_api_margin_cross_entropy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for margin_cross_entropy in dygraph."}

,

{"final_state_masked_select", (PyCFunction)(void(*)(void)) eager_final_state_api_masked_select, METH_VARARGS | METH_KEYWORDS, "C++ interface function for masked_select in dygraph."}

,

{"final_state_matmul", (PyCFunction)(void(*)(void)) eager_final_state_api_matmul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matmul in dygraph."}

,

{"final_state_matrix_nms", (PyCFunction)(void(*)(void)) eager_final_state_api_matrix_nms, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_nms in dygraph."}

,

{"final_state_matrix_power", (PyCFunction)(void(*)(void)) eager_final_state_api_matrix_power, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_power in dygraph."}

,

{"final_state_matrix_rank", (PyCFunction)(void(*)(void)) eager_final_state_api_matrix_rank, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_rank in dygraph."}

,

{"final_state_matrix_rank_tol", (PyCFunction)(void(*)(void)) eager_final_state_api_matrix_rank_tol, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_rank_tol in dygraph."}

,

{"final_state_max", (PyCFunction)(void(*)(void)) eager_final_state_api_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max in dygraph."}

,

{"final_state_max_pool2d_with_index", (PyCFunction)(void(*)(void)) eager_final_state_api_max_pool2d_with_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_pool2d_with_index in dygraph."}

,

{"final_state_max_pool3d_with_index", (PyCFunction)(void(*)(void)) eager_final_state_api_max_pool3d_with_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_pool3d_with_index in dygraph."}

,

{"final_state_maximum", (PyCFunction)(void(*)(void)) eager_final_state_api_maximum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for maximum in dygraph."}

,

{"final_state_maxout", (PyCFunction)(void(*)(void)) eager_final_state_api_maxout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for maxout in dygraph."}

,

{"final_state_mean", (PyCFunction)(void(*)(void)) eager_final_state_api_mean, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mean in dygraph."}

,

{"final_state_mean_all", (PyCFunction)(void(*)(void)) eager_final_state_api_mean_all, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mean_all in dygraph."}

,

{"final_state_meshgrid", (PyCFunction)(void(*)(void)) eager_final_state_api_meshgrid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for meshgrid in dygraph."}

,

{"final_state_min", (PyCFunction)(void(*)(void)) eager_final_state_api_min, METH_VARARGS | METH_KEYWORDS, "C++ interface function for min in dygraph."}

,

{"final_state_minimum", (PyCFunction)(void(*)(void)) eager_final_state_api_minimum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for minimum in dygraph."}

,

{"final_state_mish", (PyCFunction)(void(*)(void)) eager_final_state_api_mish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mish in dygraph."}

,

{"final_state_mode", (PyCFunction)(void(*)(void)) eager_final_state_api_mode, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mode in dygraph."}

,

{"final_state_modulo", (PyCFunction)(void(*)(void)) eager_final_state_api_modulo, METH_VARARGS | METH_KEYWORDS, "C++ interface function for modulo in dygraph."}

,

{"final_state_momentum_", (PyCFunction)(void(*)(void)) eager_final_state_api_momentum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for momentum_ in dygraph."}

,

{"final_state_multi_dot", (PyCFunction)(void(*)(void)) eager_final_state_api_multi_dot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multi_dot in dygraph."}

,

{"final_state_multiclass_nms3", (PyCFunction)(void(*)(void)) eager_final_state_api_multiclass_nms3, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiclass_nms3 in dygraph."}

,

{"final_state_multinomial", (PyCFunction)(void(*)(void)) eager_final_state_api_multinomial, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multinomial in dygraph."}

,

{"final_state_multiplex", (PyCFunction)(void(*)(void)) eager_final_state_api_multiplex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiplex in dygraph."}

,

{"final_state_multiply", (PyCFunction)(void(*)(void)) eager_final_state_api_multiply, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiply in dygraph."}

,

{"final_state_nearest_interp", (PyCFunction)(void(*)(void)) eager_final_state_api_nearest_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nearest_interp in dygraph."}

,

{"final_state_nll_loss", (PyCFunction)(void(*)(void)) eager_final_state_api_nll_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nll_loss in dygraph."}

,

{"final_state_nms", (PyCFunction)(void(*)(void)) eager_final_state_api_nms, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nms in dygraph."}

,

{"final_state_norm", (PyCFunction)(void(*)(void)) eager_final_state_api_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for norm in dygraph."}

,

{"final_state_not_equal", (PyCFunction)(void(*)(void)) eager_final_state_api_not_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for not_equal in dygraph."}

,

{"final_state_one_hot", (PyCFunction)(void(*)(void)) eager_final_state_api_one_hot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for one_hot in dygraph."}

,

{"final_state_ones", (PyCFunction)(void(*)(void)) eager_final_state_api_ones, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ones in dygraph."}

,

{"final_state_ones_like", (PyCFunction)(void(*)(void)) eager_final_state_api_ones_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ones_like in dygraph."}

,

{"final_state_p_norm", (PyCFunction)(void(*)(void)) eager_final_state_api_p_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for p_norm in dygraph."}

,

{"final_state_pad", (PyCFunction)(void(*)(void)) eager_final_state_api_pad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad in dygraph."}

,

{"final_state_pad3d", (PyCFunction)(void(*)(void)) eager_final_state_api_pad3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad3d in dygraph."}

,

{"final_state_pixel_shuffle", (PyCFunction)(void(*)(void)) eager_final_state_api_pixel_shuffle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pixel_shuffle in dygraph."}

,

{"final_state_pool2d", (PyCFunction)(void(*)(void)) eager_final_state_api_pool2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool2d in dygraph."}

,

{"final_state_pool2d_gpudnn_unused", (PyCFunction)(void(*)(void)) eager_final_state_api_pool2d_gpudnn_unused, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool2d_gpudnn_unused in dygraph."}

,

{"final_state_pool3d", (PyCFunction)(void(*)(void)) eager_final_state_api_pool3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool3d in dygraph."}

,

{"final_state_pow", (PyCFunction)(void(*)(void)) eager_final_state_api_pow, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pow in dygraph."}

,

{"final_state_prelu", (PyCFunction)(void(*)(void)) eager_final_state_api_prelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prelu in dygraph."}

,

{"final_state_prior_box", (PyCFunction)(void(*)(void)) eager_final_state_api_prior_box, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prior_box in dygraph."}

,

{"final_state_psroi_pool", (PyCFunction)(void(*)(void)) eager_final_state_api_psroi_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for psroi_pool in dygraph."}

,

{"final_state_put_along_axis", (PyCFunction)(void(*)(void)) eager_final_state_api_put_along_axis, METH_VARARGS | METH_KEYWORDS, "C++ interface function for put_along_axis in dygraph."}


,
{"final_state_put_along_axis_", (PyCFunction)(void(*)(void)) eager_final_state_api_put_along_axis_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for put_along_axis_ in dygraph."}

,

{"final_state_qr", (PyCFunction)(void(*)(void)) eager_final_state_api_qr, METH_VARARGS | METH_KEYWORDS, "C++ interface function for qr in dygraph."}

,

{"final_state_randint", (PyCFunction)(void(*)(void)) eager_final_state_api_randint, METH_VARARGS | METH_KEYWORDS, "C++ interface function for randint in dygraph."}

,

{"final_state_randperm", (PyCFunction)(void(*)(void)) eager_final_state_api_randperm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for randperm in dygraph."}

,

{"final_state_real", (PyCFunction)(void(*)(void)) eager_final_state_api_real, METH_VARARGS | METH_KEYWORDS, "C++ interface function for real in dygraph."}

,

{"final_state_reciprocal", (PyCFunction)(void(*)(void)) eager_final_state_api_reciprocal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reciprocal in dygraph."}


,
{"final_state_reciprocal_", (PyCFunction)(void(*)(void)) eager_final_state_api_reciprocal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reciprocal_ in dygraph."}

,

{"final_state_reduce_prod", (PyCFunction)(void(*)(void)) eager_final_state_api_reduce_prod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_prod in dygraph."}

,

{"final_state_relu", (PyCFunction)(void(*)(void)) eager_final_state_api_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu in dygraph."}


,
{"final_state_relu_", (PyCFunction)(void(*)(void)) eager_final_state_api_relu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu_ in dygraph."}

,

{"final_state_relu6", (PyCFunction)(void(*)(void)) eager_final_state_api_relu6, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu6 in dygraph."}

,

{"final_state_renorm", (PyCFunction)(void(*)(void)) eager_final_state_api_renorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for renorm in dygraph."}

,

{"final_state_repeat_interleave", (PyCFunction)(void(*)(void)) eager_final_state_api_repeat_interleave, METH_VARARGS | METH_KEYWORDS, "C++ interface function for repeat_interleave in dygraph."}

,

{"final_state_repeat_interleave_with_tensor_index", (PyCFunction)(void(*)(void)) eager_final_state_api_repeat_interleave_with_tensor_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for repeat_interleave_with_tensor_index in dygraph."}

,

{"final_state_reshape", (PyCFunction)(void(*)(void)) eager_final_state_api_reshape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape in dygraph."}


,
{"final_state_reshape_", (PyCFunction)(void(*)(void)) eager_final_state_api_reshape_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape_ in dygraph."}

,

{"final_state_reverse", (PyCFunction)(void(*)(void)) eager_final_state_api_reverse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reverse in dygraph."}

,

{"final_state_reverse_array", (PyCFunction)(void(*)(void)) eager_final_state_api_reverse_array, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reverse_array in dygraph."}

,

{"final_state_rmsprop_", (PyCFunction)(void(*)(void)) eager_final_state_api_rmsprop_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rmsprop_ in dygraph."}

,

{"final_state_roi_align", (PyCFunction)(void(*)(void)) eager_final_state_api_roi_align, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_align in dygraph."}

,

{"final_state_roi_pool", (PyCFunction)(void(*)(void)) eager_final_state_api_roi_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_pool in dygraph."}

,

{"final_state_roll", (PyCFunction)(void(*)(void)) eager_final_state_api_roll, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roll in dygraph."}

,

{"final_state_round", (PyCFunction)(void(*)(void)) eager_final_state_api_round, METH_VARARGS | METH_KEYWORDS, "C++ interface function for round in dygraph."}


,
{"final_state_round_", (PyCFunction)(void(*)(void)) eager_final_state_api_round_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for round_ in dygraph."}

,

{"final_state_rsqrt", (PyCFunction)(void(*)(void)) eager_final_state_api_rsqrt, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rsqrt in dygraph."}


,
{"final_state_rsqrt_", (PyCFunction)(void(*)(void)) eager_final_state_api_rsqrt_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rsqrt_ in dygraph."}

,

{"final_state_scale", (PyCFunction)(void(*)(void)) eager_final_state_api_scale, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scale in dygraph."}


,
{"final_state_scale_", (PyCFunction)(void(*)(void)) eager_final_state_api_scale_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scale_ in dygraph."}

,

{"final_state_scatter", (PyCFunction)(void(*)(void)) eager_final_state_api_scatter, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scatter in dygraph."}


,
{"final_state_scatter_", (PyCFunction)(void(*)(void)) eager_final_state_api_scatter_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scatter_ in dygraph."}

,

{"final_state_scatter_nd_add", (PyCFunction)(void(*)(void)) eager_final_state_api_scatter_nd_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scatter_nd_add in dygraph."}

,

{"final_state_searchsorted", (PyCFunction)(void(*)(void)) eager_final_state_api_searchsorted, METH_VARARGS | METH_KEYWORDS, "C++ interface function for searchsorted in dygraph."}

,

{"final_state_segment_pool", (PyCFunction)(void(*)(void)) eager_final_state_api_segment_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for segment_pool in dygraph."}

,

{"final_state_selu", (PyCFunction)(void(*)(void)) eager_final_state_api_selu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for selu in dygraph."}

,

{"final_state_sgd_", (PyCFunction)(void(*)(void)) eager_final_state_api_sgd_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sgd_ in dygraph."}

,

{"final_state_shape", (PyCFunction)(void(*)(void)) eager_final_state_api_shape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shape in dygraph."}

,

{"final_state_shard_index", (PyCFunction)(void(*)(void)) eager_final_state_api_shard_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shard_index in dygraph."}

,

{"final_state_sigmoid", (PyCFunction)(void(*)(void)) eager_final_state_api_sigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid in dygraph."}

,

{"final_state_sigmoid_cross_entropy_with_logits", (PyCFunction)(void(*)(void)) eager_final_state_api_sigmoid_cross_entropy_with_logits, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_cross_entropy_with_logits in dygraph."}

,

{"final_state_sign", (PyCFunction)(void(*)(void)) eager_final_state_api_sign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sign in dygraph."}

,

{"final_state_silu", (PyCFunction)(void(*)(void)) eager_final_state_api_silu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for silu in dygraph."}

,

{"final_state_sin", (PyCFunction)(void(*)(void)) eager_final_state_api_sin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sin in dygraph."}

,

{"final_state_sinh", (PyCFunction)(void(*)(void)) eager_final_state_api_sinh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sinh in dygraph."}

,

{"final_state_size", (PyCFunction)(void(*)(void)) eager_final_state_api_size, METH_VARARGS | METH_KEYWORDS, "C++ interface function for size in dygraph."}

,

{"final_state_slice", (PyCFunction)(void(*)(void)) eager_final_state_api_slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for slice in dygraph."}

,

{"final_state_slogdet", (PyCFunction)(void(*)(void)) eager_final_state_api_slogdet, METH_VARARGS | METH_KEYWORDS, "C++ interface function for slogdet in dygraph."}

,

{"final_state_soft_shrink", (PyCFunction)(void(*)(void)) eager_final_state_api_soft_shrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for soft_shrink in dygraph."}

,

{"final_state_softmax", (PyCFunction)(void(*)(void)) eager_final_state_api_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax in dygraph."}


,
{"final_state_softmax_", (PyCFunction)(void(*)(void)) eager_final_state_api_softmax_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax_ in dygraph."}

,

{"final_state_softplus", (PyCFunction)(void(*)(void)) eager_final_state_api_softplus, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softplus in dygraph."}

,

{"final_state_softsign", (PyCFunction)(void(*)(void)) eager_final_state_api_softsign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softsign in dygraph."}

,

{"final_state_spectral_norm", (PyCFunction)(void(*)(void)) eager_final_state_api_spectral_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for spectral_norm in dygraph."}

,

{"final_state_split", (PyCFunction)(void(*)(void)) eager_final_state_api_split, METH_VARARGS | METH_KEYWORDS, "C++ interface function for split in dygraph."}

,

{"final_state_sqrt", (PyCFunction)(void(*)(void)) eager_final_state_api_sqrt, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt in dygraph."}


,
{"final_state_sqrt_", (PyCFunction)(void(*)(void)) eager_final_state_api_sqrt_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt_ in dygraph."}

,

{"final_state_square", (PyCFunction)(void(*)(void)) eager_final_state_api_square, METH_VARARGS | METH_KEYWORDS, "C++ interface function for square in dygraph."}

,

{"final_state_squared_l2_norm", (PyCFunction)(void(*)(void)) eager_final_state_api_squared_l2_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squared_l2_norm in dygraph."}

,

{"final_state_squeeze", (PyCFunction)(void(*)(void)) eager_final_state_api_squeeze, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze in dygraph."}


,
{"final_state_squeeze_", (PyCFunction)(void(*)(void)) eager_final_state_api_squeeze_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze_ in dygraph."}

,

{"final_state_stack", (PyCFunction)(void(*)(void)) eager_final_state_api_stack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stack in dygraph."}

,

{"final_state_strided_slice", (PyCFunction)(void(*)(void)) eager_final_state_api_strided_slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for strided_slice in dygraph."}

,

{"final_state_subtract", (PyCFunction)(void(*)(void)) eager_final_state_api_subtract, METH_VARARGS | METH_KEYWORDS, "C++ interface function for subtract in dygraph."}


,
{"final_state_subtract_", (PyCFunction)(void(*)(void)) eager_final_state_api_subtract_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for subtract_ in dygraph."}

,

{"final_state_sum", (PyCFunction)(void(*)(void)) eager_final_state_api_sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sum in dygraph."}

,

{"final_state_svd", (PyCFunction)(void(*)(void)) eager_final_state_api_svd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for svd in dygraph."}

,

{"final_state_swish", (PyCFunction)(void(*)(void)) eager_final_state_api_swish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for swish in dygraph."}

,

{"final_state_sync_batch_norm_", (PyCFunction)(void(*)(void)) eager_final_state_api_sync_batch_norm_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sync_batch_norm_ in dygraph."}

,

{"final_state_take_along_axis", (PyCFunction)(void(*)(void)) eager_final_state_api_take_along_axis, METH_VARARGS | METH_KEYWORDS, "C++ interface function for take_along_axis in dygraph."}

,

{"final_state_tan", (PyCFunction)(void(*)(void)) eager_final_state_api_tan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tan in dygraph."}

,

{"final_state_tanh", (PyCFunction)(void(*)(void)) eager_final_state_api_tanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh in dygraph."}


,
{"final_state_tanh_", (PyCFunction)(void(*)(void)) eager_final_state_api_tanh_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh_ in dygraph."}

,

{"final_state_tanh_shrink", (PyCFunction)(void(*)(void)) eager_final_state_api_tanh_shrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh_shrink in dygraph."}

,

{"final_state_temporal_shift", (PyCFunction)(void(*)(void)) eager_final_state_api_temporal_shift, METH_VARARGS | METH_KEYWORDS, "C++ interface function for temporal_shift in dygraph."}

,

{"final_state_thresholded_relu", (PyCFunction)(void(*)(void)) eager_final_state_api_thresholded_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for thresholded_relu in dygraph."}

,

{"final_state_tile", (PyCFunction)(void(*)(void)) eager_final_state_api_tile, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tile in dygraph."}

,

{"final_state_top_k", (PyCFunction)(void(*)(void)) eager_final_state_api_top_k, METH_VARARGS | METH_KEYWORDS, "C++ interface function for top_k in dygraph."}

,

{"final_state_transpose", (PyCFunction)(void(*)(void)) eager_final_state_api_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transpose in dygraph."}

,

{"final_state_triangular_solve", (PyCFunction)(void(*)(void)) eager_final_state_api_triangular_solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triangular_solve in dygraph."}

,

{"final_state_tril_indices", (PyCFunction)(void(*)(void)) eager_final_state_api_tril_indices, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tril_indices in dygraph."}

,

{"final_state_tril_triu", (PyCFunction)(void(*)(void)) eager_final_state_api_tril_triu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tril_triu in dygraph."}

,

{"final_state_trilinear_interp", (PyCFunction)(void(*)(void)) eager_final_state_api_trilinear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trilinear_interp in dygraph."}

,

{"final_state_truncated_gaussian_random", (PyCFunction)(void(*)(void)) eager_final_state_api_truncated_gaussian_random, METH_VARARGS | METH_KEYWORDS, "C++ interface function for truncated_gaussian_random in dygraph."}

,

{"final_state_unbind", (PyCFunction)(void(*)(void)) eager_final_state_api_unbind, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unbind in dygraph."}

,

{"final_state_unfold", (PyCFunction)(void(*)(void)) eager_final_state_api_unfold, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unfold in dygraph."}

,

{"final_state_uniform_random", (PyCFunction)(void(*)(void)) eager_final_state_api_uniform_random, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_random in dygraph."}

,

{"final_state_unique", (PyCFunction)(void(*)(void)) eager_final_state_api_unique, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unique in dygraph."}

,

{"final_state_unique_consecutive", (PyCFunction)(void(*)(void)) eager_final_state_api_unique_consecutive, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unique_consecutive in dygraph."}

,

{"final_state_unsqueeze", (PyCFunction)(void(*)(void)) eager_final_state_api_unsqueeze, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze in dygraph."}


,
{"final_state_unsqueeze_", (PyCFunction)(void(*)(void)) eager_final_state_api_unsqueeze_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze_ in dygraph."}

,

{"final_state_unstack", (PyCFunction)(void(*)(void)) eager_final_state_api_unstack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unstack in dygraph."}

,

{"final_state_viterbi_decode", (PyCFunction)(void(*)(void)) eager_final_state_api_viterbi_decode, METH_VARARGS | METH_KEYWORDS, "C++ interface function for viterbi_decode in dygraph."}

,

{"final_state_warpctc", (PyCFunction)(void(*)(void)) eager_final_state_api_warpctc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for warpctc in dygraph."}

,

{"final_state_where", (PyCFunction)(void(*)(void)) eager_final_state_api_where, METH_VARARGS | METH_KEYWORDS, "C++ interface function for where in dygraph."}

,

{"final_state_where_index", (PyCFunction)(void(*)(void)) eager_final_state_api_where_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for where_index in dygraph."}

,

{"final_state_yolo_box", (PyCFunction)(void(*)(void)) eager_final_state_api_yolo_box, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolo_box in dygraph."}

,

{"final_state_yolov3_loss", (PyCFunction)(void(*)(void)) eager_final_state_api_yolov3_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolov3_loss in dygraph."}

,

{"final_state_zeros", (PyCFunction)(void(*)(void)) eager_final_state_api_zeros, METH_VARARGS | METH_KEYWORDS, "C++ interface function for zeros in dygraph."}

,

{"final_state_zeros_like", (PyCFunction)(void(*)(void)) eager_final_state_api_zeros_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for zeros_like in dygraph."}

,

{"final_state_broadcast_tensors", (PyCFunction)(void(*)(void)) eager_final_state_api_broadcast_tensors, METH_VARARGS | METH_KEYWORDS, "C++ interface function for broadcast_tensors in dygraph."}

,

{"final_state_dirichlet", (PyCFunction)(void(*)(void)) eager_final_state_api_dirichlet, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dirichlet in dygraph."}

,

{"final_state_eig", (PyCFunction)(void(*)(void)) eager_final_state_api_eig, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eig in dygraph."}

,

{"final_state_fold", (PyCFunction)(void(*)(void)) eager_final_state_api_fold, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fold in dygraph."}

,

{"final_state_overlap_add", (PyCFunction)(void(*)(void)) eager_final_state_api_overlap_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for overlap_add in dygraph."}

,

{"final_state_uniform_random_inplace", (PyCFunction)(void(*)(void)) eager_final_state_api_uniform_random_inplace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_random_inplace in dygraph."}


,
{"final_state_uniform_random_inplace_", (PyCFunction)(void(*)(void)) eager_final_state_api_uniform_random_inplace_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_random_inplace_ in dygraph."}

,

{"final_state_unpool", (PyCFunction)(void(*)(void)) eager_final_state_api_unpool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unpool in dygraph."}

,

{"final_state_unpool3d", (PyCFunction)(void(*)(void)) eager_final_state_api_unpool3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unpool3d in dygraph."}

,


{"final_state_sparse_abs", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for abs in dygraph."}

,

{"final_state_sparse_acos", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_acos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acos in dygraph."}

,

{"final_state_sparse_acosh", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_acosh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acosh in dygraph."}

,

{"final_state_sparse_add", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add in dygraph."}

,

{"final_state_sparse_asin", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_asin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asin in dygraph."}

,

{"final_state_sparse_asinh", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_asinh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asinh in dygraph."}

,

{"final_state_sparse_atan", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_atan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan in dygraph."}

,

{"final_state_sparse_atanh", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_atanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atanh in dygraph."}

,

{"final_state_sparse_cast", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_cast, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cast in dygraph."}

,

{"final_state_sparse_conv3d", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_conv3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d in dygraph."}

,

{"final_state_sparse_coo_to_dense", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_coo_to_dense, METH_VARARGS | METH_KEYWORDS, "C++ interface function for coo_to_dense in dygraph."}

,

{"final_state_sparse_create_sparse_coo_tensor", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_create_sparse_coo_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for create_sparse_coo_tensor in dygraph."}

,

{"final_state_sparse_dense_to_coo", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_dense_to_coo, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dense_to_coo in dygraph."}

,

{"final_state_sparse_divide", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_divide, METH_VARARGS | METH_KEYWORDS, "C++ interface function for divide in dygraph."}

,

{"final_state_sparse_divide_scalar", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_divide_scalar, METH_VARARGS | METH_KEYWORDS, "C++ interface function for divide_scalar in dygraph."}

,

{"final_state_sparse_expm1", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_expm1, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expm1 in dygraph."}

,

{"final_state_sparse_leaky_relu", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_leaky_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for leaky_relu in dygraph."}

,

{"final_state_sparse_log1p", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_log1p, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log1p in dygraph."}

,

{"final_state_sparse_multiply", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_multiply, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiply in dygraph."}

,

{"final_state_sparse_pow", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_pow, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pow in dygraph."}

,

{"final_state_sparse_relu", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu in dygraph."}

,

{"final_state_sparse_relu6", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_relu6, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu6 in dygraph."}

,

{"final_state_sparse_scale", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_scale, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scale in dygraph."}

,

{"final_state_sparse_sin", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_sin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sin in dygraph."}

,

{"final_state_sparse_sinh", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_sinh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sinh in dygraph."}

,

{"final_state_sparse_softmax", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_softmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax in dygraph."}

,

{"final_state_sparse_sqrt", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_sqrt, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt in dygraph."}

,

{"final_state_sparse_square", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_square, METH_VARARGS | METH_KEYWORDS, "C++ interface function for square in dygraph."}

,

{"final_state_sparse_subtract", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_subtract, METH_VARARGS | METH_KEYWORDS, "C++ interface function for subtract in dygraph."}

,

{"final_state_sparse_tan", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_tan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tan in dygraph."}

,

{"final_state_sparse_tanh", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_tanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for tanh in dygraph."}

,

{"final_state_sparse_to_dense", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_to_dense, METH_VARARGS | METH_KEYWORDS, "C++ interface function for to_dense in dygraph."}

,

{"final_state_sparse_to_sparse_coo", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_to_sparse_coo, METH_VARARGS | METH_KEYWORDS, "C++ interface function for to_sparse_coo in dygraph."}

,

{"final_state_sparse_to_sparse_csr", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_to_sparse_csr, METH_VARARGS | METH_KEYWORDS, "C++ interface function for to_sparse_csr in dygraph."}

,

{"final_state_sparse_values", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_values, METH_VARARGS | METH_KEYWORDS, "C++ interface function for values in dygraph."}

,

{"final_state_sparse_addmm", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_addmm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for addmm in dygraph."}

,

{"final_state_sparse_coalesce", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_coalesce, METH_VARARGS | METH_KEYWORDS, "C++ interface function for coalesce in dygraph."}

,

{"final_state_sparse_full_like", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_full_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for full_like in dygraph."}

,

{"final_state_sparse_fused_attention", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_fused_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_attention in dygraph."}

,

{"final_state_sparse_masked_matmul", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_masked_matmul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for masked_matmul in dygraph."}

,

{"final_state_sparse_matmul", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_matmul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matmul in dygraph."}

,

{"final_state_sparse_maxpool", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_maxpool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for maxpool in dygraph."}

,

{"final_state_sparse_mv", (PyCFunction)(void(*)(void)) sparse::eager_final_state_api_mv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mv in dygraph."}

,


{"final_state_strings_empty", (PyCFunction)(void(*)(void)) strings::eager_final_state_api_empty, METH_VARARGS | METH_KEYWORDS, "C++ interface function for empty in dygraph."}

,

{"final_state_strings_empty_like", (PyCFunction)(void(*)(void)) strings::eager_final_state_api_empty_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for empty_like in dygraph."}

,

{"final_state_strings_lower", (PyCFunction)(void(*)(void)) strings::eager_final_state_api_lower, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lower in dygraph."}

,

{"final_state_strings_upper", (PyCFunction)(void(*)(void)) strings::eager_final_state_api_upper, METH_VARARGS | METH_KEYWORDS, "C++ interface function for upper in dygraph."}

,


    {"get_final_state_core_ops_args_info",
    (PyCFunction)(void(*)(void))eager_get_final_state_core_ops_args_info, METH_NOARGS,
    "C++ interface function for eager_get_final_state_core_ops_args_info."},
    {"get_final_state_core_ops_args_type_info",
    (PyCFunction)(void(*)(void))eager_get_final_state_core_ops_args_type_info,
    METH_NOARGS,
    "C++ interface function for eager_get_final_state_core_ops_args_type_info."},
    {"get_final_state_core_ops_returns_info",
    (PyCFunction)(void(*)(void))eager_get_final_state_core_ops_returns_info,
    METH_NOARGS, "C++ interface function for eager_get_final_state_core_ops_returns_info."},

 {nullptr,nullptr,0,nullptr}
};

void BindFinalStateEagerOpFunctions(pybind11::module *module) {
  if (PyModule_AddFunctions(module->ptr(), EagerFinalStateMethods) < 0) {
    PADDLE_THROW(platform::errors::Fatal ("Add functions to core.eager.ops failed!"));
  }

  if (PyModule_AddFunctions(module->ptr(), CustomEagerFinalStateMethods) < 0) {
    PADDLE_THROW(platform::errors::Fatal ("Add functions to core.eager.ops failed!"));
  }
}

} // namespace pybind
} // namespace paddle
