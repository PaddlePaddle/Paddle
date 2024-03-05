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

#include <iostream>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace pybind {

static PyObject *eager_api_linear(PyObject *self,
                                  PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto x = GetTensorFromArgs("linear", "X", args, 0, false);
    auto weight = GetTensorFromArgs("linear", "weight", args, 1, false);
    auto bias = GetTensorFromArgs("linear", "Bias", args, 2, true);
    tstate = PyEval_SaveThread();
    if (bias.initialized()) {
      auto mm_out = matmul_ad_func(x, weight, false, false);
      auto out = add_ad_func(mm_out, bias);
      PyEval_RestoreThread(tstate);
      tstate = nullptr;
      return ToPyObject(out);
    } else {
      auto mm_out = matmul_ad_func(x, weight, false, false);
      PyEval_RestoreThread(tstate);
      tstate = nullptr;
      return ToPyObject(mm_out);
    }
  } catch (paddle::platform::EnforceNotMet &exception) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    std::ostringstream sout;
    sout << exception.what();
    sout << "  [operator < linear > error]";
    exception.set_error_str(sout.str());
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_flash_attn(PyObject *self,
                                      PyObject *args,
                                      PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event(
      "flash_attn pybind_imperative_func",
      paddle::platform::TracerEventType::UserDefined,
      1);
  PyThreadState *tstate = nullptr;
  try {
    VLOG(6) << "Running Eager Final State API: flash_attn";

    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get EagerTensors from args
    auto q = GetTensorFromArgs("flash_attn", "q", args, 0, false);
    auto k = GetTensorFromArgs("flash_attn", "k", args, 1, false);
    auto v = GetTensorFromArgs("flash_attn", "v", args, 2, false);
    auto fixed_seed_offset = GetOptionalTensorFromArgs(
        "flash_attn", "fixed_seed_offset", args, 3, true);
    auto attn_mask =
        GetOptionalTensorFromArgs("flash_attn", "attn_mask", args, 4, true);

    paddle::optional<paddle::Tensor> attn_mask_start_row_indices;
    int attn_mask_start_row = 0;
    ssize_t index_offset = 0;
    if (PyTuple_Size(args) == 12) {
      attn_mask_start_row_indices = GetOptionalTensorFromArgs(
          "flash_attn", "attn_mask_start_row_indices", args, 5, true);
      PyObject *attn_mask_start_row_obj = PyTuple_GET_ITEM(args, 11);
      attn_mask_start_row =
          CastPyArg2Int(attn_mask_start_row_obj, "flash_attn", 11);
      index_offset = 1;
    }

    // Parse Attributes if needed
    PyObject *dropout_obj = PyTuple_GET_ITEM(args, 6 - index_offset);
    float dropout =
        CastPyArg2Float(dropout_obj, "flash_attn", 6 - index_offset);
    PyObject *causal_obj = PyTuple_GET_ITEM(args, 7 - index_offset);
    bool causal = CastPyArg2Boolean(causal_obj, "flash_attn", 7 - index_offset);
    PyObject *return_softmax_obj = PyTuple_GET_ITEM(args, 8 - index_offset);
    bool return_softmax =
        CastPyArg2Boolean(return_softmax_obj, "flash_attn", 8 - index_offset);
    PyObject *is_test_obj = PyTuple_GET_ITEM(args, 9 - index_offset);
    bool is_test =
        CastPyArg2Boolean(is_test_obj, "flash_attn", 9 - index_offset);
    PyObject *rng_name_obj = PyTuple_GET_ITEM(args, 10 - index_offset);
    std::string rng_name =
        CastPyArg2String(rng_name_obj, "flash_attn", 10 - index_offset);

    tstate = PyEval_SaveThread();

    // Set Device ID
    auto place = egr::Controller::Instance().GetExpectedPlace();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(4) << "CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId()
              << " from " << (int)place.device;  // NOLINT
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }
    if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(4) << "CurrentDeviceId: "
              << phi::DeviceManager::GetDevice(place.GetDeviceType())
              << " from " << (int)place.device;  // NOLINT
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with CUSTOM_DEVICE if use "
          "CustomPlace."));
#endif
    }
    if (paddle::platform::is_xpu_place(place)) {
#if defined(PADDLE_WITH_XPU)
      phi::backends::xpu::SetXPUDeviceId(place.device);
      VLOG(4) << "CurrentDeviceId: "
              << phi::backends::xpu::GetXPUCurrentDeviceId() << " from "
              << (int)place.device;  // NOLINT
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with XPU if use XPUPlace."));
#endif
    }

    // Call dygraph function
    decltype(::flash_attn_ad_func(q,
                                  k,
                                  v,
                                  fixed_seed_offset,
                                  attn_mask,
                                  attn_mask_start_row_indices,
                                  dropout,
                                  causal,
                                  return_softmax,
                                  is_test,
                                  rng_name,
                                  attn_mask_start_row)) out =
        ::flash_attn_ad_func(q,
                             k,
                             v,
                             fixed_seed_offset,
                             attn_mask,
                             attn_mask_start_row_indices,
                             dropout,
                             causal,
                             return_softmax,
                             is_test,
                             rng_name,
                             attn_mask_start_row);

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyMethodDef CustomEagerFinalStateMethods[] = {
    {"linear",
     (PyCFunction)(void (*)(void))eager_api_linear,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for run_program in dygraph."},
    {"flash_attn",
     (PyCFunction)(void (*)(void))eager_api_flash_attn,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for flash_attn in dygraph."},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace pybind
}  // namespace paddle
