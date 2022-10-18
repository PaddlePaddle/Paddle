/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Corporation. All rights reserved.

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
#include <iostream>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/phi/backends/dynload/cudnn.h"

DECLARE_bool(cudnn_frontend_enable);
DECLARE_bool(cudnn_frontend_debug);

// Redirect the CUDNN APIs in the cudnn_frontend namespace to
// the functions in phi::dynload
#define CUDNN_FRONTEND_OVERRIDE_SYMBOL(__name) using phi::dynload::__name

#define CUDNN_FRONTEND_APPLY_EACH(__macro) \
  __macro(cudnnBackendCreateDescriptor);   \
  __macro(cudnnBackendDestroyDescriptor);  \
  __macro(cudnnBackendExecute);            \
  __macro(cudnnBackendFinalize);           \
  __macro(cudnnBackendGetAttribute);       \
  __macro(cudnnBackendSetAttribute);       \
  __macro(cudnnCreateFilterDescriptor);    \
  __macro(cudnnDestroyFilterDescriptor);   \
  __macro(cudnnGetStream);                 \
  __macro(cudnnGetVersion);                \
  __macro(cudnnReorderFilterAndBias);      \
  __macro(cudnnSetFilterNdDescriptor);

namespace cudnn_frontend {
CUDNN_FRONTEND_APPLY_EACH(CUDNN_FRONTEND_OVERRIDE_SYMBOL);
}  // namespace cudnn_frontend

// clang-format off
#include <cudnn_frontend.h>                                        // NOLINT
#include <cudnn_frontend_find_plan.h>                              // NOLINT
#include <cudnn_frontend_get_plan.h>                               // NOLINT
// clang-format on

namespace paddle {
namespace platform {
inline bool IsCudnnFrontendEnabled() {
  int cudnn_version = paddle::platform::DnnVersion();
  bool flag_enabled = FLAGS_cudnn_frontend_enable && (cudnn_version >= 8200);
  VLOG(3) << "[cudnn_frontend] FLAGS_cudnn_frontend_debug="
          << FLAGS_cudnn_frontend_debug
          << "; flag_enabled=" << FLAGS_cudnn_frontend_enable;
  return flag_enabled;
}
}  // namespace platform
}  // namespace paddle
