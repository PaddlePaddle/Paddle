/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/platform/device/mlu/device_context.h"
#endif

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_MLU
thread_local std::unordered_map<const MLUDeviceContext*,
                                std::shared_ptr<MLUContext>>
    MLUDeviceContext::thread_ctx_;
thread_local std::mutex MLUDeviceContext::ctx_mtx_;

MLUContext::MLUContext(const MLUPlace& place, const int priority) {
  place_ = place;
  MLUDeviceGuard guard(place_.device);
  stream_.reset(new stream::MLUStream(place_, priority));
  InitCNNLContext();
  InitMLUOPContext();
}

MLUContext::~MLUContext() {
  MLUDeviceGuard guard(place_.device);
  DestoryCNNLContext();
  DestoryMLUOPContext();
}

MLUDeviceContext::MLUDeviceContext(MLUPlace place) : place_(place) {
  MLUDeviceGuard guard(place_.device);
  compute_capability_ = GetMLUComputeCapability(place_.device);
  driver_version_ = GetMLUDriverVersion(place_.device);
  runtime_version_ = GetMLURuntimeVersion(place_.device);
  cnnl_version_ = GetMLUCnnlVersion(place_.device);
  mluOp_version_ = GetMLUOpVersion(place_.device);

  LOG_FIRST_N(WARNING, 1)
      << "Please NOTE: device: " << static_cast<int>(place_.device)
      << ", MLU Compute Capability: " << compute_capability_ / 10 << "."
      << compute_capability_ % 10
      << ", Driver API Version: " << driver_version_ / 10000 << "."
      << (driver_version_ / 100) % 100 << "." << driver_version_ % 100
      << ", Runtime API Version: " << runtime_version_ / 10000 << "."
      << (runtime_version_ / 100) % 100 << "." << runtime_version_ % 100
      << ", Cnnl API Version: " << cnnl_version_ / 10000 << "."
      << (cnnl_version_ / 100) % 100 << "." << cnnl_version_ % 100
      << ", MluOp API Version: " << mluOp_version_ / 10000 << "."
      << (mluOp_version_ / 100) % 100 << "." << mluOp_version_ % 100;

  default_ctx_.reset(new MLUContext(place_));
}

MLUDeviceContext::~MLUDeviceContext() {}

const Place& MLUDeviceContext::GetPlace() const { return place_; }

void MLUDeviceContext::Wait() const { context()->Stream()->Wait(); }

int MLUDeviceContext::GetComputeCapability() const {
  return compute_capability_;
}

mluCnnlHandle MLUDeviceContext::cnnl_handle() const {
  return context()->CnnlHandle();
}

mluOpHandle MLUDeviceContext::mluOp_handle() const {
  return context()->MluOpHandle();
}

mluStream MLUDeviceContext::stream() const { return context()->RawStream(); }

#endif
}  // namespace platform
}  // namespace paddle
