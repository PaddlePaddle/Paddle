// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/runtime/xpu/xpu_module.h"

#include "paddle/cinn/utils/profiler.h"

namespace cinn {
namespace runtime {
namespace xpu {

XpuModule::XpuModule(const std::string& data, Kind kind)
    : data_(data), kind_(kind) {
  PADDLE_ENFORCE_EQ(
      data.empty(),
      false,
      ::common::errors::PreconditionNotMet("XpuModule: data is empty."));

  XPU_DRIVER_CHECK(cuDeviceGetCount(&num_devices_));
  PADDLE_ENFORCE_GT(
      num_devices_,
      0,
      ::common::errors::Fatal("XpuModule: No available CUDA devices."));

  int current_device_id = 0;
  XPU_CHECK(cudaGetDevice(&current_device_id));
  XPU_DRIVER_CHECK(cuDeviceGet(&device_, current_device_id));
  XPU_DRIVER_CHECK(cuDevicePrimaryCtxRetain(&context_, device_));
}

CUfunction XpuModule::GetFunction(int device_id,
                                  const std::string& func_name) {
  VLOG(3) << "XpuModule::GetFunction: " << func_name
          << " device_id=" << device_id;
  cinn::utils::RecordEvent record_run("xpuGetFunction",
                                      cinn::utils::EventType::kOrdinary);
  if (!module_per_card_[device_id]) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!module_per_card_[device_id]) {
      XPU_DRIVER_CHECK(cuCtxSetCurrent(context_));
      if (kind_ == Kind::PTX) {
        XPU_DRIVER_CHECK(cuModuleLoadDataEx(
            &module_per_card_[device_id], data_.c_str(), 0, nullptr, nullptr));
      } else {
        // CUBIN: load from raw bytes
        XPU_DRIVER_CHECK(cuModuleLoadData(&module_per_card_[device_id],
                                          data_.c_str()));
      }
    }
  }

  CUfunction func;
  XPU_DRIVER_CHECK(cuModuleGetFunction(
      &func, module_per_card_[device_id], func_name.c_str()));
  return func;
}

CUfunction XpuModule::GetFunction(const std::string& func_name) {
  int device_id = 0;
  XPU_CHECK(cudaGetDevice(&device_id));
  return GetFunction(device_id, func_name);
}

XpuModule::~XpuModule() {
  for (int i = 0; i < static_cast<int>(module_per_card_.size()); ++i) {
    if (module_per_card_[i]) {
      XPU_CHECK(cudaSetDevice(i));
      XPU_DRIVER_CHECK(cuModuleUnload(module_per_card_[i]));
    }
  }
}

}  // namespace xpu
}  // namespace runtime
}  // namespace cinn
