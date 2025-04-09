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

#include "paddle/cinn/runtime/hip/hip_module.h"

#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/profiler.h"

namespace cinn {
namespace runtime {
namespace hip {

HIPModule::HIPModule(const std::string& data) : data_(data) {
  PADDLE_ENFORCE_EQ(
      data.empty(),
      false,
      ::common::errors::PreconditionNotMet("HIP Module Error: data is empty."));

  hipGetDeviceCount(&num_devices_);
  PADDLE_ENFORCE_GT(
      num_devices_,
      0,
      ::common::errors::Fatal("HIP Module Error: No available devices."));

  int current_device_id;
  hipGetDevice(&current_device_id);
  hipSetDevice(current_device_id);
  hipDeviceGet(&device_, current_device_id);
  hipCtxGetCurrent(&context_);
  hipDevicePrimaryCtxRetain(&context_, device_);
}

hipFunction_t HIPModule::GetFunction(int device_id,
                                     const std::string& func_name) {
  VLOG(3) << "GetFunction : " << func_name << " with device_id : " << device_id;
  cinn::utils::RecordEvent record_run("hipGetFunction",
                                      cinn::utils::EventType::kOrdinary);
  if (!module_per_card_[device_id]) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Compilation with parameters
    const size_t jit_num_options = 5;
    std::vector<hipJitOption> jit_options(jit_num_options);
    std::vector<void*> jit_opt_vals(jit_num_options);

    // set up size of compilation log buffer
    jit_options[0] = hipJitOptionErrorLogBufferSizeBytes;
    size_t log_buffer_size = 1024;
    jit_opt_vals[0] = reinterpret_cast<void*>(log_buffer_size);

    // set up pointer to the compilation log buffer
    jit_options[1] = hipJitOptionErrorLogBuffer;
    std::vector<char> log_buffer(log_buffer_size, '\0');
    jit_opt_vals[1] = log_buffer.data();

    int value = 1;
    // Specifies whether to create debug information in output (-g)
    jit_options[2] = hipJitOptionGenerateDebugInfo;
    jit_opt_vals[2] = reinterpret_cast<void*>(value);

    // Generate verbose log messages
    jit_options[3] = hipJitOptionLogVerbose;
    jit_opt_vals[3] = reinterpret_cast<void*>(value);

    // Generate line number information (-lineinfo)
    jit_options[4] = hipJitOptionGenerateLineInfo;
    jit_opt_vals[4] = reinterpret_cast<void*>(value);

    if (runtime::UseHipccCompiler()) {
      HIP_DRIVER_CHECK(
          hipModuleLoad(&module_per_card_[device_id], data_.c_str()));
    } else {
      HIP_DRIVER_CHECK(hipModuleLoadDataEx(&module_per_card_[device_id],
                                           data_.c_str(),
                                           jit_num_options,
                                           jit_options.data(),
                                           jit_opt_vals.data()));
    }
  }

  hipFunction_t func;
  HIP_DRIVER_CHECK(hipModuleGetFunction(
      &func, module_per_card_[device_id], func_name.c_str()));
  return func;
}

HIPModule::~HIPModule() {
  for (int i = 0; i < module_per_card_.size(); i++) {
    auto* module = module_per_card_[i];
    if (module) {
      HIP_CHECK(hipSetDevice(i));
      HIP_DRIVER_CHECK(hipModuleUnload(module));
    }
  }
}

}  // namespace hip
}  // namespace runtime
}  // namespace cinn
