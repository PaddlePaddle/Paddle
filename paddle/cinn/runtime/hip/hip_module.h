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

#pragma once

#include "paddle/cinn/runtime/hip/hip_util.h"

#include <mutex>
#include <string>
#include <vector>

namespace cinn {
namespace runtime {
namespace hip {

const int kHIPMaxCards{8};

/**
 * The HIP module, helps to compile HIP codes and fetch symbols.
 * Currently, it is a wrapper of HIPRTC.
 */
class HIPModule {
 public:
  explicit HIPModule(const std::string& data);

  //! Get a function.
  hipFunction_t GetFunction(int device_id, const std::string& func_name);

  ~HIPModule();

 private:
  //! The input data.
  std::string data_;
  //! To make parallel, we prepare one module for each card.
  std::vector<hipModule_t> module_per_card_{kHIPMaxCards, nullptr};
  std::string hip_source_;
  std::mutex mutex_;

  hipDevice_t device_;
  hipCtx_t context_;
  int num_devices_{0};
};

}  // namespace hip
}  // namespace runtime
}  // namespace cinn
