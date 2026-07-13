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

#include <string>
#include <vector>

namespace cinn {
namespace backends {
namespace xpurtc {

/**
 * Helper class to compile XPU (CUDA-compatible) device source code using
 * NVRTC. Input is CUDA device source code; output is a PTX or CUBIN string.
 */
class Compiler {
 public:
  Compiler() {}

  /**
   * Compile \p code and return PTX string.
   * @param code  XPU/CUDA device source code.
   * @param include_headers  Whether to inject CINN runtime headers.
   * @return Compiled PTX code string.
   */
  std::string operator()(const std::string& code, bool include_headers = true);

 private:
  /**
   * Get the CUDA include directories.
   */
  std::vector<std::string> FindCUDAIncludePaths();

  /**
   * Get the CINN runtime include directories.
   */
  std::vector<std::string> FindCINNRuntimeIncludePaths();

  /**
   * Compile using NVRTC.
   */
  std::string CompileWithNvrtc(const std::string& code, bool include_headers);

  /**
   * Query the SM architecture string for the current device (e.g. "sm_80").
   */
  std::string GetDeviceArch();

  std::string prefix_name_{""};
};

}  // namespace xpurtc
}  // namespace backends
}  // namespace cinn
