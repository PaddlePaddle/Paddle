// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <glog/logging.h>

#include <paddle/cinn/common/common.h>

namespace cinn {
namespace backends {
namespace syclrtc {

/**
 * Input SYCL source code, get shared library.
 */
class Compiler {
 public:
  Compiler();

  /**
   * Compile the \p code to shared library.
   * @param code The SYCL source code.
   * @return Compiled shared library path.
   */
  std::string operator()(const std::string& code, const Target::Arch gpu_type);

 private:

  /**
   * Get the directories of CINN runtime's header files.
   * @return list of header file directories.
   */
  std::vector<std::string> FindCINNRuntimeIncludePaths();
  // compile to share library
  std::string CompileToSo(const std::string& code, const Target::Arch gpu_type);

  std::string GetDeviceArch();

  std::string ReadFile(const std::string&, std::ios_base::openmode);
};

}  // namespace syclrtc
}  // namespace backends
}  // namespace cinn