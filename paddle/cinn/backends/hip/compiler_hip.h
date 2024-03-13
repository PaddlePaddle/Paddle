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
#ifdef CINN_WITH_ROCM
#if defined(__linux__)
#include <sys/stat.h>
#endif
#include <glog/logging.h>

#include <string>
#include <vector>

namespace cinn {
namespace backends {
namespace hiprtc {

/**
 * An helper class to call HIPRTC. Input HIP device source code, get PTX string.
 */
class Compiler {
 public:
  Compiler();

  /**
   * Compile the \p code and get PTX string.
   * @param code The HIP source code.
   * @param include_headers Whether to include the headers of HIP and CINN
   * runtime modules.
   * @return Compiled PTX code string.
   */
  std::string operator()(const std::string& code, bool include_headers = true);

 private:
  /**
   * Get the directories of HIP's header files.
   * @return list of header file directories.
   */
  std::vector<std::string> FindHIPIncludePaths();

  /**
   * Get the directories of CINN runtime's header files.
   * @return list of header file directories.
   */
  std::vector<std::string> FindCINNRuntimeIncludePaths();

  /**
   * Compile HIP source code and get PTX or CUBIN.
   * @param code source code string.
   * @return PTX or CUBIN string.
   */
  std::string CompileHipSource(const std::string& code, bool include_headers);

  std::string ReadFile(const std::string&, std::ios_base::openmode);

  std::string prefix_name_{""};
};

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn

#endif  // CINN_WITH_ROCM
