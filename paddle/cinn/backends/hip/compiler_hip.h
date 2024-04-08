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
 * An helper class to call HIPRTC. Input HIP device source code, get hsaco
 * string.
 */
class Compiler {
 public:
  Compiler();

  /**
   * Compile the \p code and get hsaco string.
   * @param code The HIP source code.
   * @param include_headers Whether to include the headers of HIP and CINN
   * runtime modules.
   * @return Compiled hsaco code string.
   */
  std::string operator()(const std::string& code, bool include_headers = true);

  /** Compile into hipbin or not
   * @return Compile into hipbin or not.
   */
  bool compile_to_hipbin();

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
   * Compile HIP source code and get hsaco.
   * @param code source code string.
   * @return hsaco string.
   */
  std::string CompileHipSource(const std::string& code, bool include_headers);

  // compile with hipcc
  std::string CompileWithHipcc(const std::string&);

  std::string GetDeviceArch();

  std::string ReadFile(const std::string&, std::ios_base::openmode);

  std::string prefix_name_{""};

  bool compile_to_hipbin_{true};
};

}  // namespace hiprtc
}  // namespace backends
}  // namespace cinn

#endif  // CINN_WITH_ROCM
