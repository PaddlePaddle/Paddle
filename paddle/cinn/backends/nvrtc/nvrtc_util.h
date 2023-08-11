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
#ifdef CINN_WITH_CUDA
#if defined(__linux__)
#include <sys/stat.h>
#endif
#include <glog/logging.h>

#include <string>
#include <vector>

namespace cinn {
namespace backends {
namespace nvrtc {

/**
 * An helper class to call NVRTC. Input CUDA device source code, get PTX string.
 */
class Compiler {
 public:
  Compiler();

  /**
   * Compile the \p code and get PTX string.
   * @param code The CUDA source code.
   * @param include_headers Whether to include the headers of CUDA and CINN
   * runtime modules.
   * @return Compiled PTX code string.
   */
  std::string operator()(const std::string& code, bool include_headers = true);

  /** Compile into cubin or not
   * @return Compile into cubin or not.
   */
  bool compile_to_cubin();

 private:
  /**
   * Get the directories of CUDA's header files.
   * @return list of header file directories.
   */
  std::vector<std::string> FindCUDAIncludePaths();

  /**
   * Get the directories of CINN runtime's header files.
   * @return list of header file directories.
   */
  std::vector<std::string> FindCINNRuntimeIncludePaths();

  /**
   * Compile CUDA source code and get PTX or CUBIN.
   * @param code source code string.
   * @return PTX or CUBIN string.
   */
  std::string CompileCudaSource(const std::string& code, bool include_headers);

  /**
   * whether to compile the source code into cubin, only works with cuda version
   * > 11.1
   */
  bool compile_to_cubin_{false};

  // compile with nvcc
  std::string CompileWithNvcc(const std::string&);

  // compile to ptx
  void CompileToPtx();
  // compile to cubin
  void CompileToCubin();
  std::string GetDeviceArch();

  std::string ReadFile(const std::string&, std::ios_base::openmode);

  std::string prefix_name_{""};
};

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn

#endif  // CINN_WITH_CUDA
