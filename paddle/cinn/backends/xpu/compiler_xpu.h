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

#ifdef CINN_WITH_XPU
#include "xpu/xpurtc.h"
#endif

namespace cinn {
namespace backends {
namespace xpurtc {

/**
 * Compiles XPU (M100/Houyi) device source code using the XTDK xpurtc JIT
 * compiler (libxpujitc.so).  The source is treated as Houyi kernel code
 * (equivalent to -x houyi --offload-arch=xcn).  Returns the compiled kernel
 * binary blob wrapped in an xpurtc::Kernel object.
 *
 * The corresponding XpuModule stores the Kernel and launches it via
 * xpurtc::launch_kernel().
 */
class Compiler {
 public:
  Compiler() {}

  /**
   * Compile \p code and return the compiled kernel blob as a raw binary
   * string (kernel.code() / kernel.size() packed together).
   *
   * Internally calls xpurtc::CompileContext::add_source() then get_kernel().
   *
   * @param code            XPU/Houyi device source (includes xtdk.h etc.)
   * @param include_headers Whether to inject CINN runtime include paths via
   *                        preprocessor defines.  (Reserved; include paths are
   *                        set through environment / SDK layout.)
   * @return Compiled kernel binary as std::string (binary blob).
   */
  std::string operator()(const std::string& code, bool include_headers = true);

 private:
  /**
   * Find the XTDK clang kernel include directories
   * (lib/clang/19/include/ under XTDK_PATH).
   */
  std::vector<std::string> FindXtdkIncludePaths();

  /**
   * Find the CINN runtime include directory (runtime_include_dir).
   */
  std::vector<std::string> FindCINNRuntimeIncludePaths();

  /**
   * Perform the actual compilation through xpurtc::CompileContext.
   */
  std::string CompileWithXpurtc(const std::string& code, bool include_headers);

  /**
   * Query the XPU architecture integer for the current device.
   * Returned value is passed to xpurtc::CompileContext(xpu_arch).
   * For M100 (XCN/Houyi) this is typically 4.
   */
  int GetDeviceArch();

  std::string prefix_name_{""};
};

}  // namespace xpurtc
}  // namespace backends
}  // namespace cinn
