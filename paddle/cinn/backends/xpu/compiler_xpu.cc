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

#include "paddle/cinn/backends/xpu/compiler_xpu.h"

#if defined(__linux__)
#include <sys/stat.h>
#endif
#include <glog/logging.h>

#ifdef CINN_WITH_XPU
#include "cuda_runtime_api.h"
#include "xpu/xpurtc.h"
#endif

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace backends {
namespace xpurtc {

std::string Compiler::operator()(const std::string& code,
                                 bool include_headers) {
  return CompileWithXpurtc(code, include_headers);
}

std::vector<std::string> Compiler::FindXtdkIncludePaths() {
  // XTDK_PATH env var, e.g. /path/to/xtdk-llvm19-ubuntu2004_x86_64
  const char* xtdk_path = std::getenv("XTDK_PATH");
  if (xtdk_path != nullptr) {
    // Kernel device headers are under lib/clang/19/include/
    std::string clang_inc = std::string(xtdk_path) + "/lib/clang/19/include";
    return {clang_inc};
  }
#if defined(__linux__)
  // Fallback: try a well-known install location
  struct stat st;
  const std::string fallback = "/usr/local/xtdk/lib/clang/19/include";
  if (stat(fallback.c_str(), &st) == 0) {
    return {fallback};
  }
#endif
  PADDLE_THROW(::common::errors::Fatal(
      "Cannot find XTDK include path. Set XTDK_PATH to the XTDK installation "
      "directory (e.g. /opt/xtdk-llvm19-ubuntu2004_x86_64)."));
  return {};
}

std::vector<std::string> Compiler::FindCINNRuntimeIncludePaths() {
  return {Context::Global().runtime_include_dir()};
}

std::string Compiler::CompileWithXpurtc(const std::string& code,
                                        bool include_headers) {
#ifndef CINN_WITH_XPU
  PADDLE_THROW(::common::errors::Unimplemented(
      "CompileWithXpurtc requires CINN_WITH_XPU to be enabled."));
  return "";
#else
  // Build source preamble: inject include paths as #include directives so
  // that xpurtc::CompileContext can resolve headers at JIT compile time.
  std::string full_source;
  if (include_headers) {
    for (const auto& path : FindXtdkIncludePaths()) {
      full_source += "#pragma clang include_dir \"" + path + "\"\n";
    }
    for (const auto& path : FindCINNRuntimeIncludePaths()) {
      full_source += "#pragma clang include_dir \"" + path + "\"\n";
    }
  }
  full_source += code;

  VLOG(5) << "xpu (xpurtc) CompileWithXpurtc, source length="
          << full_source.size() << ", xpu_arch=" << GetDeviceArch();

  ::xpurtc::CompileContext ctx(GetDeviceArch());
  // add_source returns a key string; SourceKind::XPU = 0 (enum value 1 in
  // xpurtc.h maps to the Houyi/XCN source kind).
  ctx.add_source(full_source, ::xpurtc::SourceKind::XPU, "cinn_xpu_kernel");

  ::xpurtc::Kernel kernel = ctx.get_kernel();
  PADDLE_ENFORCE_EQ(
      kernel.is_valid(),
      true,
      ::common::errors::External(
          "xpurtc compilation failed: kernel is not valid after compilation."));

  // Pack the binary blob into a std::string so XpuModule can store it.
  // XpuModule will later call xpurtc::launch_kernel with code() + size().
  const uint8_t* data = kernel.code();
  uint32_t size = kernel.size();
  VLOG(3) << "xpurtc compiled kernel: size=" << size
          << " hash=" << kernel.hash()
          << " mangled_name=" << kernel.mangled_name();
  return std::string(reinterpret_cast<const char*>(data), size);
#endif  // CINN_WITH_XPU
}

int Compiler::GetDeviceArch() {
#ifdef CINN_WITH_XPU
  // xpurtc::CompileContext(int xpu_arch): 0 = auto-detect, 4 = M100/Houyi
  // Query from env var if set; otherwise default to M100.
  const char* arch_env = std::getenv("XPU_ARCH");
  if (arch_env != nullptr) {
    return std::atoi(arch_env);
  }
#endif
  // Default: M100 (XCN generation = 4)
  return 4;
}

}  // namespace xpurtc
}  // namespace backends
}  // namespace cinn
