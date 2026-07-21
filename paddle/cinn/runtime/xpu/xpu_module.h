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

#ifdef CINN_WITH_XPU
#include "cuda_runtime_api.h"
#include "xpu/xpurtc.h"
#endif

#include <mutex>
#include <string>

#include "paddle/cinn/runtime/xpu/xpu_util.h"

namespace cinn {
namespace runtime {
namespace xpu {

/**
 * XpuModule wraps a compiled kernel binary blob produced by
 * xpurtc::CompileContext.  Launching is done via xpurtc::launch_kernel(),
 * which accepts the raw binary (code + size + hash) together with the
 * mangled entry name.
 *
 * This replaces the previous CUDA-driver-API approach (CUmodule / CUfunction).
 * The XTDK runtime (libxpujitc.so + XRE xcuda) manages caching and loading
 * of the binary on the device internally.
 */
class XpuModule {
 public:
  /**
   * Construct from a compiled kernel binary blob.
   *
   * @param data        Raw binary bytes returned by xpurtc::Kernel::code().
   * @param size        Byte length of \p data.
   * @param hash        Hash from xpurtc::Kernel::hash(), used by the runtime
   *                    to avoid redundant re-uploads.
   * @param mangled_name The mangled C++ entry name from
   *                    xpurtc::Kernel::mangled_name().
   * @param is_cdnn     Whether the kernel uses cdnn instructions
   *                    (xpurtc::Kernel::is_cdnn_kernel()).
   */
  XpuModule(const std::string& data,
            uint64_t hash,
            const std::string& mangled_name,
            bool is_cdnn = false);

  /**
   * Launch the kernel on \p stream with the given grid/block dimensions and
   * serialised parameter buffer.
   *
   * This is a thin wrapper around xpurtc::launch_kernel().
   *
   * @param ncluster         Grid X dimension (number of clusters / blocks).
   * @param ncore            Block X dimension (threads per block).
   * @param stream           XPUStream (cudaStream_t cast to void*).
   * @param params           Serialised argument buffer (layout matches
   *                         xpurtc::detail::SafeParamSerializer).
   * @param param_byte_size  Byte size of \p params.
   */
  void Launch(int ncluster,
              int ncore,
              void* stream,
              const void* params,
              uint32_t param_byte_size) const;

  const std::string& mangled_name() const { return mangled_name_; }
  uint64_t hash() const { return hash_; }

 private:
  std::string data_;  // raw binary blob
  uint32_t size_;     // byte length
  uint64_t hash_;     // kernel hash for runtime caching
  std::string mangled_name_;
  bool is_cdnn_;
};

}  // namespace xpu
}  // namespace runtime
}  // namespace cinn
