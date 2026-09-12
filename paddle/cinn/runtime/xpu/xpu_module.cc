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

#include "paddle/cinn/runtime/xpu/xpu_module.h"

#include <glog/logging.h>

#ifdef CINN_WITH_XPU
#include "xpu/xpurtc.h"
#endif

#include "paddle/cinn/utils/profiler.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace runtime {
namespace xpu {

XpuModule::XpuModule(const std::string& data,
                     uint64_t hash,
                     const std::string& mangled_name,
                     bool is_cdnn)
    : data_(data),
      size_(static_cast<uint32_t>(data.size())),
      hash_(hash),
      mangled_name_(mangled_name),
      is_cdnn_(is_cdnn) {
  PADDLE_ENFORCE_EQ(
      data.empty(),
      false,
      ::common::errors::PreconditionNotMet("XpuModule: kernel data is empty."));
  VLOG(3) << "XpuModule created: size=" << size_ << " hash=" << hash_
          << " mangled_name=" << mangled_name_;
}

void XpuModule::Launch(int ncluster,
                       int ncore,
                       void* stream,
                       const void* params,
                       uint32_t param_byte_size) const {
#ifndef CINN_WITH_XPU
  PADDLE_THROW(::common::errors::Unimplemented(
      "XpuModule::Launch requires CINN_WITH_XPU."));
#else
  cinn::utils::RecordEvent record_run("xpurtc::launch_kernel",
                                      cinn::utils::EventType::kInstruction);
  VLOG(3) << "XpuModule::Launch ncluster=" << ncluster << " ncore=" << ncore
          << " param_bytes=" << param_byte_size << " mangled=" << mangled_name_;

  // xpurtc::launch_kernel signature:
  //   int launch_kernel(const void* kernel_code, uint32_t kernel_len,
  //                     uint64_t kernel_hash,
  //                     int ncluster, int ncore, XPUStream stream,
  //                     const void* params, uint32_t param_byte_size,
  //                     bool kernel_use_cdnn,
  //                     const char* kernel_mangled_entry);
  int ret =
      ::xpurtc::launch_kernel(reinterpret_cast<const void*>(data_.c_str()),
                              size_,
                              hash_,
                              ncluster,
                              ncore,
                              static_cast<XPUStream>(stream),
                              params,
                              param_byte_size,
                              is_cdnn_,
                              mangled_name_.c_str());
  PADDLE_ENFORCE_EQ(
      ret,
      0,
      ::common::errors::External(
          "xpurtc::launch_kernel failed with code %d for kernel %s",
          ret,
          mangled_name_.c_str()));
#endif
}

}  // namespace xpu
}  // namespace runtime
}  // namespace cinn
