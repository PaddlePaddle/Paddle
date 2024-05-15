/* Copyright (c) 2023 Enflame. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <tops/tops_ext.h>
#include <string>
#include <vector>

#include "dtu/hlir/types.h"
#include "topstx/topstx.hpp"

#include "paddle/fluid/platform/enforce.h"

#define RT_DISALLOW_COPY_AND_ASSIGN(TypeName)    \
  TypeName(const TypeName&) = delete;            \
  TypeName(const TypeName&&) = delete;           \
  TypeName& operator=(const TypeName&) = delete; \
  TypeName& operator=(const TypeName&&) = delete

#define RT_CHECK(a)                                                           \
  do {                                                                        \
    topsError_t err = (a);                                                    \
    PADDLE_ENFORCE_EQ(topsSuccess,                                            \
                      err,                                                    \
                      platform::errors::Unavailable("%s return faild.", #a)); \
  } while (false)

// Trowing an exception may result in undefined behavior sometimes, such as
// throwing exception in a destructor.
#define RT_CHECK_NO_THROW(a)                                         \
  do {                                                               \
    topsError_t ret = (a);                                           \
    if (ret != topsSuccess) {                                        \
      VLOG(0) << "Error: " << #a << " return faild, ret is " << ret; \
      std::exit(-1);                                                 \
    }                                                                \
  } while (false)

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {

class GcuDeviceGuard {
 public:
  explicit GcuDeviceGuard(int device) {
    RT_CHECK(topsGetDevice(&device_));
    RT_CHECK(topsSetDevice(device));
  }

  ~GcuDeviceGuard() { RT_CHECK_NO_THROW(topsSetDevice(device_)); }

  GcuDeviceGuard() = delete;
  RT_DISALLOW_COPY_AND_ASSIGN(GcuDeviceGuard);

 private:
  int device_;
};
struct GcuRunTimeInfo {
  GcuRunTimeInfo() = default;
  GcuRunTimeInfo(int d_id,
                 bool dist = false,
                 uint32_t rank = 0,
                 uint32_t w_size = 1,
                 uint32_t n_id = 0)
      : device_id(d_id),
        local_rank(rank),
        world_size(w_size),
        node_id(n_id),
        is_distributed(dist) {}
  int device_id = 0;
  uint32_t local_rank = 0;
  uint32_t world_size = 1;
  uint32_t node_id = 0;
  bool is_distributed = false;
};

template <typename T>
static std::string VectorToString(std::vector<T> vec) {
  std::ostringstream os;
  os << "[";
  for (auto tmp : vec) {
    os << std::fixed << tmp << "; ";
  }
  os << "]";
  return os.str();
}

void AddGcuCallback(int device,
                    topsStream_t stream,
                    const std::function<void()>& callback);

std::string DumpHbm(void* ptr);

std::string HLIRTensorToString(const std::vector<hlir::Tensor*>& tensors,
                               bool is_inputs);

#define AOTOPS_DEBUG(op_name, params)                                 \
  VLOG(6) << op_name << "\ninputs: \n"                                \
          << HLIRTensorToString(params.inputs, true) << "outputs: \n" \
          << HLIRTensorToString(params.outputs, false);

}  // namespace runtime

enum {
  EXEC,
  DISPATCH,
};

class PaddleGcuTrace {
 public:
  static const topstx::Domain& domain() {
    static PaddleGcuTrace inst;
    return inst.domain_;
  }

  PaddleGcuTrace() : domain_("PADDLE") {
    topstx::domainNameCategory(domain_, EXEC, "EXEC");
    topstx::domainNameCategory(domain_, DISPATCH, "DISPATCH");
  }

  topstx::Domain domain_;
};

#define PADDLE_GCU_TRACE_START(category, name)                  \
  topstxRangeId_t rngid_##category_##name = topstx::rangeStart( \
      PaddleGcuTrace::domain(), topstx::Message(category, #name))
#define PADDLE_GCU_TRACE_END(category, name) \
  topstx::rangeEnd(PaddleGcuTrace::domain(), rngid_##category_##name)

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
