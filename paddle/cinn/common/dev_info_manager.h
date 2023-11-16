// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <memory>

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/nvgpu_dev_info.h"
#include "paddle/cinn/common/target.h"

namespace cinn {
namespace common {

class DevInfoMgr final {
 private:
  explicit DevInfoMgr(Target::Arch arch = Target::Arch::Unk,
                      int device_num = 0);
  std::unique_ptr<DevInfoBase> impl_;
  Target::Arch arch_;
  int device_num_;

 public:
  static std::unique_ptr<DevInfoMgr> GetDevInfo(
      Target::Arch arch = Target::Arch::NVGPU, int device_num = 0);

// Extra device should be added here
#ifdef CINN_WITH_CUDA
  using RET_TYPE = NVGPUDevInfo;
  const RET_TYPE* operator->() const {
    CHECK(!std::is_void<RET_TYPE>()) << "Current device can't be recognized!\n";
    return dynamic_cast<const RET_TYPE*>(impl_.get());
  }
  RET_TYPE* operator->() {
    CHECK(!std::is_void<RET_TYPE>()) << "Current device can't be recognized!\n";
    return dynamic_cast<RET_TYPE*>(impl_.get());
  }
#endif
};

}  // namespace common
}  // namespace cinn
