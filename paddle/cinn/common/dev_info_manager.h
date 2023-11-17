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
template <Target::Arch arch>
class DevInfoMgr final {
 private:
  explicit DevInfoMgr(int device_num = 0);
  std::unique_ptr<DevInfoBase> impl_;
  int device_num_;

  template <Target::Arch arch>
  struct GetRetType {
    using RetType = DevInfoBase;
  };

 public:
  static DevInfoMgr<arch> GetDevInfo(int device_num = 0);

  using RetType = GetRetType<arch>::RetType;
  const RetType* operator->() const {
    CHECK(!std::is_void<RET_TYPE>()) << "Current device can't be recognized!\n";
    return dynamic_cast<const RET_TYPE*>(impl_.get());
  }
  RetType* operator->() {
    CHECK(!std::is_void<RET_TYPE>()) << "Current device can't be recognized!\n";
    return dynamic_cast<RET_TYPE*>(impl_.get());
  }
};

// Extra device should be added here
template <>
struct DevInfoMgr<Target::Arch::NVGPU>::GetRetType<Target::Arch::NVGPU>;

}  // namespace common
}  // namespace cinn
