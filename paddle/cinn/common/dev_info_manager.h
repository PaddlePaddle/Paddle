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

#include "paddle/cinn/common/dev_info_base.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/nvgpu_dev_info.h"
#include "paddle/cinn/common/target.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace common {

template <typename arch>
struct GetDevType {
  using DevType = DevInfoBase;
};

// Extra device should be added here
class NVGPUDevInfo;
template <>
struct GetDevType<NVGPUArch> {
  using DevType = NVGPUDevInfo;
};

template <typename arch>
class DevInfoMgr final {
 private:
  explicit DevInfoMgr(int device_num = 0) : device_num_(device_num) {
    impl_ = std::make_unique<typename GetDevType<arch>::DevType>(device_num);
  }

  std::unique_ptr<DevInfoBase> impl_;
  int device_num_;

 public:
  static DevInfoMgr<arch> GetDevInfo(int device_num = 0) {
    return DevInfoMgr(device_num);
  }

  using RetType = typename GetDevType<arch>::DevType;

  const RetType* operator->() const {
    PADDLE_ENFORCE_EQ(!std::is_void<RetType>(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Current device can't be recognized!"));
    return dynamic_cast<const RetType*>(impl_.get());
  }
  RetType* operator->() {
    PADDLE_ENFORCE_EQ(!std::is_void<RetType>(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Current device can't be recognized!"));
    return dynamic_cast<RetType*>(impl_.get());
  }
};

}  // namespace common
}  // namespace cinn
