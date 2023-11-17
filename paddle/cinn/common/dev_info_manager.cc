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

#include <glog/logging.h>

#include "paddle/cinn/common/dev_info_base.h"
#include "paddle/cinn/common/dev_info_manager.h"

namespace cinn {
namespace common {
template <Target::Arch arch>
DevInfoMgr<arch>::DevInfoMgr(int device_num) : device_num_(device_num) {
  impl = std::make_unique<GetRetType<arch>::RetType>(device_num)
}

template <Target::Arch arch>
DevInfoMgr<arch> DevInfoMgr<arch>::GetDevInfo(int device_num) {
  std::unique_ptr<DevInfoMgr> ret(new DevInfoMgr(device_num));
  return ret;
}

class NVGPUDevInfo;
template <>
struct DevInfoMgr<Target::Arch::NVGPU>::GetRetType<Target::Arch::NVGPU> {
  using RetType = NVGPUDevInfo;
};

}  // namespace common
}  // namespace cinn
