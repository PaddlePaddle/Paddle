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

DevInfoMgr::DevInfoMgr(Target::Arch arch, int device_num)
    : arch_(arch), device_num_(device_num) {
  switch (arch) {
    case Target::Arch::ARM:
    case Target::Arch::X86:
    case Target::Arch::Unk:
      impl_ = std::make_unique<DevInfoBase>();
      break;
    case Target::Arch::NVGPU:
#ifdef CINN_WITH_CUDA
      impl_ = std::make_unique<NVGPUDevInfo>(device_num);
#endif
    default:
      CHECK(false) << "Current device can't be recognized!\n";
      break;
  }
}

std::unique_ptr<DevInfoMgr> DevInfoMgr::GetDevInfo(Target::Arch arch,
                                                   int device_num) {
  std::unique_ptr<DevInfoMgr> ret(new DevInfoMgr(arch, device_num));
  return ret;
}

}  // namespace common
}  // namespace cinn
