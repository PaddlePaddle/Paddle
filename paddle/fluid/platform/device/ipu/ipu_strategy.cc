/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/ipu/ipu_strategy.h"
#include <glog/logging.h>

namespace paddle {
namespace platform {
namespace ipu {

void IpuStrategy::enablePattern(const std::string& t) {
  VLOG(10) << "enable popart pattern: " << t;
  popart_patterns.enablePattern(t, true);
}

void IpuStrategy::disablePattern(const std::string& t) {
  VLOG(10) << "disable popart pattern: " << t;
  popart_patterns.enablePattern(t, false);
}

const bool IpuStrategy::isPatternEnabled(const std::string& t) {
  return popart_patterns.isPatternEnabled(t);
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
