/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/version.h"
#include <algorithm>

namespace paddle {
namespace framework {
bool IsProgramVersionSupported(int version) {
  static int num_supported =
      sizeof(kSupportedProgramVersion) / sizeof(kSupportedProgramVersion[0]);
  return std::find(kSupportedProgramVersion,
                   kSupportedProgramVersion + num_supported,
                   version) != kSupportedProgramVersion + num_supported;
}
}  // namespace framework
}  // namespace paddle
