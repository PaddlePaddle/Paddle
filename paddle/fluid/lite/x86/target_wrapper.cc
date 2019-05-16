// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/target_wrapper.h"
#include <algorithm>
#include "paddle/fluid/lite/utils/all.h"
#include "paddle/fluid/lite/x86/target_wrapper.h"

namespace paddle {
namespace lite {

template <>
void TargetWrapper<TARGET(kX86)>::MemcpySync(void *dst, const void *src,
                                             size_t size, IoDirection dir) {
  std::copy_n(reinterpret_cast<const uint8_t *>(src), size,
              reinterpret_cast<uint8_t *>(dst));
}

template class TargetWrapper<TARGET(kX86)>;

}  // namespace lite
}  // namespace paddle
