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
#include <cstring>
#include <memory>

namespace paddle {
namespace lite {

const int MALLOC_ALIGN = 64;

void* TargetWrapper<TARGET(kHost)>::Malloc(size_t size) {
  size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
  char* p = static_cast<char*>(malloc(offset + size));
  if (!p) {
    return nullptr;
  }
  void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                    (~(MALLOC_ALIGN - 1)));
  static_cast<void**>(r)[-1] = p;
  memset(r, 0, size);
  return r;
}
void TargetWrapper<TARGET(kHost)>::Free(void* ptr) {
  if (ptr) {
    free(static_cast<void**>(ptr)[-1]);
  }
}
void TargetWrapper<TARGET(kHost)>::MemcpySync(void* dst, const void* src,
                                              size_t size, IoDirection dir) {
  memcpy(dst, src, size);
}

}  // namespace lite
}  // namespace paddle
