/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/mlu/mlu_stream.h"

namespace paddle {
namespace platform {
namespace stream {

bool MLUStream::Init(const Place& place, const int priority) {
  // TODO(mlu)
  return true;
}

void MLUStream::Destroy() {
  // TODO(mlu)
}

void MLUStream::Wait() const {
  // TODO(mlu)
}

MLUStream* get_current_mlu_stream(int deviceId) {
#ifdef PADDLE_WITH_MLU
  // TODO(mlu)
  return nullptr;
#endif
}

MLUStream* set_current_mlu_stream(MLUStream* stream) {
#ifdef PADDLE_WITH_MLU
  // TODO(mlu)
  return nullptr;
#endif
}
}  // namespace stream
}  // namespace platform
}  // namespace paddle
