/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/dynload/nvrtc.h"

namespace paddle {
namespace platform {
namespace dynload {

std::once_flag nvrtc_dso_flag;
void* nvrtc_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

NVRTC_ROUTINE_EACH(DEFINE_WRAP);

#ifdef PADDLE_USE_DSO
bool HasNVRTC() {
  std::call_once(nvrtc_dso_flag,
                 []() { nvrtc_dso_handle = GetNVRTCDsoHandle(); });
  return nvrtc_dso_handle != nullptr;
}
#else
bool HasNVRTC() { return false; }
#endif

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
