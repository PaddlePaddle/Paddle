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

#ifdef PADDLE_WITH_WBAES

#include "paddle/fluid/platform/dynload/wbaes.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace dynload {

std::once_flag wbaes_dso_flag;
void *wbaes_dso_handle = nullptr;
void *GSECF[9];

// #define DEFINE_WRAP(__name) DynLoad__##__name __name

// WBAES_ROUTINE_EACH(DEFINE_WRAP);

void LoadWBAESInterfaces() {
  std::call_once(wbaes_dso_flag, []() {
    wbaes_dso_handle = paddle::platform::dynload::GetWBAESDsoHandle();
  });
  static void *p_func_array = dlsym(wbaes_dso_handle, "GSECF");
  PADDLE_ENFORCE_NOT_NULL(p_func_array, "Load WBAES interfaces failed.");
  memcpy(GSECF, p_func_array, sizeof(GSECF));
}

}  // namespace dynload
}  // namespace platform
}  // namespace paddle

#endif
