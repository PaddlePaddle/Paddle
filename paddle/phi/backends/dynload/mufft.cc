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

#include "paddle/phi/backends/dynload/mufft.h"

#include "paddle/phi/core/enforce.h"

namespace phi {
namespace dynload {
std::once_flag mufft_dso_flag;
void* mufft_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

MUFFT_FFT_ROUTINE_EACH(DEFINE_WRAP);

bool HasMUFFT() {
  std::call_once(mufft_dso_flag,
                 []() { mufft_dso_handle = GetMUFFTDsoHandle(); });
  return mufft_dso_handle != nullptr;
}

void EnforceMUFFTLoaded(const char* fn_name) {
  PADDLE_ENFORCE_NOT_NULL(
      mufft_dso_handle,
      phi::errors::PreconditionNotMet(
          "Cannot load mufft shared library. Cannot invoke method %s.",
          fn_name));
}

}  // namespace dynload
}  // namespace phi
