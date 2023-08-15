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

#include "paddle/phi/backends/dynload/mudnn.h"

#include "paddle/phi/core/enforce.h"

namespace phi {
namespace dynload {

std::once_flag mudnn_dso_flag;
void* mudnn_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

bool HasCUDNN() {
  std::call_once(mudnn_dso_flag,
                 []() { mudnn_dso_handle = GetCUDNNDsoHandle(); });
  return mudnn_dso_handle != nullptr;
}

void EnforceCUDNNLoaded(const char* fn_name) {
  PADDLE_ENFORCE_NOT_NULL(
      mudnn_dso_handle,
      phi::errors::PreconditionNotMet(
          "Cannot load mudnn shared library. Cannot invoke method %s.",
          fn_name));
}

}  // namespace dynload
}  // namespace phi
