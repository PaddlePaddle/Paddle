/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/platform/gpu_info.h"
#include "paddle/platform/place.h"

namespace paddle {
namespace memory {

template <typename DstPlace, typename SrcPlace>
void Copy(DstPlace, void* dst, SrcPlace, const void* src, size_t num);

#ifndef PADDLE_ONLY_CPU
template <typename DstPlace, typename SrcPlace>
void Copy(DstPlace, void* dst, SrcPlace, const void* src, size_t num,
          cudaStream_t stream);
#endif  // PADDLE_ONLY_CPU

}  // namespace memory
}  // namespace paddle
