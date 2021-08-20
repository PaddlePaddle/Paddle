// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle_inference_api.h"

namespace paddle_infer {
namespace contrib {
namespace utils {

void* CudaMallocPinnedMemory(size_t size);
void CudaFreePinnedMemory(void* mem);

void CopyTensor(Tensor& dst, const Tensor& src);
void CopyTensorAsync(Tensor& dst, const Tensor& src, void* exec_stream);
void CopyTensorAsync(Tensor& dst, const Tensor& src, CallbackFunc cb,
                     void* cb_params);

}  // namespace utils
}  // namespace contrib
}  // namespace paddle_infer
