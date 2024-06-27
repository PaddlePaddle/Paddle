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

#pragma once
#ifdef PADDLE_WITH_MUSA
#include <mudnn.h>
#include <musa_runtime_api.h>

namespace phi {
namespace dynload {

using ::musa::dnn::BatchMatMul;
using ::musa::dnn::BatchNorm;
using ::musa::dnn::Convolution;
using ::musa::dnn::Handle;
using ::musa::dnn::MatMul;
using ::musa::dnn::MemoryHandler;
using ::musa::dnn::Pooling;
using ::musa::dnn::Softmax;
using ::musa::dnn::Tensor;
using ::musa::dnn::ScaledDotProductAttention;
extern bool HasCUDNN();

void mudnnCreate(Handle** handle, int device);

void mudnnSetStream(Handle* handle, musaStream_t stream);

void mudnnDestroy(Handle* handle);

}  // namespace dynload
}  // namespace phi
#endif
