/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <vector>
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

#if defined(PADDLE_WITH_CUDA)
template <typename T, typename Generator>
void run_fmha_fprop_fp16(size_t batch_size, size_t max_seqlen, size_t num_head,
                         size_t head_size, float dropout_rate, void *d_qkv,
                         void *d_cu_seqlens, void *d_seqlens, void *d_out,
                         void *d_softmax_mask, Generator generator,
                         cudaStream_t stream);

template <typename T>
void run_fmha_dgrad_fp16(size_t batch_size, size_t max_seqlen, size_t num_head,
                         size_t head_size, float dropout_rate, void *d_qkv,
                         void *d_cu_seqlens, void *d_seqlens, void *d_out,
                         void *d_softmax_mask, void *d_dqkv,
                         cudaStream_t stream);
#endif

constexpr size_t SLEN_DIMS = 0;
constexpr size_t THREE_DIMS = 1;
constexpr size_t HIDDEN_DIMS = 2;

template <typename DeviceContext, typename T>
class FMHAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    PADDLE_ENFORCE_EQ(true, false, platform::errors::Unavailable(
                                       "FMHA CPU Kernel is not available"));
  }
};

template <typename DeviceContext, typename T>
class FMHAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    PADDLE_ENFORCE_EQ(
        true, false,
        platform::errors::Unavailable("FMHA Grad CPU Kernel is not available"));
  }
};

}  // namespace operators
}  // namespace paddle
