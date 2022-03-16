/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Corporation. All rights reserved.

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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class MHADataPrepKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    platform::Place host_pinned_place = platform::CUDAPinnedPlace();

    const Tensor* qkvo_seqlen = context.Input<Tensor>("qo_kv_seqlen");
    const int* qo_kv_slen_data = qkvo_seqlen->data<int>();
    Tensor* qkvo_seqlen_host = context.Output<Tensor>("qo_kv_seqlen_host");
    qkvo_seqlen_host->mutable_data<T>(host_pinned_place);
    int* qkvo_seqlen_host_data = qkvo_seqlen_host->data<int>();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(
        reinterpret_cast<void*>(qkvo_seqlen_host_data),
        reinterpret_cast<const void*>(qo_kv_slen_data),
        qkvo_seqlen->dims()[0] * sizeof(int), cudaMemcpyDeviceToHost));

    const Tensor* low_high_windows = context.Input<Tensor>("low_high_windows");
    const int* low_high_windows_data = low_high_windows->data<int>();
    Tensor* low_high_windows_host =
        context.Output<Tensor>("low_high_windows_host");
    low_high_windows_host->mutable_data<T>(host_pinned_place);
    int* low_high_windows_host_data = low_high_windows_host->data<int>();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(
        reinterpret_cast<void*>(low_high_windows_host_data),
        reinterpret_cast<const void*>(low_high_windows_data),
        low_high_windows->dims()[0] * sizeof(int), cudaMemcpyDeviceToHost));
  }
};

}  // namespace operators
}  // namespace paddle
