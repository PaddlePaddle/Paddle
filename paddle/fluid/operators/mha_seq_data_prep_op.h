/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2021 NVIDIA Corporation. All rights reserved.

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
#include "paddle/fluid/operators/mha_seq_data_cache.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class MHASeqDataPrepKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {

    const std::string key = context.Attr<std::string>("cache_key");
    platform::Place host_pinned_place = platform::CUDAPinnedPlace();

    const Tensor* qkvo_seqlen = context.Input<Tensor>("QKVO_seqlen");
    const int* qo_kv_slen_data = qkvo_seqlen->data<int>();
    size_t qkvo_seqlen_size = qkvo_seqlen->dims()[0] * sizeof(int);
    MHASeqDataSingleton::Instance().Data(key).qkvo_seq_len = memory::Alloc(host_pinned_place, qkvo_seqlen_size);
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(MHASeqDataSingleton::Instance().Data(key).qkvo_seq_len->ptr(),
                   reinterpret_cast<const void*>(qo_kv_slen_data),
                   qkvo_seqlen_size, cudaMemcpyDeviceToHost));

    const Tensor* low_high_windows = context.Input<Tensor>("lo_hi_windows");
    const int* low_high_windows_data = low_high_windows->data<int>();
    size_t low_high_windows_size = low_high_windows->dims()[0] * sizeof(int);
    MHASeqDataSingleton::Instance().Data(key).lo_hi_windows = memory::Alloc(host_pinned_place, low_high_windows_size);
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(MHASeqDataSingleton::Instance().Data(key).lo_hi_windows->ptr(),
                    reinterpret_cast<const void*>(low_high_windows_data),
                    low_high_windows->dims()[0] * sizeof(int),
                    cudaMemcpyDeviceToHost));

    Tensor* fake_output = context.Output<Tensor>("fake_output");
    fake_output->mutable_data<bool>(context.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle
