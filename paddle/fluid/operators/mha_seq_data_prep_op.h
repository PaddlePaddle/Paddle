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

    const Tensor* qkvo_seqlen = context.Input<Tensor>("QKVO_seqlen");
    const int* qo_kv_slen_data = qkvo_seqlen->data<int>();
    MHASeqDataSingleton::Instance().Data(key).qkvo_seq_len.resize(qkvo_seqlen->dims()[0]);
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(MHASeqDataSingleton::Instance().Data(key).qkvo_seq_len.data(),
                   reinterpret_cast<const void*>(qo_kv_slen_data),
                   qkvo_seqlen->dims()[0] * sizeof(int),
                   cudaMemcpyDeviceToHost));

    const Tensor* low_high_windows = context.Input<Tensor>("lo_hi_windows");
    MHASeqDataSingleton::Instance().Data(key).lo_hi_windows.resize(low_high_windows->dims()[0]);
    const int* low_high_windows_data = low_high_windows->data<int>();
    PADDLE_ENFORCE_CUDA_SUCCESS(
    cudaMemcpy(MHASeqDataSingleton::Instance().Data(key).lo_hi_windows.data(),
                reinterpret_cast<const void*>(low_high_windows_data),
                low_high_windows->dims()[0] * sizeof(int),
                cudaMemcpyDeviceToHost));
  }
};

}  // namespace operators
}  // namespace paddle
