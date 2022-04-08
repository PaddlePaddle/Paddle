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

#include <numeric>
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

    const Tensor* attn_mask = context.Input<Tensor>("attn_mask");
    // TODO(Ming Huang): Use reduce_sum kernel to compute qkvo seqlen in GPU
    // buffer.
    // That could remove this DtoH data movement.
    size_t attn_mask_size = attn_mask->numel() * sizeof(int);
    memory::allocation::AllocationPtr attn_mask_host =
        memory::Alloc(host_pinned_place, attn_mask_size);
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpy(attn_mask_host->ptr(),
                   reinterpret_cast<const void*>(attn_mask->data<int>()),
                   attn_mask_size, cudaMemcpyDeviceToHost));

    Tensor* qo_kv_seqlen = context.Output<Tensor>("qo_kv_seqlen");
    qo_kv_seqlen->mutable_data<T>(context.GetPlace());

    Tensor* qo_kv_seqlen_host = context.Output<Tensor>("qo_kv_seqlen_host");
    qo_kv_seqlen_host->mutable_data<T>(host_pinned_place);
    int* qo_kv_seqlen_host_ptr = qo_kv_seqlen_host->data<int>();

    int batch = attn_mask->dims()[0];
    int seqlen = attn_mask->dims()[3];
    int stride = phi::product(attn_mask->dims()) / batch;

    int* attn_mask_host_ptr = reinterpret_cast<int*>(attn_mask_host->ptr());
    for (int i = 0; i < batch; ++i) {
      int* begin = attn_mask_host_ptr + stride * i;
      int* end = begin + seqlen;
      qo_kv_seqlen_host_ptr[i] = std::accumulate(begin, end, 0);
      qo_kv_seqlen_host_ptr[i + batch] = qo_kv_seqlen_host_ptr[i];
    }

    size_t seqlen_array_size = qo_kv_seqlen_host->numel() * sizeof(int);
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpy(reinterpret_cast<void*>(qo_kv_seqlen->data<int>()),
                   reinterpret_cast<const void*>(qo_kv_seqlen_host_ptr),
                   seqlen_array_size, cudaMemcpyHostToDevice));

    Tensor* low_high_windows_host =
        context.Output<Tensor>("low_high_windows_host");
    low_high_windows_host->mutable_data<T>(host_pinned_place);
    int* low_high_windows_host_ptr = low_high_windows_host->data<int>();
    std::fill_n(low_high_windows_host_ptr, seqlen, 0);
    std::fill_n(low_high_windows_host_ptr + seqlen, seqlen, seqlen);
  }
};

}  // namespace operators
}  // namespace paddle
