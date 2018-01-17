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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "paddle/operators/sequence_erase_op.h"
#include "paddle/platform/cuda_helper.h"

namespace paddle {
namespace operators {
using platform::PADDLE_CUDA_NUM_THREADS;
using LoDTensor = framework::LoDTensor;

template <typename T>
__global__ void LabelErasedIdx(const T* in_dat, const int in_len,
                               const T* tokens, const int tokens_len,
                               int* num_erased) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < in_len) {
    int erased = 0;
    for (int i = 0; i < tokens_len; ++i) {
      if (in_dat[index] == tokens[i]) {
        erased = 1;
      }
    }
    num_erased[index + 1] = erased;
    if (index == 0) {
      num_erased[0] = 0;
    }
  }
}

template <typename T>
__global__ void GetOutLod(const T* num_erased, const int* in_lod,
                          const int lod_len, int* out_lod0) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < lod_len) {
    out_lod0[index] = in_lod[index] - num_erased[in_lod[index]];
  }
}

template <typename T>
__global__ void SetOutput(const T* in_dat, const int in_len,
                          const int* num_erased, T* out_dat) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < in_len) {
    if (num_erased[index] == num_erased[index + 1]) {
      out_dat[index - num_erased[index]] = in_dat[index];
    }
  }
}

template <typename T>
class SequenceEraseOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<LoDTensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");

    auto lod = in->lod();
    PADDLE_ENFORCE_EQ(lod.size(), 1UL, "Only support one level sequence now.");
    PADDLE_ENFORCE_EQ(lod[0].back(), (size_t)in->numel(),
                      "The actual size mismatches with the LoD information.");
    auto tokens = ctx.Attr<std::vector<T>>("tokens");
    auto tokens_len = tokens.size();
    auto in_len = in->numel();
    auto in_dat = in->data<T>();
    auto lod0 = lod[0];

    thrust::host_vector<T> host_tokens(tokens_len);
    for (size_t i = 0; i < tokens.size(); ++i) {
      host_tokens[i] = tokens[i];
    }
    thrust::device_vector<T> dev_tokens = host_tokens;
    thrust::device_vector<int> num_erased(in_len + 1);

    T* dev_tokens_ptr = thrust::raw_pointer_cast(dev_tokens.data());
    int* num_erased_ptr = thrust::raw_pointer_cast(num_erased.data());

    auto stream = ctx.cuda_device_context().stream();
    LabelErasedIdx<<<(in_len - 1) / PADDLE_CUDA_NUM_THREADS + 1,
                     PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        in_dat, in_len, dev_tokens_ptr, tokens_len, num_erased_ptr);
    thrust::inclusive_scan(num_erased.begin() + 1, num_erased.end(),
                           num_erased.begin() + 1);

    // Calc LoD
    auto lod_len = lod0.size();
    thrust::host_vector<int> host_lod(lod_len);
    for (size_t i = 0; i < lod_len; ++i) {
      host_lod[i] = lod0[i];
    }
    thrust::device_vector<int> dev_in_lod = host_lod;
    thrust::device_vector<int> dev_out_lod(lod_len);
    int* dev_in_lod_ptr = thrust::raw_pointer_cast(dev_in_lod.data());
    int* dev_out_lod_ptr = thrust::raw_pointer_cast(dev_out_lod.data());
    GetOutLod<<<(lod_len - 1) / PADDLE_CUDA_NUM_THREADS + 1,
                PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        num_erased_ptr, dev_in_lod_ptr, lod_len, dev_out_lod_ptr);
    thrust::host_vector<int> host_out_lod = dev_out_lod;
    std::vector<int> out_lod0(lod_len, 0);
    for (size_t i = 0; i < lod_len; i++) {
      out_lod0[i] = host_out_lod[i];
    }
    framework::LoD out_lod;
    out_lod.push_back(out_lod0);
    out->set_lod(out_lod);

    // Set output
    out->Resize({out_lod0.back(), 1});
    auto out_dat = out->mutable_data<T>(ctx.GetPlace());
    SetOutput<<<(in_len - 1) / PADDLE_CUDA_NUM_THREADS + 1,
                PADDLE_CUDA_NUM_THREADS, 0, stream>>>(in_dat, in_len,
                                                      num_erased_ptr, out_dat);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(sequence_erase,
                        paddle::operators::SequenceEraseOpCUDAKernel<int32_t>);
