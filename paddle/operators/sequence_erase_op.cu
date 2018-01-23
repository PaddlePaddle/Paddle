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
__global__ void LabelErasedIdx(const T* in_dat, const int64_t in_len,
                               const int* tokens, const size_t tokens_len,
                               size_t* num_erased) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < in_len) {
    for (size_t i = 0; i < tokens_len; ++i) {
      if (in_dat[index] == tokens[i]) {
        num_erased[index + 1] = 1;
        break;
      }
    }
  }
}

__global__ void GetOutLod(const size_t* num_erased, const size_t* in_lod,
                          const size_t lod_len, size_t* out_lod0) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < lod_len) {
    out_lod0[index] = in_lod[index] - num_erased[in_lod[index]];
  }
}

template <typename T>
__global__ void SetOutput(const T* in_dat, const int64_t in_len,
                          const size_t* num_erased, T* out_dat) {
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
    auto tokens = ctx.Attr<std::vector<int>>("tokens");
    auto in_len = in->numel();
    auto in_dat = in->data<T>();
    // Copy tokens to GPU
    thrust::device_vector<int> dev_tokens(tokens.begin(), tokens.end());
    int* dev_tokens_ptr = thrust::raw_pointer_cast(dev_tokens.data());

    // Count number of elements to be erased
    thrust::device_vector<size_t> num_erased(in_len + 1, 0);
    size_t* num_erased_ptr = thrust::raw_pointer_cast(num_erased.data());
    auto stream = ctx.cuda_device_context().stream();
    LabelErasedIdx<<<(in_len - 1) / PADDLE_CUDA_NUM_THREADS + 1,
                     PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        in_dat, in_len, dev_tokens_ptr, tokens.size(), num_erased_ptr);
    thrust::inclusive_scan(num_erased.begin() + 1, num_erased.end(),
                           num_erased.begin() + 1);

    // Copy LoD to GPU
    auto lod0 = lod[0];
    auto lod_len = lod0.size();
    thrust::device_vector<size_t> dev_in_lod = lod0;
    size_t* dev_in_lod_ptr = thrust::raw_pointer_cast(dev_in_lod.data());

    // Calc output LoD
    thrust::device_vector<size_t> dev_out_lod(lod_len);
    size_t* dev_out_lod_ptr = thrust::raw_pointer_cast(dev_out_lod.data());
    GetOutLod<<<(lod_len - 1) / PADDLE_CUDA_NUM_THREADS + 1,
                PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        num_erased_ptr, dev_in_lod_ptr, lod_len, dev_out_lod_ptr);
    // Set LoD for output
    std::vector<size_t> out_lod0(dev_out_lod.begin(), dev_out_lod.end());
    framework::LoD out_lod;
    out_lod.push_back(out_lod0);
    out->set_lod(out_lod);

    // Set output
    out->Resize({static_cast<int64_t>(out_lod0.back()), 1});
    auto out_dat = out->mutable_data<T>(ctx.GetPlace());
    SetOutput<<<(in_len - 1) / PADDLE_CUDA_NUM_THREADS + 1,
                PADDLE_CUDA_NUM_THREADS, 0, stream>>>(in_dat, in_len,
                                                      num_erased_ptr, out_dat);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(sequence_erase,
                        paddle::operators::SequenceEraseOpCUDAKernel<int32_t>,
                        paddle::operators::SequenceEraseOpCUDAKernel<int64_t>);
