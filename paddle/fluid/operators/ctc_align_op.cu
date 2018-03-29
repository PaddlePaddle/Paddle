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

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "paddle/fluid/operators/ctc_align_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void MergeAndDelCudaKernel(const int64_t num_token, const T* tokens,
                                      const size_t num_seq, size_t* lod0,
                                      const int blank, const int merge_repeated,
                                      size_t* out_lod0, T* output) {
  int ouput_idx = 0;
  out_lod0[0] = 0;

  for (int i = 0; i < num_seq; ++i) {
    T pre_token = -1;
    for (int j = lod0[i]; j < lod0[i + 1]; ++j) {
      if (tokens[j] != blank && !(merge_repeated && tokens[j] == pre_token)) {
        output[ouput_idx] = tokens[j];
        ++ouput_idx;
      }
      pre_token = tokens[j];
    }
    out_lod0[i + 1] = ouput_idx;
  }
}

template <typename T>
class CTCAlignOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    const size_t level = 0;
    auto* input = ctx.Input<LoDTensor>("Input");
    auto* output = ctx.Output<LoDTensor>("Output");
    auto input_lod = framework::ToAbsOffset(input->lod());

    const T* tokens = input->data<T>();
    const int64_t num_tokens = input->dims()[0];
    const size_t num_seq = input_lod[level].size() - 1;

    const int blank = ctx.Attr<int>("blank");
    const int merge_repeated =
        static_cast<int>(ctx.Attr<bool>("merge_repeated"));

    // prepare a lod to record lod information while merging elements
    thrust::device_vector<size_t> dev_out_lod0(input_lod[level].size());
    size_t* dev_out_lod0_ptr = thrust::raw_pointer_cast(dev_out_lod0.data());

    // merge elements and delete blank
    T* output_data = output->mutable_data<T>({num_tokens, 1}, ctx.GetPlace());

    auto stream = ctx.cuda_device_context().stream();
    MergeAndDelCudaKernel<T><<<1, 1, 0, stream>>>(
        num_tokens, tokens, num_seq,
        input_lod[level].CUDAMutableData(ctx.GetPlace()), blank, merge_repeated,
        dev_out_lod0_ptr, output_data);

    // set output lod
    std::vector<size_t> host_out_lod0(dev_out_lod0.begin(), dev_out_lod0.end());
    framework::LoD out_lod;
    if (host_out_lod0.back() == 0) {
      host_out_lod0.resize(1);
    }
    out_lod.push_back(host_out_lod0);
    output->set_lod(out_lod);

    // resize output dims
    output->Resize({static_cast<int64_t>(host_out_lod0.back()), 1});

    if (host_out_lod0.back() == 0) {
      output->Resize({1, 1});
      output->mutable_data<T>(ctx.GetPlace());
      math::SetConstant<platform::CUDADeviceContext, T> set_constant;
      set_constant(ctx.template device_context<platform::CUDADeviceContext>(),
                   output, -1);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(ctc_align, paddle::operators::CTCAlignOpCUDAKernel<int>,
                        paddle::operators::CTCAlignOpCUDAKernel<int64_t>);
