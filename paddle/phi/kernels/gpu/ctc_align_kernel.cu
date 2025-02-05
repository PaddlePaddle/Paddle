// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/ctc_align_kernel.h"
#include "paddle/phi/kernels/impl/ctc_align_kernel_impl.h"

namespace phi {

template <typename T>
__global__ void MergeAndDelCudaKernel(const int64_t num_token,
                                      const T* tokens,
                                      const size_t num_seq,
                                      size_t* lod0,
                                      const int blank,
                                      const int merge_repeated,
                                      size_t* out_lod0,
                                      T* output) {
  int output_idx = 0;
  out_lod0[0] = 0;

  for (int i = 0; i < num_seq; ++i) {
    T pre_token = -1;
    for (int j = lod0[i]; j < lod0[i + 1]; ++j) {
      if (tokens[j] != blank && !(merge_repeated && tokens[j] == pre_token)) {
        output[output_idx] = tokens[j];
        ++output_idx;
      }
      pre_token = tokens[j];
    }
    out_lod0[i + 1] = output_idx;
  }
}

template <typename T>
__global__ void PaddingMergeAndDelCudaKernel(const int64_t num_token,
                                             const T* tokens,
                                             const T* tokens_length,
                                             const int blank,
                                             const int merge_repeated,
                                             const int padding_value,
                                             const int64_t batch_size,
                                             T* output,
                                             T* output_length) {
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= batch_size) return;
  int output_idx = ind * num_token;
  T prev_token = -1;
  for (int i = ind * num_token; i < ind * num_token + tokens_length[ind]; i++) {
    if ((unsigned)tokens[i] != blank &&
        !(merge_repeated && tokens[i] == prev_token)) {
      output[output_idx] = tokens[i];
      ++output_idx;
    }
    prev_token = tokens[i];
  }
  output_length[ind] = output_idx - ind * num_token;
  for (int i = output_idx; i < ind * num_token + num_token; i++) {
    output[i] = padding_value;
  }
}

template <typename T, typename Context>
void CTCAlignOpCUDAKernel(const Context& dev_ctx,
                          const DenseTensor& input,
                          const paddle::optional<DenseTensor>& input_length,
                          int blank,
                          bool merge_repeated,
                          int padding_value,
                          DenseTensor* output,
                          DenseTensor* output_length) {
  PADDLE_ENFORCE_EQ(dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU,
                    true,
                    common::errors::InvalidArgument(
                        "CTCAlign operator CUDA kernel must use CUDAPlace "
                        "rather than CPUPlace."));
  const T* tokens = input.data<T>();
  auto stream = dev_ctx.stream();

  // tensor input which has no lod
  if (input.lod().empty()) {
    auto input_dims = input.dims();
    output->Resize({input_dims[0], input_dims[1]});
    T* output_data = dev_ctx.template Alloc<T>(output);
    const T* input_length_data = input_length.get().data<T>();
    output_length->Resize({input_dims[0], 1});
    T* output_length_data = dev_ctx.template Alloc<T>(output_length);
    PaddingMergeAndDelCudaKernel<T>
        <<<32, (input_dims[0] + 32 - 1) / 32, 0, stream>>>(input_dims[1],
                                                           tokens,
                                                           input_length_data,
                                                           blank,
                                                           merge_repeated,
                                                           padding_value,
                                                           input_dims[0],
                                                           output_data,
                                                           output_length_data);
  } else {
    const size_t level = 0;
    auto input_lod = phi::ToAbsOffset(input.lod());

    const int64_t num_tokens = input.dims()[0];
    const size_t num_seq = input_lod[level].size() - 1;

    // prepare a lod to record lod information while merging elements
    thrust::device_vector<size_t> dev_out_lod0(input_lod[level].size());
    size_t* dev_out_lod0_ptr = thrust::raw_pointer_cast(dev_out_lod0.data());

    // merge elements and delete blank
    output->Resize({num_tokens, 1});
    T* output_data = dev_ctx.template Alloc<T>(output);

    phi::MixVector<size_t> mixv_input_lod(&input_lod[level]);
    MergeAndDelCudaKernel<T><<<1, 1, 0, stream>>>(
        num_tokens,
        tokens,
        num_seq,
        mixv_input_lod.CUDAMutableData(dev_ctx.GetPlace()),
        blank,
        merge_repeated,
        dev_out_lod0_ptr,
        output_data);
    mixv_input_lod.CopyToCPU();

    // set output lod
    std::vector<size_t> host_out_lod0(dev_out_lod0.begin(), dev_out_lod0.end());
    phi::LegacyLoD out_lod;
    out_lod.push_back(host_out_lod0);
    output->set_lod(out_lod);

    // resize output dims
    output->Resize({static_cast<int64_t>(host_out_lod0.back()), 1});

    if (host_out_lod0.back() == 0) {
      output->Resize({1, 1});
      dev_ctx.template Alloc<T>(output);
      phi::funcs::SetConstant<phi::GPUContext, T> set_constant;
      set_constant(dev_ctx, output, -1);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    ctc_align, GPU, ALL_LAYOUT, phi::CTCAlignOpCUDAKernel, int, int64_t) {}
