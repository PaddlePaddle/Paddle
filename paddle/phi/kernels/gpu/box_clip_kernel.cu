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

#include <algorithm>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/lod_utils.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/box_clip_kernel_impl.h"

namespace phi {

static constexpr int ImInfoSize = 3;

template <typename T, int BlockSize>
static __global__ void GPUBoxClip(const T *input,
                                  const size_t *lod,
                                  const size_t width,
                                  const T *im_info,
                                  T *output) {
  T im_w = round(im_info[blockIdx.x * ImInfoSize + 1] /
                 im_info[blockIdx.x * ImInfoSize + 2]);
  T im_h = round(im_info[blockIdx.x * ImInfoSize] /
                 im_info[blockIdx.x * ImInfoSize + 2]);
  for (int i = threadIdx.x; i < (lod[blockIdx.x + 1] - lod[blockIdx.x]) * width;
       i += BlockSize) {
    int idx = lod[blockIdx.x] * width + i;
    T im_size = (idx % 2 == 0) ? im_w : im_h;
    output[idx] = max(min(input[idx], im_size - 1), T(0.));
  }
}

template <typename T, typename Context>
void GPUBoxClipKernel(const Context &dev_ctx,
                      const DenseTensor &input,
                      const DenseTensor &im_info,
                      DenseTensor *output) {
  auto *input_p = &input;
  auto *im_info_p = &im_info;

  const int64_t num = input_p->dims()[0];
  const int64_t bbox_width = input_p->numel() / num;
  auto lod = input_p->lod();
  phi::LegacyLoD abs_offset_lod = phi::ToAbsOffset(lod);

  auto stream = dev_ctx.stream();
  const size_t batch_size = lod.back().size() - 1;
  T *output_data = dev_ctx.template Alloc<T>(output);
  phi::MixVector<size_t> mix_vector(&abs_offset_lod[0]);
  GPUBoxClip<T, 512><<<batch_size, 512, 0, stream>>>(
      input_p->data<T>(),
      mix_vector.CUDAMutableData(dev_ctx.GetPlace()),
      bbox_width,
      im_info_p->data<T>(),
      output_data);
  mix_vector.CopyToCPU();
}

}  // namespace phi

PD_REGISTER_KERNEL(
    box_clip, GPU, ALL_LAYOUT, phi::GPUBoxClipKernel, float, double) {}
