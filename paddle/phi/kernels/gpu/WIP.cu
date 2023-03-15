// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
// ref:
// https://github.com/facebookresearch/xformers/blob/b6be33aecb5297f3f994568cf29e194a75e47667/xformers/ops/fmha/common.py#L102

#include "paddle/phi/backends/gpu/cuda/cuda_helper.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/slice.h"
#include "paddle/phi/kernels/pad_kernel.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void ViewSliceHelper(T* data,
                                int stride,
                                int in_last_dim,
                                int out_second_dim) {
  CUDA_KERNEL_LOOP_TYPE(i, stride * in_last_dim, int64_t) {
    if (i % in_last_dim >= out_second_dim) {
      *(data + i) = std::numeric_limits<T>::infinity();
    }
  }
}

template <typename T>
phi::DenseTensor* get_pad_lse(const phi::GPUContext& dev_ctx,
                              phi::DenseTensor* lse,
                              int out_second_dim,
                              int pad_to,
                              bool force_pad_inf = false,
                              phi::DenseTensor* out = nullptr) {
  int pad_amount = (pad_to - (lse->dims()[2] % pad_to)) % pad_to;
  if (pad_amount > 0) {
    phi::DenseTensor tmp;
    if (force_pad_inf) {
      tmp = phi::funcs::Slice<T, phi::GPUContext>(
          dev_ctx, *lse, {2}, {0}, {out_second_dim});
      pad_amount = (pad_to - (tmp.dims()[2] % pad_to)) % pad_to;
    }
    PADDLE_ENFORCE_NE(out,
                      nullptr,
                      phi::errors::InvalidArgument(
                          "The out ptr shouldn't be null if need pad amount"));
    phi::PadKernel<T, phi::GPUContext>(
        dev_ctx, tmp, {0, pad_amount}, std::numeric_limits<T>::infinity(), out);
    return out;
  } else if (force_pad_inf && out_second_dim != lse->dims()[2]) {
    auto in_dim = lse->dims();
    auto in_data = lse->template data<T>();
    int stride = in_dim[0] * in_dim[1];

    int block = PADDLE_CUDA_NUM_THREADS;
    int64_t n = lse->numel();
    dim3 grid = dim3((n + block - 1) / block);
    phi::backends::gpu::LimitGridDim(dev_ctx, &grid);
    ViewSliceHelper<T><<<grid, block, 0, dev_ctx.stream()>>>(
        in_data, stride, in_dim[2], out_second_dim);
    return lse;
  }
}
}  // namespace phi
