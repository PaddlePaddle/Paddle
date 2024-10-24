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

#include <algorithm>  // for min, max
#include <string>

#include "paddle/common/errors.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"

namespace phi::fusion {

template <typename T, typename Context>
void FusionSeqConvEltAddReluKernel(const Context& dev_ctx,
                                   const DenseTensor& x,
                                   const DenseTensor& filter,
                                   const DenseTensor& bias,
                                   const int context_length,
                                   const int context_start,
                                   const int context_stride,
                                   DenseTensor* out,
                                   DenseTensor* col_mat) {
  auto x_lod = x.lod();
  auto x_dims = common::vectorize<int64_t>(x.dims());
  auto w_dims = common::vectorize<int64_t>(filter.dims());
  PADDLE_ENFORCE_EQ(
      bias.numel(),
      w_dims[1],
      common::errors::InvalidArgument(
          "bias size should be equal to weights feature size, but received "
          "bias size is: %d, weights feature size is: %d.",
          bias.numel(),
          w_dims[1]));
  PADDLE_ENFORCE_EQ(
      x_lod.size(),
      1UL,
      common::errors::InvalidArgument(
          "Only support one level sequence now, but received value is: %d.",
          x_lod.size()));

  const T* x_data = x.data<T>();
  const T* w_data = filter.data<T>();
  const T* b_data = bias.data<T>();
  T* y_data = dev_ctx.template Alloc<T>(out);
  T* col_data = dev_ctx.template Alloc<T>(col_mat);

  int up_pad = std::max(0, -context_start);
  int down_pad = std::max(0, context_start + context_length - 1);
  // im2col
  int src_mat_w = static_cast<int>(x_dims[1]);
  int src_mat_w_sz = src_mat_w * sizeof(T);
  int col_mat_w = static_cast<int>(w_dims[0]);
  int col_mat_w_sz = col_mat_w * sizeof(T);
  for (int i = 0; i < static_cast<int>(x_lod[0].size()) - 1; ++i) {
    int st = static_cast<int>(x_lod[0][i]);
    int ed = static_cast<int>(x_lod[0][i + 1]);
    const T* src_data = x_data + st * src_mat_w;
    T* dst_data = col_data + st * col_mat_w;
    int seq_len = ed - st;
    if (seq_len > up_pad + down_pad) {
      // zero all up_pad and fill data
      std::memset(dst_data, 0, up_pad * col_mat_w_sz);
      dst_data = dst_data + up_pad * src_mat_w;
      int copy_size = col_mat_w_sz - up_pad * src_mat_w_sz;
      for (int j = 0; j < up_pad; ++j) {
        // blas.VCOPY?
        std::memcpy(dst_data, src_data, copy_size);
        dst_data += (col_mat_w - src_mat_w);
        copy_size += src_mat_w_sz;
      }
      // fill data
      if (context_start > 0) {
        src_data += context_start * src_mat_w;
      }
      for (int j = 0; j < seq_len - up_pad - down_pad; ++j) {
        std::memcpy(dst_data, src_data, copy_size);
        dst_data += col_mat_w;
        src_data += src_mat_w;
      }
      // zero all down_pad and fill data
      std::memset(dst_data, 0, down_pad * col_mat_w_sz);
      copy_size -= src_mat_w_sz;
      for (int j = 0; j < down_pad; ++j) {
        if (copy_size < 0) {
          copy_size = 0;
        }
        std::memcpy(dst_data, src_data, copy_size);
        dst_data += col_mat_w;
        src_data += src_mat_w;
        copy_size -= src_mat_w_sz;
      }
    } else {
      std::memset(dst_data, 0, seq_len * col_mat_w_sz);
      dst_data = dst_data + up_pad * src_mat_w;
      int zero_sz = up_pad * src_mat_w_sz;
      int cur_src_sz = seq_len * src_mat_w_sz;
      for (int j = 0; j < std::min(up_pad, seq_len); ++j) {
        int copy_size = std::min(cur_src_sz, col_mat_w_sz - zero_sz);
        std::memcpy(dst_data, src_data, copy_size);
        dst_data += (col_mat_w - src_mat_w);
        zero_sz -= src_mat_w_sz;
      }
      // from bottom
      dst_data = col_data + ed * col_mat_w;
      src_data = x_data + st * src_mat_w;
      if (context_start > 0) {
        src_data += context_start * src_mat_w;
      }
      zero_sz = down_pad * src_mat_w_sz;
      for (int j = 1; j <= std::min(down_pad, seq_len); ++j) {
        int copy_size = std::min(cur_src_sz, col_mat_w_sz - zero_sz);
        if (copy_size < 0) {
          copy_size = 0;
        }
        std::memcpy(dst_data - (zero_sz + copy_size) / sizeof(T),
                    src_data + std::max(seq_len - j - up_pad, 0) * src_mat_w,
                    copy_size);
        dst_data -= col_mat_w;
        zero_sz -= src_mat_w_sz;
      }
    }
  }
  phi::funcs::FCFunctor<Context, T> fc;
  fc(dev_ctx,
     x_dims[0],
     w_dims[1],
     w_dims[0],
     col_data,
     w_data,
     y_data,
     b_data,
     true);
}

}  // namespace phi::fusion

PD_REGISTER_KERNEL(fusion_seqconv_eltadd_relu,
                   CPU,
                   ALL_LAYOUT,
                   phi::fusion::FusionSeqConvEltAddReluKernel,
                   float,
                   double) {}
