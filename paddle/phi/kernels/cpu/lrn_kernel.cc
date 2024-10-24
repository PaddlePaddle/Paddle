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

#include "paddle/phi/kernels/impl/lrn_kernel_impl.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#ifdef PADDLE_WITH_DNNL
#include "paddle/phi/backends/onednn/onednn_helper.h"
#endif

namespace phi {

template <typename T>
struct LRNFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* out,
                  phi::DenseTensor* mid,
                  int N,
                  int C,
                  int H,
                  int W,
                  int n,
                  T k,
                  T alpha,
                  T beta,
                  const DataLayout data_layout) {
    auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(dev_ctx);
    phi::funcs::Transpose<phi::CPUContext, T, 4> transpose;
    phi::DenseTensor in_transpose, mid_transpose, out_transpose;
    // if channel_last, transpose to channel_first
    if (data_layout == DataLayout::kNHWC) {
      auto in_dims = input.dims();
      std::vector<int64_t> shape(
          {in_dims[0], in_dims[3], in_dims[1], in_dims[2]});
      in_transpose.Resize(common::make_ddim(shape));
      mid_transpose.Resize(common::make_ddim(shape));
      out_transpose.Resize(common::make_ddim(shape));
      dev_ctx.Alloc<T>(&in_transpose);
      dev_ctx.Alloc<T>(&mid_transpose);
      dev_ctx.Alloc<T>(&out_transpose);
      std::vector<int> axis = {0, 3, 1, 2};
      transpose(dev_ctx, input, &in_transpose, axis);
    } else {
      in_transpose = input;
      mid_transpose = *mid;
      out_transpose = *out;
      mid_transpose.Resize(mid->dims());
      out_transpose.Resize(out->dims());
      dev_ctx.Alloc<T>(&mid_transpose);
      dev_ctx.Alloc<T>(&out_transpose);
    }

    const T* idata = in_transpose.data<T>();
    T* odata = out_transpose.data<T>();
    T* mdata = mid_transpose.data<T>();

    phi::DenseTensor squared;
    squared.Resize({1, C + n - 1, H, W});
    T* sdata = dev_ctx.Alloc<T>(&squared);
    std::memset(sdata, 0, sizeof(T) * squared.numel());
    for (int i = 0; i < mid->numel(); ++i) {
      mdata[i] = k;
    }
    int img_size = H * W;
    int fea_size = C * img_size;
    int pre_pad = (n - 1) / 2;
    // compute batches one by one
    for (int i = 0; i < N; ++i) {
      blas.VSQUARE(fea_size, idata + i * fea_size, sdata + pre_pad * img_size);
      // init the first channel of mid
      for (int c = 0; c < n; ++c) {
        blas.AXPY(img_size, alpha, sdata + c * img_size, mdata + i * fea_size);
      }
      for (int c = 1; c < C; ++c) {
        // copy previous scale
        int mid_offset = i * fea_size + c * img_size;
        std::memcpy(mdata + mid_offset,
                    mdata + mid_offset - img_size,
                    img_size * sizeof(T));
        // add last
        blas.AXPY(img_size,
                  alpha,
                  sdata + (c + n - 1) * img_size,
                  mdata + mid_offset);
        // sub rest
        blas.AXPY(
            img_size, -alpha, sdata + (c - 1) * img_size, mdata + mid_offset);
      }
    }
    // compute the final output
    blas.VPOW(mid->numel(), mdata, -beta, odata);
    blas.VMUL(mid->numel(), odata, idata, odata);

    // if channel_last, transpose the output(NCHW) to channel_last
    if (data_layout == DataLayout::kNHWC) {
      std::vector<int> axis = {0, 2, 3, 1};
      transpose(dev_ctx, mid_transpose, mid, axis);
      transpose(dev_ctx, out_transpose, out, axis);
    }
  }
};
template struct LRNFunctor<phi::CPUContext, float>;
template struct LRNFunctor<phi::CPUContext, double>;
}  // namespace phi

PD_REGISTER_KERNEL(lrn, CPU, ALL_LAYOUT, phi::LRNKernel, float) {}
