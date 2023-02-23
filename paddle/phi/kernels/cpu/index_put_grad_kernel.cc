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

#include "paddle/phi/kernels/index_put_grad_kernel.h"
#include <numeric>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/array.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

// TODO(LiuYang): Here when ix is negative, we need extra error handling code
template <typename T, size_t Rank>
void index_put_grad_kernel(const int64_t N,
                           const T* out_grad,
                           const int64_t** indices,
                           phi::Array<int64_t, Rank> stride,
                           phi::Array<int64_t, Rank> shape,
                           T* value_grad) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t idx = 0; idx < N; ++idx) {
    int64_t cur_ix = 0;
    int64_t offset = 0;
    // here why can't use unroll
    for (size_t i = 0; i < Rank; ++i) {
      cur_ix = (int64_t(*(indices[i] + idx)));
      if (cur_ix < 0) {
        cur_ix += shape[i];
      }
      offset += stride[i] * cur_ix;
    }
    *(value_grad + idx) = *(out_grad + offset);
  }
}

template <typename T, typename Context, size_t Rank>
void LaunchIndexPutGradKernel(const Context& dev_ctx,
                              const std::vector<const DenseTensor*>& indices_v,
                              const DenseTensor& out_grad,
                              DenseTensor* value_grad,
                              DenseTensor* x_grad) {
  if (x_grad) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
  }
  if (value_grad) {
    if (value_grad->numel() == 1) {
      std::vector<int> v_dims(out_grad.dims().size());
      std::iota(v_dims.begin(), v_dims.end(), 0);
      IntArray v_axis(v_dims);
      SumKernel<T>(
          dev_ctx, out_grad, v_axis, value_grad->dtype(), false, value_grad);
    } else if (value_grad->numel() == indices_v[0]->numel()) {
      T* value_grad_data = dev_ctx.template Alloc<T>(value_grad);
      auto out_grad_data = out_grad.data<T>();

      auto out_grad_dims = out_grad.dims();
      const int64_t numel = indices_v[0]->numel();
      auto out_grad_stride = phi::stride(out_grad_dims);

      phi::Array<int64_t, Rank> stride_a;
      phi::Array<int64_t, Rank> shape_a;

      for (size_t idx = 0; idx < Rank; ++idx) {
        stride_a[idx] = out_grad_stride[idx];
        shape_a[idx] = out_grad_dims[idx];
      }

      const int64_t* pd_indices[Rank];
      for (size_t i = 0; i < Rank; ++i) {
        pd_indices[i] = indices_v[i]->data<int64_t>();
      }
      index_put_grad_kernel<T, Rank>(
          numel, out_grad_data, pd_indices, stride_a, shape_a, value_grad_data);
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "shape of slice of source tensor isn't the same as shape of value"));
    }
  }
}

// Note(LiuYang): Here I don't take it in consideration that whether we support
// value_tensor can be broadcast to X[indice] when calls like X[indice] = value
template <typename T, typename Context>
void IndexPutGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const std::vector<const DenseTensor*>& indices_v,
                        const DenseTensor& value,
                        const DenseTensor& out_grad,
                        DenseTensor* value_grad,
                        DenseTensor* x_grad) {
  const size_t total_dims = out_grad.dims().size();
  PADDLE_ENFORCE_EQ(indices_v.size(),
                    total_dims,
                    phi::errors::InvalidArgument(
                        "The size %d of indices must be equal to the size %d "
                        "of the dimension of source tensor x.",
                        indices_v.size(),
                        total_dims));

  switch (total_dims) {
    case 1:
      LaunchIndexPutGradKernel<T, Context, 1>(
          dev_ctx, indices_v, out_grad, value_grad, x_grad);
      break;
    case 2:
      LaunchIndexPutGradKernel<T, Context, 2>(
          dev_ctx, indices_v, out_grad, value_grad, x_grad);
      break;
    case 3:
      LaunchIndexPutGradKernel<T, Context, 3>(
          dev_ctx, indices_v, out_grad, value_grad, x_grad);
      break;
    case 4:
      LaunchIndexPutGradKernel<T, Context, 4>(
          dev_ctx, indices_v, out_grad, value_grad, x_grad);
      break;
    case 5:
      LaunchIndexPutGradKernel<T, Context, 5>(
          dev_ctx, indices_v, out_grad, value_grad, x_grad);
      break;
    case 6:
      LaunchIndexPutGradKernel<T, Context, 6>(
          dev_ctx, indices_v, out_grad, value_grad, x_grad);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "dims of input tensor should be less than 7, But received"
          "%d",
          x.dims().size()));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(index_put_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexPutGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
