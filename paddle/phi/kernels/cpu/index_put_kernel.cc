// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_put_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/array.h"
#include "paddle/phi/kernels/expand_kernel.h"

namespace phi {

template <typename T, size_t Rank>
void index_put_kernel(const int64_t N,
                      const T* x,
                      const T* vals,
                      const int64_t** indices,
                      phi::Array<int64_t, Rank> stride,
                      phi::Array<int64_t, Rank> shape,
                      int64_t isSingleValTensor,
                      T* out) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t idx = 0; idx < N; ++idx) {
    int64_t cur_ix = 0;
    int64_t offset = 0;
    // #pragma unroll Rank
    for (size_t i = 0; i < Rank; ++i) {
      cur_ix = (int64_t(*(indices[i] + idx)));
      if (cur_ix < 0) {
        cur_ix += shape[i];
      }
      offset += stride[i] * cur_ix;
    }

    *(out + offset) = *(vals + (idx & isSingleValTensor));
  }
}

template <typename T, typename Context, size_t Rank>
void LaunchIndexPutKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const std::vector<const DenseTensor*>& indices_v,
                          const DenseTensor& value,
                          DenseTensor* out) {
  // auto* x_data = const_cast<T*>(x.data<T>());
  auto* x_data = x.data<T>();
  auto* val_data = value.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);
  // T* out_data = out->data<T>();

  auto x_dims = x.dims();
  const int64_t numel = indices_v[0]->numel();
  auto x_stride = phi::stride(x_dims);

  phi::Array<int64_t, Rank> stride_a;
  phi::Array<int64_t, Rank> shape_a;

  for (size_t idx = 0; idx < Rank; ++idx) {
    stride_a[idx] = x_stride[idx];
    shape_a[idx] = x_dims[idx];
  }

  // phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);

  int64_t isSingleValTensor = (value.numel() == 1) ? 0 : INT64_MAX;

  // get cpu pointer array
  const int64_t* pd_indices[Rank];
  for (size_t i = 0; i < Rank; ++i) {
    pd_indices[i] = indices_v[i]->data<int64_t>();
  }

  index_put_kernel<T, Rank>(numel,
                            x_data,
                            val_data,
                            pd_indices,
                            stride_a,
                            shape_a,
                            isSingleValTensor,
                            out_data);
}

template <typename T, typename Context>
void IndexPutKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<const DenseTensor*>& indices_v,
                    const DenseTensor& value,
                    DenseTensor* out) {
  const size_t total_dims = x.dims().size();
  PADDLE_ENFORCE_EQ(indices_v.size(),
                    total_dims,
                    phi::errors::InvalidArgument(
                        "The size %d of indices must be equal to the size %d "
                        "of the dimension of source tensor x.",
                        indices_v.size(),
                        total_dims));

  switch (total_dims) {
    case 1:
      LaunchIndexPutKernel<T, Context, 1>(dev_ctx, x, indices_v, value, out);
      break;
    case 2:
      LaunchIndexPutKernel<T, Context, 2>(dev_ctx, x, indices_v, value, out);
      break;
    case 3:
      LaunchIndexPutKernel<T, Context, 3>(dev_ctx, x, indices_v, value, out);
      break;
    case 4:
      LaunchIndexPutKernel<T, Context, 4>(dev_ctx, x, indices_v, value, out);
      break;
    case 5:
      LaunchIndexPutKernel<T, Context, 5>(dev_ctx, x, indices_v, value, out);
      break;
    case 6:
      LaunchIndexPutKernel<T, Context, 6>(dev_ctx, x, indices_v, value, out);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "dims of input tensor should be less than 7, But received"
          "%d",
          x.dims().size()));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(index_put,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexPutKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
