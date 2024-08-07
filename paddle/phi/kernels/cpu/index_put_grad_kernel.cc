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
#include <array>
#include <numeric>
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T>
void set_zero_kernel(const int64_t N,
                     const int64_t** indices,
                     const phi::DDim& stride,
                     const phi::DDim& shape,
                     T* out) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t idx = 0; idx < N; ++idx) {
    int64_t cur_ix = 0;
    int64_t offset = 0;

    for (int i = 0; i < shape.size(); ++i) {
      cur_ix = (static_cast<int64_t>(*(indices[i] + idx)));
      if (cur_ix < 0) {
        cur_ix += shape[i];
      }
      offset += stride[i] * cur_ix;
    }
    *(out + offset) = 0;
  }
}

template <typename T>
void index_put_grad_kernel(const int64_t N,
                           const T* out_grad,
                           const int64_t** indices,
                           const phi::DDim& stride,
                           const phi::DDim& shape,
                           T* value_grad) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t idx = 0; idx < N; ++idx) {
    int64_t cur_ix = 0;
    int64_t offset = 0;

    for (int i = 0; i < shape.size(); ++i) {
      cur_ix = (static_cast<int64_t>(*(indices[i] + idx)));
      if (cur_ix < 0) {
        cur_ix += shape[i];
      }
      offset += stride[i] * cur_ix;
    }
    *(value_grad + idx) = *(out_grad + offset);
  }
}

template <typename T, typename Context>
void LaunchIndexPutGradKernel(const Context& dev_ctx,
                              const std::vector<const DenseTensor*>& indices,
                              const DenseTensor& out_grad,
                              bool accumulate,
                              DenseTensor* value_grad,
                              DenseTensor* x_grad) {
  std::array<const int64_t*, 7> pd_indices = {};
  for (size_t i = 0; i < indices.size(); ++i) {
    pd_indices[i] = indices[i]->data<int64_t>();
  }

  if (x_grad) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    if (!accumulate) {
      T* x_grad_data = x_grad->data<T>();

      auto x_grad_dims = x_grad->dims();
      const int64_t numel = indices[0]->numel();
      auto x_grad_stride = common::stride(x_grad_dims);

      set_zero_kernel<T>(
          numel, pd_indices.data(), x_grad_stride, x_grad_dims, x_grad_data);
    }
  }

  const auto& out_grad_dims = out_grad.dims();
  const int64_t numel = indices[0]->numel();
  auto out_grad_stride = common::stride(out_grad_dims);

  if (value_grad) {
    if (value_grad->numel() == 1) {
      DenseTensor tmp_value_grad(value_grad->dtype());
      tmp_value_grad.Resize(indices[0]->dims());

      T* tmp_value_grad_data = dev_ctx.template Alloc<T>(&tmp_value_grad);
      auto out_grad_data = out_grad.data<T>();

      index_put_grad_kernel<T>(numel,
                               out_grad_data,
                               pd_indices.data(),
                               out_grad_stride,
                               out_grad_dims,
                               tmp_value_grad_data);

      std::vector<int> v_dims(tmp_value_grad.dims().size());
      std::iota(v_dims.begin(), v_dims.end(), 0);
      IntArray v_axis(v_dims);
      SumKernel<T>(dev_ctx,
                   tmp_value_grad,
                   v_axis,
                   value_grad->dtype(),
                   false,
                   value_grad);
    } else if (value_grad->numel() == indices[0]->numel()) {
      T* value_grad_data = dev_ctx.template Alloc<T>(value_grad);
      auto out_grad_data = out_grad.data<T>();

      index_put_grad_kernel<T>(numel,
                               out_grad_data,
                               pd_indices.data(),
                               out_grad_stride,
                               out_grad_dims,
                               value_grad_data);
    } else {
      DenseTensor tmp_value_grad(value_grad->dtype());
      tmp_value_grad.Resize(indices[0]->dims());

      T* tmp_value_grad_data = dev_ctx.template Alloc<T>(&tmp_value_grad);
      auto out_grad_data = out_grad.data<T>();

      index_put_grad_kernel<T>(numel,
                               out_grad_data,
                               pd_indices.data(),
                               out_grad_stride,
                               out_grad_dims,
                               tmp_value_grad_data);

      std::vector<int64_t> after_dims =
          common::vectorize(tmp_value_grad.dims());
      std::vector<int64_t> before_dims = common::vectorize(value_grad->dims());
      std::vector<int64_t> compress_dims;
      std::vector<int64_t> dims_without_1;

      funcs::CalCompressedDimsWith1AndWithout1(
          &after_dims, &before_dims, &compress_dims, &dims_without_1);

      auto pre_dims = value_grad->dims();
      value_grad->Resize(common::make_ddim(dims_without_1));
      IntArray v_axis(compress_dims);
      SumKernel<T>(dev_ctx,
                   tmp_value_grad,
                   v_axis,
                   value_grad->dtype(),
                   false,
                   value_grad);
      value_grad->Resize(pre_dims);
    }
  }
}

template <typename T, typename Context>
void IndexPutGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const std::vector<const DenseTensor*>& indices,
                        const DenseTensor& value,
                        const DenseTensor& out_grad,
                        bool accumulate,
                        DenseTensor* x_grad,
                        DenseTensor* value_grad) {
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      value.dtype(),
      common::errors::InvalidArgument(
          "The data type of tensor value must be same to the data type "
          "of tensor x."));
  std::vector<DenseTensor> tmp_args;
  std::vector<const phi::DenseTensor*> int_indices_v =
      funcs::DealWithBoolIndices<T, Context>(dev_ctx, indices, &tmp_args);
  if (int_indices_v.empty()) {
    if (x_grad) {
      phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    }
    if (value_grad) {
      FullKernel<T, Context>(dev_ctx,
                             common::vectorize(value_grad->dims()),
                             0.0f,
                             value_grad->dtype(),
                             value_grad);
    }
    return;
  }
  auto bd_dim = funcs::BroadCastTensorsDims(int_indices_v);

  std::vector<int64_t> res_dim_v(common::vectorize(bd_dim));
  std::vector<const phi::DenseTensor*> res_indices_v(x.dims().size(), nullptr);
  std::vector<DenseTensor> tmp_res_indices_v;
  std::vector<DenseTensor> range_tensor_v;

  for (int i = static_cast<int>(int_indices_v.size()); i < x.dims().size();
       ++i) {
    range_tensor_v.emplace_back(funcs::GetRangeTensor<int64_t, Context>(
        dev_ctx, x.dims()[i], phi::DataType::INT64));
  }

  funcs::DealWithIndices<T, Context>(dev_ctx,
                                     x,
                                     int_indices_v,
                                     &res_indices_v,
                                     &tmp_res_indices_v,
                                     range_tensor_v,
                                     bd_dim,
                                     &res_dim_v);

  LaunchIndexPutGradKernel<T, Context>(
      dev_ctx, res_indices_v, out_grad, accumulate, value_grad, x_grad);
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
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
