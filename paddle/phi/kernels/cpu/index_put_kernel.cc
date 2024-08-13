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

#include "paddle/phi/kernels/index_put_kernel.h"
#include <array>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"

namespace phi {

template <typename T>
void index_put_kernel(const int64_t N,
                      const T* x UNUSED,
                      const T* vals,
                      const int64_t** indices,
                      const phi::DDim& stride,
                      const phi::DDim& shape,
                      int64_t is_single_val_tensor,
                      bool accumulate,
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

    if (accumulate) {
      *(out + offset) += *(vals + (idx & is_single_val_tensor));
    } else {
      *(out + offset) = *(vals + (idx & is_single_val_tensor));
    }
  }
}

template <typename T, typename Context>
void LaunchIndexPutKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const std::vector<const DenseTensor*>& indices,
                          const DenseTensor& value,
                          bool accumulate,
                          DenseTensor* out) {
  auto* x_data = x.data<T>();
  auto* val_data = value.data<T>();
  bool is_initialized = out->initialized();
  bool is_same_place = true;

  if (is_initialized) {
    is_same_place = (x.place() == out->place());
  }

  T* out_data = dev_ctx.template Alloc<T>(out);

  if (!is_initialized || !is_same_place) {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  }

  const auto& x_dims = x.dims();
  const int64_t numel = indices[0]->numel();
  auto x_stride = common::stride(x_dims);

  int64_t is_single_val_tensor = (value.numel() == 1) ? 0 : INT64_MAX;

  std::array<const int64_t*, 7> pd_indices;
  for (size_t i = 0; i < indices.size(); ++i) {
    pd_indices[i] = indices[i]->data<int64_t>();
  }

  index_put_kernel<T>(numel,
                      x_data,
                      val_data,
                      pd_indices.data(),
                      x_stride,
                      x_dims,
                      is_single_val_tensor,
                      accumulate,
                      out_data);
}

template <typename T, typename Context>
void IndexPutKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<const DenseTensor*>& indices,
                    const DenseTensor& value,
                    bool accumulate,
                    DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      value.dtype(),
      common::errors::InvalidArgument(
          "The data type of tensor value must be same to the data type "
          "of tensor x."));
  PADDLE_ENFORCE_EQ(
      indices.empty(),
      false,
      common::errors::InvalidArgument("Indices cannot be empty."));

  const size_t total_dims = x.dims().size();
  PADDLE_ENFORCE_LE(total_dims,
                    6,
                    common::errors::InvalidArgument(
                        "Dims of input tensor should be less than 7."));

  std::vector<DenseTensor> tmp_args;
  std::vector<const phi::DenseTensor*> int_indices_v =
      funcs::DealWithBoolIndices<T, Context>(dev_ctx, indices, &tmp_args);
  if (int_indices_v.empty()) {
    if (!out->initialized()) {
      phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }

  auto bd_dim = funcs::BroadCastTensorsDims(int_indices_v);

  std::vector<int64_t> res_dim_v(common::vectorize(bd_dim));
  std::vector<const phi::DenseTensor*> res_indices_v(x.dims().size(), nullptr);
  std::vector<DenseTensor> tmp_res_indices_v;
  std::vector<DenseTensor> tmp_value_v;
  std::vector<DenseTensor> range_tensor_v;
  const DenseTensor* ptr_value = nullptr;

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
  if (value.numel() != 1) {
    tmp_value_v.emplace_back(
        DenseTensor(value.dtype()).Resize(common::make_ddim(res_dim_v)));
    ExpandKernel<T, Context>(
        dev_ctx, value, IntArray(res_dim_v), &tmp_value_v[0]);
    ptr_value = &tmp_value_v[0];
  } else {
    ptr_value = &value;
  }

  LaunchIndexPutKernel<T, Context>(
      dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
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
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
