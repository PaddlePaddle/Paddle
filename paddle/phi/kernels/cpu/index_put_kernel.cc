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
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"

namespace phi {

template <typename T>
void range_kernel(int64_t N, T* out) {
  for (int64_t idx = 0; idx < N; ++idx) {
    out[idx] = idx;
  }
}

template <typename T, typename Context>
phi::DenseTensor GetRangeTensor(const Context& dev_ctx,
                                int64_t N,
                                phi::DataType dtype) {
  phi::DenseTensor res(dtype);
  res.Resize(phi::make_ddim({N}));
  DenseTensor* p_res = &res;
  T* out = dev_ctx.template Alloc<T>(p_res);
  range_kernel<T>(N, out);
  return res;
}

template <typename T, size_t Rank>
void index_put_kernel(const int64_t N,
                      const T* x,
                      const T* vals,
                      const int64_t** indices,
                      phi::Array<int64_t, Rank> stride,
                      phi::Array<int64_t, Rank> shape,
                      int64_t isSingleValTensor,
                      bool accumulate,
                      T* out) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t idx = 0; idx < N; ++idx) {
    int64_t cur_ix = 0;
    int64_t offset = 0;

    for (size_t i = 0; i < Rank; ++i) {
      cur_ix = (int64_t(*(indices[i] + idx)));
      if (cur_ix < 0) {
        cur_ix += shape[i];
      }
      offset += stride[i] * cur_ix;
    }

    if (accumulate) {
      *(out + offset) += *(vals + (idx & isSingleValTensor));
    } else {
      *(out + offset) = *(vals + (idx & isSingleValTensor));
    }
  }
}

template <typename T, typename Context, size_t Rank>
void LaunchIndexPutKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const std::vector<const DenseTensor*>& indices_v,
                          const DenseTensor& value,
                          bool accumulate,
                          DenseTensor* out) {
  auto* x_data = x.data<T>();
  auto* val_data = value.data<T>();
  bool isInitialized = out->initialized();
  T* out_data = dev_ctx.template Alloc<T>(out);

  if (!isInitialized) {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  }

  auto x_dims = x.dims();
  const int64_t numel = indices_v[0]->numel();
  auto x_stride = phi::stride(x_dims);

  phi::Array<int64_t, Rank> stride_a;
  phi::Array<int64_t, Rank> shape_a;

  for (size_t idx = 0; idx < Rank; ++idx) {
    stride_a[idx] = x_stride[idx];
    shape_a[idx] = x_dims[idx];
  }

  int64_t isSingleValTensor = (value.numel() == 1) ? 0 : INT64_MAX;

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
                            accumulate,
                            out_data);
}

template <typename T, typename Context>
void IndexPutKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<const DenseTensor*>& indices_v,
                    const DenseTensor& value,
                    bool accumulate,
                    DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      value.dtype(),
      phi::errors::InvalidArgument(
          "The data type of tensor in indices must be same to the data type "
          "of tensor x."));
  std::vector<DenseTensor> tmp_args;
  std::vector<const phi::DenseTensor*> int_indices_v =
      DealWithBoolIndices<T, Context>(dev_ctx, indices_v, &tmp_args);
  const size_t total_dims = x.dims().size();
  auto bd_dim = BroadCastTensorsDims(int_indices_v);

  std::vector<int64_t> res_dim_v(phi::vectorize(bd_dim));
  std::vector<const phi::DenseTensor*> res_indices_v(x.dims().size(), nullptr);
  std::vector<DenseTensor> tmp_res_indices_v;
  std::vector<DenseTensor> tmp_value_v;
  const DenseTensor* ptr_value = nullptr;

  if (int_indices_v.size() < total_dims) {
    std::vector<int64_t> tmp_x_dims = phi::vectorize(x.dims());
    int len_bd_dim = bd_dim.size();
    res_dim_v.insert(
        res_dim_v.end(), tmp_x_dims.begin() + len_bd_dim, tmp_x_dims.end());

    std::vector<DenseTensor> reshaped_indices_v;
    for (size_t i = 0; i < int_indices_v.size(); ++i) {
      if (int_indices_v[i]->dtype() == phi::DataType::INT32) {
        reshaped_indices_v.emplace_back(phi::Cast<int, Context>(
            dev_ctx, *int_indices_v[i], phi::DataType::INT64));
      } else {
        reshaped_indices_v.emplace_back(*int_indices_v[i]);
      }
    }
    for (size_t i = int_indices_v.size(); i < total_dims; ++i) {
      reshaped_indices_v.emplace_back(GetRangeTensor<int64_t, Context>(
          dev_ctx, res_dim_v[i], phi::DataType::INT64));
    }
    phi::DDim res_dim = phi::make_ddim(res_dim_v);

    for (size_t i = 0; i < reshaped_indices_v.size(); ++i) {
      tmp_res_indices_v.emplace_back(
          GetReshapeAndExpandTensor<int64_t, Context>(
              dev_ctx,
              reshaped_indices_v[i],
              res_dim,
              ((i < int_indices_v.size()) ? 0 : i)));
    }
    for (size_t i = 0; i < res_indices_v.size(); ++i) {
      res_indices_v[i] = &tmp_res_indices_v[i];
    }
    // value至少需要满足与已有的indices为可broadcast_to关系
    if (value.numel() != 1) {
      tmp_value_v.emplace_back(DenseTensor(value.dtype()).Resize(res_dim));
      ExpandKernel<T, Context>(dev_ctx,
                               value,
                               IntArray(phi::vectorize<int64_t>(res_dim)),
                               &tmp_value_v[0]);
      ptr_value = &tmp_value_v[0];
    } else {
      ptr_value = &value;
    }
  } else {
    std::vector<DenseTensor> int_indices_v_tmp;

    for (size_t i = 0; i < int_indices_v.size(); ++i) {
      if (int_indices_v[i]->dtype() == phi::DataType::INT32) {
        int_indices_v_tmp.emplace_back(phi::Cast<int, Context>(
            dev_ctx, *int_indices_v[i], phi::DataType::INT64));
      } else {
        int_indices_v_tmp.emplace_back(*int_indices_v[i]);
      }
    }

    for (size_t i = 0; i < int_indices_v.size(); ++i) {
      if (bd_dim != int_indices_v[i]->dims()) {
        tmp_res_indices_v.emplace_back(
            DenseTensor(phi::DataType::INT64).Resize(bd_dim));
        ExpandKernel<int64_t, Context>(
            dev_ctx,
            int_indices_v_tmp[i],
            IntArray(phi::vectorize<int64_t>(bd_dim)),
            &tmp_res_indices_v[i]);
      } else {
        tmp_res_indices_v.emplace_back(int_indices_v_tmp[i]);
      }
    }

    for (size_t i = 0; i < res_indices_v.size(); ++i) {
      res_indices_v[i] = &tmp_res_indices_v[i];
    }

    if (value.numel() != 1) {
      tmp_value_v.emplace_back(DenseTensor(value.dtype()).Resize(bd_dim));
      ExpandKernel<T, Context>(dev_ctx,
                               value,
                               IntArray(phi::vectorize<int64_t>(bd_dim)),
                               &tmp_value_v[0]);
      ptr_value = &tmp_value_v[0];
    } else {
      ptr_value = &value;
    }
  }

  switch (total_dims) {
    case 1:
      LaunchIndexPutKernel<T, Context, 1>(
          dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
      break;
    case 2:
      LaunchIndexPutKernel<T, Context, 2>(
          dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
      break;
    case 3:
      LaunchIndexPutKernel<T, Context, 3>(
          dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
      break;
    case 4:
      LaunchIndexPutKernel<T, Context, 4>(
          dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
      break;
    case 5:
      LaunchIndexPutKernel<T, Context, 5>(
          dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
      break;
    case 6:
      LaunchIndexPutKernel<T, Context, 6>(
          dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
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
                   bool) {}
