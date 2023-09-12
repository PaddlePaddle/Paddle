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
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"

namespace phi {

template <typename T, size_t Rank>
__global__ void index_put_cuda_kernel(const int64_t N,
                                      const T* x,
                                      const T* vals,
                                      int64_t** indices,
                                      phi::Array<int64_t, Rank> stride,
                                      phi::Array<int64_t, Rank> shape,
                                      int64_t is_single_val_tensor,
                                      bool accumulate,
                                      T* out) {
  int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  int64_t cur_ix = 0;

  if (idx >= N) {
    return;
  }
  int64_t offset = 0;
  for (int i = 0; i < Rank; ++i) {
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

template <typename T, typename Context, size_t Rank>
void LaunchIndexPutCudaKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const std::vector<const DenseTensor*>& indices,
                              const DenseTensor& value,
                              bool accumulate,
                              DenseTensor* out) {
  auto* x_data = x.data<T>();
  auto* val_data = value.data<T>();
  bool is_initialized = out->initialized();
  T* out_data = dev_ctx.template Alloc<T>(out);

  if (!is_initialized) {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  }

  auto x_dims = x.dims();
  const int64_t numel = indices[0]->numel();
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  auto x_stride = phi::stride(x_dims);

  phi::Array<int64_t, Rank> stride_a;
  phi::Array<int64_t, Rank> shape_a;

  for (size_t idx = 0; idx < Rank; ++idx) {
    stride_a[idx] = x_stride[idx];
    shape_a[idx] = x_dims[idx];
  }

  int64_t is_single_val_tensor = (value.numel() == 1) ? 0 : INT64_MAX;

  auto pd_indices =
      funcs::GetDevicePointerArray<int64_t, Context>(dev_ctx, indices);
  index_put_cuda_kernel<T, Rank>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          numel,
          x_data,
          val_data,
          pd_indices,
          stride_a,
          shape_a,
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
      phi::errors::InvalidArgument(
          "The data type of tensor value must be same to the data type "
          "of tensor x."));
  PADDLE_ENFORCE_EQ(indices.empty(),
                    false,
                    phi::errors::InvalidArgument("Indices cannot be empty."));
  std::vector<DenseTensor> tmp_args;
  std::vector<const phi::DenseTensor*> int_indices_v =
      funcs::DealWithBoolIndices<T, Context>(dev_ctx, indices, &tmp_args);
  if (int_indices_v.empty()) {
    if (!out->initialized()) {
      phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }
  const size_t total_dims = x.dims().size();
  auto bd_dim = funcs::BroadCastTensorsDims(int_indices_v);

  std::vector<int64_t> res_dim_v(phi::vectorize(bd_dim));
  std::vector<const phi::DenseTensor*> res_indices_v(x.dims().size(), nullptr);
  std::vector<DenseTensor> tmp_res_indices_v;
  std::vector<DenseTensor> tmp_value_v;
  std::vector<DenseTensor> range_tensor_v;
  const DenseTensor* ptr_value = nullptr;

  for (int i = int_indices_v.size(); i < x.dims().size(); ++i) {
    range_tensor_v.emplace_back(funcs::GetRangeCudaTensor<int64_t, Context>(
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
        DenseTensor(value.dtype()).Resize(phi::make_ddim(res_dim_v)));
    ExpandKernel<T, Context>(
        dev_ctx, value, IntArray(res_dim_v), &tmp_value_v[0]);
    ptr_value = &tmp_value_v[0];
  } else {
    ptr_value = &value;
  }

  switch (total_dims) {
    case 1:
      LaunchIndexPutCudaKernel<T, Context, 1>(
          dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
      break;
    case 2:
      LaunchIndexPutCudaKernel<T, Context, 2>(
          dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
      break;
    case 3:
      LaunchIndexPutCudaKernel<T, Context, 3>(
          dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
      break;
    case 4:
      LaunchIndexPutCudaKernel<T, Context, 4>(
          dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
      break;
    case 5:
      LaunchIndexPutCudaKernel<T, Context, 5>(
          dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
      break;
    case 6:
      LaunchIndexPutCudaKernel<T, Context, 6>(
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
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexPutKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16) {}
