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

#include "paddle/phi/kernels/index_kernel.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/array.h"
#include "paddle/phi/kernels/expand_kernel.h"

namespace phi {
template <typename T>
__global__ void elementwise_set_cuda_kernel(
    const int64_t N, T* x, T* y, int64_t* offsets, int64_t isSingleValTensor) {
  int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  int64_t cur_ix = 0;

  if (idx >= N) {
    return;
  }
  *(x + *(offsets + idx)) = *(y + (idx & isSingleValTensor));
}

template <typename T1, typename T2, size_t Rank>
__global__ void index_put_cuda_kernel(const int64_t N,
                                      T1* x,
                                      const T1* vals,
                                      T2** indices,
                                      phi::Array<int64_t, Rank> stride,
                                      phi::Array<int64_t, Rank> shape,
                                      int64_t isSingleValTensor,
                                      T1* out) {
  int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  int64_t cur_ix = 0;

  if (idx >= N) {
    return;
  }
  int64_t offset = 0;
  for (int i = 0; i < Rank; ++i) {
    cur_ix = (int64_t(*(indices[i] + idx)));
    if (cur_ix < 0) {
      cur_ix += shape[i];
    }
    offset += stride[i] * cur_ix;
  }

  *(x + offset) = *(vals + (idx & isSingleValTensor));
}

template <typename T, typename Context>
T** GetDevicePointerArray(const Context& ctx,
                          const std::vector<const DenseTensor*>& indices_v) {
  std::vector<const T*> h_indices_v(indices_v.size());
  for (int i = 0; i < indices_v.size(); ++i) {
    h_indices_v[i] = indices_v[i]->data<T>();
  }
  auto d_indices_data = paddle::memory::Alloc(
      ctx.GetPlace(),
      h_indices_v.size() * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  paddle::memory::Copy(ctx.GetPlace(),
                       d_indices_data->ptr(),
                       phi::CPUPlace(),
                       reinterpret_cast<void*>(h_indices_v.data()),
                       h_indices_v.size() * sizeof(T*),
                       ctx.stream());
  return reinterpret_cast<T**>(d_indices_data->ptr());
}

template <typename T, typename Context, size_t Rank>
void LaunchIndexPutCudaKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const std::vector<const DenseTensor*>& indices_v,
                              const DenseTensor& value,
                              DenseTensor* out) {
  auto* x_data = const_cast<T*>(x.data<T>());
  auto* val_data = value.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);

  auto x_dims = x.dims();
  // 如果是bool索引，这里需要先滤过False，可能需要一个新的kernel
  const int64_t numel = indices_v[0]->numel();
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  auto x_stride = phi::stride(x_dims);

  phi::Array<int64_t, Rank> stride_a;
  phi::Array<int64_t, Rank> shape_a;

  for (size_t idx = 0; idx < Rank; ++idx) {
    stride_a[idx] = x_stride[idx];
    shape_a[idx] = x_dims[idx];
  }

  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);

  int64_t isSingleValTensor = (value.numel() == 1) ? 0 : INT64_MAX;

  if (indices_v[0]->dtype() == paddle::experimental::DataType::INT32) {
    auto pd_indices = GetDevicePointerArray<int, Context>(dev_ctx, indices_v);
    index_put_cuda_kernel<T, int, Rank><<<config.block_per_grid,
                                          config.thread_per_block,
                                          0,
                                          dev_ctx.stream()>>>(numel,
                                                              x_data,
                                                              val_data,
                                                              pd_indices,
                                                              stride_a,
                                                              shape_a,
                                                              isSingleValTensor,
                                                              out_data);
  } else if (indices_v[0]->dtype() == paddle::experimental::DataType::INT64) {
    auto pd_indices =
        GetDevicePointerArray<int64_t, Context>(dev_ctx, indices_v);
    index_put_cuda_kernel<T, int64_t, Rank>
        <<<config.block_per_grid,
           config.thread_per_block,
           0,
           dev_ctx.stream()>>>(numel,
                               x_data,
                               val_data,
                               pd_indices,
                               stride_a,
                               shape_a,
                               isSingleValTensor,
                               out_data);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "dtype of index tensor should be int32 or int64"));
  }
}

inline void GetBroadcastDimsArrays(const DDim& x_dims,
                                   const DDim& y_dims,
                                   int* x_dims_array,
                                   int* y_dims_array,
                                   int* out_dims_array,
                                   const int max_dim,
                                   const int axis) {
  PADDLE_ENFORCE_GE(
      axis,
      0,
      phi::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LE(
      axis,
      max_dim,
      phi::errors::InvalidArgument(
          "Axis should be less than or equal to %d, but received axis is %d.",
          max_dim,
          axis));
  if (x_dims.size() > y_dims.size()) {
    std::fill(y_dims_array, y_dims_array + axis, 1);
    if (axis + y_dims.size() < max_dim) {
      std::fill(y_dims_array + axis + y_dims.size(), y_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array + axis);
  } else {
    std::fill(x_dims_array, x_dims_array + axis, 1);
    if (axis + x_dims.size() < max_dim) {
      std::fill(x_dims_array + axis + x_dims.size(), x_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array + axis);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array);
  }

  for (int i = 0; i < max_dim; i++) {
    PADDLE_ENFORCE_EQ(
        x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 ||
            y_dims_array[i] <= 1,
        true,
        phi::errors::InvalidArgument(
            "Broadcast dimension mismatch. Operands could "
            "not be broadcast together with the shape of X = [%s] and "
            "the shape of Y = [%s]. Received [%d] in X is not equal to "
            "[%d] in Y at i:%d.",
            x_dims,
            y_dims,
            x_dims_array[i],
            y_dims_array[i],
            i));
    if ((x_dims_array[i] > 1 || y_dims_array[i] > 1) ||
        (x_dims_array[i] == 1 && y_dims_array[i] == 1)) {
      out_dims_array[i] = std::max(x_dims_array[i], y_dims_array[i]);
    } else {
      out_dims_array[i] = -1;
    }
  }
}

DDim BroadcastTwoDims(const DDim& x_dims, const DDim& y_dims, int axis) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);
  return phi::make_ddim(out_dims_array);
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
  std::vector<const DenseTensor*> indices_v_offset(indices_v.size());

  auto pre_dim = indices_v[0]->dims();
  auto tmp_dim = phi::make_ddim({0});
  auto indice_dtype = indices_v[0]->dtype();
  bool need_broadcast = false;
  for (int i = 1; i < indices_v.size(); ++i) {
    tmp_dim = indices_v[i]->dims();
    VLOG(2) << "tmp_dim is " << tmp_dim << std::endl;
    if (pre_dim != tmp_dim) {
      pre_dim = BroadcastTwoDims(pre_dim, tmp_dim, -1);
      need_broadcast = true;
    }
  }

  std::vector<DenseTensor> indices_v_tmp(
      indices_v.size(), DenseTensor(indice_dtype).Resize(pre_dim));

  if (need_broadcast) {
    for (int i = 0; i < indices_v.size(); ++i) {
      if (pre_dim == indices_v[i]->dims()) {
        indices_v_offset[i] = indices_v[i];
        continue;
      }
      if (indices_v[0]->dtype() == paddle::experimental::DataType::INT32) {
        ExpandKernel<int, Context>(dev_ctx,
                                   *indices_v[i],
                                   IntArray(phi::vectorize<int64_t>(pre_dim)),
                                   &indices_v_tmp[i]);
      } else if (indices_v[0]->dtype() ==
                 paddle::experimental::DataType::INT64) {
        ExpandKernel<int64_t, Context>(
            dev_ctx,
            *indices_v[i],
            IntArray(phi::vectorize<int64_t>(pre_dim)),
            &indices_v_tmp[i]);
      }
      indices_v_offset[i] = &indices_v_tmp[i];
    }
  } else {
    indices_v_offset = std::move(indices_v);
  }

  switch (total_dims) {
    case 1:
      LaunchIndexPutCudaKernel<T, Context, 1>(
          dev_ctx, x, indices_v_offset, value, out);
      break;
    case 2:
      LaunchIndexPutCudaKernel<T, Context, 2>(
          dev_ctx, x, indices_v_offset, value, out);
      break;
    case 3:
      LaunchIndexPutCudaKernel<T, Context, 3>(
          dev_ctx, x, indices_v_offset, value, out);
      break;
    case 4:
      LaunchIndexPutCudaKernel<T, Context, 4>(
          dev_ctx, x, indices_v_offset, value, out);
      break;
    case 5:
      LaunchIndexPutCudaKernel<T, Context, 5>(
          dev_ctx, x, indices_v_offset, value, out);
      break;
    case 6:
      LaunchIndexPutCudaKernel<T, Context, 6>(
          dev_ctx, x, indices_v_offset, value, out);
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
