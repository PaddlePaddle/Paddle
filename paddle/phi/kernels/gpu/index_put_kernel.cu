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
#include <cinttypes>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"

namespace phi {

__global__ void ValidateIndexPutCudaKernel(
    int64_t** indices,
    Array<int64_t, DDim::kMaxRank> shape,
    const int rank,
    const int64_t numel,
    int* has_error,
    int64_t* invalid_index,
    int* invalid_axis) {
  int64_t idx =
      static_cast<int64_t>(threadIdx.x) +
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blockIdx.x);

  if (idx >= numel || *has_error) {
    return;
  }

#pragma unroll
  for (int i = 0; i < DDim::kMaxRank; ++i) {
    if (i >= rank) {
      break;
    }
    int64_t cur_ix = static_cast<int64_t>(*(indices[i] + idx));
    if (cur_ix < -shape[i] || cur_ix >= shape[i]) {
      if (atomicCAS(has_error, 0, 1) == 0) {
        *invalid_index = cur_ix;
        *invalid_axis = i;
      }
      return;
    }
  }
}

template <typename T>
__global__ void IndexPutCudaKernel(const T* x,
                                   const T* vals,
                                   int64_t** indices,
                                   Array<int64_t, DDim::kMaxRank> stride,
                                   Array<int64_t, DDim::kMaxRank> shape,
                                   const int rank,
                                   const int64_t numel,
                                   const int64_t is_single_val_tensor,
                                   const bool accumulate,
                                   T* out) {
  int64_t idx =
      static_cast<int64_t>(threadIdx.x) +
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blockIdx.x);
  int64_t cur_ix = 0;

  if (idx >= numel) {
    return;
  }
  int64_t offset = 0;
#pragma unroll
  for (int i = 0; i < DDim::kMaxRank; ++i) {
    if (i >= rank) {
      break;
    }
    cur_ix = (static_cast<int64_t>(*(indices[i] + idx)));
    if (cur_ix < 0) {
      cur_ix += shape[i];
    }
    offset += stride[i] * cur_ix;
  }

  if (accumulate) {
    CudaAtomicAdd(out + offset, *(vals + (idx & is_single_val_tensor)));
  } else {
    *(out + offset) = *(vals + (idx & is_single_val_tensor));
  }
}

template <typename T, typename Context>
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
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  }

  auto x_dims = x.dims();
  const int rank = x_dims.size();
  auto x_stride = common::stride(x_dims);

  Array<int64_t, DDim::kMaxRank> stride_array;
  Array<int64_t, DDim::kMaxRank> shape_array;
  for (int i = 0; i < rank; ++i) {
    stride_array[i] = x_stride[i];
    shape_array[i] = x_dims[i];
  }

  int64_t is_single_val_tensor = (value.numel() == 1) ? 0 : INT64_MAX;
  const int64_t numel = indices[0]->numel();
  phi::Allocator::AllocationPtr holder;
  auto pd_indices =
      funcs::GetDevicePointerArray<int64_t, Context>(dev_ctx, indices, &holder);

  int host_has_error = 0;
  int64_t host_invalid_index = 0;
  int host_invalid_axis = 0;
  auto error_flag = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(), sizeof(int), dev_ctx.stream());
  auto invalid_index = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(), sizeof(int64_t), dev_ctx.stream());
  auto invalid_axis = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(), sizeof(int), dev_ctx.stream());
  phi::memory_utils::Copy(dev_ctx.GetPlace(),
                          error_flag->ptr(),
                          phi::CPUPlace(),
                          &host_has_error,
                          sizeof(int),
                          dev_ctx.stream());
  phi::memory_utils::Copy(dev_ctx.GetPlace(),
                          invalid_index->ptr(),
                          phi::CPUPlace(),
                          &host_invalid_index,
                          sizeof(int64_t),
                          dev_ctx.stream());
  phi::memory_utils::Copy(dev_ctx.GetPlace(),
                          invalid_axis->ptr(),
                          phi::CPUPlace(),
                          &host_invalid_axis,
                          sizeof(int),
                          dev_ctx.stream());

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  ValidateIndexPutCudaKernel<<<config.block_per_grid,
                               config.thread_per_block,
                               0,
                               dev_ctx.stream()>>>(pd_indices,
                                                   shape_array,
                                                   rank,
                                                   numel,
                                                   reinterpret_cast<int*>(
                                                       error_flag->ptr()),
                                                   reinterpret_cast<int64_t*>(
                                                       invalid_index->ptr()),
                                                   reinterpret_cast<int*>(
                                                       invalid_axis->ptr()));
  phi::memory_utils::Copy(phi::CPUPlace(),
                          &host_has_error,
                          dev_ctx.GetPlace(),
                          error_flag->ptr(),
                          sizeof(int),
                          dev_ctx.stream());
  dev_ctx.Wait();
  if (host_has_error != 0) {
    phi::memory_utils::Copy(phi::CPUPlace(),
                            &host_invalid_index,
                            dev_ctx.GetPlace(),
                            invalid_index->ptr(),
                            sizeof(int64_t),
                            dev_ctx.stream());
    phi::memory_utils::Copy(phi::CPUPlace(),
                            &host_invalid_axis,
                            dev_ctx.GetPlace(),
                            invalid_axis->ptr(),
                            sizeof(int),
                            dev_ctx.stream());
    dev_ctx.Wait();
    PADDLE_THROW(common::errors::OutOfRange(
        "The index value %" PRId64
        " is out of bounds for axis %d with size %" PRId64
        " in index_put. Expected the index to satisfy -%" PRId64
        " <= index < %" PRId64 " before negative index normalization.",
        host_invalid_index,
        host_invalid_axis,
        x_dims[host_invalid_axis],
        x_dims[host_invalid_axis],
        x_dims[host_invalid_axis]));
  }

  IndexPutCudaKernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          x_data,
          val_data,
          pd_indices,
          stride_array,
          shape_array,
          rank,
          numel,
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
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
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
  std::vector<DenseTensor> tmp_args;
  std::vector<const DenseTensor*> int_indices_v =
      funcs::DealWithBoolIndices<T, Context>(dev_ctx, indices, &tmp_args);
  if (int_indices_v.empty()) {
    if (!out->initialized()) {
      Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }
  auto bd_dim = funcs::BroadCastTensorsDims(int_indices_v);

  std::vector<int64_t> res_dim_v(common::vectorize(bd_dim));
  std::vector<const DenseTensor*> res_indices_v(x.dims().size(), nullptr);
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
        DenseTensor(value.dtype()).Resize(make_ddim(res_dim_v)));
    ExpandKernel<T, Context>(
        dev_ctx, value, IntArray(res_dim_v), &tmp_value_v[0]);
    ptr_value = &tmp_value_v[0];
  } else {
    ptr_value = &value;
  }

  LaunchIndexPutCudaKernel<T, Context>(
      dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
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
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
