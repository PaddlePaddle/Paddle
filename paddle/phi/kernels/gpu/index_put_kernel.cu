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
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/kernels/funcs/index_put_with_sort.cu.h"
#endif

COMMON_DECLARE_bool(cudnn_deterministic);

namespace phi {

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
  Allocator::AllocationPtr holder;
  auto pd_indices =
      funcs::GetDevicePointerArray<int64_t, Context>(dev_ctx, indices, &holder);

  auto config = backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
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

  std::vector<int64_t> res_dim_v(vectorize(bd_dim));
  std::vector<const DenseTensor*> res_indices_v(x.dims().size(), nullptr);
  std::vector<DenseTensor> tmp_res_indices_v;
  std::vector<DenseTensor> tmp_value_v;
  std::vector<DenseTensor> range_tensor_v;
  const DenseTensor* ptr_value = nullptr;

  for (int i = int_indices_v.size(); i < x.dims().size(); ++i) {
    range_tensor_v.emplace_back(funcs::GetRangeCudaTensor<int64_t, Context>(
        dev_ctx, x.dims()[i], DataType::INT64));
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
    tmp_value_v.emplace_back(DenseTensor(value.dtype()).Resize(res_dim_v));
    ExpandKernel<T, Context>(
        dev_ctx, value, IntArray(res_dim_v), &tmp_value_v[0]);
    ptr_value = &tmp_value_v[0];
  } else {
    ptr_value = &value;
  }

#ifdef PADDLE_WITH_CUDA
  // The default accumulate path (LaunchIndexPutCudaKernel) scatters with
  // CudaAtomicAdd, whose summation order is nondeterministic across runs, so
  // duplicated indices give run-to-run bitwise drift. When
  // FLAGS_cudnn_deterministic is set, route to the sort-based accumulate, which
  // groups colliding indices and adds them in a fixed order (same result, just
  // deterministic). Scoped to a single index into a 1-D destination: that is
  // exactly the flat scatter the strided-view backward emits
  // (StridedTensorAccumulate builds a 1-D storage and one linear index), and it
  // matches index_elementwise_get_grad's single-index use of the same helper.
  // Other shapes keep the atomic path.
  if (accumulate && FLAGS_cudnn_deterministic && int_indices_v.size() == 1 &&
      x.dims().size() == 1) {
    // IndexPutWithSortKernel accumulates into `out` in place and does not copy
    // `x` in, so reproduce LaunchIndexPutCudaKernel's prologue: an
    // uninitialized destination starts as a copy of x (plain index_put
    // semantics); an already initialized one (e.g. the pre-zeroed strided
    // gradient buffer) is left as is.
    bool is_initialized = out->initialized();
    dev_ctx.template Alloc<T>(out);
    if (!is_initialized) {
      Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    // input_dims/strides and index_dims/strides are unused by the helper (it
    // derives geometry from `out` and `indices`); pass the natural values.
    const auto x_dims_v = vectorize<int64_t>(x.dims());
    const auto x_strides_v = vectorize<int64_t>(common::stride(x.dims()));
    const auto idx_dims_v = vectorize<int64_t>(res_indices_v[0]->dims());
    const auto idx_strides_v =
        vectorize<int64_t>(common::stride(res_indices_v[0]->dims()));
    funcs::IndexPutWithSortKernel<T, int64_t>(dev_ctx,
                                              x,
                                              *ptr_value,
                                              res_indices_v,
                                              x_dims_v,
                                              x_strides_v,
                                              idx_dims_v,
                                              idx_strides_v,
                                              /*slice_offset=*/0,
                                              /*accumulate=*/true,
                                              out);
    return;
  }
#endif

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
