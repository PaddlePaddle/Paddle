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
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "paddle/phi/kernels/index_elementwise_put_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

inline bool CheckIsDimsMatchBool(const DDim& first, const DDim& second) {
  int ignore_axis1 = 0, ignore_axis2 = 0;
  for (; ignore_axis1 < first.size(); ++ignore_axis1) {
    if (first[ignore_axis1] != 1) {
      break;
    }
  }
  for (; ignore_axis2 < second.size(); ++ignore_axis2) {
    if (second[ignore_axis2] != 1) {
      break;
    }
  }

  if (second.size() == ignore_axis2) {
    // second tensor has only one value
    return true;
  }

  if (first.size() - ignore_axis1 >= second.size() - ignore_axis2) {
    auto idx1 = first.size() - 1;
    auto idx2 = second.size() - 1;
    bool is_match = true;
    for (; idx2 >= ignore_axis2; idx2--) {
      if (first[idx1--] != second[idx2] && second[idx2] != 1) {
        is_match = false;
        break;
      }
    }
    if (is_match) {
      return true;
    }
  }

  return false;
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
    phi::CudaAtomicAdd(out + offset, *(vals + (idx & is_single_val_tensor)));
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
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
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

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
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
void IndexPutKernel_V1(const Context& dev_ctx,
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
        DenseTensor(value.dtype()).Resize(common::make_ddim(res_dim_v)));
    ExpandKernel<T, Context>(
        dev_ctx, value, IntArray(res_dim_v), &tmp_value_v[0]);
    ptr_value = &tmp_value_v[0];
  } else {
    ptr_value = &value;
  }

  LaunchIndexPutCudaKernel<T, Context>(
      dev_ctx, x, res_indices_v, *ptr_value, accumulate, out);
}

template <typename T, typename Context>
void IndexPutKernel_V2(const Context& dev_ctx,
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

  funcs::AdvancedIndex ad = funcs::make_info<T, Context>(dev_ctx, x, indices);
  if (!CheckIsDimsMatchBool(common::make_ddim(ad.src_sizes), value.dims())) {
    for (size_t i = 0; i < indices.size(); i++) {
      PADDLE_ENFORCE_EQ(indices[i]->meta().is_contiguous(),
                        true,
                        common::errors::InvalidArgument(
                            "Indices in Index_put must be contiguous."));
    }
    PADDLE_ENFORCE_EQ(
        x.meta().is_contiguous(),
        true,
        common::errors::InvalidArgument("X in Index_put must be contiguous."));
    PADDLE_ENFORCE_EQ(value.meta().is_contiguous(),
                      true,
                      common::errors::InvalidArgument(
                          "Value in Index_put must be contiguous."));
    IndexPutKernel_V1<T, Context>(dev_ctx, x, indices, value, accumulate, out);
    return;
  }

  int64_t numel = 0;
  int64_t num_indices = ad.indexed_sizes.size();

  auto sizes = std::array<int64_t, phi::DDim::kMaxRank + 1>{};
  auto strides = std::array<int64_t, phi::DDim::kMaxRank + 1>{};
  for (int64_t i = 0; i < num_indices; i++) {
    sizes[i] = ad.indexed_sizes[i];
    strides[i] = ad.indexed_strides[i];
  }

  auto index_ptrs = funcs::GetIndexDataPtrs_v2<int64_t>(ad.indices);

  std::vector<int64_t*> strides_array;
  std::vector<int64_t> desired_shape;
  std::vector<std::vector<int64_t>> strides_vec;

  int64_t ntensor = 2 + num_indices;
  strides_array.resize(ntensor);
  strides_vec.resize(ntensor);

  funcs::IndexPutStrideV2(ntensor,
                          ad.src_sizes,
                          ad.src_strides,
                          phi::SizeOf(x.dtype()),
                          common::vectorize<int64_t>(value.dims()),
                          common::vectorize<int64_t>(value.strides()),
                          phi::SizeOf(value.dtype()),
                          ad.indices,
                          &desired_shape,
                          &strides_array,
                          &numel,
                          strides_vec);

  auto offset_calc =
      funcs::make_offset_calculator_put_v2<3>(desired_shape, strides_array);

  const int64_t N = numel;
  PADDLE_ENFORCE(N >= 0 && N <= std::numeric_limits<int32_t>::max(),
                 "N >= 0 && N <= std::numeric_limits<int32_t>::max()");
  constexpr int nt = 128;
  constexpr int vt = 4;
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = dev_ctx.stream();

  auto* val_data = value.data<T>();

  bool is_initialized = out->initialized();
  T* out_data = dev_ctx.template Alloc<T>(out);
  if (!is_initialized) {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  }

  const char* in_ptr = reinterpret_cast<const char*>(val_data);
  char* out_ptr = reinterpret_cast<char*>(out_data);
  funcs::index_put_kernel<nt, vt, T><<<grid, block, 0, stream>>>(
      N, accumulate, [=] __device__(int idx, bool accumulate) {
        const auto offsets = offset_calc.get(idx);
        char* const out_data = out_ptr + offsets[0];
        const char* const in_data = in_ptr + offsets[1];

        int64_t offset = 0;
#pragma unroll
        for (int64_t i = 0; i < num_indices; i++) {
          int64_t index =
              *reinterpret_cast<int64_t*>(index_ptrs[i] + offsets[2]);
          if (index < 0) {
            index += sizes[i];
          }
          offset += index * strides[i];
        }
        if (accumulate) {
          *reinterpret_cast<T*>(out_data + offset) +=
              *reinterpret_cast<const T*>(in_data);
        } else {
          *reinterpret_cast<T*>(out_data + offset) =
              *reinterpret_cast<const T*>(in_data);
        }
      });
}
}  // namespace phi

PD_REGISTER_KERNEL(index_put,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexPutKernel_V1,
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

PD_REGISTER_KERNEL(index_put,
                   GPU,
                   STRIDED,
                   phi::IndexPutKernel_V2,
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
