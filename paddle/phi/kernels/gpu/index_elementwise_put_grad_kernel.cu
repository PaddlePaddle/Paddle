// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_elementwise_put_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename IndexT = int>
void GPUIndexElementwisePutGradKernel(
    const phi::GPUContext& dev_ctx,
    const std::vector<const DenseTensor*>& index,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* output) {
  int64_t numel = 0;

  auto num_indices = index_dims.size();

  auto sizes = std::array<int64_t, 25>{};
  auto strides = std::array<int64_t, 25>{};
  for (unsigned i = 0; i < num_indices; i++) {
    sizes[i] = index_dims[i];
    strides[i] = index_strides[i];
  }
  auto index_ptrs = funcs::GetIndexDataPtrs<IndexT>(index);

  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;

  funcs::IndexPutStride<3>(input_dims,
                           input_strides,
                           phi::SizeOf(output->dtype()),
                           std::vector<int64_t>(),
                           std::vector<int64_t>(),
                           4,
                           common::vectorize<int64_t>(index[0]->dims()),
                           common::vectorize<int64_t>(index[0]->strides()),
                           phi::SizeOf(index[0]->dtype()),
                           &desired_shape,
                           &strides_array,
                           &numel,
                           strides_vec);
  auto offset_calc =
      funcs::make_offset_calculator_put<3>(desired_shape, strides_array);
  const int64_t N = numel;
  PADDLE_ENFORCE(N >= 0 && N <= std::numeric_limits<int32_t>::max(),
                 "N >= 0 && N <= std::numeric_limits<int32_t>::max()");
  constexpr int nt = 128;
  constexpr int vt = 4;
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = dev_ctx.stream();

  using dtype = funcs::OpaqueType<sizeof(T)>;

  char* out_ptr = reinterpret_cast<char*>(output->data<T>());
  funcs::index_elementwise_kernel<nt, vt>
      <<<grid, block, 0, stream>>>(N, [=] __device__(int idx) {
        const auto offsets = offset_calc.get(idx);
        char* const out_data = out_ptr + offsets[0] + slice_offset;

        int64_t offset = 0;
#pragma unroll
        for (int i = 0; i < num_indices; i++) {
          int64_t index =
              *reinterpret_cast<int64_t*>(index_ptrs[i] + offsets[2]);
          if (index < 0) {
            index += sizes[i];
          }
          offset += index * strides[i];
        }
        T num = T(0);

        *reinterpret_cast<dtype*>(out_data + offset) =
            *reinterpret_cast<dtype*>(&num);
      });
}

template <typename T>
__global__ void SetZeroElementwiseCudaKernel(
    int64_t** indices,
    Array<int64_t, DDim::kMaxRank> stride,
    Array<int64_t, DDim::kMaxRank> shape,
    const int rank,
    const int64_t numel,
    T* out) {
  int64_t idx =
      static_cast<int64_t>(threadIdx.x) +
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blockIdx.x);
  if (idx >= numel) {
    return;
  }

  int64_t cur_ix = 0;
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

  *(out + offset) = 0;
}

template <typename T>
__global__ void IndexElementwisePutGradCudaKernel(
    const T* out_grad,
    int64_t** indices,
    Array<int64_t, DDim::kMaxRank> stride,
    Array<int64_t, DDim::kMaxRank> shape,
    const int rank,
    const int64_t numel,
    T* value_grad) {
  int64_t idx =
      static_cast<int64_t>(threadIdx.x) +
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blockIdx.x);
  if (idx >= numel) {
    return;
  }

  int64_t cur_ix = 0;
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

  *(value_grad + idx) = *(out_grad + offset);
}

template <typename T, typename Context>
void LaunchIndexElementwisePutGradCudaKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& x_indices,
    const std::vector<const DenseTensor*>& indices,
    const DenseTensor& out_grad,
    const int rank,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* value_grad,
    DenseTensor* x_grad) {
  phi::Allocator::AllocationPtr indices_holder_1, indices_holder_2;
  const auto& index_type = indices[0]->dtype();
  if (x_grad) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);

    GPUIndexElementwisePutGradKernel<T, int64_t>(dev_ctx,
                                                 x_indices,
                                                 input_dims,
                                                 input_strides,
                                                 index_dims,
                                                 index_strides,
                                                 slice_offset,
                                                 x_grad);
  }

  auto out_grad_dims = out_grad.dims();
  auto out_grad_stride = common::stride(out_grad_dims);

  Array<int64_t, DDim::kMaxRank> stride_array;
  Array<int64_t, DDim::kMaxRank> shape_array;
  for (int i = 0; i < rank; ++i) {
    stride_array[i] = out_grad_stride[i];
    shape_array[i] = out_grad_dims[i];
  }

  const int64_t numel = indices[0]->numel();
  auto pd_indices = funcs::GetDevicePointerArray<int64_t, Context>(
      dev_ctx, indices, &indices_holder_2);
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);

  if (value_grad) {
    if (value_grad->numel() == 1) {
      DenseTensor tmp_value_grad(value_grad->dtype());
      tmp_value_grad.Resize(indices[0]->dims());

      T* tmp_value_grad_data = dev_ctx.template Alloc<T>(&tmp_value_grad);
      auto out_grad_data = out_grad.data<T>();

      IndexElementwisePutGradCudaKernel<T>
          <<<config.block_per_grid,
             config.thread_per_block,
             0,
             dev_ctx.stream()>>>(out_grad_data,
                                 pd_indices,
                                 stride_array,
                                 shape_array,
                                 rank,
                                 numel,
                                 tmp_value_grad_data);

      std::vector<int> v_dims(tmp_value_grad.dims().size());
      std::iota(v_dims.begin(), v_dims.end(), 0);
      IntArray v_axis(v_dims);
      SumKernel<T, Context>(dev_ctx,
                            tmp_value_grad,
                            v_axis,
                            value_grad->dtype(),
                            false,
                            value_grad);
    } else if (value_grad->numel() == indices[0]->numel()) {
      T* value_grad_data = dev_ctx.template Alloc<T>(value_grad);
      auto out_grad_data = out_grad.data<T>();

      IndexElementwisePutGradCudaKernel<T>
          <<<config.block_per_grid,
             config.thread_per_block,
             0,
             dev_ctx.stream()>>>(out_grad_data,
                                 pd_indices,
                                 stride_array,
                                 shape_array,
                                 rank,
                                 numel,
                                 value_grad_data);
    } else {
      DenseTensor tmp_value_grad(value_grad->dtype());
      tmp_value_grad.Resize(common::make_ddim(input_dims));

      T* tmp_value_grad_data = dev_ctx.template Alloc<T>(&tmp_value_grad);
      auto out_grad_data = out_grad.data<T>();

      IndexElementwisePutGradCudaKernel<T>
          <<<config.block_per_grid,
             config.thread_per_block,
             0,
             dev_ctx.stream()>>>(out_grad_data,
                                 pd_indices,
                                 stride_array,
                                 shape_array,
                                 rank,
                                 numel,
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
      SumKernel<T, Context>(dev_ctx,
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
void IndexElementwisePutGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const std::vector<const DenseTensor*>& indices,
    const DenseTensor& value,
    const DenseTensor& out_grad,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* x_grad,
    DenseTensor* value_grad) {
  const auto& index_type = indices[0]->dtype();
  PADDLE_ENFORCE_EQ(index_type == phi::DataType::INT64,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s].",
                        index_type,
                        phi::DataType::INT64));

  std::vector<DenseTensor> tmp_args;
  if (indices.empty()) {
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

  auto bd_dim = funcs::BroadCastTensorsDims(indices);

  std::vector<int64_t> res_dim_v(common::vectorize(bd_dim));
  std::vector<const phi::DenseTensor*> res_indices_v(x.dims().size(), nullptr);
  std::vector<DenseTensor> tmp_res_indices_v;
  std::vector<DenseTensor> range_tensor_v;

  for (int i = indices.size(); i < x.dims().size(); ++i) {
    range_tensor_v.emplace_back(funcs::GetRangeCudaTensor<int64_t, Context>(
        dev_ctx, x.dims()[i], phi::DataType::INT64));
  }

  funcs::DealWithIndices<T, Context>(dev_ctx,
                                     x,
                                     indices,
                                     &res_indices_v,
                                     &tmp_res_indices_v,
                                     range_tensor_v,
                                     bd_dim,
                                     &res_dim_v);

  const int rank = x.dims().size();
  LaunchIndexElementwisePutGradCudaKernel<T, Context>(dev_ctx,
                                                      indices,
                                                      res_indices_v,
                                                      out_grad,
                                                      rank,
                                                      input_dims,
                                                      input_strides,
                                                      index_dims,
                                                      index_strides,
                                                      slice_offset,
                                                      value_grad,
                                                      x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_elementwise_put_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexElementwisePutGradKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
