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
    const DenseTensor& out_grad,
    const std::vector<const DenseTensor*>& index,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* x_grad,
    DenseTensor* value_grad) {
  int64_t numel = 0;

  int64_t num_indices = 0;
  std::vector<int64_t> shape_tmp;
  std::vector<int64_t> stride_tmp;
  funcs::cal_shape_stride(index_dims, &num_indices, &shape_tmp, &stride_tmp);

  auto sizes = std::array<int64_t, phi::DDim::kMaxRank + 1>{};
  auto strides = std::array<int64_t, phi::DDim::kMaxRank + 1>{};
  for (int64_t i = 0; i < num_indices; i++) {
    sizes[i] = index_dims[i];
    strides[i] = index_strides[i];
  }

  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;
  std::vector<int64_t> value_dims;
  std::vector<int64_t> value_strides;
  if (value_grad) {
    value_dims = common::vectorize<int64_t>(value_grad->dims());
    value_strides = common::vectorize<int64_t>(value_grad->strides());
  }

  funcs::IndexPutStride<3>(input_dims,
                           input_strides,
                           phi::SizeOf(out_grad.dtype()),
                           value_dims,
                           value_strides,
                           4,
                           shape_tmp,
                           stride_tmp,
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
  if (!value_grad) {
    char* out_ptr = reinterpret_cast<char*>(x_grad->data<T>());
    if (index.size() == 1 && index[0]->dtype() == phi::DataType::BOOL) {
      const bool* mask_data = index[0]->data<bool>();
      funcs::index_elementwise_with_tensor_kernel<nt, vt>
          <<<grid, block, 0, stream>>>(N, [=] __device__(int idx) {
            const auto offsets = offset_calc.get(idx);
            char* const out_data = out_ptr + offsets[0] + slice_offset;
            if (mask_data[idx]) {
              *reinterpret_cast<T*>(out_data) = T(0);
            }
          });
    } else {
      auto index_ptrs = funcs::GetIndexDataPtrs<IndexT>(index);
      funcs::index_elementwise_with_tensor_kernel<nt, vt>
          <<<grid, block, 0, stream>>>(N, [=] __device__(int idx) {
            const auto offsets = offset_calc.get(idx);
            char* const out_data = out_ptr + offsets[0] + slice_offset;

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
            T num = T(0);
            *reinterpret_cast<dtype*>(out_data + offset) =
                *reinterpret_cast<dtype*>(&num);
          });
    }
  } else if (!x_grad) {
    auto index_ptrs = funcs::GetIndexDataPtrs<IndexT>(index);
    const char* out_ptr = reinterpret_cast<const char*>(out_grad.data<T>());
    char* value_ptr = reinterpret_cast<char*>(value_grad->data<T>());
    funcs::index_elementwise_with_tensor_kernel<nt, vt>
        <<<grid, block, 0, stream>>>(N, [=] __device__(int idx) {
          const auto offsets = offset_calc.get(idx);
          const char* const out_data = out_ptr + offsets[0] + slice_offset;
          char* const value_data = value_ptr + offsets[1];

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
          *reinterpret_cast<dtype*>(value_data) =
              *reinterpret_cast<const dtype*>(out_data + offset);
        });
  } else {
    auto index_ptrs = funcs::GetIndexDataPtrs<IndexT>(index);
    char* out_ptr = reinterpret_cast<char*>(x_grad->data<T>());
    char* value_ptr = reinterpret_cast<char*>(value_grad->data<T>());
    funcs::index_elementwise_with_tensor_kernel<nt, vt>
        <<<grid, block, 0, stream>>>(N, [=] __device__(int idx) {
          const auto offsets = offset_calc.get(idx);
          char* const out_data = out_ptr + offsets[0] + slice_offset;
          char* const value_data = value_ptr + offsets[1];

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
          T num = T(0);
          *reinterpret_cast<dtype*>(value_data) =
              *reinterpret_cast<dtype*>(out_data + offset);
          *reinterpret_cast<dtype*>(out_data + offset) =
              *reinterpret_cast<dtype*>(&num);
        });
  }
}

template <typename T, typename Context>
void LaunchIndexElementwisePutWithTensorGradCudaKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& indices,
    const DenseTensor& out_grad,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* value_grad,
    DenseTensor* x_grad) {
  if (x_grad && !value_grad) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);

    GPUIndexElementwisePutGradKernel<T, int64_t>(dev_ctx,
                                                 out_grad,
                                                 indices,
                                                 input_dims,
                                                 input_strides,
                                                 index_dims,
                                                 index_strides,
                                                 slice_offset,
                                                 x_grad,
                                                 value_grad);
  } else if (value_grad) {
    if (x_grad) {
      phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    }
    if (value_grad->numel() == 1) {
      DenseTensor tmp_value_grad(value_grad->dtype());
      tmp_value_grad.Resize(common::make_ddim(input_dims));
      dev_ctx.template Alloc<T>(&tmp_value_grad);

      GPUIndexElementwisePutGradKernel<T, int64_t>(dev_ctx,
                                                   out_grad,
                                                   indices,
                                                   input_dims,
                                                   input_strides,
                                                   index_dims,
                                                   index_strides,
                                                   slice_offset,
                                                   x_grad,
                                                   &tmp_value_grad);

      std::vector<int> v_dims(tmp_value_grad.dims().size());
      std::iota(v_dims.begin(), v_dims.end(), 0);
      IntArray v_axis(v_dims);
      SumKernel<T, Context>(dev_ctx,
                            tmp_value_grad,
                            v_axis,
                            value_grad->dtype(),
                            false,
                            value_grad);
    } else if (value_grad->dims() == common::make_ddim(input_dims)) {
      dev_ctx.template Alloc<T>(value_grad);
      GPUIndexElementwisePutGradKernel<T, int64_t>(dev_ctx,
                                                   out_grad,
                                                   indices,
                                                   input_dims,
                                                   input_strides,
                                                   index_dims,
                                                   index_strides,
                                                   slice_offset,
                                                   x_grad,
                                                   value_grad);
    } else {
      DenseTensor tmp_value_grad(value_grad->dtype());
      tmp_value_grad.Resize(common::make_ddim(input_dims));
      dev_ctx.template Alloc<T>(&tmp_value_grad);

      GPUIndexElementwisePutGradKernel<T, int64_t>(dev_ctx,
                                                   out_grad,
                                                   indices,
                                                   input_dims,
                                                   input_strides,
                                                   index_dims,
                                                   index_strides,
                                                   slice_offset,
                                                   x_grad,
                                                   &tmp_value_grad);

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
void LaunchIndexElementwisePutGradCudaKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& indices,
    const DenseTensor& out_grad,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* x_grad) {
  if (x_grad) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);

    GPUIndexElementwisePutGradKernel<T, int64_t>(dev_ctx,
                                                 out_grad,
                                                 indices,
                                                 input_dims,
                                                 input_strides,
                                                 index_dims,
                                                 index_strides,
                                                 slice_offset,
                                                 x_grad,
                                                 nullptr);
  }
}

template <typename T, typename Context>
void IndexElementwisePutGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const std::vector<const DenseTensor*>& indices,
    const DenseTensor& out_grad,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* x_grad) {
  const auto& index_type = indices[0]->dtype();
  PADDLE_ENFORCE_EQ(
      index_type == phi::DataType::INT64 ||
          (index_type == phi::DataType::BOOL && indices.size() == 1),
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
    return;
  }

  LaunchIndexElementwisePutGradCudaKernel<T, Context>(dev_ctx,
                                                      indices,
                                                      out_grad,
                                                      input_dims,
                                                      input_strides,
                                                      index_dims,
                                                      index_strides,
                                                      slice_offset,
                                                      x_grad);
}

template <typename T, typename Context>
void IndexElementwisePutWithTensorGradKernel(
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

  LaunchIndexElementwisePutWithTensorGradCudaKernel<T, Context>(dev_ctx,
                                                                indices,
                                                                out_grad,
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

PD_REGISTER_KERNEL(index_elementwise_put_with_tensor_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexElementwisePutWithTensorGradKernel,
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
