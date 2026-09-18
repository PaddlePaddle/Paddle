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

#include "paddle/phi/kernels/index_elementwise_get_grad_kernel.h"

#include "paddle/common/enforce.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/gpu/cuda/cuda_device_function.h"
#endif
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/index_put_with_sort.cu.h"
#include "paddle/phi/kernels/funcs/radix_sort.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
template <typename T, typename IndexT, int nt, int vt, typename offset_calc_t>
__global__ void IndexEleGetGradAccKernel(
    int64_t N,
    const char* in_ptr,
    char* out_ptr,
    const std::array<char*, DDim::kMaxRank> index_ptrs,
    const std::array<int64_t, DDim::kMaxRank + 1> sizes,
    const std::array<int64_t, DDim::kMaxRank + 1> strides,
    int num_indices,
    offset_calc_t offset_calc) {
  const int tid = threadIdx.x;
  const int nv = nt * vt;
  int64_t idx = nv * static_cast<int64_t>(blockIdx.x) + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      const auto offsets = offset_calc.get(idx);
      char* const out_data = out_ptr + offsets[0];
      const char* const in_data = in_ptr + offsets[1];

      int64_t offset = 0;
#pragma unroll
      for (int i = 0; i < num_indices; i++) {
        int64_t index = *reinterpret_cast<int64_t*>(index_ptrs[i] + offsets[2]);
        if (index < 0) index += sizes[i];
        offset += index * strides[i];
      }

      CudaAtomicAdd(reinterpret_cast<T*>(out_data + offset),
                    *reinterpret_cast<const T*>(in_data));
      idx += nt;
    }
  }
}

template <typename T, typename OffsetT = uint32_t>
void GPUIndexElementwiseGetGrad(const GPUContext& dev_ctx,
                                const DenseTensor& input,
                                const DenseTensor& value,
                                const std::vector<const DenseTensor*>& index,
                                const std::vector<int64_t>& input_dims,
                                const std::vector<int64_t>& input_strides,
                                const std::vector<int64_t>& index_dims,
                                const std::vector<int64_t>& index_strides,
                                const int64_t slice_offset,
                                const bool accumulate,
                                DenseTensor* output) {
  int64_t numel = 0;

  int64_t num_indices = 0;
  std::vector<int64_t> shape_tmp;
  std::vector<int64_t> stride_tmp;
  funcs::cal_shape_stride(index_dims, &num_indices, &shape_tmp, &stride_tmp);

  auto sizes = std::array<int64_t, DDim::kMaxRank + 1>{};
  auto strides = std::array<int64_t, DDim::kMaxRank + 1>{};
  for (int64_t i = 0; i < num_indices; i++) {
    sizes[i] = index_dims[i];
    strides[i] = index_strides[i];
  }
  auto index_ptrs = funcs::GetIndexDataPtrs<int64_t>(index);

  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;

  funcs::IndexPutStride<3>(input_dims,
                           input_strides,
                           phi::SizeOf(input.dtype()),
                           vectorize<int64_t>(value.dims()),
                           vectorize<int64_t>(value.strides()),
                           phi::SizeOf(value.dtype()),
                           shape_tmp,
                           stride_tmp,
                           phi::SizeOf(index[0]->dtype()),
                           &desired_shape,
                           &strides_array,
                           &numel,
                           strides_vec);
  auto offset_calc = funcs::make_offset_calculator_put<3, false, OffsetT>(
      desired_shape, strides_array);

  auto max_grid_size = phi::backends::gpu::GetGpuMaxGridDimSize(
      dev_ctx.GetPlace().GetDeviceId());

  const int64_t N = numel;
  constexpr int nt = 128;
  constexpr int vt = 4;
  const int64_t grid_x =
      (N + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt);
  PADDLE_ENFORCE_LE(
      grid_x,
      max_grid_size[0],
      common::errors::InvalidArgument("grid_x (%d) is too large to be "
                                      "launched in a CUDA grid.",
                                      grid_x));
  const dim3 block(nt);
  const dim3 grid(grid_x);
  auto stream = dev_ctx.stream();

  using dtype = funcs::OpaqueType<sizeof(T)>;

  const char* in_ptr = reinterpret_cast<const char*>(value.data<T>());
  char* out_ptr = reinterpret_cast<char*>(output->data<T>()) + slice_offset;

  if (accumulate) {
    IndexEleGetGradAccKernel<T, int64_t, nt, vt>
        <<<grid, block, 0, stream>>>(N,
                                     in_ptr,
                                     out_ptr,
                                     index_ptrs,
                                     sizes,
                                     strides,
                                     num_indices,
                                     offset_calc);
  } else {
    funcs::index_elementwise_with_tensor_kernel<nt, vt>
        <<<grid, block, 0, stream>>>(N, [=] __device__(int64_t idx) {
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
          *reinterpret_cast<dtype*>(out_data + offset) =
              *reinterpret_cast<const dtype*>(in_data);
        });
  }
}

template <typename T, typename Context>
void IndexElementwiseGetGradKernel(const Context& dev_ctx,
                                   const DenseTensor& x,
                                   const std::vector<const DenseTensor*>& index,
                                   const DenseTensor& out_grad,
                                   const std::vector<int64_t>& input_dims,
                                   const std::vector<int64_t>& input_strides,
                                   const std::vector<int64_t>& index_dims,
                                   const std::vector<int64_t>& index_strides,
                                   const int64_t slice_offset,
                                   const bool accumulate,
                                   const bool is_combined,
                                   DenseTensor* x_grad) {
  // CudaAtomicAdd for sub-4-byte types (bool, int8_t, uint8_t, int16_t) uses
  // atomicCAS on uint32_t, which reads 4 bytes at a 4-byte-aligned address.
  // If the total allocation size is not a multiple of 4, the last few elements
  // may cause out-of-bounds reads. Pad the allocation to prevent this.
  if (sizeof(T) < 4 && accumulate) {
    size_t alloc_bytes = static_cast<size_t>(x_grad->numel()) * sizeof(T);
    size_t padded_bytes = (alloc_bytes + 3) & ~static_cast<size_t>(3);
    dev_ctx.template Alloc<T>(x_grad, padded_bytes);
  } else {
    dev_ctx.template Alloc<T>(x_grad);
  }
  funcs::set_constant(dev_ctx, x_grad, static_cast<float>(0));
  if (out_grad.numel() == 0) return;

  const auto& index_type = index[0]->dtype();
  PADDLE_ENFORCE_EQ(index_type == DataType::INT64,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s].",
                        index_type,
                        DataType::INT32,
                        DataType::INT64));

  if (accumulate && index.size() == 1 && !is_combined) {
#ifdef PADDLE_WITH_CUDA
    funcs::IndexPutWithSortKernel<T, int64_t>(dev_ctx,
                                              x,
                                              out_grad,
                                              index,
                                              input_dims,
                                              input_strides,
                                              index_dims,
                                              index_strides,
                                              slice_offset,
                                              accumulate,
                                              x_grad);
    return;
#endif
  }
  if (funcs::IsInUint32Range(x_grad->numel() * sizeof(T),
                             out_grad.numel() * sizeof(T))) {
    GPUIndexElementwiseGetGrad<T>(dev_ctx,
                                  x,
                                  out_grad,
                                  index,
                                  input_dims,
                                  input_strides,
                                  index_dims,
                                  index_strides,
                                  slice_offset,
                                  accumulate,
                                  x_grad);
  } else {
    GPUIndexElementwiseGetGrad<T, uint64_t>(dev_ctx,
                                            x,
                                            out_grad,
                                            index,
                                            input_dims,
                                            input_strides,
                                            index_dims,
                                            index_strides,
                                            slice_offset,
                                            accumulate,
                                            x_grad);
  }
}

}  // namespace phi
PD_REGISTER_KERNEL(index_elementwise_get_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexElementwiseGetGradKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
