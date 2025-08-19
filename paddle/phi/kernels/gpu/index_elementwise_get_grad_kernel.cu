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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
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
    const std::array<int64_t, phi::DDim::kMaxRank + 1> sizes,
    const std::array<int64_t, phi::DDim::kMaxRank + 1> strides,
    int num_indices,
    offset_calc_t offset_calc) {
  const int tid = threadIdx.x;
  const int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
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

      phi::CudaAtomicAdd(reinterpret_cast<T*>(out_data + offset),
                         *reinterpret_cast<const T*>(in_data));
      idx += nt;
    }
  }
}

template <typename T, typename IndexT>
void GPUIndexElementwiseGetGrad(const phi::GPUContext& dev_ctx,
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

  auto sizes = std::array<int64_t, phi::DDim::kMaxRank + 1>{};
  auto strides = std::array<int64_t, phi::DDim::kMaxRank + 1>{};
  for (int64_t i = 0; i < num_indices; i++) {
    sizes[i] = index_dims[i];
    strides[i] = index_strides[i];
  }
  auto index_ptrs = funcs::GetIndexDataPtrs<IndexT>(index);

  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;

  funcs::IndexPutStride<3>(input_dims,
                           input_strides,
                           phi::SizeOf(input.dtype()),
                           common::vectorize<int64_t>(value.dims()),
                           common::vectorize<int64_t>(value.strides()),
                           phi::SizeOf(value.dtype()),
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
  constexpr int nt = 128;
  constexpr int vt = 4;
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = dev_ctx.stream();

  using dtype = funcs::OpaqueType<sizeof(T)>;

  const char* in_ptr = reinterpret_cast<const char*>(value.data<T>());
  char* out_ptr = reinterpret_cast<char*>(output->data<T>()) + slice_offset;

  if (accumulate) {
    IndexEleGetGradAccKernel<T, IndexT, nt, vt>
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
        <<<grid, block, 0, stream>>>(N, [=] __device__(int idx) {
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

#ifdef PADDLE_WITH_CUDA
#define WARP_SIZE 32

template <typename scalar_t, int SZ>
__global__ void IndexingBackwardKernel(const int64_t* sorted_indices,
                                       const int64_t* indices,
                                       const scalar_t* grad_output,
                                       scalar_t* grad_weight,
                                       int64_t numel,
                                       int64_t stride,
                                       int64_t stride_before,
                                       int64_t outer_dim,
                                       bool accumulate) {
  using opmath_t = typename phi::dtype::MPTypeTrait<scalar_t>::Type;

  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z) {
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel &&
        (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])) {
      do {
        int64_t start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        if (!accumulate && (idx < numel - 1) &&
            sorted_indices[idx] == sorted_indices[idx + 1]) {
          idx++;
          continue;
        }

        const int64_t weight_row =
            sorted_indices[idx] * stride + z * stride_before;
        const int64_t grad_row = indices[idx] * stride + z * numel * stride;
        const opmath_t scale = static_cast<opmath_t>(1.0);

        opmath_t gradient[SZ];
        opmath_t weight[SZ];

        while (start_feature < stride) {
#pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * WARP_SIZE;
            if (feature_dim < stride) {
              gradient[ii] =
                  static_cast<opmath_t>(grad_output[grad_row + feature_dim]);
              if (accumulate) {
                weight[ii] = static_cast<opmath_t>(
                    grad_weight[weight_row + feature_dim]);
              }
            }
          }

#pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            if (accumulate) {
              weight[ii] += gradient[ii] * scale;
            } else {
              weight[ii] = gradient[ii] * scale;
            }
          }

#pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * WARP_SIZE;
            if (feature_dim < stride) {
              grad_weight[weight_row + feature_dim] =
                  static_cast<scalar_t>(weight[ii]);
            }
          }
          start_feature += gridDim.y * blockDim.x * SZ;
        }
        idx++;
      } while (idx < numel && sorted_indices[idx] == sorted_indices[idx - 1]);
    }
  }
}

template <typename T, typename IndexT>
void IndexPutWithSortKernel(const phi::GPUContext& dev_ctx,
                            const DenseTensor& input,
                            const DenseTensor& value,
                            const std::vector<const DenseTensor*>& indices,
                            const std::vector<int64_t>& input_dims,
                            const std::vector<int64_t>& input_strides,
                            const std::vector<int64_t>& index_dims,
                            const std::vector<int64_t>& index_strides,
                            const int64_t slice_offset,
                            const bool accumulate,
                            DenseTensor* output) {
  DenseTensor& self = *output;

  if (indices.size() > static_cast<size_t>(self.dims().size())) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Too many indices for tensor of dimension %d (got %d).",
        self.dims().size(),
        indices.size()));
  }

  const bool unsafe = true;
  const bool self_contiguous = self.meta().is_contiguous();
  auto self_ = self_contiguous
                   ? self
                   : phi::Contiguous<T, phi::GPUContext>(dev_ctx, self);
  DenseTensor linearIndex, src, expandedValue = value;
  int64_t nElemBefore, strideBefore, sliceSize;
  std::vector<int64_t> inversePerm;
  std::tie(
      linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm) =
      funcs::makeLinearIndex<T>(dev_ctx, self_, indices, !unsafe);

  int64_t num_indices = linearIndex.numel();

  if (expandedValue.numel() < num_indices * nElemBefore * sliceSize) {
    auto expanded_size = common::vectorize<int64_t>(expandedValue.dims());
    auto size1 = common::vectorize<int64_t>(expandedValue.dims());
    auto size2 = common::vectorize<int64_t>(linearIndex.dims());
    if (funcs::are_expandable(size1, size2)) {
      expanded_size = funcs::infer_size_dimvector(size1, size2);
    }
    if (nElemBefore > 1) {
      expanded_size.insert(expanded_size.begin(), nElemBefore);
    }
    if (sliceSize > 1) {
      expanded_size.insert(expanded_size.end(), sliceSize);
    }

    DenseTensor expanded_tensor;
    phi::ExpandKernel<T, phi::GPUContext>(
        dev_ctx, expandedValue, phi::IntArray(expanded_size), &expanded_tensor);
    expandedValue = expanded_tensor;
  }
  if (!expandedValue.meta().is_contiguous()) {
    expandedValue = phi::Contiguous<T, phi::GPUContext>(dev_ctx, expandedValue);
  }

  if (num_indices > 0 && sliceSize > 0) {
    const bool permuted = !src.meta().is_contiguous();
    DenseTensor src_ =
        permuted ? phi::Contiguous<T, phi::GPUContext>(dev_ctx, src) : src;
    linearIndex =
        phi::Reshape<IndexT, phi::GPUContext>(dev_ctx, linearIndex, {-1});

    DenseTensor sorted_indices;
    sorted_indices.Resize(linearIndex.dims());
    dev_ctx.Alloc<IndexT>(&sorted_indices);
    DenseTensor orig_indices;
    orig_indices.Resize(linearIndex.dims());
    dev_ctx.Alloc<IndexT>(&orig_indices);

    auto stream = dev_ctx.stream();
    constexpr int blockSize = 256;
    int gridSize = (num_indices + blockSize - 1) / blockSize;

    auto shape = phi::IntArray(common::vectorize<int64_t>(linearIndex.dims()));
    auto divisor = phi::Full<IndexT, phi::GPUContext>(
        dev_ctx, shape, phi::Scalar(sliceSize));

    DenseTensor linearIndex_d = phi::FloorDivide<IndexT, phi::GPUContext>(
        dev_ctx, linearIndex, divisor);

    DenseTensor range;
    range.Resize({num_indices});
    dev_ctx.Alloc<IndexT>(&range);
    phi::ArangeKernel<IndexT>(dev_ctx,
                              phi::Scalar(0),
                              phi::Scalar(num_indices),
                              phi::Scalar(1),
                              &range);
    int64_t nbits = funcs::GetNumBits(funcs::LargestIndex(self_) / sliceSize);

    funcs::RadixSortPairs<IndexT, IndexT>(dev_ctx,
                                          linearIndex_d.data<IndexT>(),
                                          sorted_indices.data<IndexT>(),
                                          range.data<IndexT>(),
                                          orig_indices.data<IndexT>(),
                                          num_indices,
                                          false,
                                          0,
                                          nbits);

    const int UNROLL = 4;
    const int INDICES_PER_BLOCK = 4;
    auto max_grid_size = phi::backends::gpu::GetGpuMaxGridDimSize(
        dev_ctx.GetPlace().GetDeviceId());

    dim3 grid((num_indices + INDICES_PER_BLOCK - 1) / INDICES_PER_BLOCK,
              std::min<int>(
                  max_grid_size[1],
                  (sliceSize + WARP_SIZE * UNROLL - 1) / (WARP_SIZE * UNROLL)),
              std::min<int>(std::max<int>(1, static_cast<int>(nElemBefore)),
                            max_grid_size[2]));
    dim3 block(WARP_SIZE, INDICES_PER_BLOCK);

    IndexingBackwardKernel<T, UNROLL>
        <<<grid, block, 0, stream>>>(sorted_indices.data<IndexT>(),
                                     orig_indices.data<IndexT>(),
                                     expandedValue.data<T>(),
                                     src_.data<T>(),
                                     num_indices,
                                     sliceSize,
                                     strideBefore,
                                     nElemBefore,
                                     true);

    if (permuted) {
      phi::DenseTensor transposed_src;
      std::vector<int> inversePerm_int(inversePerm.size());
      std::transform(inversePerm.begin(),
                     inversePerm.end(),
                     inversePerm_int.begin(),
                     [](int64_t x) { return static_cast<int>(x); });

      phi::Transpose<T, phi::GPUContext>(
          dev_ctx, src_, inversePerm_int, &transposed_src);
      phi::Copy(dev_ctx, transposed_src, dev_ctx.GetPlace(), false, output);
    } else if (!self_contiguous) {
      phi::Copy(dev_ctx, self_, dev_ctx.GetPlace(), false, output);
    }
  }
}
#endif

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
  dev_ctx.template Alloc<T>(x_grad);
  phi::funcs::set_constant(dev_ctx, x_grad, static_cast<float>(0));
  if (out_grad.numel() == 0) return;

  const auto& index_type = index[0]->dtype();
  PADDLE_ENFORCE_EQ(index_type == phi::DataType::INT64,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s].",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  if (accumulate && index.size() == 1 && !is_combined) {
#ifdef PADDLE_WITH_CUDA
    IndexPutWithSortKernel<T, int64_t>(dev_ctx,
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

  GPUIndexElementwiseGetGrad<T, int64_t>(dev_ctx,
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
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
