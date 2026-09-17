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

#pragma once

#ifdef PADDLE_WITH_CUDA

#include <algorithm>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/phi/backends/gpu/cuda/cuda_device_function.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/radix_sort.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace funcs {

constexpr int kIndexPutWarpSize = 32;

// Where the indexed axes sit inside the indexed view, plus that view's shape
// and where that view lives inside x_grad.  The sort based kernel arguments
// only describe the *restrided* view (indexed axes already replaced by the
// broadcast index shape), so this has to be recovered by the caller (see
// DeriveSortedPathLayout in index_elementwise_get_grad_kernel.cu).  A whole
// tensor consumer such as take_along_axis_grad builds a trivial layout with
// dims_before == 0 and view_dims == the full contiguous x_grad shape.
struct SortedPathLayout {
  int64_t dims_before;
  std::vector<int64_t> view_dims;     // the view's shape before indexing
  std::vector<int64_t> view_strides;  // its element strides inside x_grad
  int64_t view_offset;                // its byte offset inside x_grad
  bool is_whole_tensor;               // view == the whole contiguous x_grad
};

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
    for (int64_t idx =
             static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
         idx < numel;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.y) {
      if (idx < numel &&
          (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])) {
        int64_t curr_idx = idx;
        do {
          int64_t start_feature =
              threadIdx.x + static_cast<int64_t>(blockIdx.y) * blockDim.x * SZ;
          if (!accumulate && (curr_idx < numel - 1) &&
              sorted_indices[curr_idx] == sorted_indices[curr_idx + 1]) {
            curr_idx++;
            continue;
          }

          const int64_t weight_row =
              sorted_indices[curr_idx] * stride + z * stride_before;
          const int64_t grad_row =
              indices[curr_idx] * stride + z * numel * stride;
          const opmath_t scale = static_cast<opmath_t>(1.0);

          opmath_t gradient[SZ];
          opmath_t weight[SZ];

          while (start_feature < stride) {
#pragma unroll
            for (int ii = 0; ii < SZ; ii++) {
              int64_t feature_dim = start_feature + ii * kIndexPutWarpSize;
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
              int64_t feature_dim = start_feature + ii * kIndexPutWarpSize;
              if (feature_dim < stride) {
                grad_weight[weight_row + feature_dim] =
                    static_cast<scalar_t>(weight[ii]);
              }
            }
            start_feature += static_cast<int64_t>(gridDim.y) * blockDim.x * SZ;
          }
          curr_idx++;
        } while (curr_idx < numel &&
                 sorted_indices[curr_idx] == sorted_indices[curr_idx - 1]);
      }
    }
  }
}

// The sliceSize == 1 case can reduce all duplicate gradients with one warp.
// This mirrors the specialized CUDA path used by PyTorch and avoids routing
// the reduction through the generic feature-unrolled kernel.
template <typename scalar_t>
__global__ void IndexingBackwardKernelStride1(const int64_t* sorted_indices,
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
    for (int64_t idx =
             static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
         idx < numel;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.y) {
      const int64_t current_index = sorted_indices[idx];
      if (idx != 0 && current_index == sorted_indices[idx - 1]) {
        continue;
      }

      int64_t num_duplicates = 1;
      while (idx + num_duplicates < numel &&
             sorted_indices[idx + num_duplicates] == current_index) {
        ++num_duplicates;
      }

      const int64_t weight_row = current_index * stride + z * stride_before;
      const opmath_t scale = static_cast<opmath_t>(1.0);

      if (!accumulate) {
        if (threadIdx.x == 0) {
          const int64_t grad_row =
              indices[idx + num_duplicates - 1] * stride + z * numel * stride;
          grad_weight[weight_row] = static_cast<scalar_t>(
              static_cast<opmath_t>(grad_output[grad_row]) * scale);
        }
      } else {
        opmath_t gradient = static_cast<opmath_t>(0.0);
        const int lane = threadIdx.x;
        const int64_t num_warp_passes = num_duplicates / kIndexPutWarpSize;

        for (int64_t i = 0; i < num_warp_passes; ++i) {
          const int64_t duplicate_idx = idx + i * kIndexPutWarpSize + lane;
          const int64_t grad_row =
              indices[duplicate_idx] * stride + z * numel * stride;
          gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
        }

        unsigned mask = 0;
        CREATE_SHFL_MASK(mask, true);
        for (int offset = kIndexPutWarpSize / 2; offset > 0; offset /= 2) {
          gradient = gradient +
                     backends::gpu::CudaShuffleDownSync(mask, gradient, offset);
        }

        if (lane == 0) {
          for (int64_t i = num_warp_passes * kIndexPutWarpSize;
               i < num_duplicates;
               ++i) {
            const int64_t duplicate_idx = idx + i;
            const int64_t grad_row =
                indices[duplicate_idx] * stride + z * numel * stride;
            gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
          }
          grad_weight[weight_row] = static_cast<scalar_t>(
              static_cast<opmath_t>(grad_weight[weight_row]) + gradient);
        }
      }
    }
  }
}

// The 1 < slice_size <= kIndexPutWarpSize case lets a single thread own one
// feature column, so all duplicates of an index can be reduced in `opmath_t`
// registers and written back exactly once. The generic feature-unrolled kernel
// instead read-modify-writes `grad_weight` per duplicate, which rounds to
// `scalar_t` on every step and loses precision for float16/bfloat16. This
// mirrors the specialized CUDA path used by PyTorch.
template <typename scalar_t>
__global__ void IndexingBackwardKernelSmallStride(const int64_t* sorted_indices,
                                                  const int64_t* indices,
                                                  const scalar_t* grad_output,
                                                  scalar_t* grad_weight,
                                                  int64_t numel,
                                                  int64_t stride,
                                                  int64_t stride_before,
                                                  int64_t outer_dim,
                                                  bool accumulate) {
  using opmath_t = typename phi::dtype::MPTypeTrait<scalar_t>::Type;

  const int64_t tidx = threadIdx.x;
  if (tidx >= stride) return;

  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z) {
    for (int64_t idx =
             static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
         idx < numel;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.y) {
      const int64_t current_index = sorted_indices[idx];
      if (idx != 0 && current_index == sorted_indices[idx - 1]) {
        continue;
      }

      int64_t num_duplicates = 1;
      while (idx + num_duplicates < numel &&
             sorted_indices[idx + num_duplicates] == current_index) {
        ++num_duplicates;
      }

      const int64_t weight_row = current_index * stride + z * stride_before;
      const opmath_t scale = static_cast<opmath_t>(1.0);

      if (!accumulate) {
        const int64_t grad_row =
            indices[idx + num_duplicates - 1] * stride + z * numel * stride;
        grad_weight[weight_row + tidx] = static_cast<scalar_t>(
            static_cast<opmath_t>(grad_output[grad_row + tidx]) * scale);
      } else {
        opmath_t gradient = static_cast<opmath_t>(0.0);
        for (int64_t i = 0; i < num_duplicates; ++i) {
          const int64_t grad_row =
              indices[idx + i] * stride + z * numel * stride;
          gradient +=
              static_cast<opmath_t>(grad_output[grad_row + tidx]) * scale;
        }
        grad_weight[weight_row + tidx] = static_cast<scalar_t>(
            static_cast<opmath_t>(grad_weight[weight_row + tidx]) + gradient);
      }
    }
  }
}

template <typename T, typename IndexT>
void IndexPutWithSortKernel(const GPUContext& dev_ctx,
                            const DenseTensor& value,
                            const std::vector<const DenseTensor*>& indices,
                            const SortedPathLayout& layout,
                            const bool accumulate,
                            DenseTensor* output) {
  DenseTensor& self = *output;

  if (indices.size() > layout.view_dims.size()) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Too many indices for tensor of dimension %d (got %d).",
        layout.view_dims.size(),
        indices.size()));
  }

  const bool self_contiguous = self.meta().is_contiguous();
  // Must be spelled with an explicit type rather than `auto`: `auto` would give
  // this variable a dependent type, and `self_.data<T>()` would then be parsed
  // as `(self_.data) < T > (...)`, i.e. a comparison against a type name.
  DenseTensor self_ =
      self_contiguous ? self : phi::Contiguous<T, GPUContext>(dev_ctx, self);
  DenseTensor expanded_value = value;

  // Reinterpret x_grad with the shape of the indexed view so that the linear
  // index is built against the axes the indices actually address. This is a
  // pure relabelling: the two agree elementwise and both are contiguous.
  DenseTensor view_src = self_;
  auto view_meta = self_.meta();
  view_meta.dims = make_ddim(layout.view_dims);
  view_meta.strides = DenseTensorMeta::calc_strides(view_meta.dims);
  view_src.set_meta(view_meta);

  std::vector<DenseTensor> axis_indices(layout.view_dims.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    axis_indices[layout.dims_before + i] = *indices[i];
  }

  auto [linear_index, n_elem_before, stride_before, slice_size] =
      funcs::computeLinearIndex(dev_ctx, view_src, axis_indices, false);

  int64_t num_indices = linear_index.numel();

  if (expanded_value.numel() < num_indices * n_elem_before * slice_size) {
    auto expanded_size = vectorize<int64_t>(expanded_value.dims());
    auto size1 = vectorize<int64_t>(expanded_value.dims());
    auto size2 = vectorize<int64_t>(linear_index.dims());
    if (funcs::are_expandable(size1, size2)) {
      expanded_size = funcs::infer_size_dimvector(size1, size2);
    }
    if (n_elem_before > 1) {
      expanded_size.insert(expanded_size.begin(), n_elem_before);
    }
    if (slice_size > 1) {
      expanded_size.insert(expanded_size.end(), slice_size);
    }

    DenseTensor expanded_tensor;
    phi::ExpandKernel<T, GPUContext>(
        dev_ctx, expanded_value, IntArray(expanded_size), &expanded_tensor);
    expanded_value = expanded_tensor;
  }
  if (!expanded_value.meta().is_contiguous()) {
    expanded_value = phi::Contiguous<T, GPUContext>(dev_ctx, expanded_value);
  }

  if (num_indices > 0 && slice_size > 0) {
    linear_index =
        phi::Reshape<IndexT, GPUContext>(dev_ctx, linear_index, {-1});

    DenseTensor sorted_indices;
    sorted_indices.Resize(linear_index.dims());
    dev_ctx.Alloc<IndexT>(&sorted_indices);
    DenseTensor orig_indices;
    orig_indices.Resize(linear_index.dims());
    dev_ctx.Alloc<IndexT>(&orig_indices);

    auto stream = dev_ctx.stream();

    auto shape = IntArray(vectorize<int64_t>(linear_index.dims()));
    auto divisor =
        phi::Full<IndexT, GPUContext>(dev_ctx, shape, Scalar(slice_size));

    DenseTensor linear_index_d =
        phi::FloorDivide<IndexT, GPUContext>(dev_ctx, linear_index, divisor);

    DenseTensor range;
    range.Resize({num_indices});
    dev_ctx.Alloc<IndexT>(&range);
    phi::ArangeKernel<IndexT>(
        dev_ctx, Scalar(0), Scalar(num_indices), Scalar(1), &range);
    int64_t nbits = funcs::GetNumBits(funcs::LargestIndex(self_) / slice_size);

    funcs::RadixSortPairs<IndexT, IndexT>(dev_ctx,
                                          linear_index_d.data<IndexT>(),
                                          sorted_indices.data<IndexT>(),
                                          range.data<IndexT>(),
                                          orig_indices.data<IndexT>(),
                                          num_indices,
                                          false,
                                          0,
                                          nbits);

    const int UNROLL = 4;
    const int INDICES_PER_BLOCK = 4;
    auto max_grid_size =
        backends::gpu::GetGpuMaxGridDimSize(dev_ctx.GetPlace().GetDeviceId());

    dim3 grid(
        std::min(static_cast<int64_t>(max_grid_size[0]),
                 (num_indices + INDICES_PER_BLOCK - 1) / INDICES_PER_BLOCK),
        std::min(static_cast<int64_t>(max_grid_size[1]),
                 (slice_size + kIndexPutWarpSize * UNROLL - 1) /
                     (kIndexPutWarpSize * UNROLL)),
        std::min(std::max(static_cast<int64_t>(1),
                          static_cast<int64_t>(n_elem_before)),
                 static_cast<int64_t>(max_grid_size[2])));
    dim3 block(kIndexPutWarpSize, INDICES_PER_BLOCK);

    if (slice_size == 1) {
      IndexingBackwardKernelStride1<T>
          <<<grid, block, 0, stream>>>(sorted_indices.data<IndexT>(),
                                       orig_indices.data<IndexT>(),
                                       expanded_value.data<T>(),
                                       self_.data<T>(),
                                       num_indices,
                                       slice_size,
                                       stride_before,
                                       n_elem_before,
                                       accumulate);
    } else if (slice_size <= kIndexPutWarpSize) {
      IndexingBackwardKernelSmallStride<T>
          <<<grid, block, 0, stream>>>(sorted_indices.data<IndexT>(),
                                       orig_indices.data<IndexT>(),
                                       expanded_value.data<T>(),
                                       self_.data<T>(),
                                       num_indices,
                                       slice_size,
                                       stride_before,
                                       n_elem_before,
                                       accumulate);
    } else {
      IndexingBackwardKernel<T, UNROLL>
          <<<grid, block, 0, stream>>>(sorted_indices.data<IndexT>(),
                                       orig_indices.data<IndexT>(),
                                       expanded_value.data<T>(),
                                       self_.data<T>(),
                                       num_indices,
                                       slice_size,
                                       stride_before,
                                       n_elem_before,
                                       accumulate);
    }

    if (!self_contiguous) {
      phi::Copy(dev_ctx, self_, dev_ctx.GetPlace(), false, output);
    }
  }
}

}  // namespace funcs
}  // namespace phi

#endif  // PADDLE_WITH_CUDA
