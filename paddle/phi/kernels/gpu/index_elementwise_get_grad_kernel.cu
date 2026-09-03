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

#include <algorithm>
#include <cstdlib>
#include <utility>
#include <vector>

#include "paddle/common/enforce.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/gpu/cuda/cuda_device_function.h"
#endif
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/radix_sort.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"

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
  auto offset_calc = funcs::make_offset_calculator_put<3, true, OffsetT>(
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
        const int64_t num_warp_passes = num_duplicates / WARP_SIZE;

        for (int64_t i = 0; i < num_warp_passes; ++i) {
          const int64_t duplicate_idx = idx + i * WARP_SIZE + lane;
          const int64_t grad_row =
              indices[duplicate_idx] * stride + z * numel * stride;
          gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
        }

        unsigned mask = 0;
        CREATE_SHFL_MASK(mask, true);
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
          gradient = gradient +
                     backends::gpu::CudaShuffleDownSync(mask, gradient, offset);
        }

        if (lane == 0) {
          for (int64_t i = num_warp_passes * WARP_SIZE; i < num_duplicates;
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

// The 1 < sliceSize <= WARP_SIZE case lets a single thread own one feature
// column, so all duplicates of an index can be reduced in `opmath_t` registers
// and written back exactly once. The generic feature-unrolled kernel instead
// read-modify-writes `grad_weight` per duplicate, which rounds to `scalar_t`
// on every step and loses precision for float16/bfloat16. This mirrors the
// specialized CUDA path used by PyTorch.
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

// Where the indexed axes sit inside the indexed view, plus that view's shape
// and where that view lives inside x_grad.  The kernel arguments only describe
// the *restrided* view (indexed axes already replaced by the broadcast index
// shape), so this has to be recovered.
struct SortedPathLayout {
  int64_t dims_before;
  std::vector<int64_t> view_dims;     // the view's shape before indexing
  std::vector<int64_t> view_strides;  // its element strides inside x_grad
  int64_t view_offset;                // its byte offset inside x_grad
  bool is_whole_tensor;               // view == the whole contiguous x_grad
};

// A strided region is free of self overlap when, walking its axes from the
// smallest stride magnitude up, every stride clears the span already covered.
// Only magnitudes matter: negating an axis mirrors the region onto the same set
// of elements, so a reversed view (x[::-1, idx]) is just as non overlapping as
// the forward one.  Such a view cannot reach this kernel today because the
// forward index_elementwise_get_kernel truncates negative strides to unsigned
// (its offset calculator is instantiated with signed_strides = false), but the
// sign of a stride is a property of the layout, not of what the caller happens
// to be able to build, so it is handled here rather than assumed away.
static bool IsNonOverlapping(const std::vector<int64_t>& dims,
                             const std::vector<int64_t>& strides) {
  std::vector<std::pair<int64_t, int64_t>> axes;  // (|stride|, extent)
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] == 1) continue;         // its stride is never used
    if (strides[i] == 0) return false;  // broadcast axis, not a real region
    axes.emplace_back(std::abs(strides[i]), dims[i]);
  }
  std::sort(axes.begin(), axes.end());
  int64_t span = 1;
  for (auto& [stride, extent] : axes) {
    if (stride < span) return false;
    span += (extent - 1) * stride;
  }
  return true;
}

// restride_src() (paddle/fluid/pybind/slice_utils.h) replaces the indexed axes
// with the broadcast index shape and gives exactly those axes a zero stride, so
// the run of zeros in `input_strides` marks the block and its start is
// dims_before.  `index_dims` starts with the extents of exactly those axes (see
// AdvancedIndex::indexed_sizes, terminated by -1), so the view's pre-indexing
// shape and its strides inside x_grad are both reconstructible.
//
// Returns false only when the region is degenerate or overlaps itself, in which
// case the caller falls back to the elementwise path.
static bool DeriveSortedPathLayout(const std::vector<int64_t>& input_dims,
                                   const std::vector<int64_t>& input_strides,
                                   const std::vector<int64_t>& index_dims,
                                   const std::vector<int64_t>& index_strides,
                                   int64_t slice_offset,
                                   int64_t elesize,
                                   int64_t grad_numel,
                                   SortedPathLayout* layout) {
  const size_t ndim = input_dims.size();
  const size_t nidx = index_strides.size();
  if (nidx == 0 || ndim == 0 || input_strides.size() != ndim ||
      index_dims.size() < nidx) {
    return false;
  }

  size_t db = 0;
  while (db < ndim && input_strides[db] != 0) ++db;
  size_t nblock = 0;
  while (db + nblock < ndim && input_strides[db + nblock] == 0) ++nblock;
  if (nblock == 0) return false;
  for (size_t i = db + nblock; i < ndim; ++i) {
    if (input_strides[i] == 0) return false;
  }

  std::vector<int64_t> view(input_dims.begin(), input_dims.begin() + db);
  std::vector<int64_t> vstride(input_strides.begin(),
                               input_strides.begin() + db);
  for (size_t i = 0; i < nidx; ++i) {
    if (index_dims[i] < 0) return false;
    view.push_back(index_dims[i]);
    vstride.push_back(index_strides[i] / elesize);
  }
  view.insert(view.end(), input_dims.begin() + db + nblock, input_dims.end());
  vstride.insert(
      vstride.end(), input_strides.begin() + db + nblock, input_strides.end());

  if (!IsNonOverlapping(view, vstride)) return false;

  int64_t numel = 1;
  for (int64_t s : view) numel *= s;
  if (numel <= 0 || numel > grad_numel) return false;

  // Every element the sort based kernel touches sits at
  //   slice_offset / elesize + sum_k i_k * view_strides[k]
  // so both ends of that range have to stay inside x_grad.  Checking the span
  // rather than just numel also covers the strided cases, where the region is
  // sparse and reaches further than its element count; a reversed axis has a
  // negative stride and pulls the low end below slice_offset.
  const int64_t base = slice_offset / elesize;
  int64_t lo = base;
  int64_t hi = base;
  for (size_t i = 0; i < view.size(); ++i) {
    const int64_t reach = (view[i] - 1) * vstride[i];
    if (reach < 0) {
      lo += reach;
    } else {
      hi += reach;
    }
  }
  PADDLE_ENFORCE_GE(
      lo,
      0,
      common::errors::InvalidArgument(
          "The indexed view starts before the beginning of x_grad: its lowest "
          "element is at position %d. slice_offset is %d bytes and the view is "
          "%s with strides %s.",
          lo,
          slice_offset,
          make_ddim(view).to_str(),
          make_ddim(vstride).to_str()));
  PADDLE_ENFORCE_LT(
      hi,
      grad_numel,
      common::errors::InvalidArgument(
          "The indexed view runs past the end of x_grad: its highest element "
          "is at position %d but x_grad only holds %d elements. slice_offset "
          "is %d bytes and the view is %s with strides %s.",
          hi,
          grad_numel,
          slice_offset,
          make_ddim(view).to_str(),
          make_ddim(vstride).to_str()));

  std::vector<int64_t> contig(view.size(), 1);
  for (int i = static_cast<int>(view.size()) - 2; i >= 0; --i) {
    contig[i] = contig[i + 1] * view[i + 1];
  }
  layout->is_whole_tensor =
      (slice_offset == 0 && numel == grad_numel && vstride == contig);
  layout->dims_before = static_cast<int64_t>(db);
  layout->view_dims = std::move(view);
  layout->view_strides = std::move(vstride);
  layout->view_offset = slice_offset;
  return true;
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
  auto self_ =
      self_contiguous ? self : phi::Contiguous<T, GPUContext>(dev_ctx, self);
  DenseTensor expandedValue = value;

  // Reinterpret x_grad with the shape of the indexed view so that the linear
  // index is built against the axes the indices actually address. This is a
  // pure relabelling: the two agree elementwise and both are contiguous.
  DenseTensor view_src = self_;
  auto view_meta = self_.meta();
  view_meta.dims = make_ddim(layout.view_dims);
  view_meta.strides = DenseTensorMeta::calc_strides(view_meta.dims);
  view_src.set_meta(view_meta);

  std::vector<DenseTensor> aligned(layout.view_dims.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    aligned[layout.dims_before + i] = *indices[i];
  }

  auto [linearIndex, nElemBefore, strideBefore, sliceSize] =
      funcs::computeLinearIndex(dev_ctx, view_src, aligned, false);

  int64_t num_indices = linearIndex.numel();

  if (expandedValue.numel() < num_indices * nElemBefore * sliceSize) {
    auto expanded_size = vectorize<int64_t>(expandedValue.dims());
    auto size1 = vectorize<int64_t>(expandedValue.dims());
    auto size2 = vectorize<int64_t>(linearIndex.dims());
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
    phi::ExpandKernel<T, GPUContext>(
        dev_ctx, expandedValue, phi::IntArray(expanded_size), &expanded_tensor);
    expandedValue = expanded_tensor;
  }
  if (!expandedValue.meta().is_contiguous()) {
    expandedValue = phi::Contiguous<T, GPUContext>(dev_ctx, expandedValue);
  }

  if (num_indices > 0 && sliceSize > 0) {
    DenseTensor& src_ = self_;
    linearIndex = phi::Reshape<IndexT, GPUContext>(dev_ctx, linearIndex, {-1});

    DenseTensor sorted_indices;
    sorted_indices.Resize(linearIndex.dims());
    dev_ctx.Alloc<IndexT>(&sorted_indices);
    DenseTensor orig_indices;
    orig_indices.Resize(linearIndex.dims());
    dev_ctx.Alloc<IndexT>(&orig_indices);

    auto stream = dev_ctx.stream();

    auto shape = phi::IntArray(vectorize<int64_t>(linearIndex.dims()));
    auto divisor =
        Full<IndexT, GPUContext>(dev_ctx, shape, phi::Scalar(sliceSize));

    DenseTensor linearIndex_d =
        phi::FloorDivide<IndexT, GPUContext>(dev_ctx, linearIndex, divisor);

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

    dim3 grid(
        std::min(static_cast<int64_t>(max_grid_size[0]),
                 (num_indices + INDICES_PER_BLOCK - 1) / INDICES_PER_BLOCK),
        std::min(static_cast<int64_t>(max_grid_size[1]),
                 (sliceSize + WARP_SIZE * UNROLL - 1) / (WARP_SIZE * UNROLL)),
        std::min(std::max(static_cast<int64_t>(1),
                          static_cast<int64_t>(nElemBefore)),
                 static_cast<int64_t>(max_grid_size[2])));
    dim3 block(WARP_SIZE, INDICES_PER_BLOCK);

    if (sliceSize == 1) {
      IndexingBackwardKernelStride1<T>
          <<<grid, block, 0, stream>>>(sorted_indices.data<IndexT>(),
                                       orig_indices.data<IndexT>(),
                                       expandedValue.data<T>(),
                                       src_.data<T>(),
                                       num_indices,
                                       sliceSize,
                                       strideBefore,
                                       nElemBefore,
                                       accumulate);
    } else if (sliceSize <= WARP_SIZE) {
      IndexingBackwardKernelSmallStride<T>
          <<<grid, block, 0, stream>>>(sorted_indices.data<IndexT>(),
                                       orig_indices.data<IndexT>(),
                                       expandedValue.data<T>(),
                                       src_.data<T>(),
                                       num_indices,
                                       sliceSize,
                                       strideBefore,
                                       nElemBefore,
                                       accumulate);
    } else {
      IndexingBackwardKernel<T, UNROLL>
          <<<grid, block, 0, stream>>>(sorted_indices.data<IndexT>(),
                                       orig_indices.data<IndexT>(),
                                       expandedValue.data<T>(),
                                       src_.data<T>(),
                                       num_indices,
                                       sliceSize,
                                       strideBefore,
                                       nElemBefore,
                                       accumulate);
    }

    if (!self_contiguous) {
      Copy(dev_ctx, self_, dev_ctx.GetPlace(), false, output);
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

  // slice_offset is the byte offset of the indexed view inside x_grad's buffer,
  // measured in the forward pass. Both the sort based path and the elementwise
  // fallback add it to x_grad's base pointer, so a bogus value would turn into
  // an out of bounds write. Reject it here instead.
  const int64_t grad_bytes = x_grad->numel() * static_cast<int64_t>(sizeof(T));
  PADDLE_ENFORCE_GE(
      slice_offset,
      0,
      common::errors::InvalidArgument(
          "slice_offset must be non-negative, but got %d.", slice_offset));
  PADDLE_ENFORCE_LT(
      slice_offset,
      grad_bytes,
      common::errors::InvalidArgument(
          "slice_offset (%d bytes) must point inside x_grad, which only holds "
          "%d bytes.",
          slice_offset,
          grad_bytes));

  // slice_offset and index_strides are byte quantities that both paths turn
  // into T* arithmetic, so they have to be whole elements. By construction they
  // always are (slice_offset is a pointer delta between two views of one
  // allocation, index_strides is an element stride times sizeof(T)); assert it
  // here so neither path can build a misaligned T*.
  PADDLE_ENFORCE_EQ(
      slice_offset % static_cast<int64_t>(sizeof(T)),
      0,
      common::errors::InvalidArgument(
          "slice_offset (%d bytes) must be a whole number of %d byte elements.",
          slice_offset,
          sizeof(T)));
  for (size_t i = 0; i < index_strides.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        index_strides[i] % static_cast<int64_t>(sizeof(T)),
        0,
        common::errors::InvalidArgument(
            "index_strides[%d] (%d bytes) must be a whole number of %d byte "
            "elements.",
            i,
            index_strides[i],
            sizeof(T)));
  }

  if (accumulate) {
#ifdef PADDLE_WITH_CUDA
    // PyTorch routes every accumulating advanced index backward through the
    // sort based kernel, so how much the duplicate reduction rounds depends on
    // sliceSize alone and not on how the index expression was spelled. Do the
    // same here.
    SortedPathLayout layout;
    if (DeriveSortedPathLayout(input_dims,
                               input_strides,
                               index_dims,
                               index_strides,
                               slice_offset,
                               static_cast<int64_t>(sizeof(T)),
                               x_grad->numel(),
                               &layout)) {
      if (layout.is_whole_tensor) {
        IndexPutWithSortKernel<T, int64_t>(
            dev_ctx, out_grad, index, layout, accumulate, x_grad);
      } else {
        // The indices address a strided sub region of x (slices were applied
        // first), which the sort based kernel cannot write to because it
        // addresses grad_weight as one flat buffer. Reduce into a contiguous
        // buffer shaped like that view and scatter it back afterwards, which is
        // how PyTorch composes slice_backward with index_backward. x_grad is
        // already zeroed, so the copy needs no accumulation.
        DenseTensor view_grad;
        view_grad.Resize(make_ddim(layout.view_dims));
        dev_ctx.template Alloc<T>(&view_grad);
        funcs::set_constant(dev_ctx, &view_grad, static_cast<float>(0));
        IndexPutWithSortKernel<T, int64_t>(
            dev_ctx, out_grad, index, layout, accumulate, &view_grad);
        auto grad_meta = x_grad->meta();
        StridedCopyKernel<T, Context>(dev_ctx,
                                      view_grad,
                                      layout.view_dims,
                                      layout.view_strides,
                                      layout.view_offset,
                                      x_grad);
        x_grad->set_meta(grad_meta);
      }
      return;
    }
#endif
  }

  if (funcs::IsInInt32Range(x_grad->numel() * sizeof(T),
                            out_grad.numel() * sizeof(T),
                            funcs::IndexOperandByteSpan(index_dims))) {
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
