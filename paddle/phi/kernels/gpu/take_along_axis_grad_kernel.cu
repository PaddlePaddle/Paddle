// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/take_along_axis_grad_kernel.h"

#include <limits>
#include <type_traits>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/kernels/funcs/index_put_with_sort.cu.h"
#endif

COMMON_DECLARE_bool(cudnn_deterministic);

namespace phi {

#ifdef PADDLE_WITH_CUDA

template <typename T>
constexpr bool kTakeAlongAxisDeterministicSupported =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, phi::dtype::float16> ||
    std::is_same_v<T, phi::dtype::bfloat16>;

// Validate each index is in [-axis_size, axis_size) and normalize negatives to
// [0, axis_size).
template <typename IndexT>
__global__ void TakeAlongAxisCheckNormalizeKernel(const IndexT* __restrict__ in,
                                                  int64_t* __restrict__ out,
                                                  int64_t numel,
                                                  int64_t axis_size) {
  CUDA_KERNEL_LOOP_TYPE(i, numel, int64_t) {
    int64_t v = static_cast<int64_t>(in[i]);
    PADDLE_ENFORCE(
        v >= -axis_size && v < axis_size,
        "The index is out of bounds, please check whether the index and "
        "input's shape meet the requirements. It should be greater or equal "
        "to [%lld] and less than [%lld], but received [%lld]",
        static_cast<long long>(-axis_size),  // NOLINT
        static_cast<long long>(axis_size),   // NOLINT
        static_cast<long long>(v));          // NOLINT
    out[i] = v < 0 ? v + axis_size : v;
  }
}

// Deterministic take_along_axis backward, numerically bit-aligned with torch.
//
// Routes through the shared IndexPutWithSortKernel, which is in same with
// torch's `index_put_with_sort_kernel`.
// Turn the single `index` into a full set of per-dimension index tensors to
// use IndexPutWithSortKernel like torch's `_scatter_via_index_put`:
//  the `axis` dimension uses `index` itself;
//  other dimension uses an arange broadcast to match `index`'s shape.
//
// Returns false (falling back to the atomic scatter-add path) when:
//   1. the value dtype T is integral -- accumulation is exact and
//      order-independent, so determinism is already guaranteed,
//   2. inputs are non-contiguous -- coordinate recovery assumes contiguity,
//   3. the element count exceeds INT_MAX -- CUB refuses to sort that many keys,
//   4. the index dtype is neither int32 nor int64.
template <typename T, typename Context>
bool TakeAlongAxisGradDeterministic(const Context& dev_ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& index,
                                    const DenseTensor& out_grad,
                                    int axis,
                                    DenseTensor* x_grad) {
  if constexpr (!kTakeAlongAxisDeterministicSupported<T>) {
    return false;
  } else {
    // Validate dtype first, before the empty-input shortcut below: an empty
    // index with an unsupported dtype must still fall back to the scatter path
    // so it raises the proper InvalidArgument instead of silently succeeding.
    const auto& index_type = index.dtype();
    if (index_type != DataType::INT32 && index_type != DataType::INT64) {
      return false;
    }

    int64_t numel = index.numel();
    if (numel == 0) return true;
    if (numel > std::numeric_limits<int>::max()) return false;
    if (out_grad.numel() != numel) return false;
    if (!index.meta().is_contiguous() || !out_grad.meta().is_contiguous()) {
      return false;
    }

    int ndim = static_cast<int>(index.dims().size());
    if (ndim == 0 || ndim > DDim::kMaxRank) return false;
    // The per-dim index construction below requires `index` and `x_grad` to
    // agree on rank, and a valid scatter axis.
    if (ndim != static_cast<int>(x_grad->dims().size())) return false;
    if (axis < 0 || axis >= ndim) return false;

    auto index_shape = vectorize<int64_t>(index.dims());
    int64_t axis_size = x_grad->dims()[axis];

    // Build one index tensor per dimension (torch's _scatter_via_index_put).
    std::vector<DenseTensor> index_holders(ndim);
    std::vector<const DenseTensor*> indices_ptrs(ndim);
    for (int d = 0; d < ndim; ++d) {
      if (d == axis) {
        // Validate range and normalize negatives; also unifies int32/int64.
        DenseTensor axis_index;
        axis_index.Resize(index.dims());
        dev_ctx.template Alloc<int64_t>(&axis_index);
        auto config = backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
        auto stream = dev_ctx.stream();
        if (index_type == DataType::INT32) {
          TakeAlongAxisCheckNormalizeKernel<int32_t>
              <<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
                  index.data<int32_t>(),
                  axis_index.data<int64_t>(),
                  numel,
                  axis_size);
        } else {
          TakeAlongAxisCheckNormalizeKernel<int64_t>
              <<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
                  index.data<int64_t>(),
                  axis_index.data<int64_t>(),
                  numel,
                  axis_size);
        }
        index_holders[d] = axis_index;
      } else {
        DenseTensor arange_d;
        arange_d.Resize({index_shape[d]});
        dev_ctx.template Alloc<int64_t>(&arange_d);
        ArangeKernel<int64_t>(
            dev_ctx, Scalar(0), Scalar(index_shape[d]), Scalar(1), &arange_d);

        std::vector<int64_t> view_shape(ndim, 1);
        view_shape[d] = index_shape[d];
        DenseTensor reshaped =
            Reshape<int64_t, Context>(dev_ctx, arange_d, view_shape);

        DenseTensor expanded;
        ExpandKernel<int64_t, Context>(
            dev_ctx, reshaped, IntArray(index_shape), &expanded);
        index_holders[d] = expanded;
      }
      indices_ptrs[d] = &index_holders[d];
    }

    auto x_grad_dims = vectorize<int64_t>(x_grad->dims());

    // take_along_axis builds a full per-dim index set covering every axis, so
    // the indexed view is the whole contiguous x_grad: no axes precede the
    // indexed block (dims_before == 0), the view shape is x_grad's shape, and
    // there is no slice offset.
    funcs::SortedPathLayout layout;
    layout.dims_before = 0;
    layout.view_dims = x_grad_dims;
    layout.view_strides = vectorize<int64_t>(x_grad->strides());
    layout.view_offset = 0;
    layout.is_whole_tensor = true;

    funcs::IndexPutWithSortKernel<T, int64_t>(dev_ctx,
                                              out_grad,
                                              indices_ptrs,
                                              layout,
                                              /*accumulate=*/true,
                                              x_grad);
    return true;
  }
}
#endif

template <typename T, typename Context>
void TakeAlongAxisGradKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& index,
                             const DenseTensor& out_grad,
                             int axis,
                             DenseTensor* x_grad) {
  // We need to know the shape of input matrix to determine the shape of grad
  // matrix of input.
  x_grad->Resize(x.dims());
  dev_ctx.template Alloc<T>(x_grad);

  if (x_grad->numel() == 0) {
    return;
  }

  // Set to zero tensor.
  funcs::SetConstant<Context, T> functor;
  functor(dev_ctx, x_grad, static_cast<T>(0));
  const auto& index_type = index.dtype();

#ifdef PADDLE_WITH_CUDA
  if (FLAGS_cudnn_deterministic &&
      TakeAlongAxisGradDeterministic<T, Context>(
          dev_ctx, x, index, out_grad, axis, x_grad)) {
    return;
  }
#endif

  if (index_type == DataType::INT32) {
    funcs::gpu_scatter_add_kernel<T, int32_t>(
        *x_grad,
        axis,
        index,
        out_grad,
        true,
        dev_ctx);  // the gradient of gather is scatter
  } else if (index_type == DataType::INT64) {
    funcs::gpu_scatter_add_kernel<T, int64_t>(
        *x_grad, axis, index, out_grad, true, dev_ctx);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The data type of input index is expected "
        "to be int32 or int64, but received %s.",
        DataTypeToString(index_type)));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(take_along_axis_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::TakeAlongAxisGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   int16_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {}
