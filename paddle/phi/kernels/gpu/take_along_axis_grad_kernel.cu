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

#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/kernels/funcs/radix_sort.h"
#endif

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);
COMMON_DECLARE_bool(cudnn_deterministic);

namespace phi {

#ifdef PADDLE_WITH_CUDA

constexpr int kTakeAlongAxisWarpSize = 32;
constexpr int kTakeAlongAxisIndicesPerBlock = 4;

inline int TakeAlongAxisNumBits(uint64_t max_val) {
  int num_bits = 1;
  while (max_val > 1) {
    max_val >>= 1;
    num_bits++;
  }
  return num_bits;
}

struct TakeAlongAxisDimArray {
  int64_t data[DDim::kMaxRank];
};

template <typename IndexT>
__global__ void TakeAlongAxisDestOffsetKernel(
    const IndexT* __restrict__ index,
    int64_t* __restrict__ dest,
    int64_t* __restrict__ source_pos,
    TakeAlongAxisDimArray index_dims,
    TakeAlongAxisDimArray x_grad_strides,
    int ndim,
    int axis,
    int64_t axis_size,
    int64_t numel) {
  int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= numel) return;

  int64_t axis_coord = static_cast<int64_t>(index[p]);
  PADDLE_ENFORCE(axis_coord >= -axis_size && axis_coord < axis_size,
                 "The index is out of bounds, "
                 "please check whether the dimensions of index and "
                 "input meet the requirements. It should "
                 "be less than [%lld] and greater than or equal to [%lld], but "
                 "received [%lld]",
                 axis_size,
                 -axis_size,
                 axis_coord);
  if (axis_coord < 0) axis_coord += axis_size;

  int64_t remainder = p;
  int64_t offset = 0;
  for (int d = ndim - 1; d >= 0; --d) {
    int64_t coord = remainder % index_dims.data[d];
    remainder /= index_dims.data[d];
    offset += (d == axis ? axis_coord : coord) * x_grad_strides.data[d];
  }
  dest[p] = offset;
  source_pos[p] = p;
}

// One warp per sorted position; only the warp owning the first position of a
// group does work, so each destination element is written exactly once and the
// plain read-modify-write below needs no atomics.
template <typename T>
__global__ void TakeAlongAxisScatterAddGroupKernel(
    const int64_t* __restrict__ sorted_dest,
    const int64_t* __restrict__ source_pos,
    const T* __restrict__ out_grad,
    T* __restrict__ x_grad,
    int64_t numel) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (idx >= numel) return;

  int64_t dest = sorted_dest[idx];
  if (idx != 0 && sorted_dest[idx - 1] == dest) return;  // not a group head

  int64_t num_duplicates = 1;
  while (idx + num_duplicates < numel &&
         sorted_dest[idx + num_duplicates] == dest) {
    num_duplicates++;
  }

  MT gradient = static_cast<MT>(0);
  int lane_idx = threadIdx.x % kTakeAlongAxisWarpSize;
  int64_t num_warp_passes = num_duplicates / kTakeAlongAxisWarpSize;
  for (int64_t i = 0; i < num_warp_passes; ++i) {
    int64_t grad_row = source_pos[idx + i * kTakeAlongAxisWarpSize + lane_idx];
    gradient += static_cast<MT>(out_grad[grad_row]);
  }
  for (int offset = kTakeAlongAxisWarpSize / 2; offset > 0; offset /= 2) {
    gradient +=
        backends::gpu::CudaShuffleDownSync(0xffffffff, gradient, offset);
  }

  if (lane_idx == 0) {
    for (int64_t i = num_warp_passes * kTakeAlongAxisWarpSize;
         i < num_duplicates;
         ++i) {
      gradient += static_cast<MT>(out_grad[source_pos[idx + i]]);
    }
    x_grad[dest] = static_cast<T>(static_cast<MT>(x_grad[dest]) + gradient);
  }
}

template <typename T>
constexpr bool kTakeAlongAxisDeterministicSupported =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, phi::dtype::float16> ||
    std::is_same_v<T, phi::dtype::bfloat16>;

// Returns false when follow path is hit:
// 1 .integral types accumulate exactly and gain nothing here,
// 2. the coordinate recovery above needs contiguous inputs,
// 3. CUB refuses to sort more than INT_MAX items.
//
// Note:
// This entire function reimplements IndexPutWithSortKernel's sliceSize==1 (see
// index_elementwise_get_grad_kernel.cu) for the no-broadcast case. If that
// kernel is ever promoted to a public header, this function could delegate to
// it instead -- pending verification that its five kernel-header dependencies
// (arange/contiguous/reshape/transpose/elementwise) don't cause include
// issues in this translation unit, and that its generic index_put path
// doesn't add measurable overhead over this specialization.
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
    int64_t numel = index.numel();
    if (numel == 0) return true;
    if (numel > std::numeric_limits<int>::max()) return false;
    if (out_grad.numel() != numel) return false;
    if (!index.meta().is_contiguous() || !out_grad.meta().is_contiguous()) {
      return false;
    }

    int ndim = static_cast<int>(index.dims().size());
    if (ndim == 0 || ndim > DDim::kMaxRank) return false;
    // The destination offset is built from `index`'s coordinates with only the
    // `axis` one replaced, which requires both tensors to agree on rank.
    if (ndim != static_cast<int>(x_grad->dims().size())) return false;
    if (axis < 0 || axis >= ndim) return false;

    TakeAlongAxisDimArray index_dims;
    TakeAlongAxisDimArray x_grad_strides;
    auto strides = common::stride(x_grad->dims());
    for (int d = 0; d < ndim; ++d) {
      index_dims.data[d] = index.dims()[d];
      x_grad_strides.data[d] = strides[d];
    }

    DenseTensor dest;
    dest.Resize({numel});
    dev_ctx.template Alloc<int64_t>(&dest);
    DenseTensor source_pos;
    source_pos.Resize({numel});
    dev_ctx.template Alloc<int64_t>(&source_pos);

    auto stream = dev_ctx.stream();
    constexpr int kOffsetBlock = 256;
    int64_t offset_grid = (numel + kOffsetBlock - 1) / kOffsetBlock;
    const auto& index_type = index.dtype();
    if (index_type == DataType::INT32) {
      TakeAlongAxisDestOffsetKernel<int32_t>
          <<<offset_grid, kOffsetBlock, 0, stream>>>(index.data<int32_t>(),
                                                     dest.data<int64_t>(),
                                                     source_pos.data<int64_t>(),
                                                     index_dims,
                                                     x_grad_strides,
                                                     ndim,
                                                     axis,
                                                     x.dims()[axis],
                                                     numel);
    } else {
      TakeAlongAxisDestOffsetKernel<int64_t>
          <<<offset_grid, kOffsetBlock, 0, stream>>>(index.data<int64_t>(),
                                                     dest.data<int64_t>(),
                                                     source_pos.data<int64_t>(),
                                                     index_dims,
                                                     x_grad_strides,
                                                     ndim,
                                                     axis,
                                                     x.dims()[axis],
                                                     numel);
    }

    DenseTensor sorted_dest;
    sorted_dest.Resize({numel});
    dev_ctx.template Alloc<int64_t>(&sorted_dest);
    DenseTensor sorted_source_pos;
    sorted_source_pos.Resize({numel});
    dev_ctx.template Alloc<int64_t>(&sorted_source_pos);

    // Destination offsets are non-negative and bounded by the last element of
    // the contiguous `x_grad`, so sorting on fewer bits is safe and faster;
    // this mirrors torch2.12 derives for the same sort.
    int64_t nbits =
        TakeAlongAxisNumBits(static_cast<uint64_t>(x_grad->numel() - 1));
    funcs::RadixSortPairs<int64_t, int64_t>(dev_ctx,
                                            dest.data<int64_t>(),
                                            sorted_dest.data<int64_t>(),
                                            source_pos.data<int64_t>(),
                                            sorted_source_pos.data<int64_t>(),
                                            numel,
                                            false,
                                            0,
                                            nbits);

    dim3 block(kTakeAlongAxisWarpSize, kTakeAlongAxisIndicesPerBlock);
    dim3 grid((numel + kTakeAlongAxisIndicesPerBlock - 1) /
              kTakeAlongAxisIndicesPerBlock);
    TakeAlongAxisScatterAddGroupKernel<T>
        <<<grid, block, 0, stream>>>(sorted_dest.data<int64_t>(),
                                     sorted_source_pos.data<int64_t>(),
                                     out_grad.data<T>(),
                                     x_grad->data<T>(),
                                     numel);
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
  if (FLAGS_use_accuracy_compatible_kernel && FLAGS_cudnn_deterministic &&
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
