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

#include "paddle/phi/kernels/as_strided_grad_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/as_strided_kernel.h"
#include "paddle/phi/kernels/funcs/strided_grad_utils.h"
#include "paddle/phi/kernels/funcs/strided_utils.h"
#include "paddle/phi/kernels/funcs/strided_view_utils.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void AsStridedGradKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& out_grad,
                         const std::vector<int64_t>& dims,
                         const std::vector<int64_t>& stride,
                         int64_t offset,
                         DenseTensor* input_grad) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  dev_ctx.Alloc(input_grad, input_grad->dtype());
  input_grad->set_strides(DenseTensorMeta::calc_strides(input_grad->dims()));
  PD_VISIT_ALL_TYPES(input_grad->dtype(), "AsStridedGradKernel", ([&] {
                       phi::StridedTensorFill<data_t>(
                           *input_grad, 0, input_grad);
                     }));
  if (out_grad.numel() == 0) {
    return;
  }
  // `offset` is a byte offset into the allocation shared with the forward
  // `input`, but `input_grad` is a dense row-major buffer over `input`'s own
  // logical indices. Subtracting `input.offset()` lines the two up only when
  // `input` is contiguous *and* the whole window of the view lands inside
  // `input`. Otherwise a storage index is not a row-major index; and since the
  // forward validates a view against the shared allocation rather than against
  // the extent of `input`, a window may legally reach outside `input` on either
  // side, where its contributions belong to no element of `input_grad` and have
  // to be dropped instead of written past its end. Everything but the fast case
  // is routed through a buffer laid out in storage coordinates.
  //
  // A view whose stride is shorter than its shape is malformed, but the forward
  // accepts it for a non-empty input, so it has to keep working here as well.
  // Its element range is not well defined and every helper below indexes
  // stride[i] for i < dims.size(), so such a view stays on the original copy
  // path.
  const bool ranked_alike = dims.size() == stride.size();
  bool through_storage = false;
  if (ranked_alike) {
    const int64_t itemsize = static_cast<int64_t>(SizeOf(input_grad->dtype()));
    const int64_t input_base = static_cast<int64_t>(input.offset()) / itemsize;
    const StridedViewRange out_range =
        ComputeStridedViewRange(dims, stride, offset / itemsize);
    const bool inside_input = !out_range.empty &&
                              out_range.min_index >= input_base &&
                              out_range.max_index < input_base + input.numel();
    through_storage = !input.meta().is_contiguous() || !inside_input;
  }
  if (through_storage) {
    PD_VISIT_ALL_TYPES(out_grad.dtype(), "AsStridedGradKernel", ([&] {
                         phi::StridedTensorAccumulateThroughStorage<data_t>(
                             out_grad, dims, stride, offset, input, input_grad);
                       }));
    return;
  }
  const int64_t grad_offset = offset - static_cast<int64_t>(input.offset());
  if (ranked_alike && MaybeOverlappingStrides(dims, stride)) {
    // Several elements of out_grad map to the same storage slot, so the
    // gradient has to be accumulated instead of copied.
    PD_VISIT_ALL_TYPES(out_grad.dtype(), "AsStridedGradKernel", ([&] {
                         phi::StridedTensorAccumulate<data_t>(
                             out_grad, dims, stride, grad_offset, input_grad);
                       }));
    return;
  }
  DenseTensor tmp;
  tmp.set_meta(out_grad.meta());
  AsStridedKernel<Context>(
      dev_ctx, *input_grad, dims, stride, grad_offset, &tmp);
  PD_VISIT_ALL_TYPES(out_grad.dtype(), "AsStridedGradKernel", ([&] {
                       phi::StridedTensorCopy<data_t>(
                           out_grad,
                           vectorize<int64_t>(tmp.dims()),
                           vectorize<int64_t>(tmp.strides()),
                           tmp.offset(),
                           &tmp);
                     }));
}
}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(as_strided_grad,
                                         STRIDED,
                                         phi::AsStridedGradKernel) {}
