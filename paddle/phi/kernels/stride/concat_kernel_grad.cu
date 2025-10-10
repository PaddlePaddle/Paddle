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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include <limits>
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/concat_grad_kernel.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "paddle/phi/kernels/funcs/strided_utils.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/phi/kernels/stride/elementwise_stride_base.cu.h"

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/dims_simplifier.h"

#endif

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

template <typename Context>
void NarrowStrideKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const int64_t& dim,
                        const int64_t& begin,
                        const int64_t& length,
                        DenseTensor* out) {
  auto cur_size = dim;
  auto start = begin;
  if (start < 0) {
    start = start + cur_size;
  }

  // auto start_val = begin;
  // auto end_val = start + length;

  // const auto& ndim = x.dims().size();
  // auto sizes = x.dims();
  // auto strides = x.strides();

  // if (start_val < 0) {
  //   start_val += sizes[dim];
  // }
  // if (end_val < 0) {
  //   end_val += sizes[dim];
  // }
  // if (start_val < 0) {
  //   start_val = 0;
  // } else if (start_val >= sizes[dim]) {
  //   start_val = sizes[dim];
  // }
  // if (end_val < start_val) {
  //   end_val = start_val;
  // } else if (end_val >= sizes[dim]) {
  //   end_val = sizes[dim];
  // }

  // int64_t storage_offset = static_cast<int64_t>(x.offset()) + start_val *
  // strides[dim]; auto len = end_val - start_val; sizes[dim] = len;

  // printf("slice as_stride shape\n");
  // for (int i=0; i<sizes.size(); i++) {
  //   printf("%ld ", sizes[i]);
  // }
  // printf("\n");

  // printf("slice as_stride stride\n");
  // for (int i=0; i<strides.size(); i++) {
  //   printf("%ld ", strides[i]);
  // }
  // printf("\n");

  // auto meta = out->meta();
  // meta.offset = storage_offset;
  // meta.dims = sizes;
  // meta.strides = strides;
  // out->set_meta(meta);
  // out->ResetHolder(x.Holder());
  // out->ShareInplaceVersionCounterWith(x);
  // printf("after meta set\n");

  SliceStridedKernel<Context>(dev_ctx,
                              x,
                              {dim},
                              IntArray({start}),
                              IntArray({start + length}),
                              {},
                              {},
                              out);
}

template <typename T, typename Context>
void ConcatGradStrideKernel(const Context& dev_ctx,
                            const std::vector<const DenseTensor*>& x,
                            const DenseTensor& out_grad,
                            const Scalar& axis_scalar,
                            std::vector<DenseTensor*> x_grad) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  // printf("x shape\n");
  // for (int i = 0; i < x.size(); i++) {
  //   auto& shape = x[i]->dims();
  //   for (int j = 0; j < shape.size(); j++) {
  //     printf("%d ", shape[j]);
  //   }
  //   printf("\n");
  // }

  bool invalid_stride = false;

  for (int i = 0; i < x.size(); i++) {
    if (x[i]) {
      if (IsComplexType(x[i]->dtype())) {
        // printf("x is complex\n");
        invalid_stride = true;
        break;
      }
      if (x[i]->numel() == 0) {
        invalid_stride = true;
        break;
      }
    }
  }

  if (!FLAGS_use_stride_compute_kernel || invalid_stride) {
    // printf("enter common concat grad\n");
    DenseTensor out_grad_;
    if (!out_grad.meta().is_contiguous()) {
      out_grad_ = Tensor2Contiguous<Context>(dev_ctx, out_grad);
    } else {
      out_grad_ = out_grad;
    }
    ConcatGradKernel<T, Context>(dev_ctx, x, out_grad_, axis_scalar, x_grad);
    return;
  }

  // printf("enter stride concat grad\n");

  auto outs = x_grad;
  {
    auto dx = x_grad;
    for (size_t i = 0; i < dx.size(); ++i) {
      if (dx[i] != nullptr) {
        dx[i]->set_lod(x[i]->lod());
      }
    }
  }
  PADDLE_ENFORCE_NOT_NULL(
      x[0],
      common::errors::NotFound("The first input tensor is not initialized."));

  auto axis = axis_scalar.to<int>();
  axis = funcs::ComputeAxis(static_cast<int64_t>(axis),
                            static_cast<int64_t>(x[0]->dims().size()));
  // printf("axis:%d\n", axis);
  // get output tensor that the name is not kEmptyVarName
  std::vector<DenseTensor*> outputs;
  // if the out_grad.numel() == 0 ,the all x and x_grad must be zero size
  // tensor, so just return
  if (out_grad.numel() == 0) {
    return;
  }

  int64_t accumulate = 0;
  // printf("enter here\n");
  for (int i = 0; i < x.size(); i++) {
    // printf("out grad:%d\n", i);
    auto& shape = x[i]->dims();
    // for (int j = 0; j < shape.size(); j++) {
    //   printf("%d ", shape[j]);
    // }
    // printf("\n");
    const auto& size = shape[axis];
    // printf("size:%d\n", size);
    accumulate += size;
    // printf("accumulate:%d\n", accumulate);
    // printf("axis:%d\n", axis);
    // printf("accumulate - size:%d\n", accumulate - size);
    if (outs[i]) {
      NarrowStrideKernel<Context>(
          dev_ctx, out_grad, axis, accumulate - size, size, outs[i]);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(concat_grad,
                   GPU,
                   STRIDED,
                   phi::ConcatGradStrideKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2,
                   phi::complex64,
                   phi::complex128) {}

#endif
