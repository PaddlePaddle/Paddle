//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/cuda/manipulation.h"
#include "paddle/pten/kernels/cuda/utils.h"
#include "paddle/pten/kernels/hybird/cuda/cast_kernel_impl.h"
#include "paddle/pten/kernels/hybird/general/manipulation.h"

namespace pten {

template <typename T>
void Flatten(const CUDAContext& dev_ctx,
             const DenseTensor& x,
             int start_axis,
             int stop_axis,
             DenseTensor* out) {
  auto out_dims = out->dims();
  pten::Copy(dev_ctx, x, false, out);
  out->Resize(out_dims);
}

// TODO(yuanrisheng): this kernel is for training and xshape is a Intermediate
// Output Tensor，
// is there a more flexible way to deal with this case?
template <typename T>
void FlattenWithXShape(const CUDAContext& dev_ctx,
                       const DenseTensor& x,
                       int start_axis,
                       int stop_axis,
                       DenseTensor* out,
                       DenseTensor* xshape) {
  Flatten<T>(dev_ctx, x, start_axis, stop_axis, out);
  general::SetXShape(x, xshape);
}

void Reshape(const CUDAContext& dev_ctx,
             const DenseTensor& x,
             const ScalarArray& shape,
             DenseTensor* out) {
  auto out_meta = InferMetaFromVecValue(x.meta(), shape.GetData());
  if (x.data() == out->data() && x.numel() == out->numel()) {
    out->Resize(out_meta.dims);
    return;
  }
  pten::Copy(dev_ctx, x, false, out);
  out->Resize(out_meta.dims);
  out->ResetLoD(x.lod());
}

void ReshapeWithXShape(const CUDAContext& dev_ctx,
                       const DenseTensor& x,
                       const ScalarArray& shape,
                       DenseTensor* xshape,
                       DenseTensor* out) {
  general::SetXShape(x, xshape);
  Reshape(dev_ctx, x, shape, out);
}

template <typename T>
void Cast(const CUDAContext& dev_ctx,
          const DenseTensor& x,
          DataType out_dtype,
          DataType in_dtype,
          DenseTensor* out) {
  PD_VISIT_ALL_TYPES(out_dtype, "CastKernelImpl", ([&] {
                       detail::CastCUDAKernelImpl<T, data_t>(dev_ctx, x, out);
                     }));
}

}  // namespace pten

using float16 = paddle::platform::float16;

PT_REGISTER_KERNEL(flatten,
                   CUDA,
                   ALL_LAYOUT,
                   pten::Flatten,
                   float,
                   float16,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}
PT_REGISTER_KERNEL(flatten_with_xshape,
                   CUDA,
                   ALL_LAYOUT,
                   pten::FlattenWithXShape,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}

#define PTEN_REGISTER_CAST_CUDA_BASE_TYPE(op_name, ...) \
  PT_REGISTER_KERNEL(cast,                              \
                     CUDA,                              \
                     ALL_LAYOUT,                        \
                     pten::Cast,                        \
                     float,                             \
                     double,                            \
                     int,                               \
                     int64_t,                           \
                     int16_t,                           \
                     bool,                              \
                     uint8_t,                           \
                     paddle::platform::float16,         \
                     paddle::platform::complex<float>,  \
                     paddle::platform::complex<double>, \
                     ##__VA_ARGS__) {                   \
    kernel->OutputAt(0).SetDataType(                    \
        paddle::experimental::DataType::UNDEFINED);     \
  }

#if !defined(PADDLE_WITH_HIP)
PTEN_REGISTER_CAST_CUDA_BASE_TYPE(cast, paddle::platform::bfloat16)
#else
PTEN_REGISTER_CAST_CUDA_BASE_TYPE(cast)
#endif

PT_REGISTER_NO_TEMPLATE_KERNEL(reshape, CUDA, ANY, pten::Reshape, ALL_DTYPE) {}
PT_REGISTER_NO_TEMPLATE_KERNEL(
    reshape_with_xshape, CUDA, ANY, pten::ReshapeWithXShape, ALL_DTYPE) {}
