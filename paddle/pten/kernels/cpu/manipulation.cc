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

#include "paddle/pten/kernels/cpu/manipulation.h"
#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/cpu/utils.h"
#include "paddle/pten/kernels/hybird/general/manipulation.h"
#include "paddle/pten/kernels/hybird/math/cast_func.h"

namespace pten {

template <typename T>
void Flatten(const CPUContext& dev_ctx,
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
void FlattenWithXShape(const CPUContext& dev_ctx,
                       const DenseTensor& x,
                       int start_axis,
                       int stop_axis,
                       DenseTensor* out,
                       DenseTensor* xshape) {
  Flatten<T>(dev_ctx, x, start_axis, stop_axis, out);
  general::SetXShape(x, xshape);
}

void ReshapeFromVectorVal(const CPUContext& dev_ctx,
                          const DenseTensor& x,
                          const std::vector<int64_t>& shape,
                          DenseTensor* out) {
  auto out_meta = InferMetaFromVecValue(x.meta(), shape);
  if (x.data() == out->data() && x.numel() == out->numel()) {
    out->Resize(out_meta.dims);
    return;
  }
  pten::Copy(dev_ctx, x, false, out);
  out->Resize(out_meta.dims);
}

void ReshapeFromVectorValWithXShape(const CPUContext& dev_ctx,
                                    const DenseTensor& x,
                                    const std::vector<int64_t>& shape,
                                    DenseTensor* xshape,
                                    DenseTensor* out) {
  general::SetXShape(x, xshape);
  ReshapeFromVectorVal(dev_ctx, x, shape, out);
}

void ReshapeFromDT(const CPUContext& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& shape,
                   DenseTensor* out) {
  auto* shape_data = shape.data<int>();
  auto vector_shape =
      std::vector<int64_t>(shape_data, shape_data + shape.numel());
  ReshapeFromVectorVal(dev_ctx, x, vector_shape, out);
  out->ResetLoD(x.lod());
}

void ReshapeFromDTWithXShape(const CPUContext& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& shape,
                             DenseTensor* xshape,
                             DenseTensor* out) {
  general::SetXShape(x, xshape);
  ReshapeFromDT(dev_ctx, x, shape, out);
}

void ReshapeFromVectorDT(const CPUContext& dev_ctx,
                         const DenseTensor& x,
                         const std::vector<DenseTensor>& shape,
                         DenseTensor* out) {
  std::vector<int64_t> vector_shape;
  for (auto& tensor : shape) {
    PADDLE_ENFORCE_EQ(
        tensor.dims(),
        paddle::framework::make_ddim({1}),
        paddle::platform::errors::InvalidArgument(
            "If the element type of 'shape' in ReshapeOp is Tensor, "
            "the element's shape must be [1]. But received the element's shape "
            "is [%s]",
            tensor.dims()));
    vector_shape.push_back(*tensor.data<int32_t>());
  }
  ReshapeFromVectorVal(dev_ctx, x, vector_shape, out);
}

void ReshapeFromVectorDTWithXShape(const CPUContext& dev_ctx,
                                   const DenseTensor& x,
                                   const std::vector<DenseTensor>& shape,
                                   DenseTensor* xshape,
                                   DenseTensor* out) {
  general::SetXShape(x, xshape);
  ReshapeFromVectorDT(dev_ctx, x, shape, out);
}

template <typename T>
void Cast(const CPUContext& dev_ctx,
          const DenseTensor& x,
          DataType out_dtype,
          DataType in_dtype,
          DenseTensor* out) {
  PD_VISIT_ALL_TYPES(out_dtype, "CastKernelImpl", ([&] {
                       math::CastKernelImpl<CPUContext, T, data_t>(
                           dev_ctx, x, out);
                     }));
}

}  // namespace pten

// TODO(yuanrisheng): "flatten_contiguous_range" is compatible with old kernel
// architecture, kernel_name should be "flatten".
PT_REGISTER_KERNEL(flatten,
                   CPU,
                   ANY,
                   pten::Flatten,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}
PT_REGISTER_KERNEL(flatten_mid,
                   CPU,
                   ANY,
                   pten::FlattenWithXShape,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}

PT_REGISTER_KERNEL(cast,
                   CPU,
                   ANY,
                   pten::Cast,
                   float,
                   double,
                   int,
                   int64_t,
                   int16_t,
                   bool,
                   uint8_t,
                   paddle::platform::float16,
                   paddle::platform::bfloat16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}

PT_REGISTER_KERNEL_ALL_DTYPE(reshape, CPU, ANY, pten::ReshapeFromVectorVal) {}
PT_REGISTER_KERNEL_ALL_DTYPE(reshape_mid,
                             CPU,
                             ANY,
                             pten::ReshapeFromVectorValWithXShape) {}
PT_REGISTER_KERNEL_ALL_DTYPE(reshape_host, CPU, ANY, pten::ReshapeFromDT) {
  kernel->InputAt(1).SetBackend(pten::Backend::CPU);
  kernel->InputAt(1).SetDataType(paddle::experimental::DataType::INT32);
}
PT_REGISTER_KERNEL_ALL_DTYPE(reshape_host_mid,
                             CPU,
                             ANY,
                             pten::ReshapeFromDTWithXShape) {
  kernel->InputAt(1).SetBackend(pten::Backend::CPU);
  kernel->InputAt(1).SetDataType(paddle::experimental::DataType::INT32);
}
PT_REGISTER_KERNEL_ALL_DTYPE(reshape_mulhost,
                             CPU,
                             ANY,
                             pten::ReshapeFromVectorDT) {
  kernel->InputAt(1).SetBackend(pten::Backend::CPU);
  kernel->InputAt(1).SetDataType(paddle::experimental::DataType::INT32);
}
PT_REGISTER_KERNEL_ALL_DTYPE(reshape_mulhost_mid,
                             CPU,
                             ANY,
                             pten::ReshapeFromVectorDTWithXShape) {
  kernel->InputAt(1).SetBackend(pten::Backend::CPU);
  kernel->InputAt(1).SetDataType(paddle::experimental::DataType::INT32);
}
