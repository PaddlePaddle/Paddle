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

#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/cuda/manipulation.h"
#include "paddle/pten/kernels/cuda/utils.h"
#include "paddle/pten/kernels/functions/general/manipulation.h"

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

void ReshapeFromVectorValImpl(const CUDAContext& dev_ctx,
                              const DenseTensor& x,
                              const std::vector<int64_t>& shape,
                              DenseTensor* out,
                              bool set_lod) {
  auto out_meta = InferShapeFromVecValue(x.meta(), shape);
  if (&x != out) {
    pten::Copy(dev_ctx, x, false, out);
  }
  if (set_lod) {
    out->Resize(out_meta.dims, out_meta.lod);
  } else {
    out->Resize(out_meta.dims);
  }
}

void ReshapeFromVectorVal(const CUDAContext& dev_ctx,
                          const DenseTensor& x,
                          const std::vector<int64_t>& shape,
                          DenseTensor* out) {
  ReshapeFromVectorValImpl(dev_ctx, x, shape, out, false);
}

void ReshapeFromVectorValWithXShape(const CUDAContext& dev_ctx,
                                    const DenseTensor& x,
                                    const std::vector<int64_t>& shape,
                                    DenseTensor* xshape,
                                    DenseTensor* out) {
  ReshapeFromVectorVal(dev_ctx, x, shape, out);
  general::SetXShape(x, xshape);
}

void ReshapeFromDT(const CUDAContext& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& shape,
                   DenseTensor* out) {
  auto* shape_data = shape.data<int>();
  auto vector_shape =
      std::vector<int64_t>(shape_data, shape_data + shape.numel());
  ReshapeFromVectorValImpl(dev_ctx, x, vector_shape, out, true);
}

void ReshapeFromDTWithXShape(const CUDAContext& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& shape,
                             DenseTensor* xshape,
                             DenseTensor* out) {
  ReshapeFromDT(dev_ctx, x, shape, out);
  general::SetXShape(x, xshape);
}

void ReshapeFromVectorDT(const CUDAContext& dev_ctx,
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

void ReshapeFromVectorDTWithXShape(const CUDAContext& dev_ctx,
                                   const DenseTensor& x,
                                   const std::vector<DenseTensor>& shape,
                                   DenseTensor* xshape,
                                   DenseTensor* out) {
  ReshapeFromVectorDT(dev_ctx, x, shape, out);
  general::SetXShape(x, xshape);
}

}  // namespace pten

// TODO(chenweihang): replace by better impl
PT_REGISTER_MODULE(ManipulationCUDA);

using float16 = paddle::platform::float16;
// TODO(yuanrisheng): "flatten_contiguous_range" is compatible with old kernel
// architecture, kernel_name should be "flatten".
PT_REGISTER_KERNEL("flatten_contiguous_range",
                   CUDA,
                   ANY,
                   pten::Flatten,
                   float,
                   float16,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}

PT_REGISTER_KERNEL("flatten_contiguous_range.mid",
                   CUDA,
                   ANY,
                   pten::FlattenWithXShape,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}

PT_REGISTER_KERNEL_WITH_NO_TYPE("reshape2",
                                CUDA,
                                ANY,
                                pten::ReshapeFromVectorVal) {}

PT_REGISTER_KERNEL_WITH_NO_TYPE("reshape2.mid",
                                CUDA,
                                ANY,
                                pten::ReshapeFromVectorValWithXShape) {}
