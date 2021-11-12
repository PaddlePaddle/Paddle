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
#include "paddle/pten/kernels/functions/math/cast_func.h"

namespace pten {

template <typename T>
void Flatten(const CUDAContext& dev_ctx,
             const DenseTensor& x,
             int start_axis,
             int stop_axis,
             DenseTensor* out) {
  auto out_dims = out->dims();
  pten::Copy(dev_ctx, x, out);
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
  const auto& in_dims = x.meta().dims;
  std::vector<int64_t> xshape_dims(in_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    xshape_dims[i + 1] = in_dims[i];
  }
  xshape->Resize(paddle::framework::make_ddim(xshape_dims));
  xshape->set_lod(x.lod());
}

template <typename T>
void Cast(const CUDAContext& dev_ctx,
          const DenseTensor& x,
          DataType out_dtype,
          DataType in_dtype,
          DenseTensor* out) {
  PTEN_DISPATCH_ALL_TYPES(out_dtype, "CastKernelImpl", ([&] {
                            math::CastKernelImpl<CUDAContext, T, data_t>(
                                dev_ctx, x, out);
                          }));
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
// todo: Hip need support bfloat16
PT_REGISTER_KERNEL("cast",
                   CUDA,
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
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}
