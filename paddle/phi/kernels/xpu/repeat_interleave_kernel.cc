// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/repeat_interleave_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/repeat_tensor2index_tensor.h"

namespace phi {
template <typename T, typename Context>
void RepeatInterleaveKernel(const Context& ctx,
                            const DenseTensor& x,
                            int repeats,
                            int dim,
                            DenseTensor* out) {
  PADDLE_ENFORCE_GT(repeats,
                    0,
                    common::errors::InvalidArgument(
                        "repeats must grater than 0, but got %d", repeats));
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto input_dim = x.dims();
  if (dim < 0) {
    dim += input_dim.size();
  }

  DenseTensor index;
  int64_t index_size = input_dim[dim] * repeats;
  std::vector<int> index_vec(index_size);
  for (int i = 0; i < input_dim[dim]; i++) {
    std::fill_n(index_vec.begin() + i * repeats, repeats, i);
  }
  index.Resize(common::make_ddim({index_size}));

  phi::TensorFromVector<int>(index_vec, ctx, &index);
  auto xshape = common::vectorize(input_dim);
  auto out_shape = xshape;
  out_shape[dim] = index_size;
  out->Resize(common::make_ddim(out_shape));
  ctx.template Alloc<T>(out);
  int ret = xpu::paddle_gather<XPUType, int>(
      ctx.x_context(),
      reinterpret_cast<const XPUType*>(x.data<T>()),
      index.data<int>(),
      reinterpret_cast<XPUType*>(out->data<T>()),
      xshape,
      index_size,
      dim);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "paddle_gather");
}

template <typename T, typename Context>
void RepeatInterleaveWithTensorIndexKernel(const Context& ctx,
                                           const DenseTensor& x,
                                           const DenseTensor& repeats_tensor,
                                           int dim,
                                           DenseTensor* out) {
  auto input_dim = x.dims();
  if (dim < 0) {
    dim += input_dim.size();
  }
  using XPUType = typename XPUTypeTrait<T>::Type;
  DenseTensor index;
  PADDLE_ENFORCE_EQ(repeats_tensor.dims()[0] == x.dims()[dim],
                    true,
                    common::errors::InvalidArgument(
                        "The length of Input(RepeatsTensor) must be the "
                        "same as length of Input(X) in axis. "
                        "But received: [%s], required: [%d].",
                        repeats_tensor.dims()[0],
                        x.dims()[dim]));
  const auto& index_type = repeats_tensor.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      common::errors::InvalidArgument(
          "Input(RepeatsTensor) holds the wrong type, it holds %s, but "
          "desires to be %s or %s",
          DataTypeToString(index_type),
          DataTypeToString(phi::DataType::INT32),
          DataTypeToString(phi::DataType::INT64)));
  auto xshape = common::vectorize(x.dims());
  auto out_shape = xshape;
  if (index_type == phi::DataType::INT64) {
    phi::funcs::RepeatsTensor2IndexTensor<Context, int64_t>(
        ctx, repeats_tensor, &index);
    out_shape[dim] = index.dims()[0];
    out->Resize(common::make_ddim(out_shape));
    ctx.template Alloc<T>(out);
    int ret = xpu::paddle_gather<XPUType, int64_t>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        index.data<int64_t>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        xshape,
        index.numel(),
        dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "paddle_gather");
  } else {
    phi::funcs::RepeatsTensor2IndexTensor<Context, int>(
        ctx, repeats_tensor, &index);
    out_shape[dim] = index.dims()[0];
    out->Resize(common::make_ddim(out_shape));
    ctx.template Alloc<T>(out);
    int ret = xpu::paddle_gather<XPUType, int>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        index.data<int>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        xshape,
        index.numel(),
        dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "paddle_gather");
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(repeat_interleave,
                   XPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveKernel,
                   float,
                   int,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(repeat_interleave_with_tensor_index,
                   XPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveWithTensorIndexKernel,
                   float,
                   int,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
