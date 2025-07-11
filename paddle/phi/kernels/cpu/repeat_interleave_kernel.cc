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

#include "paddle/phi/kernels/repeat_interleave_kernel.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/cpu/index_select_impl.h"
#include "paddle/phi/kernels/funcs/repeat_tensor2index_tensor.h"

namespace phi {

template <typename T, typename Context>
void RepeatInterleaveKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            int repeats,
                            int dim,
                            DenseTensor* out) {
  PADDLE_ENFORCE_GT(repeats,
                    0,
                    common::errors::InvalidArgument(
                        "repeats must grater than 0, but got %d", repeats));
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

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
  DenseTensor x_copy = x;
  phi::TensorFromVector<int>(index_vec, dev_ctx, &index);

  auto output_dim = common::vectorize(x.dims());
  output_dim[dim] = index_size;
  out->Resize(common::make_ddim(output_dim));
  phi::IndexSelectInner<Context, T, int>(dev_ctx, &x_copy, index, out, dim);
}

template <typename T, typename Context>
void RepeatInterleaveWithTensorIndexKernel(const Context& dev_ctx,
                                           const DenseTensor& x,
                                           const DenseTensor& repeats_tensor,
                                           int dim,
                                           DenseTensor* out) {
  auto input_dim = x.dims();
  if (dim < 0) {
    dim += input_dim.size();
  }
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

  if (x.numel() == 0) {
    // infer out shape
    if (index_type == phi::DataType::INT32) {
      phi::funcs::RepeatsTensor2IndexTensorFunctor<Context, int>()(
          dev_ctx, repeats_tensor, &index);

    } else if (index_type == phi::DataType::INT64) {
      phi::funcs::RepeatsTensor2IndexTensorFunctor<Context, int64_t>()(
          dev_ctx, repeats_tensor, &index);
    }
    auto output_dim = common::vectorize(x.dims());
    output_dim[dim] = index.dims()[0];
    out->Resize(common::make_ddim(output_dim));
    dev_ctx.template Alloc<T>(out);
    return;
  }
  auto x_copy = x;
  if (index_type == phi::DataType::INT32) {
    phi::funcs::RepeatsTensor2IndexTensorFunctor<Context, int>()(
        dev_ctx, repeats_tensor, &index);
    auto output_dim = common::vectorize(x.dims());
    output_dim[dim] = index.dims()[0];
    out->Resize(common::make_ddim(output_dim));
    IndexSelectInner<Context, T, int>(dev_ctx, &x_copy, index, out, dim);
  } else if (index_type == phi::DataType::INT64) {
    phi::funcs::RepeatsTensor2IndexTensorFunctor<Context, int64_t>()(
        dev_ctx, repeats_tensor, &index);
    auto output_dim = common::vectorize(x.dims());
    output_dim[dim] = index.dims()[0];
    out->Resize(common::make_ddim(output_dim));
    IndexSelectInner<Context, T, int64_t>(dev_ctx, &x_copy, index, out, dim);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(repeat_interleave,
                   CPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(repeat_interleave_with_tensor_index,
                   CPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveWithTensorIndexKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
