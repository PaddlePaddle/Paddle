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

#include "paddle/phi/kernels/repeat_interleave_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/cpu/index_select_impl.h"
#include "paddle/phi/kernels/funcs/repeat_tensor2index_tensor.h"

namespace phi {

template <typename T, typename Context>
void RepeatInterleaveWithTensorIndexGradKernel(
    const Context& ctx,
    const DenseTensor& x,
    const DenseTensor& repeats_tensor,
    const DenseTensor& out_grad,
    int dim,
    DenseTensor* x_grad) {
  auto input_dim = x_grad->dims();
  if (dim < 0) {
    dim += input_dim.size();
  }

  DenseTensor index;
  PADDLE_ENFORCE_EQ(repeats_tensor.dims()[0] == x_grad->dims()[dim],
                    true,
                    phi::errors::InvalidArgument(
                        "The length of Input(RepeatsTensor) must be the "
                        "same as length of Input(X) in axis. "
                        "But received: [%s], required: [%d].",
                        repeats_tensor.dims()[0],
                        x_grad->dims()[dim]));

  const auto& index_type =
      paddle::framework::TransToProtoVarType(repeats_tensor.dtype());

  bool index_type_match =
      index_type == paddle::framework::proto::VarType::INT32 ||
      index_type == paddle::framework::proto::VarType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Input(Repeats) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        paddle::framework::DataTypeToString(index_type),
                        paddle::framework::DataTypeToString(
                            paddle::framework::proto::VarType::INT32),
                        paddle::framework::DataTypeToString(
                            paddle::framework::proto::VarType::INT64)));

  paddle::platform::DeviceContextPool::Instance().Get(repeats_tensor.place());
  if (index_type == paddle::framework::proto::VarType::INT32) {
    phi::funcs::RepeatsTensor2IndexTensor<Context, int>(
        ctx, repeats_tensor, &index);
    IndexSelectGradInner<Context, T, int>(ctx, out_grad, index, x_grad, dim);
  } else if (index_type == paddle::framework::proto::VarType::INT64) {
    phi::funcs::RepeatsTensor2IndexTensor<Context, int64_t>(
        ctx, repeats_tensor, &index);
    IndexSelectGradInner<Context, T, int64_t>(
        ctx, out_grad, index, x_grad, dim);
  }
}

template <typename T, typename Context>
void RepeatInterleaveGradKernel(const Context& ctx,
                                const DenseTensor& x,
                                const DenseTensor& out_grad,
                                int repeats,
                                int dim,
                                DenseTensor* x_grad) {
  auto input_dim = x_grad->dims();
  if (dim < 0) {
    dim += input_dim.size();
  }

  DenseTensor index;
  int64_t index_size = x_grad->dims()[dim] * repeats;
  std::vector<int> index_vec(index_size);
  for (int i = 0; i < x_grad->dims()[dim]; i++) {
    std::fill_n(index_vec.begin() + i * repeats, repeats, i);
  }
  index.Resize(phi::make_ddim({index_size}));
  paddle::framework::TensorFromVector<int>(index_vec, &index);
  const DenseTensor index_copy = index;
  IndexSelectGradInner<Context, T, int>(ctx, out_grad, index_copy, x_grad, dim);
}
}  // namespace phi

PD_REGISTER_KERNEL(repeat_interleave_with_tensor_index_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveWithTensorIndexGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(repeat_interleave_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
