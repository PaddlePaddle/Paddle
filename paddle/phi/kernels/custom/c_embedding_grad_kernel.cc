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

#include "paddle/phi/kernels/c_embedding_grad_kernel.h"
#include "glog/logging.h"
#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template <typename T, typename Context>
void CEmbeddingGradKernel(const Context& dev_ctx,
                          const DenseTensor& w,
                          const DenseTensor& ids,
                          const DenseTensor& out_grad,
                          int64_t start_index,
                          DenseTensor* w_grad) {
  w_grad->Resize(w.dims());
  dev_ctx.template Alloc(w_grad, w.dtype());
  const auto& index_type = ids.dtype();
  if (index_type == phi::DataType::INT32 ||
      index_type == phi::DataType::INT64) {
    auto K = ids.numel();
    auto N = w.dims()[0];
    auto D = w.dims()[1];

    auto x_tmp = std::make_shared<phi::DenseTensor>();
    x_tmp->ShareDataWith(ids).Resize({K});
    auto w_tmp = std::make_shared<phi::DenseTensor>();
    w_tmp->set_meta(w.meta());
    dev_ctx.Alloc(w_tmp.get(), w_tmp->dtype());
    auto out_grad_tmp = std::make_shared<phi::DenseTensor>();
    out_grad_tmp->ShareDataWith(out_grad).Resize({K, D});
    paddle::Tensor x_tensor(x_tmp), w_tensor(w_tmp),
        out_grad_tensor(out_grad_tmp);

    auto start_index_tensor = paddle::experimental::full_like(
        x_tensor, start_index, x_tensor.dtype(), x_tensor.place());
    auto end_index_tensor = paddle::experimental::full_like(
        x_tensor, start_index + N, x_tensor.dtype(), x_tensor.place());
    auto ids_mask_tensor = paddle::experimental::logical_and(
        x_tensor.greater_equal(start_index_tensor),
        x_tensor.less_than(end_index_tensor));
    auto real_ids_tensor = (x_tensor - start_index_tensor)
                               .multiply(paddle::experimental::cast(
                                   ids_mask_tensor, x_tensor.dtype()));
    auto out_grad_tensor_mul_mask =
        paddle::experimental::reshape(out_grad_tensor, {K, D})
            .multiply(paddle::experimental::reshape(
                paddle::experimental::cast(ids_mask_tensor, w.dtype()),
                {K, 1}));
    paddle::Tensor w_grad_tensor;
    paddle::experimental::embedding_grad(real_ids_tensor,
                                         w_tensor,
                                         out_grad_tensor_mul_mask,
                                         -1,
                                         false,
                                         &w_grad_tensor);
    w_grad->ShareDataWith(
        *reinterpret_cast<phi::DenseTensor*>(w_grad_tensor.impl().get()));

  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "Custom Device c_embedding_grad ids only support int32 or int64."));
  }
}
#endif
}  // namespace phi

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(c_embedding_grad,
                   Custom,
                   ALL_LAYOUT,
                   phi::CEmbeddingGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
