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

#include "paddle/phi/kernels/c_embedding_kernel.h"
#include "glog/logging.h"
#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template <typename T, typename Context>
void CEmbeddingKernel(const Context& dev_ctx,
                      const DenseTensor& w,
                      const DenseTensor& ids,
                      int64_t start_index,
                      int64_t vocab_size,
                      DenseTensor* out) {
  const auto& index_type = ids.dtype();
  if (index_type == phi::DataType::INT32 ||
      index_type == phi::DataType::INT64) {
    auto out_dims = out->dims();
    auto K = ids.numel();
    auto N = w.dims()[0];
    auto D = w.dims()[1];

    auto x_tmp = std::make_shared<phi::DenseTensor>();
    x_tmp->ShareDataWith(ids).Resize({K});
    auto w_tmp = std::make_shared<phi::DenseTensor>();
    w_tmp->ShareDataWith(w).Resize({N, D});
    paddle::Tensor x_tensor(x_tmp), w_tensor(w_tmp);

    auto start_index_tensor = paddle::experimental::full_like(
        x_tensor, start_index, x_tensor.dtype(), x_tensor.place());
    auto end_index_tensor = paddle::experimental::full_like(
        x_tensor, start_index + N, x_tensor.dtype(), x_tensor.place());
    auto ids_mask_tensor = paddle::experimental::logical_and(
        x_tensor.greater_equal(start_index_tensor),
        x_tensor.less_than(end_index_tensor));
    auto ids_tensor = (x_tensor - start_index_tensor)
                          .multiply(paddle::experimental::cast(
                              ids_mask_tensor, x_tensor.dtype()));
    auto out_tensor =
        paddle::experimental::reshape(
            paddle::experimental::cast(ids_mask_tensor, w_tensor.dtype()),
            {K, 1})
            .multiply(paddle::experimental::reshape(
                paddle::experimental::embedding(
                    ids_tensor, w_tensor, -1, false),
                {K, D}));
    out->ShareDataWith(
           *reinterpret_cast<phi::DenseTensor*>(out_tensor.impl().get()))
        .Resize(out_dims);
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "Custom Device c_embedding ids only support int32 or int64."));
  }
}
#endif
}  // namespace phi

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(c_embedding,
                   Custom,
                   ALL_LAYOUT,
                   phi::CEmbeddingKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
