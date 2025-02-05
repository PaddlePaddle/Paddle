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

#include <algorithm>
#include <type_traits>

#include "paddle/common/errors.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/emb_eltwise_layer_norm_functor.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void EmbeddingEltWiseLayerNormKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& ids,
    const std::vector<const DenseTensor*>& embs,
    const DenseTensor& bias,
    const DenseTensor& scale,
    const float epsilon,
    DenseTensor* out) {
  PADDLE_ENFORCE_GE(
      epsilon,
      0.0f,
      common::errors::InvalidArgument(
          "'epsilon' is %f, but it should be between 0.0 and 0.001", epsilon));
  PADDLE_ENFORCE_LE(
      epsilon,
      0.001f,
      common::errors::InvalidArgument(
          "'epsilon' is %f, but it should be between 0.0 and 0.001.", epsilon));
  int input_num = static_cast<int>(ids.size());

  DenseTensor in_ids_(phi::DataType::INT64), in_embs_(phi::DataType::INT64);
  DDim in_dim{input_num};

  in_ids_.Resize(in_dim);
  in_embs_.Resize(in_dim);

  int64_t* in_ids_d = dev_ctx.template Alloc<int64_t>(
      &in_ids_, in_ids_.numel() * sizeof(int64_t));
  int64_t* in_embs_d = dev_ctx.template Alloc<int64_t>(
      &in_embs_, in_embs_.numel() * sizeof(int64_t));

  std::vector<int64_t> in1s, in2s;
  for (int i = 0; i < input_num; ++i) {
    in1s.push_back(reinterpret_cast<uintptr_t>(ids[i]->data<int64_t>()));
    in2s.push_back(reinterpret_cast<uintptr_t>(embs[i]->data<T>()));
  }

  phi::memory_utils::Copy(phi::GPUPlace{},
                          in_ids_d,
                          phi::CPUPlace{},
                          in1s.data(),
                          sizeof(int64_t) * input_num,
                          dev_ctx.stream());
  phi::memory_utils::Copy(phi::GPUPlace{},
                          in_embs_d,
                          phi::CPUPlace{},
                          in2s.data(),
                          sizeof(int64_t) * input_num,
                          dev_ctx.stream());

  // should be (B * S * hidden)
  auto id0_dims = ids[0]->dims();
  auto emb0_dims = embs[0]->dims();

  int batch = id0_dims[0];
  int seq_len = id0_dims[1];
  int hidden = emb0_dims[1];

  auto* bias_d = bias.data<T>();
  auto* scale_d = scale.data<T>();
  auto* output_d = dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));

  if (std::is_same<T, phi::dtype::float16>::value) {
    const half* scale_new = reinterpret_cast<const half*>(scale_d);
    const half* bias_new = reinterpret_cast<const half*>(bias_d);
    half* output_new = reinterpret_cast<half*>(output_d);

    phi::funcs::EmbEltwiseLayerNormFunctor<half> emb_eltwise_layernorm_func;
    emb_eltwise_layernorm_func(batch,
                               seq_len,
                               hidden,
                               in_ids_d,
                               scale_new,
                               bias_new,
                               in_embs_d,
                               output_new,
                               epsilon,
                               input_num,
                               dev_ctx.stream());
  } else {
    phi::funcs::EmbEltwiseLayerNormFunctor<T> emb_eltwise_layernorm_func;
    emb_eltwise_layernorm_func(batch,
                               seq_len,
                               hidden,
                               in_ids_d,
                               scale_d,
                               bias_d,
                               in_embs_d,
                               output_d,
                               epsilon,
                               input_num,
                               dev_ctx.stream());
  }
}

}  // namespace fusion
}  // namespace phi

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 10000
PD_REGISTER_KERNEL(fused_embedding_eltwise_layernorm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::EmbeddingEltWiseLayerNormKernel,
                   float,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(fused_embedding_eltwise_layernorm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::EmbeddingEltWiseLayerNormKernel,
                   float) {}
#endif
