// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_runtime.h>
#include <paddle/fluid/platform/device_context.h>
#include <algorithm>
#include <cub/cub.cuh>  // NOLINT
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/bert_encoder_functor.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class EmbeddingEltWiseLayerNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    using Tensor = framework::Tensor;
    auto *word_id = context.Input<framework::Tensor>("WordId");
    auto *pos_id = context.Input<framework::Tensor>("PosId");
    auto *sent_id = context.Input<framework::Tensor>("SentId");

    auto *word_emb = context.Input<framework::Tensor>("WordEmb");
    auto *pos_emb = context.Input<framework::Tensor>("PosEmb");
    auto *sent_emb = context.Input<framework::Tensor>("SentEmb");

    auto *bias = context.Input<framework::Tensor>("Bias");
    auto *scale = context.Input<framework::Tensor>("Scale");
    auto *out = context.Output<framework::Tensor>("Out");

    auto *word_id_d = word_id->data<int64_t>();
    auto *pos_id_d = pos_id->data<int64_t>();
    auto *sent_id_d = sent_id->data<int64_t>();

    auto *word_emb_d = word_emb->data<T>();
    auto *pos_emb_d = pos_emb->data<T>();
    auto *sent_emb_d = sent_emb->data<T>();

    auto *bias_d = bias->data<T>();
    auto *scale_d = scale->data<T>();
    auto *output_d = out->mutable_data<T>(context.GetPlace());
    // compute q*k with eltadd
    auto &device_ctx = context.template device_context<DeviceContext>();
    float eps = context.Attr<float>("epsilon");

    // should be (B * S * hidden)
    auto word_id_dims = word_id->dims();
    auto word_emb_dims = word_emb->dims();

    int batch = word_id_dims[0];
    int seq_len = word_id_dims[1];
    int hidden = word_emb_dims[1];

    math::EmbEltwiseLayerNormFunctor<T> emb_eltwise_layernorm_func;
    emb_eltwise_layernorm_func(
        batch, seq_len, hidden, word_id_d, pos_id_d, sent_id_d, scale_d, bias_d,
        word_emb_d, pos_emb_d, sent_emb_d, output_d, eps, device_ctx.stream());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_embedding_eltwise_layernorm,
                        ops::EmbeddingEltWiseLayerNormKernel<
                            paddle::platform::CUDADeviceContext, float>);
