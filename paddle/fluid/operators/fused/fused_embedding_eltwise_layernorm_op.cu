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

#include <paddle/fluid/platform/device_context.h>
#include <algorithm>
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/math/bert_encoder_functor.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class EmbeddingEltWiseLayerNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    using Tensor = framework::Tensor;
    auto &device_ctx = context.template device_context<DeviceContext>();
    auto ids = context.MultiInput<framework::Tensor>("Ids");
    auto embs = context.MultiInput<framework::Tensor>("Embs");
    int input_num = static_cast<int>(ids.size());

    framework::Tensor in_ids_(
        framework::TransToPtenDataType(framework::proto::VarType::INT64)),
        in_embs_(
            framework::TransToPtenDataType(framework::proto::VarType::INT64));
    framework::DDim in_dim{input_num};
    int device_id;
#ifdef PADDLE_WITH_HIP
    hipGetDevice(&device_id);
#else
    cudaGetDevice(&device_id);
#endif
    in_ids_.Resize(in_dim);
    in_embs_.Resize(in_dim);
    int64_t *in_ids_d =
        in_ids_.mutable_data<int64_t>(platform::CUDAPlace(device_id));
    int64_t *in_embs_d =
        in_embs_.mutable_data<int64_t>(platform::CUDAPlace(device_id));

    std::vector<int64_t> in1s, in2s;
    for (int i = 0; i < input_num; ++i) {
      in1s.push_back(reinterpret_cast<uintptr_t>(ids[i]->data<int64_t>()));
      in2s.push_back(reinterpret_cast<uintptr_t>(embs[i]->data<T>()));
    }
#ifdef PADDLE_WITH_HIP
    hipMemcpyAsync(in_ids_d, in1s.data(), sizeof(int64_t) * input_num,
                   hipMemcpyHostToDevice, device_ctx.stream());
    hipMemcpyAsync(in_embs_d, in2s.data(), sizeof(int64_t) * input_num,
                   hipMemcpyHostToDevice, device_ctx.stream());
#else
    cudaMemcpyAsync(in_ids_d, in1s.data(), sizeof(int64_t) * input_num,
                    cudaMemcpyHostToDevice, device_ctx.stream());
    cudaMemcpyAsync(in_embs_d, in2s.data(), sizeof(int64_t) * input_num,
                    cudaMemcpyHostToDevice, device_ctx.stream());
#endif

    auto *bias = context.Input<framework::Tensor>("Bias");
    auto *scale = context.Input<framework::Tensor>("Scale");
    auto *out = context.Output<framework::Tensor>("Out");

    // should be (B * S * hidden)
    auto id0_dims = ids[0]->dims();
    auto emb0_dims = embs[0]->dims();

    int batch = id0_dims[0];
    int seq_len = id0_dims[1];
    int hidden = emb0_dims[1];

    auto *bias_d = bias->data<T>();
    auto *scale_d = scale->data<T>();
    auto *output_d = out->mutable_data<T>(context.GetPlace());
    float eps = context.Attr<float>("epsilon");

    int shared_bytes = input_num * sizeof(int64_t);
    math::EmbEltwiseLayerNormFunctor<T> emb_eltwise_layernorm_func;
    emb_eltwise_layernorm_func(batch, seq_len, hidden, in_ids_d, scale_d,
                               bias_d, in_embs_d, output_d, eps, input_num,
                               device_ctx.stream());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_embedding_eltwise_layernorm,
                        ops::EmbeddingEltWiseLayerNormKernel<
                            paddle::platform::CUDADeviceContext, float>);
