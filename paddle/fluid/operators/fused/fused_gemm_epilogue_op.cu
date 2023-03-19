/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_gemm_epilogue_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

#if CUDA_VERSION >= 11060

template <typename DeviceContext, typename T>
class FusedGemmEpilogueKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();

    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* y = ctx.Input<phi::DenseTensor>("Y");
    const phi::DenseTensor* bias = ctx.Input<phi::DenseTensor>("Bias");

    phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");
    phi::DenseTensor* reserve_space =
        ctx.Output<phi::DenseTensor>("ReserveSpace");

    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    std::string activation = ctx.Attr<std::string>("activation");
    dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));

    // (M * K) * (K * N)
    auto x_mat_dims =
        phi::flatten_to_2d(x->dims(), trans_x ? 1 : x->dims().size() - 1);
    int64_t M = trans_x ? x_mat_dims[1] : x_mat_dims[0];
    int64_t K = trans_y ? y->dims()[1] : y->dims()[0];
    int64_t N = trans_y ? y->dims()[0] : y->dims()[1];

    ComputeFusedGemmEpilogueForward<T>(dev_ctx,
                                       x,
                                       y,
                                       bias,
                                       M,
                                       N,
                                       K,
                                       trans_x,
                                       trans_y,
                                       activation,
                                       out,
                                       reserve_space);
  }
};

template <typename DeviceContext, typename T>
class FusedGemmEpilogueGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();

    const phi::DenseTensor* dout = ctx.Input<phi::DenseTensor>("DOut");
    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* y = ctx.Input<phi::DenseTensor>("Y");
    const phi::DenseTensor* reserve_space =
        ctx.Input<phi::DenseTensor>("ReserveSpace");

    phi::DenseTensor* dx = ctx.Output<phi::DenseTensor>("DX");
    phi::DenseTensor* dy = ctx.Output<phi::DenseTensor>("DY");
    phi::DenseTensor* dbias = ctx.Output<phi::DenseTensor>("DBias");

    std::string activation_grad = ctx.Attr<std::string>("activation_grad");

    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    // (M * K) * (K * N)
    auto x_mat_dims =
        phi::flatten_to_2d(x->dims(), trans_x ? 1 : x->dims().size() - 1);
    int64_t M = trans_x ? x_mat_dims[1] : x_mat_dims[0];
    int64_t K = trans_y ? y->dims()[1] : y->dims()[0];
    int64_t N = trans_y ? y->dims()[0] : y->dims()[1];

    ComputeFusedGemmEpilogueBackward<T>(dev_ctx,
                                        dout,
                                        x,
                                        y,
                                        reserve_space,
                                        M,
                                        N,
                                        K,
                                        trans_x,
                                        trans_y,
                                        activation_grad,
                                        dx,
                                        dy,
                                        dbias);
  }
};
#endif

}  // namespace operators
}  // namespace paddle

#if CUDA_VERSION >= 11060
namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fused_gemm_epilogue,
    ops::FusedGemmEpilogueKernel<phi::GPUContext, float>,
    ops::FusedGemmEpilogueKernel<phi::GPUContext, double>,
    ops::FusedGemmEpilogueKernel<phi::GPUContext, paddle::platform::float16>,
    ops::FusedGemmEpilogueKernel<phi::GPUContext, paddle::platform::bfloat16>);

REGISTER_OP_CUDA_KERNEL(
    fused_gemm_epilogue_grad,
    ops::FusedGemmEpilogueGradKernel<phi::GPUContext, float>,
    ops::FusedGemmEpilogueGradKernel<phi::GPUContext, double>,
    ops::FusedGemmEpilogueGradKernel<phi::GPUContext,
                                     paddle::platform::float16>,
    ops::FusedGemmEpilogueGradKernel<phi::GPUContext,
                                     paddle::platform::bfloat16>);
#endif
