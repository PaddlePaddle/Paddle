/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include "paddle/fluid/operators/mul_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace framework {
class Tensor;
}  // namespace framework
namespace platform {
class MKLDNNDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::DDim;
using framework::ExecutionContext;
using framework::Tensor;

using platform::MatMulV2MKLDNNHandler;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;

using dnnl::memory;
using dnnl::prop_kind;
using dnnl::stream;


template <typename XT, typename YT>
class MulMKLDNNKernel
    : public framework::OpKernel<XT>{
 public:
  void Compute(const ExecutionContext& ctx) const override { RunKernel(ctx); }

 private:
  template <typename OT = XT>
  void RunKernel(const ExecutionContext& ctx) const {
    const auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<Tensor>("X");
    const auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Output<Tensor>("Out");

    const Tensor x_matrix =
        x->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *x, ctx.template Attr<int>("x_num_col_dims"))
            : *x;
    const Tensor y_matrix =
        y->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *y, ctx.template Attr<int>("y_num_col_dims"))
            : *y;

    // adding mb dim because MatMulV2 handler needs it
    std::vector<int64_t> y_dims(3, 1);
    std::vector<int64_t> x_dims(3, 1);

    y_dims[1] = y_matrix.dims()[0];
    y_dims[2] = y_matrix.dims()[1];

    x_dims[1] = x_matrix.dims()[0];
    x_dims[2] = x_matrix.dims()[1];

    static const std::vector<int64_t> vec_placeholder;

    MatMulV2MKLDNNHandler<XT> handler(onednn_engine, ctx.GetPlace(), x_dims,
                                     false, y_dims, false,
                                     false, vec_placeholder, vec_placeholder);

    const auto src_memory_p = handler.AcquireSrcMemory(&x_matrix);
    const auto weights_memory_p = handler.AcquireWeightsMemory(&y_matrix);
    const auto dst_memory_p = handler.AcquireDstMemory(out);

    auto matmul_p = handler.AcquireForwardPrimitive();

    std::unordered_map<int, dnnl::memory> matmul_args = {
        {DNNL_ARG_SRC, *src_memory_p},
        {DNNL_ARG_WEIGHTS, *weights_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    auto& astream = MKLDNNDeviceContext::tls().get_stream();
    matmul_p->execute(astream, matmul_args);
    astream.wait();

    out->set_layout(paddle::framework::DataLayout::kMKLDNN);
    out->set_format(platform::GetMKLDNNFormat(dst_memory_p->get_desc()));
  }
};

template <typename XT, typename YT>
class MulGradMKLDNNKernel
    : public framework::OpKernel<XT>{
 public:
  void Compute(const ExecutionContext& ctx) const override { RunKernel(ctx); }

 private:
  template <typename OT = XT>
  void RunKernel(const ExecutionContext& ctx) const {
    const auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<Tensor>("X");
    const auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Output<Tensor>("Out");

    const Tensor x_matrix =
        x->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *x, ctx.template Attr<int>("x_num_col_dims"))
            : *x;
    const Tensor y_matrix =
        y->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *y, ctx.template Attr<int>("y_num_col_dims"))
            : *y;

    // adding mb dim because MatMulV2 handler needs it
    std::vector<int64_t> y_dims(3, 1);
    std::vector<int64_t> x_dims(3, 1);

    y_dims[1] = y_matrix.dims()[0];
    y_dims[2] = y_matrix.dims()[1];

    x_dims[1] = x_matrix.dims()[0];
    x_dims[2] = x_matrix.dims()[1];

    static const std::vector<int64_t> vec_placeholder;

    MatMulV2MKLDNNHandler<XT> handler(onednn_engine, ctx.GetPlace(), x_dims,
                                     false, y_dims, false,
                                     false, vec_placeholder, vec_placeholder);

    const auto src_memory_p = handler.AcquireSrcMemory(&x_matrix);
    const auto weights_memory_p = handler.AcquireWeightsMemory(&y_matrix);
    const auto dst_memory_p = handler.AcquireDstMemory(out);

    auto matmul_p = handler.AcquireForwardPrimitive();

    std::unordered_map<int, dnnl::memory> matmul_args = {
        {DNNL_ARG_SRC, *src_memory_p},
        {DNNL_ARG_WEIGHTS, *weights_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    auto& astream = MKLDNNDeviceContext::tls().get_stream();
    matmul_p->execute(astream, matmul_args);
    astream.wait();

    out->set_layout(paddle::framework::DataLayout::kMKLDNN);
    out->set_format(platform::GetMKLDNNFormat(dst_memory_p->get_desc()));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul, MKLDNN, ::paddle::platform::CPUPlace,
                                    U8, ops::kMULMKLDNNINT8,
                                    ops::MulMKLDNNKernel<uint8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul, MKLDNN, ::paddle::platform::CPUPlace,
                                    S8, ops::kMULMKLDNNINT8,
                                    ops::MulMKLDNNKernel<int8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul, MKLDNN, ::paddle::platform::CPUPlace,
                                    FP32, ops::kMULMKLDNNFP32,
                                    ops::MulMKLDNNKernel<float, float>);    

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul, MKLDNN, ::paddle::platform::CPUPlace,
                                    BF16, ops::kMULMKLDNNFP32,
                                    ops::MulMKLDNNKernel<paddle::platform::bfloat16, paddle::platform::bfloat16>);                                    

REGISTER_OP_KERNEL(mul, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::MulMKLDNNKernel<uint8_t, float>, ops::MulMKLDNNKernel<paddle::platform::bfloat16, paddle::platform::bfloat16>, ops::MulMKLDNNKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul_grad, MKLDNN, ::paddle::platform::CPUPlace,
                                    FP32, ops::kMULMKLDNNFP32,
                                    ops::MulGradMKLDNNKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul_grad, MKLDNN, ::paddle::platform::CPUPlace,
                                    BF16, ops::kMULMKLDNNFP32,
                                    ops::MulGradMKLDNNKernel<paddle::platform::bfloat16, paddle::platform::bfloat16>);
