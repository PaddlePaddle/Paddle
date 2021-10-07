/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using framework::DataLayout;

template <typename T>
class TransposeMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL Transpose must use CPUPlace"));
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    std::vector<int> transpose_axis = ctx.Attr<std::vector<int>>("axis");
    int ndims = transpose_axis.size();
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    if (ndims == 1) {
      framework::TensorCopy(*x, x->place(), out);
      out->set_mem_desc(x->mem_desc());
      return;
    }

    auto x_vec_dims = framework::vectorize(x->dims());

    mkldnn::memory::data_type x_type = framework::ToMKLDNNDataType(x->type());
    platform::ReorderMKLDNNHandler reorder_handler(x_vec_dims, x->type(),
                                                   x_type, mkldnn_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x->mem_desc(), platform::to_void_cast(x->data<T>()));

    auto reorder_dst_memory_p =
        reorder_handler.AcquireDstMemory(out, x->mem_desc(), ctx.GetPlace());

    auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                    reorder_src_memory_p);

    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();

    out->set_layout(DataLayout::kMKLDNN);
    out->set_mem_desc(reorder_dst_memory_p->get_desc().permute_axes(
        TransposeToPermuteAxis(transpose_axis)));
  }

 private:
  // it is needed because oneDNN's permute axis understand axes order in
  // different way than matmul's one
  std::vector<int> TransposeToPermuteAxis(
      const std::vector<int>& transpose_axis) const {
    std::vector<int> permute_axis(transpose_axis.size());

    for (size_t i = 0; i < transpose_axis.size(); ++i) {
      permute_axis[transpose_axis[i]] = i;
    }
    return permute_axis;
  }
};

template <typename T>
class TransposeMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL TransposeGrad must use CPUPlace"));

    const auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    if (!dx) return;
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    std::vector<int> transpose_axis = ctx.Attr<std::vector<int>>("axis");

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    int ndims = transpose_axis.size();
    if (ndims == 1) {
      framework::TensorCopy(*dout, dout->place(), dx);
      dx->set_mem_desc(dout->mem_desc());
      return;
    }

    auto dout_vec_dims = framework::vectorize(dout->dims());

    mkldnn::memory::data_type dout_type =
        framework::ToMKLDNNDataType(dout->type());
    platform::ReorderMKLDNNHandler reorder_handler(dout_vec_dims, dout->type(),
                                                   dout_type, mkldnn_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        dout->mem_desc(), platform::to_void_cast(dout->data<T>()));

    auto reorder_dst_memory_p =
        reorder_handler.AcquireDstMemory(dx, dout->mem_desc(), ctx.GetPlace());

    auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                    reorder_dst_memory_p);

    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();
    dx->set_layout(DataLayout::kMKLDNN);
    dx->set_mem_desc(
        reorder_dst_memory_p->get_desc().permute_axes(transpose_axis));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(transpose2, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kTransposeMKLDNNFP32,
                                    ops::TransposeMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(transpose2, MKLDNN,
                                    ::paddle::platform::CPUPlace, U8,
                                    ops::kTransposeMKLDNNINT8,
                                    ops::TransposeMKLDNNOpKernel<uint8_t>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(transpose2, MKLDNN,
                                    ::paddle::platform::CPUPlace, S8,
                                    ops::kTransposeMKLDNNINT8,
                                    ops::TransposeMKLDNNOpKernel<int8_t>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(
    transpose2, MKLDNN, ::paddle::platform::CPUPlace, BF16,
    ops::kTransposeMKLDNNFP32,
    ops::TransposeMKLDNNOpKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(transpose, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(transpose_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNGradOpKernel<float>);

REGISTER_OP_KERNEL(transpose2_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNGradOpKernel<float>);
