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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/operators/elementwise/mkldnn/elementwise_mkldnn_op.h"

namespace paddle {
namespace framework {
class ExecutionContext;
}  // namespace framework
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

enum elementwise_op {
  elementwise_add,
  elementwise_sub,
  elementwise_mul,
  elementwise_div
};

namespace paddle {
namespace operators {
template <typename T, elementwise_op op>
class EltwiseMKLDNNGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Input<Tensor>("Out");

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    int axis = ctx.Attr<int>("axis");

    auto tz = phi::vectorize<int64_t>(dout->dims());
    memory::data_type dout_type = framework::ToMKLDNNDataType(
        framework::TransToProtoVarType(dout->dtype()));

    platform::ReorderMKLDNNHandler handler(
        tz, framework::TransToProtoVarType(dout->dtype()), dout_type,
        onednn_engine);

    auto reorder_src_memory_p = handler.AcquireSrcMemory(
        dout->format(), platform::to_void_cast(dout->data<T>()));

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    if (dx) {
      std::shared_ptr<dnnl::memory> dst_memory;
      if (op == elementwise_add || op == elementwise_sub) {
        dst_memory =
            handler.AcquireDstMemory(dx, dout->format(), ctx.GetPlace());
        auto reorder_p =
            handler.AcquireReorder(dst_memory, reorder_src_memory_p);
        platform::RecordEvent record_reorder(
            "int_reorder", platform::TracerEventType::UserDefined, 2,
            platform::EventRole::kUniqueOp);

        reorder_p->execute(astream, *reorder_src_memory_p, *dst_memory);
      } else {
        dnnl::algorithm algorithm = (op == elementwise_mul)
                                        ? dnnl::algorithm::binary_mul
                                        : dnnl::algorithm::binary_div;
        platform::BinaryMKLDNNHandler<T> handler(algorithm, axis, onednn_engine,
                                                 ctx.GetPlace(), dout, y, dx,
                                                 1.0f, 1.0f, 1.0f);

        const auto src_dout_memory = handler.AcquireSrcMemory(dout);
        const auto src_y_memory = handler.AcquireSecondSrcMemory(y);
        dst_memory = handler.AcquireDstMemory(dx);

        const auto binary_prim = handler.AcquireForwardPrimitive();

        const std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC_0, *src_dout_memory},
            {DNNL_ARG_SRC_1, *src_y_memory},
            {DNNL_ARG_DST, *dst_memory}};

        binary_prim->execute(astream, args);
      }
      astream.wait();
      dx->set_layout(DataLayout::kMKLDNN);
      dx->set_format(platform::GetMKLDNNFormat(*dst_memory));
    }

    if (dy) {
      if (op == elementwise_add || op == elementwise_sub) {
        // Direct copy
        if (dout->dims() == dy->dims()) {
          auto reorder_dst_memory_p =
              handler.AcquireDstMemory(dy, dout->format(), ctx.GetPlace());

          dnnl::primitive_attr reorder_attr;
          std::vector<float> scales(1);
          scales[0] = (op == elementwise_add) ? 1 : -1;
          reorder_attr.set_output_scales(0, scales);
          auto reorder_p = std::make_shared<dnnl::reorder>(
              *(reorder_src_memory_p), *(reorder_dst_memory_p), reorder_attr);
          platform::RecordEvent record_reorder(
              "int_reorder", platform::TracerEventType::UserDefined, 2,
              platform::EventRole::kUniqueOp);
          reorder_p->execute(astream, *reorder_src_memory_p,
                             *reorder_dst_memory_p);
          astream.wait();

          dy->set_layout(DataLayout::kMKLDNN);
          dy->set_format(platform::GetMKLDNNFormat(*reorder_dst_memory_p));
        } else {
          // Broadcasting

          dnnl::primitive_attr attr;
          if (op == elementwise_sub) {
            dnnl::post_ops po;
            po.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, -1.0f, 0);
            attr.set_post_ops(po);
          }

          platform::ReductionMKLDNNHandler<T> handler_sum(
              dnnl::algorithm::reduction_sum, 0.0f, 0.0f, onednn_engine,
              ctx.GetPlace(), dout, dy, CalculateBroadcastedDims(dout, dy),
              attr);

          auto dy_memory_p = handler_sum.AcquireDstMemory(dy);
          auto reduction_p = handler_sum.AcquireForwardPrimitive();

          reduction_p->execute(astream,
                               {
                                   {DNNL_ARG_SRC, *reorder_src_memory_p},
                                   {DNNL_ARG_DST, *dy_memory_p},
                               });
          astream.wait();

          dy->set_layout(DataLayout::kMKLDNN);
          dy->set_format(
              platform::GetMKLDNNFormat(dy_memory_p->get_desc().reshape(
                  phi::vectorize<int64_t>(dy->dims()))));
        }
      } else {
        std::unordered_map<int, dnnl::memory> args;
        std::shared_ptr<dnnl::binary> binary_prim;
        std::shared_ptr<dnnl::memory> y_memory;
        std::shared_ptr<dnnl::memory> src_0_memory;
        std::shared_ptr<dnnl::memory> src_1_memory;

        platform::BinaryMKLDNNHandler<T> handler(
            dnnl::algorithm::binary_mul, axis, onednn_engine, ctx.GetPlace(),
            dout, x, nullptr, 1.0f, 1.0f, 1.0f);

        src_1_memory = handler.AcquireSecondSrcMemory(x);

        if (op == elementwise_div) {
          platform::BinaryMKLDNNHandler<T> y_handler(
              dnnl::algorithm::binary_div, axis, onednn_engine, ctx.GetPlace(),
              y, y, nullptr, 1.0f, 1.0f, 1.0f);

          y_memory = y_handler.AcquireSrcMemory(y);

          dnnl::post_ops po;
          po.append_binary(dnnl::algorithm::binary_div, y_memory->get_desc());

          handler = platform::BinaryMKLDNNHandler<T>(
              dnnl::algorithm::binary_mul, axis, onednn_engine, ctx.GetPlace(),
              dout, out, nullptr, -1.0f, 1.0f, 1.0f, po);

          src_1_memory = handler.AcquireSecondSrcMemory(out);
        }

        src_0_memory = handler.AcquireSrcMemory(dout);

        // If broadcasting is in use then let's write to temporary
        // buffer allocated by oneDNN

        const auto dst_dy_memory = (dout->dims() == dy->dims())
                                       ? handler.AcquireDstMemory(dy)
                                       : handler.AcquireDstMemory();

        binary_prim = handler.AcquireForwardPrimitive();

        if (op == elementwise_mul) {
          args = {{DNNL_ARG_SRC_0, *src_0_memory},
                  {DNNL_ARG_SRC_1, *src_1_memory},
                  {DNNL_ARG_DST, *dst_dy_memory}};
        } else {
          args = {
              {DNNL_ARG_SRC_0, *src_0_memory},
              {DNNL_ARG_SRC_1, *src_1_memory},
              {DNNL_ARG_DST, *dst_dy_memory},
              {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, *y_memory}};
        }

        binary_prim->execute(astream, args);
        astream.wait();

        dy->set_layout(framework::DataLayout::kMKLDNN);

        // Reduction is needed for broadcasting scenario
        if (dout->dims() != dy->dims()) {
          platform::ReductionMKLDNNHandler<T> handler_sum(
              dnnl::algorithm::reduction_sum, 0.0f, 0.0f, onednn_engine,
              ctx.GetPlace(), dout, dy, CalculateBroadcastedDims(dout, dy));
          auto dy_memory_p = handler_sum.AcquireDstMemory(dy);
          auto reduction_p = handler_sum.AcquireForwardPrimitive();
          // As source we use mem object with results from binary operation
          reduction_p->execute(astream, {{DNNL_ARG_SRC, *dst_dy_memory},
                                         {DNNL_ARG_DST, *dy_memory_p}});
          astream.wait();
          dy->set_format(
              platform::GetMKLDNNFormat(dy_memory_p->get_desc().reshape(
                  phi::vectorize<int64_t>(dy->dims()))));

        } else {
          dy->set_format(platform::GetMKLDNNFormat(*dst_dy_memory));
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(
    elementwise_add, MKLDNN, ::paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNKernel<float, dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNKernel<paddle::platform::bfloat16,
                             dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNKernel<int8_t, dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNKernel<uint8_t, dnnl::algorithm::binary_add>)

REGISTER_OP_KERNEL(
    elementwise_add_grad, MKLDNN, ::paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNGradKernel<paddle::platform::bfloat16, elementwise_add>,
    ops::EltwiseMKLDNNGradKernel<float, elementwise_add>)

REGISTER_OP_KERNEL(
    elementwise_sub, MKLDNN, paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNKernel<float, dnnl::algorithm::binary_sub>,
    ops::EltwiseMKLDNNKernel<paddle::platform::bfloat16,
                             dnnl::algorithm::binary_sub>,
    ops::EltwiseMKLDNNKernel<int8_t, dnnl::algorithm::binary_sub>,
    ops::EltwiseMKLDNNKernel<uint8_t, dnnl::algorithm::binary_sub>)

REGISTER_OP_KERNEL(
    elementwise_sub_grad, MKLDNN, ::paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNGradKernel<paddle::platform::bfloat16, elementwise_sub>,
    ops::EltwiseMKLDNNGradKernel<float, elementwise_sub>)

REGISTER_OP_KERNEL(
    elementwise_mul, MKLDNN, ::paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNKernel<float, dnnl::algorithm::binary_mul>,
    ops::EltwiseMKLDNNKernel<paddle::platform::bfloat16,
                             dnnl::algorithm::binary_mul>,
    ops::EltwiseMKLDNNKernel<int8_t, dnnl::algorithm::binary_mul>,
    ops::EltwiseMKLDNNKernel<uint8_t, dnnl::algorithm::binary_mul>)

REGISTER_OP_KERNEL(
    elementwise_mul_grad, MKLDNN, ::paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNGradKernel<paddle::platform::bfloat16, elementwise_mul>,
    ops::EltwiseMKLDNNGradKernel<float, elementwise_mul>)

REGISTER_OP_KERNEL(elementwise_div, MKLDNN, paddle::platform::CPUPlace,
                   ops::EltwiseMKLDNNKernel<float, dnnl::algorithm::binary_div>,
                   ops::EltwiseMKLDNNKernel<paddle::platform::bfloat16,
                                            dnnl::algorithm::binary_div>)

REGISTER_OP_KERNEL(
    elementwise_div_grad, MKLDNN, paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNGradKernel<paddle::platform::bfloat16, elementwise_div>,
    ops::EltwiseMKLDNNGradKernel<float, elementwise_div>)
