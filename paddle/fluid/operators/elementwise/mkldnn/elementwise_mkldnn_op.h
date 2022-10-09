// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using dnnl::memory;
using dnnl::primitive;
using dnnl::stream;
using framework::DataLayout;

inline std::vector<int64_t> CalculateBroadcastedDims(
    const phi::DenseTensor* x, const phi::DenseTensor* y) {
  const auto src_tz = phi::vectorize(x->dims());
  const auto dst_tz = phi::vectorize(y->dims());

  size_t j = 0;
  std::vector<int64_t> dst_tz_ex(src_tz.size(), 1);
  for (size_t i = 0; i < src_tz.size(); ++i) {
    dst_tz_ex[i] = (src_tz[i] != dst_tz[j]) ? 1 : dst_tz[j++];
    if (j == dst_tz.size()) break;
  }

  return dst_tz_ex;
}

template <typename T, dnnl::algorithm BINARY_OP>
class EltwiseMKLDNNKernel : public framework::OpKernel<T> {
 private:
  dnnl::post_ops get_post_ops(const framework::ExecutionContext& ctx) const {
    dnnl::post_ops post_operations;
    platform::AppendActivation(ctx, post_operations);
    return post_operations;
  }

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* z = ctx.Output<phi::DenseTensor>("Out");

    float scale_x = ctx.Attr<float>("Scale_x");
    float scale_y = ctx.Attr<float>("Scale_y");
    float scale_o = ctx.Attr<float>("Scale_out");
    int axis = ctx.Attr<int>("axis");

    platform::BinaryMKLDNNHandler<T> handler(BINARY_OP,
                                             axis,
                                             mkldnn_engine,
                                             ctx.GetPlace(),
                                             x,
                                             y,
                                             z,
                                             scale_x,
                                             scale_y,
                                             scale_o,
                                             get_post_ops(ctx));

    // oneDNN's binary is optimized for broadcasting y into x, so in other case
    // we have to swap tensors to achieve optimal performance
    if (x->numel() < y->numel()) {
      std::swap(x, y);
    }

    const auto src_x_memory = handler.AcquireSrcMemory(x);
    const auto src_y_memory = handler.AcquireSecondSrcMemory(y);
    // (jczaja) For Inplace src and dst should be the same memory object.
    // So x should share buffer with z. But UT mechanics is testing inplace
    // execution for this op not checking that x can be bradcasted to match in
    // shape y tensor.
    // This is wrong as when x is to be broadcasted then z(out) will match the
    // shape of y which is bigger than x. Hence if x is smaller in shape than z
    // and they share a buffer (of
    // shape x) then this buffer is not big enough to hold result of elementwise
    // operation.
    const bool reuse_x_memopry =
        x->numel() == z->numel() && x->IsSharedBufferWith(*z);
    std::shared_ptr<dnnl::memory> dst_memory;
    if (reuse_x_memopry) {
      dst_memory = src_x_memory;
      // NOTE(chenfeiyu): when the output reuses memory from other tensor rather
      // than allocate its own, it's still need to take care of its data type.
      // Unfortunately, paddle's operator only infers the output' shape, but not
      // the data type. mutable_data<T> takes care of allocation and data type
      // normally, but if the memory is already allocated and there is no need
      // to re-allocate, it just set the data type. So this it added there to
      // get the right data type.
      z->mutable_data<T>(ctx.GetPlace());
    } else {
      dst_memory = handler.AcquireDstMemory(z);
    }

    const auto binary_prim = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    const std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC_0, *src_x_memory},
        {DNNL_ARG_SRC_1, *src_y_memory},
        {DNNL_ARG_DST, *dst_memory}};

    binary_prim->execute(astream, args);
    astream.wait();

    z->set_mem_desc(dst_memory->get_desc());
  }
};

template <typename T, dnnl::algorithm BINARY_OP>
class EltwiseMKLDNNGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = phi::DenseTensor;

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* out = ctx.Input<phi::DenseTensor>("Out");

    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    VLOG(4) << "element sub: dx " << dx << " dy " << dy << " dout " << dout;

    // oneDNN's binary is optimized for broadcasting y into x, so in other case
    // we have to swap tensors to achieve optimal performance
    bool swap_x_y = false;
    if (x->numel() < y->numel()) {
      std::swap(x, y);
      std::swap(dx, dy);
      swap_x_y = true;
    }

    std::vector<float> scales{1.0};
    if (swap_x_y) {
      scales[0] = (BINARY_OP == dnnl::algorithm::binary_add) ? 1 : -1;
    }

    int axis = ctx.Attr<int>("axis");

    auto tz = phi::vectorize<int64_t>(dout->dims());
    auto proto_type_dout = framework::TransToProtoVarType(dout->dtype());

    platform::ReorderMKLDNNHandler reorder_handler(
        tz,
        proto_type_dout,
        framework::ToMKLDNNDataType(proto_type_dout),
        onednn_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        dout->mem_desc(), platform::to_void_cast(dout->data<T>()));

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    if (dx) {
      std::shared_ptr<dnnl::memory> dst_memory;

      // elementwise_add & elementwise_sub
      if (BINARY_OP == dnnl::algorithm::binary_add ||
          BINARY_OP == dnnl::algorithm::binary_sub) {
        dst_memory = reorder_handler.AcquireDstMemory(
            dx, dout->mem_desc(), ctx.GetPlace());

        dnnl::primitive_attr reorder_attr;
        reorder_attr.set_output_scales(0, scales);
        auto reorder_p = reorder_handler.AcquireReorder(
            dst_memory, reorder_src_memory_p, reorder_attr);
        platform::RecordEvent record_reorder(
            "int_reorder",
            platform::TracerEventType::UserDefined,
            2,
            platform::EventRole::kUniqueOp);

        reorder_p->execute(astream, *reorder_src_memory_p, *dst_memory);

      } else {  // elementwise_mul & elementwise_div
        platform::BinaryMKLDNNHandler<T> binary_handler(BINARY_OP,
                                                        axis,
                                                        onednn_engine,
                                                        ctx.GetPlace(),
                                                        dout,
                                                        y,
                                                        dx,
                                                        1.0f,
                                                        1.0f,
                                                        1.0f);

        const auto src_dout_memory = binary_handler.AcquireSrcMemory(dout);
        const auto src_y_memory = binary_handler.AcquireSecondSrcMemory(y);
        dst_memory = binary_handler.AcquireDstMemory(dx);

        const auto binary_prim = binary_handler.AcquireForwardPrimitive();

        const std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC_0, *src_dout_memory},
            {DNNL_ARG_SRC_1, *src_y_memory},
            {DNNL_ARG_DST, *dst_memory}};

        binary_prim->execute(astream, args);
      }
      astream.wait();

      dx->set_mem_desc(dst_memory->get_desc());
    }

    if (dy) {
      dnnl::primitive_attr broadcast_reduction_attr;
      std::shared_ptr<dnnl::memory> broadcast_src_memory;
      std::shared_ptr<dnnl::memory> dst_memory;

      // elementwise_add & elementwise_sub
      if (BINARY_OP == dnnl::algorithm::binary_add ||
          BINARY_OP == dnnl::algorithm::binary_sub) {
        if (dout->dims() == dy->dims()) {
          auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
              dy, dout->mem_desc(), ctx.GetPlace());

          dnnl::primitive_attr reorder_attr;
          reorder_attr.set_output_scales(0, scales);

          auto reorder_p = reorder_handler.AcquireReorder(
              reorder_dst_memory_p, reorder_src_memory_p, reorder_attr);
          platform::RecordEvent record_reorder(
              "int_reorder",
              platform::TracerEventType::UserDefined,
              2,
              platform::EventRole::kUniqueOp);
          reorder_p->execute(
              astream, *reorder_src_memory_p, *reorder_dst_memory_p);

          dst_memory = reorder_dst_memory_p;
        } else {
          broadcast_src_memory = reorder_src_memory_p;
        }
      } else {  // elementwise_mul & elementwise_div
        std::unordered_map<int, dnnl::memory> args;
        std::shared_ptr<dnnl::binary> binary_prim;
        std::shared_ptr<dnnl::memory> post_op_memory;
        std::shared_ptr<dnnl::memory> src_0_memory;
        std::shared_ptr<dnnl::memory> src_1_memory;

        platform::BinaryMKLDNNHandler<T> binary_handler(
            dnnl::algorithm::binary_mul,
            axis,
            onednn_engine,
            ctx.GetPlace(),
            dout,
            x,
            nullptr,
            1.0f,
            1.0f,
            1.0f);

        src_1_memory = binary_handler.AcquireSecondSrcMemory(x);

        if (BINARY_OP == dnnl::algorithm::binary_div) {
          platform::BinaryMKLDNNHandler<T> post_op_binary_handler(
              dnnl::algorithm::binary_div,
              axis,
              onednn_engine,
              ctx.GetPlace(),
              y,
              y,
              nullptr,
              1.0f,
              1.0f,
              1.0f);

          post_op_memory = post_op_binary_handler.AcquireSrcMemory(y);

          dnnl::post_ops po;
          po.append_binary(dnnl::algorithm::binary_div,
                           post_op_memory->get_desc());

          binary_handler =
              platform::BinaryMKLDNNHandler<T>(dnnl::algorithm::binary_mul,
                                               axis,
                                               onednn_engine,
                                               ctx.GetPlace(),
                                               dout,
                                               out,
                                               nullptr,
                                               -1.0f,
                                               1.0f,
                                               1.0f,
                                               po);

          src_1_memory = binary_handler.AcquireSecondSrcMemory(out);
        }

        src_0_memory = binary_handler.AcquireSrcMemory(dout);

        const auto dst_dy_memory = (dout->dims() == dy->dims())
                                       ? binary_handler.AcquireDstMemory(dy)
                                       : binary_handler.AcquireDstMemory();

        binary_prim = binary_handler.AcquireForwardPrimitive();
        args = {{DNNL_ARG_SRC_0, *src_0_memory},
                {DNNL_ARG_SRC_1, *src_1_memory},
                {DNNL_ARG_DST, *dst_dy_memory}};

        if (BINARY_OP == dnnl::algorithm::binary_div)
          args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                       *post_op_memory});

        binary_prim->execute(astream, args);
        broadcast_src_memory = dst_dy_memory;
        dst_memory = dst_dy_memory;
      }
      astream.wait();

      if (dout->dims() != dy->dims()) {
        // Broadcasting
        if (BINARY_OP == dnnl::algorithm::binary_sub) {
          dnnl::post_ops po;
          po.append_eltwise(
              1.0f, dnnl::algorithm::eltwise_linear, scales[0], 0);
          broadcast_reduction_attr.set_post_ops(po);
        }

        platform::ReductionMKLDNNHandler<T> reduction_handler(
            dnnl::algorithm::reduction_sum,
            0.0f,
            0.0f,
            onednn_engine,
            ctx.GetPlace(),
            dout,
            dy,
            CalculateBroadcastedDims(dout, dy),
            broadcast_reduction_attr);
        dst_memory = reduction_handler.AcquireDstMemory(dy);

        auto reduction_p = reduction_handler.AcquireForwardPrimitive();

        reduction_p->execute(astream,
                             {
                                 {DNNL_ARG_SRC, *broadcast_src_memory},
                                 {DNNL_ARG_DST, *dst_memory},
                             });
        astream.wait();
        dy->set_mem_desc(dst_memory->get_desc().reshape(
            phi::vectorize<int64_t>(dy->dims())));
      } else {
        dy->set_mem_desc(dst_memory->get_desc());
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
