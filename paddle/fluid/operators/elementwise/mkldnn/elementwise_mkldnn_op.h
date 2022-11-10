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
using phi::DataLayout;
using phi::funcs::BinaryOneDNNHandler;

inline std::vector<int64_t> CalculateBroadcastedDims(
    const phi::DenseTensor* x, const phi::DenseTensor* y) {
  const auto src_tz = phi::vectorize(x->dims());
  const auto dst_tz = phi::vectorize(y->dims());

  std::vector<int64_t> dst_tz_ex(src_tz.size(), 1);

  if (src_tz.size() == dst_tz.size()) {
    for (size_t i = 0; i < src_tz.size(); i++) {
      dst_tz_ex[i] = (src_tz[i] == dst_tz[i]) ? dst_tz[i] : 1;
    }
  } else {
    size_t j = 0;
    for (size_t i = 0; i < src_tz.size(); i++) {
      dst_tz_ex[i] = (src_tz[i] != dst_tz[j]) ? 1 : dst_tz[j++];
      if (j == dst_tz.size()) break;
    }
  }

  return dst_tz_ex;
}

inline void AddSubNonBroadcast(
    phi::funcs::ReorderOneDNNHandler* reorder_handler,
    phi::DenseTensor* grad_tensor,
    const std::shared_ptr<dnnl::memory>& src_memory,
    const std::shared_ptr<dnnl::memory>& dst_memory,
    const std::vector<float>& scales) {
  dnnl::primitive_attr reorder_attr;
  reorder_attr.set_output_scales(0, scales);
  auto reorder_p =
      reorder_handler->AcquireReorder(dst_memory, src_memory, reorder_attr);

  reorder_p->execute(platform::MKLDNNDeviceContext::tls().get_stream(),
                     *src_memory,
                     *dst_memory);
}

template <typename T>
inline void BroadcastReduction(const framework::ExecutionContext& ctx,
                               const dnnl::engine& onednn_engine,
                               phi::DenseTensor* grad_tensor,
                               const phi::DenseTensor* dout,
                               const std::shared_ptr<dnnl::memory>& src_memory,
                               std::shared_ptr<dnnl::memory> dst_memory,
                               const std::vector<float>& scales,
                               const bool is_sub) {
  dnnl::primitive_attr broadcast_reduction_attr;

  // Broadcasting
  if (is_sub) {
    dnnl::post_ops po;
    po.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, scales[0], 0);
    broadcast_reduction_attr.set_post_ops(po);
  }

  phi::funcs::ReductionOneDNNHandler<T> reduction_handler(
      dnnl::algorithm::reduction_sum,
      0.0f,
      0.0f,
      onednn_engine,
      ctx.GetPlace(),
      dout,
      grad_tensor,
      CalculateBroadcastedDims(dout, grad_tensor),
      broadcast_reduction_attr);
  dst_memory = reduction_handler.AcquireDstMemory(grad_tensor);

  auto reduction_p = reduction_handler.AcquireForwardPrimitive();
  auto astream = platform::MKLDNNDeviceContext::tls().get_stream();
  reduction_p->execute(astream,
                       {
                           {DNNL_ARG_SRC, *src_memory},
                           {DNNL_ARG_DST, *dst_memory},
                       });
  astream.wait();
  grad_tensor->set_mem_desc(dst_memory->get_desc().reshape(
      phi::vectorize<int64_t>(grad_tensor->dims())));
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

    BinaryOneDNNHandler<T> handler(BINARY_OP,
                                   axis,
                                   mkldnn_engine,
                                   ctx.GetPlace(),
                                   x,
                                   y,
                                   z,
                                   scale_x,
                                   scale_y,
                                   scale_o,
                                   true,
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

    if (handler.use_broadcasting_hack == false) {
      platform::SetOutMemDescWithLogicalLayoutFusesSupport(
          ctx, z, dst_memory->get_desc());
    } else {
      auto dims = dst_memory->get_desc().dims();
      dims.insert(dims.begin(), x->dims()[0]);
      dims[1] /= dims[0];
      platform::SetOutMemDescWithLogicalLayoutFusesSupport(
          ctx, z, dst_memory->get_desc().reshape(dims));
    }
  }
};

template <typename T, dnnl::algorithm BINARY_OP>
class EltwiseMKLDNNGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* out = ctx.Input<phi::DenseTensor>("Out");

    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));

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
    auto dout_type = phi::funcs::ToOneDNNDataType(dout->dtype());

    phi::funcs::ReorderOneDNNHandler reorder_handler(
        tz, dout->dtype(), dout_type, onednn_engine);

    auto reorder_src_memory = reorder_handler.AcquireSrcMemory(
        dout->mem_desc(), phi::funcs::to_void_cast(dout->data<T>()));

    std::shared_ptr<dnnl::memory> dst_memory;
    std::shared_ptr<dnnl::memory> broadcast_src_memory = reorder_src_memory;

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    if (dx) {
      // elementwise_add & elementwise_sub
      if (BINARY_OP == dnnl::algorithm::binary_add ||
          BINARY_OP == dnnl::algorithm::binary_sub) {
        if (dout->dims() == dx->dims()) {
          dst_memory = reorder_handler.AcquireDstMemory(
              dx, dout->mem_desc(), ctx.GetPlace());
          AddSubNonBroadcast(
              &reorder_handler, dx, reorder_src_memory, dst_memory, scales);
        }
      } else {  // elementwise_mul & elementwise_div
        BinaryOneDNNHandler<T> binary_handler(BINARY_OP,
                                              axis,
                                              onednn_engine,
                                              ctx.GetPlace(),
                                              dout,
                                              y,
                                              dx,
                                              1.0f,
                                              1.0f,
                                              1.0f,
                                              false);

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

      if (dout->dims() != dx->dims()) {
        BroadcastReduction<T>(ctx,
                              onednn_engine,
                              dx,
                              dout,
                              broadcast_src_memory,
                              dst_memory,
                              scales,
                              BINARY_OP == dnnl::algorithm::binary_sub);
      } else {
        dx->set_mem_desc(dst_memory->get_desc());
      }
    }

    if (dy) {
      // elementwise_add & elementwise_sub
      if (BINARY_OP == dnnl::algorithm::binary_add ||
          BINARY_OP == dnnl::algorithm::binary_sub) {
        if (dout->dims() == dy->dims()) {
          dst_memory = reorder_handler.AcquireDstMemory(
              dy, dout->mem_desc(), ctx.GetPlace());
          AddSubNonBroadcast(
              &reorder_handler, dy, reorder_src_memory, dst_memory, scales);
        }
      } else {  // elementwise_mul & elementwise_div
        std::unordered_map<int, dnnl::memory> args;
        std::shared_ptr<dnnl::binary> binary_prim;
        std::shared_ptr<dnnl::memory> post_op_memory;
        std::shared_ptr<dnnl::memory> src_0_memory;
        std::shared_ptr<dnnl::memory> src_1_memory;

        BinaryOneDNNHandler<T> binary_handler(dnnl::algorithm::binary_mul,
                                              axis,
                                              onednn_engine,
                                              ctx.GetPlace(),
                                              dout,
                                              x,
                                              nullptr,
                                              1.0f,
                                              1.0f,
                                              1.0f,
                                              false);

        src_1_memory = binary_handler.AcquireSecondSrcMemory(x);

        if (BINARY_OP == dnnl::algorithm::binary_div) {
          BinaryOneDNNHandler<T> post_op_binary_handler(
              dnnl::algorithm::binary_div,
              axis,
              onednn_engine,
              ctx.GetPlace(),
              y,
              y,
              nullptr,
              1.0f,
              1.0f,
              1.0f,
              false);

          post_op_memory = post_op_binary_handler.AcquireSrcMemory(y);

          dnnl::post_ops po;
          po.append_binary(dnnl::algorithm::binary_div,
                           post_op_memory->get_desc());

          binary_handler = BinaryOneDNNHandler<T>(dnnl::algorithm::binary_mul,
                                                  axis,
                                                  onednn_engine,
                                                  ctx.GetPlace(),
                                                  dout,
                                                  out,
                                                  nullptr,
                                                  -1.0f,
                                                  1.0f,
                                                  1.0f,
                                                  false,
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
        BroadcastReduction<T>(ctx,
                              onednn_engine,
                              dy,
                              dout,
                              broadcast_src_memory,
                              dst_memory,
                              scales,
                              BINARY_OP == dnnl::algorithm::binary_sub);
      } else {
        dy->set_mem_desc(dst_memory->get_desc());
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
