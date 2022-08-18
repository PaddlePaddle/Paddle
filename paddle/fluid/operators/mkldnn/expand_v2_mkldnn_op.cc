/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/operators/expand_v2_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace {

using paddle::framework::ExecutionContext;
using paddle::framework::GradVarName;
using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;
using phi::vectorize;

template <typename T>
class ExpandMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const ExecutionContext& ctx) const {
    const auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    auto x_vec_dims = vectorize(x->dims());

    auto out_new_dims = paddle::operators::get_expand_shape(ctx);
    for (size_t i = 0; i < out_new_dims.size(); ++i) {
      out_new_dims[i] = out_new_dims[i] > 0 ? out_new_dims[i] : x_vec_dims[i];
    }

    if (x_vec_dims.size() != out_new_dims.size()) {
      x_vec_dims = GetExtendedXDims(x_vec_dims, out_new_dims.size());
    }

    out->Resize(phi::make_ddim(out_new_dims));
    paddle::platform::BroadcastDataMKLDNNHandler<T> handler(
        dnnl::algorithm::binary_add,
        onednn_engine,
        ctx.GetPlace(),
        x,
        out,
        0.0f,
        1.0f,
        x_vec_dims);

    auto src_memory_p = handler.AcquireSrcMemory(x);
    auto dst_memory_p = handler.AcquireZeroedDstMemory(out);
    auto binary_p = handler.AcquireForwardPrimitive();

    const std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC_0, *dst_memory_p},
        {DNNL_ARG_SRC_1, *src_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    auto& astream = MKLDNNDeviceContext::tls().get_stream();
    binary_p->execute(astream, args);
    astream.wait();

    out->set_mem_desc(dst_memory_p->get_desc());
  }

 private:
  std::vector<int64_t> GetExtendedXDims(const std::vector<int64_t>& x_vec_dims,
                                        int new_size) const {
    std::vector<int64_t> extended_x_dims(new_size, 1);
    std::copy(x_vec_dims.begin(),
              x_vec_dims.end(),
              extended_x_dims.begin() + new_size - x_vec_dims.size());

    return extended_x_dims;
  }
};

template <typename T>
class ExpandGradMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const ExecutionContext& ctx) const {
    const auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* dout = ctx.Input<Tensor>(GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(GradVarName("X"));

    auto dx_vec_dims = vectorize(dx->dims());
    auto dout_vec_dims = vectorize(dout->dims());

    if (dx_vec_dims.size() != dout_vec_dims.size()) {
      dx_vec_dims.insert(
          dx_vec_dims.begin(), dout_vec_dims.size() - dx_vec_dims.size(), 1);
    }

    auto& astream = MKLDNNDeviceContext::tls().get_stream();
    if (dout_vec_dims == dx_vec_dims) {
      dnnl::memory::data_type dout_type = paddle::framework::ToMKLDNNDataType(
          paddle::framework::TransToProtoVarType(dout->dtype()));
      paddle::platform::ReorderMKLDNNHandler reorder_handler(
          dout_vec_dims,
          paddle::framework::TransToProtoVarType(dout->dtype()),
          dout_type,
          onednn_engine);

      auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
          dout->mem_desc(), paddle::platform::to_void_cast(dout->data<T>()));

      auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
          dx,
          paddle::platform::GetPlainMKLDNNFormat(dx_vec_dims.size()),
          ctx.GetPlace());

      auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                      reorder_dst_memory_p);

      reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
      astream.wait();

      dx->set_mem_desc(reorder_dst_memory_p->get_desc());
    } else {
      paddle::platform::ReductionMKLDNNHandler<T> handler(
          dnnl::algorithm::reduction_sum,
          0.0f,
          0.0f,
          onednn_engine,
          ctx.GetPlace(),
          dout,
          dx,
          dx_vec_dims);

      auto src_memory_p = handler.AcquireSrcMemory(dout);
      auto dst_memory_p = handler.AcquireDstMemory(dx);

      std::unordered_map<int, dnnl::memory> reduction_args = {
          {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};

      auto reduction_p = handler.AcquireForwardPrimitive();

      reduction_p->execute(astream, reduction_args);
      astream.wait();
      dx->set_layout(paddle::framework::DataLayout::kMKLDNN);
      dx->set_mem_desc(
          dst_memory_p->get_desc().reshape(vectorize<int64_t>(dx->dims())));
    }
  }
};
}  // anonymous namespace

REGISTER_OP_KERNEL(expand_v2,
                   MKLDNN,
                   paddle::platform::CPUPlace,
                   ExpandMKLDNNKernel<float>,
                   ExpandMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(expand_v2_grad,
                   MKLDNN,
                   paddle::platform::CPUPlace,
                   ExpandGradMKLDNNKernel<float>,
                   ExpandGradMKLDNNKernel<paddle::platform::bfloat16>);
