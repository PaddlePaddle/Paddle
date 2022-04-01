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

#pragma once
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::LoDTensor;
using paddle::framework::Tensor;
using platform::to_void_cast;

inline std::vector<int64_t> CalculateReducedDims(
    const Tensor* input, const Tensor* output,
    std::vector<int>& reduce_dims,  // NOLINT
    bool reduce_all, bool keep_dim) {
  if (keep_dim) return phi::vectorize(output->dims());

  if (reduce_all)
    return std::vector<int64_t>(phi::vectorize(input->dims()).size(), 1);

  std::vector<int64_t> output_dims(phi::vectorize(input->dims()));
  for (size_t i = 0; i < reduce_dims.size(); ++i) {
    reduce_dims[i] = (reduce_dims[i] >= 0)
                         ? reduce_dims[i]
                         : input->dims().size() + reduce_dims[i];
    output_dims[reduce_dims[i]] = 1;
  }

  return output_dims;
}

template <typename T>
class ReduceMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void RunKernel(const framework::ExecutionContext& ctx,
                 dnnl::algorithm reduction_type) const {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    const auto* input = ctx.Input<LoDTensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    auto reduce_dims = ctx.Attr<std::vector<int>>("dim");
    bool reduce_all = ctx.Attr<bool>("reduce_all");
    bool keep_dim = ctx.Attr<bool>("keep_dim");

    auto output_dims =
        CalculateReducedDims(input, output, reduce_dims, reduce_all, keep_dim);
    auto input_dims = phi::vectorize(input->dims());

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    // oneDNN reduce op does not support edge case in which memory is being
    // copied without actual reduction.
    // In that case reorder must be executed to maintain compatibility with
    // PaddlePaddle reduce op
    if (input_dims == output_dims) {
      dnnl::memory::data_type input_type = framework::ToMKLDNNDataType(
          framework::TransToProtoVarType(input->dtype()));
      platform::ReorderMKLDNNHandler reorder_handler(
          input_dims, framework::TransToProtoVarType(input->dtype()),
          input_type, onednn_engine);

      auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
          input->mem_desc(), platform::to_void_cast(input->data<T>()));

      auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
          output, input->mem_desc(), ctx.GetPlace());

      auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                      reorder_dst_memory_p);

      reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
      astream.wait();

      output->set_mem_desc(reorder_dst_memory_p->get_desc().reshape(
          phi::vectorize<int64_t>(output->dims())));
    } else {
      platform::ReductionMKLDNNHandler<T> handler(reduction_type, 0.0f, 0.0f,
                                                  onednn_engine, ctx.GetPlace(),
                                                  input, output, output_dims);

      auto src_memory_p = handler.AcquireSrcMemory(input);
      auto dst_memory_p = handler.AcquireDstMemory(output);

      std::unordered_map<int, dnnl::memory> reduction_args = {
          {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};

      auto reduction_p = handler.AcquireForwardPrimitive();

      reduction_p->execute(astream, reduction_args);
      astream.wait();
      output->set_mem_desc(dst_memory_p->get_desc().reshape(
          phi::vectorize<int64_t>(output->dims())));
    }
  }
};

template <typename T>
class ReduceGradMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void RunKernel(const framework::ExecutionContext& ctx,
                 dnnl::algorithm binary_type, dnnl::algorithm reduction_type,
                 float scale_x, float scale_y) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    bool keep_dim = ctx.Attr<bool>("keep_dim");
    bool reduce_all = ctx.Attr<bool>("reduce_all");
    auto dims = ctx.Attr<std::vector<int>>("dim");
    const auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    const auto input_dims =
        CalculateReducedDims(dx, dout, dims, reduce_all, keep_dim);
    const auto output_dims = phi::vectorize(dx->dims());

    auto dout_mem_desc = dout->mem_desc();

    if (input_dims != output_dims) {
      dout_mem_desc = dout_mem_desc.reshape(input_dims);
    }

    platform::BroadcastDataMKLDNNHandler<T> handler(
        binary_type, onednn_engine, ctx.GetPlace(), dx, dout, scale_x, scale_y,
        dout_mem_desc);

    const auto src_memory_p = handler.AcquireSrcMemory(dout);
    const auto dst_memory_p = handler.AcquireDstMemory(dx);
    const auto binary_prim = handler.AcquireForwardPrimitive();

    const std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC_0, *dst_memory_p},
        {DNNL_ARG_SRC_1, *src_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    binary_prim->execute(astream, args);
    astream.wait();

    dx->set_mem_desc(dst_memory_p->get_desc());
  }
};

}  // namespace operators
}  // namespace paddle
