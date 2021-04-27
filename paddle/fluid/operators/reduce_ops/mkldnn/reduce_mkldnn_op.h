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

#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::LoDTensor;
using paddle::framework::Tensor;
using platform::to_void_cast;

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

    std::vector<int64_t> output_dims =
        CalculateOutputDims(input, output, reduce_dims, reduce_all, keep_dim);

    auto input_dims = framework::vectorize(input->dims());

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    // oneDNN reduce op does not support edge case in which memory is being
    // copied without actual reduction.
    // In that case reorder must be executed to maintain compatibility with
    // PaddlePaddle reduce op
    if (input_dims == output_dims) {
      mkldnn::memory::data_type input_type =
          framework::ToMKLDNNDataType(input->type());
      std::string key = platform::CreateKey(
          dev_ctx, input_dims, input->format(), input->format(), input_type);
      platform::ReorderMKLDNNHandler reorder_handler(
          input_dims, input->type(), input_type, dev_ctx, onednn_engine, key);

      auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
          input->format(), platform::to_void_cast(input->data<T>()));

      auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
          output, input->format(), ctx.GetPlace());

      auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                      reorder_dst_memory_p);

      platform::RecordEvent record_reorder("int_reorder",
                                           platform::EventRole::kUniqueOp);

      reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
      astream.wait();

      output->set_layout(framework::DataLayout::kMKLDNN);
      output->set_format(
          platform::GetMKLDNNFormat(reorder_dst_memory_p->get_desc().reshape(
              paddle::framework::vectorize<int64_t>(output->dims()))));
    } else {
      platform::ReductionMKLDNNHandler<T> handler(
          reduction_type, 0.0f, 0.0f, dev_ctx, onednn_engine, ctx.GetPlace(),
          input, output, ctx.InputName("X"), output_dims);

      auto src_memory_p = handler.AcquireSrcMemory(input);
      auto dst_memory_p = handler.AcquireDstMemory(output);

      std::unordered_map<int, dnnl::memory> reduction_args = {
          {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};

      auto reduction_p = handler.AcquireForwardPrimitive();

      reduction_p->execute(astream, reduction_args);
      astream.wait();
      output->set_layout(framework::DataLayout::kMKLDNN);
      output->set_format(
          platform::GetMKLDNNFormat(dst_memory_p->get_desc().reshape(
              paddle::framework::vectorize<int64_t>(output->dims()))));
    }
  }

 private:
  std::vector<int64_t> CalculateOutputDims(const Tensor* input,
                                           const Tensor* output,
                                           std::vector<int>& reduce_dims,
                                           bool reduce_all,
                                           bool keep_dim) const {
    if (keep_dim) return framework::vectorize(output->dims());

    if (reduce_all)
      return std::vector<int64_t>(framework::vectorize(input->dims()).size(),
                                  1);

    std::vector<int64_t> output_dims(framework::vectorize(input->dims()));
    for (size_t i = 0; i < reduce_dims.size(); ++i) {
      reduce_dims[i] = (reduce_dims[i] >= 0)
                           ? reduce_dims[i]
                           : input->dims().size() + reduce_dims[i];
      output_dims[reduce_dims[i]] = 1;
    }

    return output_dims;
  }
};

template <typename T>
class ReduceGradMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void RunKernel(const framework::ExecutionContext& ctx,
                 dnnl::algorithm binary_type, float scale_x,
                 float scale_y) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto dims = ctx.Attr<std::vector<int>>("dim");
    auto* input_dy = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* output_dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    output_dx->mutable_data<T>(ctx.GetPlace());
    output_dx->set_format(getPlainFormatTag(output_dx));
    output_dx->set_layout(input_dy->layout());

    platform::BroadcastDataMKLDNNHandler<T> handler(
        binary_type, dev_ctx, onednn_engine, ctx.GetPlace(), output_dx,
        input_dy, scale_x, scale_y,
        ctx.InputName(framework::GradVarName("Out")));

    const auto src_dx_memory = handler.AcquireSrcMemory(output_dx);
    const auto src_dy_memory = handler.AcquireSecondSrcMemory(input_dy);
    const auto binary_prim = handler.AcquireForwardPrimitive();

    const std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC_0, *src_dx_memory},
        {DNNL_ARG_SRC_1, *src_dy_memory},
        {DNNL_ARG_DST, *src_dx_memory}};

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    binary_prim->execute(astream, args);
    astream.wait();
  }

 protected:
  mkldnn::memory::format_tag getPlainFormatTag(const Tensor* tensor) const {
    auto tensor_dims_size = tensor->dims().size();
    PADDLE_ENFORCE_EQ(
        tensor_dims_size <= 5 && tensor_dims_size >= 1, true,
        platform::errors::InvalidArgument(
            "Dims for reduction_grad oneDNN op must be in range <1, 5>"));

    switch (tensor_dims_size) {
      case 1:
        return mkldnn::memory::format_tag::a;
      case 2:
        return mkldnn::memory::format_tag::ab;
      case 3:
        return mkldnn::memory::format_tag::abc;
      case 4:
        return mkldnn::memory::format_tag::abcd;
    }

    return mkldnn::memory::format_tag::abcde;
  }
};

}  // namespace operators
}  // namespace paddle
