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

#include <memory>
#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::concat;
using mkldnn::stream;
using platform::to_void_cast;

template <typename T>
class ConcatMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::concat> {
 public:
  ConcatMKLDNNHandler(const paddle::framework::ExecutionContext& ctx,
                      const mkldnn::engine mkldnn_engine,
                      const std::vector<const Tensor*> inputs, Tensor* output)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::concat>(mkldnn_engine,
                                                           ctx.GetPlace()) {
    int concat_axis = ctx.Attr<int>("axis");
    const int rank = inputs[0]->dims().size();
    PADDLE_ENFORCE_EQ(
        concat_axis >= -rank && concat_axis < rank, true,
        platform::errors::InvalidArgument(
            "The axis is expected to be in range of [%d, %d), but got %d",
            -rank, rank, concat_axis));

    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<Tensor>("AxisTensor");
      concat_axis = GetDataFromTensor(axis_tensor)[0];
      auto out_dims = inputs[0]->dims();
      for (size_t i = 1; i < inputs.size(); ++i) {
        out_dims[concat_axis] += inputs[i]->dims()[concat_axis];
      }
      output->Resize(out_dims);
    }

    if (concat_axis < 0) {
      concat_axis = concat_axis + rank;
    }

    memory::data_type dt =
        paddle::framework::ToMKLDNNDataType(inputs[0]->type());
    std::vector<memory::desc> srcs_md(inputs.size());

    // Create memory descriptors for each of inputs
    const auto dims = paddle::framework::vectorize<int64_t>(inputs[0]->dims());
    for (size_t i = 0; i < inputs.size(); i++) {
      srcs_md.emplace_back(memory::desc(dims, dt, inputs[i]->format()));
    }

    auto dst_dims = paddle::framework::vectorize<int64_t>(output->dims());
    auto dst_md = memory::desc(dst_dims, dt, MKLDNNMemoryFormat::any);

    this->AcquireForwardPrimitiveDescriptor(dst_md, concat_axis, srcs_md);
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(
      const framework::Tensor& input, int i) {
    const T* input_data = input.data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->src_desc(i),
                                            to_void_cast<T>(input_data));
  }
};

static void EnforceLayouts(const std::vector<const Tensor*> inputs) {
  for (auto* input : inputs) {
    PADDLE_ENFORCE_EQ(
        input->layout(), DataLayout::kMKLDNN,
        platform::errors::InvalidArgument("Wrong layout set for Input tensor"));
    PADDLE_ENFORCE_NE(
        input->format(), MKLDNNMemoryFormat::undef,
        platform::errors::InvalidArgument("Wrong format set for Input tensor"));
  }
}

// From a multi-input, gather only nonempty inputs
static const std::vector<const Tensor*> ReduceMultiInput(
    const std::vector<const Tensor*>& inputs) {
  std::vector<const Tensor*> reduced(inputs.size());
  auto end_it = std::copy_if(inputs.begin(), inputs.end(), reduced.begin(),
                             [](const Tensor* t) { return t->numel() > 0; });
  reduced.resize(std::distance(reduced.begin(), end_it));
  return reduced;
}

template <typename T>
class ConcatMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    // If any of the multiple inputs of concat has an input size of 0, the
    // actual size of the multi_input will change
    auto multi_input = ReduceMultiInput(ctx.MultiInput<Tensor>("X"));
    EnforceLayouts(multi_input);
    Tensor* output = ctx.Output<Tensor>("Out");

    ConcatMKLDNNHandler<T> handler(ctx, mkldnn_engine, multi_input, output);

    std::vector<std::shared_ptr<memory>> srcs(multi_input.size());

    auto dst_mem = handler.AcquireDstMemory(output);
    auto concat_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    std::unordered_map<int, memory> args;
    for (size_t i = 0; i < multi_input.size(); ++i) {
      srcs.push_back(handler.AcquireSrcMemory(multi_input[i], i));
      args.insert({MKLDNN_ARG_MULTIPLE_SRC + i, (*srcs).at(i)});
    }
    args.insert({MKLDNN_ARG_DST, *dst_mem});

    concat_p->execute(astream, args);
    astream.wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(platform::GetMKLDNNFormat(*dst_mem));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(concat, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConcatMKLDNNOpKernel<float>,
                   ops::ConcatMKLDNNOpKernel<paddle::platform::bfloat16>,
                   ops::ConcatMKLDNNOpKernel<int8_t>,
                   ops::ConcatMKLDNNOpKernel<uint8_t>);
