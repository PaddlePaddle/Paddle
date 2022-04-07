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

#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using framework::LoDTensor;
using dnnl::memory;
using dnnl::primitive;
using dnnl::concat;
using dnnl::stream;
using platform::to_void_cast;

template <typename T>
class StackMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::concat> {
 public:
  StackMKLDNNHandler(const framework::ExecutionContext& ctx,
                     const dnnl::engine mkldnn_engine,
                     const std::vector<const Tensor*>& inputs, Tensor* output)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::concat>(mkldnn_engine,
                                                           ctx.GetPlace()) {
    int stack_axis = ctx.Attr<int>("axis");

    int ndims = inputs[0]->dims().size();

    if (stack_axis < 0) {
      stack_axis = ndims + 1 + stack_axis;  // +1 to match output's ndims
    }

    // in stack op all inputs must have same dims
    auto input_dims = phi::vectorize<int64_t>(inputs[0]->dims());

    memory::data_type dt = framework::ToMKLDNNDataType(
        framework::TransToProtoVarType(inputs[0]->dtype()));
    std::vector<memory::desc> srcs_md;
    memory::desc dst_md;
    MKLDNNMemoryFormat dst_fmt;

    srcs_md.reserve(inputs.size());

    // if stack is not done on last(non existing) axis, then we can optimize
    // concat primitive by not adding additional dimension, since it causes
    // wrong output format deduction and suboptimal performance as a result
    if (stack_axis != ndims) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        srcs_md.emplace_back(memory::desc(input_dims, dt, inputs[i]->format()));
      }

      input_dims[stack_axis] *= inputs.size();
      dst_md = memory::desc(input_dims, dt, MKLDNNMemoryFormat::any);
    } else {
      auto extended_input_dims = phi::vectorize<int64_t>(output->dims());
      extended_input_dims[stack_axis] = 1;

      for (size_t i = 0; i < inputs.size(); ++i) {
        srcs_md.emplace_back(memory::desc(input_dims, dt, inputs[i]->format())
                                 .reshape(extended_input_dims));
      }

      // concat primitive choses suboptimal format tag because it cannot
      // distinguish between f.e. abcd and abdc if last dim is equal to 1 so
      // enforcing is needed for better performance
      dst_fmt = platform::GetPlainMKLDNNFormat(extended_input_dims.size());
      dst_md = memory::desc(phi::vectorize(output->dims()), dt, dst_fmt);
    }

    this->AcquireForwardPrimitiveDescriptor(dst_md, stack_axis, srcs_md);
  }

  // concat oneDNN prim is not having .desc attribute so we cannot use default
  // AcquireForwardPrimitiveDescriptor
  void AcquireForwardPrimitiveDescriptor(
      const memory::desc& dst_md, const int stack_axis,
      const std::vector<memory::desc>& srcs_md) {
    this->fwd_pd_.reset(new dnnl::concat::primitive_desc(
        dst_md, stack_axis, srcs_md, this->engine_));
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const Tensor& input, int i) {
    const T* input_data = input.data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->src_desc(i),
                                            to_void_cast<T>(input_data));
  }
};

template <typename T>
class StackMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto multi_input = ctx.MultiInput<Tensor>("X");

    Tensor* output = ctx.Output<Tensor>("Y");

    StackMKLDNNHandler<T> handler(ctx, mkldnn_engine, multi_input, output);

    std::vector<std::shared_ptr<memory>> srcs;
    srcs.reserve(multi_input.size());

    auto dst_mem = handler.AcquireDstMemory(output);
    auto concat_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    std::unordered_map<int, memory> args;
    for (size_t i = 0; i < multi_input.size(); ++i) {
      srcs.push_back(handler.AcquireSrcMemory(*(multi_input[i]), i));
      args.insert({DNNL_ARG_MULTIPLE_SRC + i, *(srcs.at(i))});
    }
    args.insert({DNNL_ARG_DST, *dst_mem});

    concat_p->execute(astream, args);
    astream.wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(platform::GetMKLDNNFormat(
        dst_mem->get_desc().reshape(phi::vectorize(output->dims()))));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(stack, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::StackMKLDNNOpKernel<float>);
