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
using framework::LoDTensor;
using dnnl::memory;
using dnnl::primitive;
using dnnl::concat;
using dnnl::stream;
using platform::to_void_cast;

template <typename T>
class ConcatMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::concat> {
 public:
  ConcatMKLDNNHandler(const framework::ExecutionContext& ctx,
                      const dnnl::engine mkldnn_engine,
                      const std::vector<const Tensor*>& inputs, Tensor* output)
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

    memory::data_type dt = framework::ToMKLDNNDataType(inputs[0]->type());
    std::vector<memory::desc> srcs_md;
    srcs_md.reserve(inputs.size());

    // Create memory descriptors for each of inputs
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto dims = framework::vectorize<int64_t>(inputs[i]->dims());
      srcs_md.emplace_back(memory::desc(dims, dt, inputs[i]->format()));
    }

    auto dst_dims = framework::vectorize<int64_t>(output->dims());
    auto dst_md = memory::desc(dst_dims, dt, MKLDNNMemoryFormat::any);

    this->AcquireForwardPrimitiveDescriptor(dst_md, concat_axis, srcs_md);
  }

  // (jczaja) concat oneDNN prim is not having .desc attribute so
  // we cannot use base AcquireForwardPrimitiveDescriptor
  void AcquireForwardPrimitiveDescriptor(
      const memory::desc& dst_md, const int concat_axis,
      const std::vector<memory::desc>& srcs_md) {
    this->fwd_pd_.reset(new dnnl::concat::primitive_desc(
        dst_md, concat_axis, srcs_md, this->engine_));
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const Tensor& input, int i) {
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
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    // If any of the multiple inputs of concat has an input size of 0, the
    // actual size of the multi_input will change
    auto multi_input = ReduceMultiInput(ctx.MultiInput<Tensor>("X"));
    EnforceLayouts(multi_input);
    Tensor* output = ctx.Output<Tensor>("Out");

    ConcatMKLDNNHandler<T> handler(ctx, mkldnn_engine, multi_input, output);

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
    output->set_format(platform::GetMKLDNNFormat(*dst_mem));
  }
};

template <typename T>
class ConcatGradMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    auto out_var_names = ctx.OutputNames(framework::GradVarName("X"));

    const auto x = ctx.MultiInput<LoDTensor>("X");
    const auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto dx = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));

    for (size_t i = 0; i < dx.size(); ++i) {
      if (dx[i] != nullptr) {
        dx[i]->set_lod(x[i]->lod());
      }
    }

    int axis = ctx.Attr<int>("axis");
    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<Tensor>("AxisTensor");
      axis = GetDataFromTensor<int>(axis_tensor)[0];
    }

    auto dout_vec_dims = framework::vectorize(dout->dims());

    axis = ComputeAxis(axis, dout_vec_dims.size());

    std::vector<int64_t> offset(dout_vec_dims.size(), 0);

    dnnl::memory::data_type dout_type =
        framework::ToMKLDNNDataType(dout->type());
    platform::ReorderMKLDNNHandler reorder_handler(dout_vec_dims, dout->type(),
                                                   dout_type, onednn_engine);
    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        dout->format(), platform::to_void_cast(dout->data<T>()));

    for (size_t i = 0; i < dx.size(); ++i) {
      if (out_var_names[i] != framework::kEmptyVarName &&
          dx[i]->numel() != 0UL) {
        auto dx_vec_dims = framework::vectorize(dx[i]->dims());
        auto slice_mem_p = reorder_handler.AcquireSubmemory(
            dx_vec_dims, offset, reorder_src_memory_p);

        auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
            dx[i], dx_vec_dims, dout->format(), ctx.GetPlace());
        auto reorder_p =
            reorder_handler.AcquireReorder(reorder_dst_memory_p, slice_mem_p);

        reorder_p->execute(astream, *slice_mem_p, *reorder_dst_memory_p);

        offset[axis] += dx[i]->dims()[axis];

        dx[i]->set_layout(framework::DataLayout::kMKLDNN);
        dx[i]->set_format(platform::GetMKLDNNFormat(*reorder_dst_memory_p));
      }
    }
    astream.wait();
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

REGISTER_OP_KERNEL(concat_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConcatGradMKLDNNOpKernel<float>,
                   ops::ConcatGradMKLDNNOpKernel<paddle::platform::bfloat16>);
