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

#include "mkldnn.hpp"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/conv_op.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;

using mkldnn::memory;  // Note: paddle has also "memory" namespace
using mkldnn::primitive;
using mkldnn::convolution_forward;
using mkldnn::convolution_backward_weights;
using mkldnn::convolution_backward_data;
using mkldnn::convolution_direct;
using mkldnn::prop_kind;
using mkldnn::padding_kind;
using mkldnn::stream;

template <typename T>
class ConvOpMkldnnKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");

    // Get an unique name from "argument" name of "Output" variable
    // This name will be used as key when saving info into device context
    const std::string key = ctx.op().Output("Output");
    const std::string key_conv_pd = key + "@conv_pd";

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");

    // TODO(pzelazko-intel) enable group convolution
    PADDLE_ENFORCE(groups == 1, "MKLDNN doesn't support group convolution yet");
    PADDLE_ENFORCE(
        dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1,
        "MKLDNN doesn't support dilation in convolution yet");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    // allocate memory for output
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    PADDLE_ENFORCE(input->dims().size() == 4,
                   "Input must be with 4 dimensions, i.e. NCHW");
    PADDLE_ENFORCE(filter->dims().size() == 4,
                   "Filter must be with 4 dimensions, i.e. OIHW");

    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> weights_tz =
        paddle::framework::vectorize2int(filter->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

    // MKLDNN primitives
    // memory descriptors for convolution src/weight/dst
    memory::dims conv_src_tz = {src_tz[0], src_tz[1], src_tz[2], src_tz[3]};
    memory::dims conv_weights_tz = {weights_tz[0], weights_tz[1], weights_tz[2],
                                    weights_tz[3]};
    memory::dims conv_dst_tz = {dst_tz[0], dst_tz[1], dst_tz[2], dst_tz[3]};

    memory::dims conv_strides = {strides[0], strides[1]};
    memory::dims conv_padding = {paddings[0], paddings[1]};

    // TODO(pzelazko-intel): support more formats
    auto conv_src_md = memory::desc({conv_src_tz}, memory::data_type::f32,
                                    memory::format::nchw);
    auto conv_weights_md = memory::desc(
        {conv_weights_tz}, memory::data_type::f32, memory::format::oihw);
    auto conv_dst_md = memory::desc({conv_dst_tz}, memory::data_type::f32,
                                    memory::format::nchw);

    // create memory primitives
    auto conv_src_memory =
        memory({conv_src_md, mkldnn_engine}, (void*)input_data);
    auto conv_weights_memory =
        memory({conv_weights_md, mkldnn_engine}, (void*)filter_data);
    auto conv_dst_memory = memory({conv_dst_md, mkldnn_engine}, output_data);

    // create convolution op descriptor
    auto conv_desc = convolution_forward::desc(
        prop_kind::forward, convolution_direct, conv_src_md, conv_weights_md,
        conv_dst_md, conv_strides, conv_padding, conv_padding,
        padding_kind::zero);

    // conv primitive desc need be used in backward path
    // so we need allocate it in heap (instead of in stack)
    convolution_forward::primitive_desc* p_conv_pd;
    p_conv_pd =
        new convolution_forward::primitive_desc(conv_desc, mkldnn_engine);
    // save conv_pd into dev_ctx to be referred in backward path
    std::shared_ptr<void> conv_pd;
    conv_pd.reset(p_conv_pd);
    dev_ctx.SetBlob(key_conv_pd, conv_pd);

    // create convolution op primitive
    auto conv_prim = convolution_forward(*p_conv_pd, conv_src_memory,
                                         conv_weights_memory, conv_dst_memory);

    // push op to stream and wait MKLDNN until it's executed
    std::vector<primitive> pipeline{conv_prim};
    stream(stream::kind::eager).submit(pipeline).wait();
  }
};

template <typename T>
class ConvGradOpMkldnnKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    const Tensor* output = ctx.Input<Tensor>("Output");
    const Tensor* output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    if (!input_grad && !filter_grad) return;

    // Get an unique name from "argument" name of "Output" variable
    // This name will be used as key when saving info into device context
    const std::string key = ctx.op().Input("Output");
    const std::string key_conv_pd = key + "@conv_pd";

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    const T* output_grad_data = output_grad->data<T>();
    T* input_grad_data = nullptr;
    T* filter_grad_data = nullptr;

    // allocate memory for gradient of input/filter
    if (input_grad) {
      input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
    }
    if (filter_grad) {
      filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
    }

    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> weights_tz =
        paddle::framework::vectorize2int(filter->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

    // MKLDNN primitives
    // memory descriptors for convolution src/weight/dst
    memory::dims conv_src_tz = {src_tz[0], src_tz[1], src_tz[2], src_tz[3]};
    memory::dims conv_weights_tz = {weights_tz[0], weights_tz[1], weights_tz[2],
                                    weights_tz[3]};
    memory::dims conv_dst_tz = {dst_tz[0], dst_tz[1], dst_tz[2], dst_tz[3]};

    memory::dims conv_strides = {strides[0], strides[1]};
    memory::dims conv_padding = {paddings[0], paddings[1]};

    // TODO(pzelazko-intel): support more formats
    auto conv_src_md = memory::desc({conv_src_tz}, memory::data_type::f32,
                                    memory::format::nchw);
    auto conv_diff_src_md = memory::desc({conv_src_tz}, memory::data_type::f32,
                                         memory::format::nchw);
    auto conv_weights_md = memory::desc(
        {conv_weights_tz}, memory::data_type::f32, memory::format::oihw);
    auto conv_diff_weights_md = memory::desc(
        {conv_weights_tz}, memory::data_type::f32, memory::format::oihw);
    auto conv_diff_dst_md = memory::desc({conv_dst_tz}, memory::data_type::f32,
                                         memory::format::nchw);

    // create memory
    auto conv_diff_dst_memory =
        memory({conv_diff_weights_md, mkldnn_engine}, (void*)output_grad_data);
    // Retrieve conv_pd from device context
    std::shared_ptr<void> conv_pd;
    convolution_forward::primitive_desc* p_conv_pd;

    conv_pd = dev_ctx.GetBlob(key_conv_pd);
    PADDLE_ENFORCE(conv_pd != nullptr,
                   "Fail to find conv_pd in device context");
    p_conv_pd =
        static_cast<convolution_forward::primitive_desc*>(conv_pd.get());

    // create backward conv primitive for weights
    if (filter_grad) {
      // create memory
      auto conv_diff_weights_memory = memory(
          {conv_diff_weights_md, mkldnn_engine}, (void*)filter_grad_data);
      auto conv_src_memory =
          memory({conv_src_md, mkldnn_engine}, (void*)input_data);

      // create primitive descriptor
      auto conv_bwd_weights_desc = convolution_backward_weights::desc(
          convolution_direct, conv_src_md, conv_diff_weights_md,
          conv_diff_dst_md, conv_strides, conv_padding, conv_padding,
          padding_kind::zero);
      auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
          conv_bwd_weights_desc, mkldnn_engine,
          *p_conv_pd);  // Need to hint forward desc

      // create backward conv primitive for weights
      auto conv_bwd_weights_prim = convolution_backward_weights(
          conv_bwd_weights_pd, conv_src_memory, conv_diff_dst_memory,
          conv_diff_weights_memory);

      // push primitive and execute it
      std::vector<primitive> pipeline{conv_bwd_weights_prim};
      stream(stream::kind::eager).submit(pipeline).wait();
    }

    if (input_grad) {
      // create memory
      auto conv_diff_src_memory =
          memory({conv_diff_src_md, mkldnn_engine}, (void*)input_grad_data);
      auto conv_weights_memory =
          memory({conv_weights_md, mkldnn_engine}, (void*)filter_data);
      // create primitive descriptor
      auto conv_bwd_data_desc = convolution_backward_data::desc(
          convolution_direct, conv_diff_src_md, conv_weights_md,
          conv_diff_dst_md, conv_strides, conv_padding, conv_padding,
          padding_kind::zero);
      auto conv_bwd_data_pd = convolution_backward_data::primitive_desc(
          conv_bwd_data_desc, mkldnn_engine,
          *p_conv_pd);  // Need to hint forward desc

      // create backward conv primitive for data
      auto conv_bwd_data_prim =
          convolution_backward_data(conv_bwd_data_pd, conv_diff_dst_memory,
                                    conv_weights_memory, conv_diff_src_memory);

      // push primitive and execute it
      std::vector<primitive> pipeline{conv_bwd_data_prim};
      stream(stream::kind::eager).submit(pipeline).wait();
    }
  }  // Compute()
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(conv2d, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvOpMkldnnKernel<float>);

REGISTER_OP_KERNEL(conv2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvGradOpMkldnnKernel<float>);
