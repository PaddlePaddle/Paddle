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

#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

template <typename T>
class ConvMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
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

    // TODO(pzelazko-intel) add support for group convolution and dilation
    PADDLE_ENFORCE(groups == 1, "group convolution is not implemented yet");
    PADDLE_ENFORCE(
        dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1,
        "dilation in convolution is not implemented yet");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    PADDLE_ENFORCE(input->dims().size() == 4,
                   "Input must be with 4 dimensions, i.e. NCHW");
    PADDLE_ENFORCE(filter->dims().size() == 4,
                   "Filter must be with 4 dimensions, i.e. OIHW");

    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> weights_tz =
        paddle::framework::vectorize2int(filter->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

    // TODO(pzelazko-intel): support more formats
    auto src_md = platform::MKLDNNMemDesc(
        src_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw);
    auto weights_md =
        platform::MKLDNNMemDesc(weights_tz, mkldnn::memory::data_type::f32,
                                mkldnn::memory::format::oihw);
    auto dst_md = platform::MKLDNNMemDesc(
        dst_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw);

    auto src_memory =
        mkldnn::memory({src_md, mkldnn_engine},
                       reinterpret_cast<void*>(const_cast<T*>(input_data)));
    auto weights_memory =
        mkldnn::memory({weights_md, mkldnn_engine},
                       reinterpret_cast<void*>(const_cast<T*>(filter_data)));
    auto dst_memory = mkldnn::memory({dst_md, mkldnn_engine}, output_data);

    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd =
        ConvFwdPrimitiveDesc(src_md, weights_md, dst_md, strides, paddings,
                             mkldnn_engine);

    // save conv_pd into global device context to be referred in backward path
    dev_ctx.SetBlob(key_conv_pd, conv_pd);

    // create convolution op primitive
    auto conv_prim = mkldnn::convolution_forward(*conv_pd, src_memory,
                                                 weights_memory, dst_memory);

    // push primitive to stream and wait until it's executed
    std::vector<mkldnn::primitive> pipeline{conv_prim};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
  }

 private:
  std::unique_ptr<mkldnn::convolution_forward::primitive_desc>
  ConvFwdPrimitiveDesc(const mkldnn::memory::desc& src,
                       const mkldnn::memory::desc& weights,
                       const mkldnn::memory::desc& dst,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const mkldnn::engine& engine) const {
    mkldnn::memory::dims stride_dims = {strides[0], strides[1]};
    mkldnn::memory::dims padding_dims = {paddings[0], paddings[1]};

    auto conv_desc = mkldnn::convolution_forward::desc(
        mkldnn::prop_kind::forward, mkldnn::convolution_direct, src, weights,
        dst, stride_dims, padding_dims, padding_dims,
        mkldnn::padding_kind::zero);

    auto p_conv_pd =
        new mkldnn::convolution_forward::primitive_desc(conv_desc, engine);

    return std::unique_ptr<mkldnn::convolution_forward::primitive_desc>(
        p_conv_pd);
  }
};

template <typename T>
class ConvMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
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

    // TODO(pzelazko-intel): support more formats
    auto src_md = platform::MKLDNNMemDesc(
        src_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw);
    auto diff_src_md = platform::MKLDNNMemDesc(
        src_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw);
    auto weights_md =
        platform::MKLDNNMemDesc(weights_tz, mkldnn::memory::data_type::f32,
                                mkldnn::memory::format::oihw);
    auto diff_weights_md =
        platform::MKLDNNMemDesc(weights_tz, mkldnn::memory::data_type::f32,
                                mkldnn::memory::format::oihw);
    auto diff_dst_md = platform::MKLDNNMemDesc(
        dst_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw);

    // create memory
    auto diff_dst_memory = mkldnn::memory(
        {diff_weights_md, mkldnn_engine},
        reinterpret_cast<void*>(const_cast<T*>(output_grad_data)));
    // Retrieve conv_pd from device context
    auto conv_pd =
        std::static_pointer_cast<mkldnn::convolution_forward::primitive_desc>(
            dev_ctx.GetBlob(key_conv_pd));
    PADDLE_ENFORCE(conv_pd != nullptr,
                   "Fail to find conv_pd in device context");

    // create backward conv primitive for weights
    if (filter_grad) {
      // create primitive descriptor
      mkldnn::convolution_backward_weights::primitive_desc conv_bwd_weights_pd =
          ConvBwdWeightsPrimitiveDesc(src_md, diff_weights_md, diff_dst_md,
                                      strides, paddings, *conv_pd,
                                      mkldnn_engine);

      // create memory
      auto diff_weights_memory =
          mkldnn::memory({diff_weights_md, mkldnn_engine},
                         reinterpret_cast<void*>(filter_grad_data));
      auto src_memory =
          mkldnn::memory({src_md, mkldnn_engine},
                         reinterpret_cast<void*>(const_cast<T*>(input_data)));

      // create backward conv primitive for weights
      auto conv_bwd_weights_prim = mkldnn::convolution_backward_weights(
          conv_bwd_weights_pd, src_memory, diff_dst_memory,
          diff_weights_memory);

      // push primitive and execute it
      std::vector<mkldnn::primitive> pipeline{conv_bwd_weights_prim};
      mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
    }

    if (input_grad) {
      // create primitive descriptor
      mkldnn::convolution_backward_data::primitive_desc conv_bwd_data_pd =
          ConvBwdDataPrimitiveDesc(diff_src_md, weights_md, diff_dst_md,
                                   strides, paddings, *conv_pd, mkldnn_engine);

      // create memory
      auto diff_src_memory = mkldnn::memory(
          {diff_src_md, mkldnn_engine},
          reinterpret_cast<void*>(const_cast<T*>(input_grad_data)));
      auto weights_memory =
          mkldnn::memory({weights_md, mkldnn_engine},
                         reinterpret_cast<void*>(const_cast<T*>(filter_data)));

      // create backward conv primitive for data
      auto conv_bwd_data_prim = mkldnn::convolution_backward_data(
          conv_bwd_data_pd, diff_dst_memory, weights_memory, diff_src_memory);

      // push primitive to stream and wait until it's executed
      std::vector<mkldnn::primitive> pipeline{conv_bwd_data_prim};
      mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
    }
  }  // Compute()

 private:
  mkldnn::convolution_backward_weights::primitive_desc
  ConvBwdWeightsPrimitiveDesc(
      const mkldnn::memory::desc& src, const mkldnn::memory::desc& diff_weights,
      const mkldnn::memory::desc& diff_dst, const std::vector<int>& strides,
      const std::vector<int>& paddings,
      const mkldnn::convolution_forward::primitive_desc& conv_pd,
      const mkldnn::engine& engine) const {
    auto conv_bwd_weights_desc = mkldnn::convolution_backward_weights::desc(
        mkldnn::convolution_direct, src, diff_weights, diff_dst, strides,
        paddings, paddings, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_weights::primitive_desc(
        conv_bwd_weights_desc, engine, conv_pd);
  }

  mkldnn::convolution_backward_data::primitive_desc ConvBwdDataPrimitiveDesc(
      const mkldnn::memory::desc& diff_src, const mkldnn::memory::desc& weights,
      const mkldnn::memory::desc& diff_dst, const std::vector<int>& strides,
      const std::vector<int>& paddings,
      const mkldnn::convolution_forward::primitive_desc& conv_pd,
      const mkldnn::engine& engine) const {
    auto conv_bwd_data_desc = mkldnn::convolution_backward_data::desc(
        mkldnn::convolution_direct, diff_src, weights, diff_dst, strides,
        paddings, paddings, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_data::primitive_desc(conv_bwd_data_desc,
                                                             engine, conv_pd);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(conv2d, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(conv2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvMKLDNNGradOpKernel<float>);
