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

#include "paddle/fluid/framework/mkldnn_tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using framework::MKLDNNTensor;
using framework::MKLDNNTensorMutable;

template <typename T>
class ConvMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    // Get an unique name from "argument" name of "Output" variable
    // This name will be used as key when saving info into device context
    const std::string key = ctx.op().Output("Output");
    const std::string key_conv_pd = key + "@conv_pd";
    const std::string key_weights = key + "@weights";

    Tensor* input = ctx.MutableInput<Tensor>("Input");
    Tensor* filter = ctx.MutableInput<Tensor>("Filter");
    Tensor* output = ctx.Output<Tensor>("Output");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");

    // TODO(pzelazko-intel) add support for group convolution and dilation
    PADDLE_ENFORCE(groups == 1, "group convolution is not implemented yet");
    PADDLE_ENFORCE(
        dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1,
        "dilation in convolution is not implemented yet");

    PADDLE_ENFORCE(input->dims().size() == 4,
                   "Input must be with 4 dimensions, i.e. NCHW");
    PADDLE_ENFORCE(filter->dims().size() == 4,
                   "Filter must be with 4 dimensions, i.e. OIHW");

    std::vector<int> input_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> filter_tz =
        paddle::framework::vectorize2int(filter->dims());
    std::vector<int> output_tz =
        paddle::framework::vectorize2int(output->dims());

    auto input_md = platform::MKLDNNMemDesc(
        input_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
    auto filter_md = platform::MKLDNNMemDesc(
        filter_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
    auto output_md = platform::MKLDNNMemDesc(
        output_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);

    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd =
        ConvFwdPrimitiveDesc(input_md, filter_md, output_md, strides, paddings,
                             mkldnn_engine);

    // save conv_pd into global device context to be referred in backward path
    dev_ctx.SetBlob(key_conv_pd, conv_pd);

    MKLDNNTensorMutable input_mkldnn =
        MKLDNNTensorMutable::Create(input, mkldnn_engine);
    MKLDNNTensorMutable filter_mkldnn =
        MKLDNNTensorMutable::Create(filter, mkldnn_engine);
    MKLDNNTensorMutable output_mkldnn =
        MKLDNNTensorMutable::Create(output, mkldnn_engine);

    Reorder(conv_pd, &input_mkldnn, &output_mkldnn, &filter_mkldnn);

    auto input_memory = input_mkldnn.GetMemory();
    auto filter_memory = filter_mkldnn.GetMemory();
    auto output_memory = output_mkldnn.GetMutableMemory(ctx.GetPlace());

    // create convolution op primitive
    auto conv_prim = mkldnn::convolution_forward(*conv_pd, input_memory,
                                                 filter_memory, output_memory);

    // push primitive to stream and wait until it's executed
    std::vector<mkldnn::primitive> pipeline{conv_prim};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
  }

 private:
  std::unique_ptr<mkldnn::convolution_forward::primitive_desc>
  ConvFwdPrimitiveDesc(const mkldnn::memory::desc& input,
                       const mkldnn::memory::desc& weights,
                       const mkldnn::memory::desc& output,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const mkldnn::engine& engine) const {
    mkldnn::memory::dims stride_dims = {strides[0], strides[1]};
    mkldnn::memory::dims padding_dims = {paddings[0], paddings[1]};

    auto conv_desc = mkldnn::convolution_forward::desc(
        mkldnn::prop_kind::forward, mkldnn::convolution_direct, input, weights,
        output, stride_dims, padding_dims, padding_dims,
        mkldnn::padding_kind::zero);

    auto p_conv_pd =
        new mkldnn::convolution_forward::primitive_desc(conv_desc, engine);

    return std::unique_ptr<mkldnn::convolution_forward::primitive_desc>(
        p_conv_pd);
  }

  void Reorder(
      std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd,
      MKLDNNTensorMutable* input, MKLDNNTensorMutable* output,
      MKLDNNTensorMutable* filter) const {
    auto input_format =
        mkldnn::memory::primitive_desc(conv_pd->src_primitive_desc())
            .desc()
            .data.format;
    auto output_format =
        mkldnn::memory::primitive_desc(conv_pd->dst_primitive_desc())
            .desc()
            .data.format;
    auto filter_format =
        mkldnn::memory::primitive_desc(conv_pd->weights_primitive_desc())
            .desc()
            .data.format;

    if (input->GetFormat() != input_format) {
      VLOG(3) << "input " << input->GetFormat() << " -> " << input_format;
    }
    if (output->GetFormat() != output_format) {
      VLOG(3) << "output " << output->GetFormat() << " -> " << output_format;
    }
    if (filter->GetFormat() != filter_format) {
      VLOG(3) << "weights " << filter->GetFormat() << " -> " << filter_format;
    }

    input->Reorder(static_cast<mkldnn::memory::format>(input_format));
    output->SetFormat(static_cast<mkldnn::memory::format>(output_format));
    filter->Reorder(static_cast<mkldnn::memory::format>(filter_format));
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

    Tensor* input = ctx.MutableInput<Tensor>("Input");
    Tensor* filter = ctx.MutableInput<Tensor>("Filter");
    Tensor* output = ctx.MutableInput<Tensor>("Output");
    Tensor* output_grad =
        ctx.MutableInput<Tensor>(framework::GradVarName("Output"));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto* filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    if (!input_grad && !filter_grad) return;

    MKLDNNTensorMutable input_mkldnn =
        MKLDNNTensorMutable::Create(input, mkldnn_engine);
    MKLDNNTensorMutable filter_mkldnn =
        MKLDNNTensorMutable::Create(filter, mkldnn_engine);
    MKLDNNTensorMutable output_mkldnn =
        MKLDNNTensorMutable::Create(output, mkldnn_engine);
    MKLDNNTensorMutable output_grad_mkldnn =
        MKLDNNTensorMutable::Create(output_grad, mkldnn_engine);
    MKLDNNTensorMutable filter_grad_mkldnn =
        MKLDNNTensorMutable::Create(filter_grad, mkldnn_engine);

    // Get an unique name from "argument" name of "Output" variable
    // This name will be used as key when saving info into device context
    const std::string key = ctx.op().Input("Output");
    const std::string key_conv_pd = key + "@conv_pd";

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");

    std::vector<int> input_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> filter_tz =
        paddle::framework::vectorize2int(filter->dims());
    std::vector<int> output_tz =
        paddle::framework::vectorize2int(output->dims());

    auto input_md = platform::MKLDNNMemDesc(
        input_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
    auto input_grad_md = platform::MKLDNNMemDesc(
        input_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
    auto filter_md = platform::MKLDNNMemDesc(
        filter_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
    auto filter_grad_md = platform::MKLDNNMemDesc(
        filter_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
    auto output_grad_md = platform::MKLDNNMemDesc(
        output_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);

    // Retrieve conv_pd from device context
    auto conv_pd =
        std::static_pointer_cast<mkldnn::convolution_forward::primitive_desc>(
            dev_ctx.GetBlob(key_conv_pd));
    PADDLE_ENFORCE(conv_pd != nullptr,
                   "Fail to find conv_pd in device context");

    // execute backward conv primitive for weights
    if (filter_grad) {
      mkldnn::convolution_backward_weights::primitive_desc conv_bwd_weights_pd =
          ConvBwdWeightsPrimitiveDesc(input_md, filter_grad_md, output_grad_md,
                                      strides, paddings, *conv_pd,
                                      mkldnn_engine);

      Reorder(conv_bwd_weights_pd, &input_mkldnn, &output_grad_mkldnn,
              &filter_grad_mkldnn);

      auto input_memory = input_mkldnn.GetMemory();
      auto output_grad_memory = output_grad_mkldnn.GetMemory();
      auto filter_grad_memory =
          filter_grad_mkldnn.GetMutableMemory(ctx.GetPlace());

      auto conv_bwd_weights_prim = mkldnn::convolution_backward_weights(
          conv_bwd_weights_pd, input_memory, output_grad_memory,
          filter_grad_memory);

      std::vector<mkldnn::primitive> pipeline{conv_bwd_weights_prim};
      mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
    }

    // execute backward conv primitive for input
    if (input_grad) {
      mkldnn::convolution_backward_data::primitive_desc conv_bwd_data_pd =
          ConvBwdDataPrimitiveDesc(input_grad_md, filter_md, output_grad_md,
                                   strides, paddings, *conv_pd, mkldnn_engine);

      MKLDNNTensorMutable input_grad_mkldnn =
          MKLDNNTensorMutable::Create(input_grad, mkldnn_engine);

      // reorder tensor layout for optimal performance
      Reorder(conv_bwd_data_pd, &output_grad_mkldnn, &filter_mkldnn,
              &input_grad_mkldnn);

      auto output_grad_memory = output_grad_mkldnn.GetMemory();
      auto filter_memory = filter_grad_mkldnn.GetMemory();
      auto input_grad_memory =
          input_grad_mkldnn.GetMutableMemory(ctx.GetPlace());

      auto conv_bwd_data_prim = mkldnn::convolution_backward_data(
          conv_bwd_data_pd, output_grad_memory, filter_memory,
          input_grad_memory);

      std::vector<mkldnn::primitive> pipeline{conv_bwd_data_prim};
      mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
    }
  }  // Compute()

 private:
  mkldnn::convolution_backward_weights::primitive_desc
  ConvBwdWeightsPrimitiveDesc(
      const mkldnn::memory::desc& input,
      const mkldnn::memory::desc& filter_grad,
      const mkldnn::memory::desc& output_grad, const std::vector<int>& strides,
      const std::vector<int>& paddings,
      const mkldnn::convolution_forward::primitive_desc& conv_pd,
      const mkldnn::engine& engine) const {
    auto conv_bwd_weights_desc = mkldnn::convolution_backward_weights::desc(
        mkldnn::convolution_direct, input, filter_grad, output_grad, strides,
        paddings, paddings, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_weights::primitive_desc(
        conv_bwd_weights_desc, engine, conv_pd);
  }

  mkldnn::convolution_backward_data::primitive_desc ConvBwdDataPrimitiveDesc(
      const mkldnn::memory::desc& input_grad,
      const mkldnn::memory::desc& weights,
      const mkldnn::memory::desc& output_grad, const std::vector<int>& strides,
      const std::vector<int>& paddings,
      const mkldnn::convolution_forward::primitive_desc& conv_pd,
      const mkldnn::engine& engine) const {
    auto conv_bwd_data_desc = mkldnn::convolution_backward_data::desc(
        mkldnn::convolution_direct, input_grad, weights, output_grad, strides,
        paddings, paddings, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_data::primitive_desc(conv_bwd_data_desc,
                                                             engine, conv_pd);
  }

  void Reorder(const mkldnn::convolution_backward_weights::primitive_desc&
                   bwd_weights_pd,
               MKLDNNTensorMutable* input, MKLDNNTensorMutable* output_grad,
               MKLDNNTensorMutable* filter_grad) const {
    auto input_format =
        mkldnn::memory::primitive_desc(bwd_weights_pd.src_primitive_desc())
            .desc()
            .data.format;
    auto output_grad_format =
        mkldnn::memory::primitive_desc(bwd_weights_pd.diff_dst_primitive_desc())
            .desc()
            .data.format;
    auto filter_grad_format = mkldnn::memory::primitive_desc(
                                  bwd_weights_pd.diff_weights_primitive_desc())
                                  .desc()
                                  .data.format;

    if (input->GetFormat() != input_format) {
      VLOG(3) << "input " << input->GetFormat() << " -> " << input_format;
    }
    if (output_grad->GetFormat() != output_grad_format) {
      VLOG(3) << "output_grad " << output_grad->GetFormat() << " -> "
              << output_grad_format;
    }
    if (filter_grad->GetFormat() != filter_grad_format) {
      VLOG(3) << "filter_grad " << filter_grad->GetFormat() << " -> "
              << filter_grad_format;
    }

    input->Reorder(static_cast<mkldnn::memory::format>(input_format));
    output_grad->Reorder(
        static_cast<mkldnn::memory::format>(output_grad_format));
    filter_grad->SetFormat(
        static_cast<mkldnn::memory::format>(filter_grad_format));
  }

  void Reorder(
      const mkldnn::convolution_backward_data::primitive_desc& bwd_data_pd,
      MKLDNNTensorMutable* output_grad, MKLDNNTensorMutable* filter,
      MKLDNNTensorMutable* input_grad) const {
    auto output_grad_format =
        mkldnn::memory::primitive_desc(bwd_data_pd.diff_dst_primitive_desc())
            .desc()
            .data.format;
    auto filter_format =
        mkldnn::memory::primitive_desc(bwd_data_pd.weights_primitive_desc())
            .desc()
            .data.format;
    auto input_grad_format =
        mkldnn::memory::primitive_desc(bwd_data_pd.diff_src_primitive_desc())
            .desc()
            .data.format;

    if (output_grad->GetFormat() != output_grad_format) {
      VLOG(3) << "output_grad " << output_grad->GetFormat() << " -> "
              << output_grad_format;
    }
    if (filter->GetFormat() != filter_format) {
      VLOG(3) << "filter " << filter->GetFormat() << " -> " << filter_format;
    }
    if (input_grad->GetFormat() != input_grad_format) {
      VLOG(3) << "input_grad " << input_grad->GetFormat() << " -> "
              << input_grad_format;
    }

    output_grad->Reorder(
        static_cast<mkldnn::memory::format>(output_grad_format));
    filter->Reorder(static_cast<mkldnn::memory::format>(filter_format));
    input_grad->SetFormat(
        static_cast<mkldnn::memory::format>(input_grad_format));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(conv2d, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(conv2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvMKLDNNGradOpKernel<float>);
