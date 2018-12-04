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

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;

template <typename T>
class ConvMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    const bool is_test = ctx.Attr<bool>("is_test");

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* bias = ctx.HasInput("Bias") ? ctx.Input<Tensor>("Bias") : nullptr;
    auto* output = ctx.Output<Tensor>("Output");

    PADDLE_ENFORCE(input->layout() == DataLayout::kMKLDNN &&
                       input->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input tensor");
    PADDLE_ENFORCE(filter->layout() == DataLayout::kMKLDNN &&
                       filter->format() != memory::format::format_undef,
                   "Wrong layout/format set for Filter tensor");
    PADDLE_ENFORCE(input->dims().size() == 4,
                   "Input must be with 4 dimensions, i.e. NCHW");
    PADDLE_ENFORCE(filter->dims().size() == 4,
                   "Filter must be with 4 dimensions, i.e. OIHW");
    if (bias) {
      PADDLE_ENFORCE(bias->layout() == DataLayout::kMKLDNN &&
                         bias->format() != memory::format::format_undef,
                     "Wrong layout/format set for Bias tensor");
      PADDLE_ENFORCE(bias->dims().size() == 1,
                     "Bias must only have 1 dimension, i.e. X");
    }

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    bool fuse_relu = ctx.Attr<bool>("fuse_relu");
    bool fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");
    int groups = ctx.Attr<int>("groups");

    // TODO(tpatejko): add support for dilation
    PADDLE_ENFORCE(
        dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1,
        "dilation in convolution is not implemented yet");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();

    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> weights_tz =
        paddle::framework::vectorize2int(filter->dims());
    int g = std::max(groups, 1);
    if (g > 1) {
      int o = weights_tz[0];
      int i = weights_tz[1];
      int h = weights_tz[2];
      int w = weights_tz[3];
      weights_tz.resize(5);
      weights_tz[0] = g;
      weights_tz[1] = o / g;
      weights_tz[2] = i;
      weights_tz[3] = h;
      weights_tz[4] = w;
    }
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

    // Get unique name for storing MKLDNN primitives
    const std::string key = platform::ConvMKLDNNHandler::GetHash(
        src_tz, weights_tz, strides, paddings, dilations, groups,
        ctx.op().Output("Output"));
    const std::string key_conv_pd = key + "@conv_pd";

    std::vector<primitive> pipeline;

    auto user_src_md = platform::MKLDNNMemDesc(
        {src_tz}, platform::MKLDNNGetDataType<T>(), input->format());
    auto user_weights_md = platform::MKLDNNMemDesc(
        {weights_tz}, platform::MKLDNNGetDataType<T>(),
        (g == 1) ? filter->format() : mkldnn::memory::format::goihw);

    /* create memory descriptor for convolution without specified format
     * ('any') which lets a primitive (convolution in this case) choose
     * the memory format preferred for best performance
     */
    std::string data_format = ctx.Attr<std::string>("data_format");
    auto chosen_memory_format =
        platform::data_format_to_memory_format(data_format);

    auto src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    std::vector<int> bias_tz;  // TODO(mgallus): avoid empty vector creation.
                               // Currently used whenever bias is != nullptr.
    auto dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);

    // create a conv primitive descriptor and save it for usage in backward
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd;
    auto fwd_prop_kind = is_test ? mkldnn::prop_kind::forward_inference
                                 : mkldnn::prop_kind::forward_training;
    if (bias) {
      bias_tz = paddle::framework::vectorize2int(bias->dims());
      auto bias_md = platform::MKLDNNMemDesc(
          bias_tz, platform::MKLDNNGetDataType<T>(), memory::format::x);
      conv_pd = ConvFwdPrimitiveDesc(
          src_md, weights_md, bias_md, dst_md, strides, paddings, mkldnn_engine,
          fuse_relu, fuse_residual_conn, fwd_prop_kind);
    } else {
      conv_pd = ConvFwdPrimitiveDesc(src_md, weights_md, dst_md, strides,
                                     paddings, mkldnn_engine, fuse_relu,
                                     fuse_residual_conn, fwd_prop_kind);
    }
    // Save conv_pd/src_memory/weights_memory for backward pass
    if (!is_test) dev_ctx.SetBlob(key_conv_pd, conv_pd);

    platform::ConvMKLDNNHandler handler(conv_pd, dev_ctx, mkldnn_engine, key);

    // create mkldnn memory from input tensors (data/weights)
    auto user_src_memory_p =
        handler.AcquireSrcMemory(user_src_md, to_void_cast<T>(input_data));
    auto user_weights_memory_p = handler.AcquireWeightsMemory(
        user_weights_md, to_void_cast<T>(filter_data));

    // create reorder primitive if the input format is not the preferred one
    auto src_memory_p =
        handler.AcquireSrcMemoryFromPrimitive(user_src_memory_p, pipeline);
    auto weights_memory_p = handler.AcquireWeightsMemoryFromPrimitive(
        user_weights_memory_p, pipeline, is_test);

    std::shared_ptr<mkldnn::memory> dst_memory_p;

    if (fuse_residual_conn) {
      auto residual_param = ctx.Input<Tensor>("ResidualData");
      auto residual_param_data = residual_param->data<T>();

      PADDLE_ENFORCE(
          residual_param_data != nullptr,
          "Provide data if you want MKLDNN conv+elementwise_add fusion");
      PADDLE_ENFORCE_EQ(output->dims(), residual_param->dims(),
                        "Output and elementwise parameter need to have the "
                        "same dimension sizes");

      if (residual_param->format() != handler.GetDstFormat()) {
        auto output_data = output->mutable_data<T>(
            ctx.GetPlace(), ::paddle::memory::Allocator::kDefault,
            handler.GetDstMemorySize());
        auto residual_data_tz =
            paddle::framework::vectorize2int(residual_param->dims());
        auto residual_data_type =
            paddle::framework::ToMKLDNNDataType(residual_param->type());

        auto user_residual_md = platform::MKLDNNMemDesc(
            residual_data_tz, residual_data_type, residual_param->format());
        auto user_residual_memory_p = handler.AcquireResidualDataMemory(
            user_residual_md, to_void_cast<T>(residual_param_data));

        dst_memory_p = handler.AcquireDstMemoryFromResidualDataMemory(
            user_residual_memory_p, to_void_cast<T>(output_data), pipeline);
      } else {
        output->ShareDataWith(*residual_param);
        auto output_data = output->mutable_data<T>(ctx.GetPlace());
        dst_memory_p =
            handler.AcquireDstMemoryFromPrimitive(to_void_cast<T>(output_data));
      }
    } else {
      auto output_data = output->mutable_data<T>(
          ctx.GetPlace(), paddle::memory::Allocator::kDefault,
          handler.GetDstMemorySize());
      dst_memory_p =
          handler.AcquireDstMemoryFromPrimitive(to_void_cast<T>(output_data));
    }

    // create convolution op primitive
    std::shared_ptr<mkldnn::convolution_forward> conv_p;
    if (bias) {
      const T* bias_data = bias->data<T>();
      auto user_bias_md = platform::MKLDNNMemDesc(
          {bias_tz}, platform::MKLDNNGetDataType<T>(), memory::format::x);
      auto user_bias_memory_p =
          handler.AcquireBiasMemory(user_bias_md, to_void_cast<T>(bias_data));

      auto bias_memory_p =
          handler.AcquireBiasMemoryFromPrimitive(user_bias_memory_p, pipeline);
      conv_p = handler.AcquireConvolution(src_memory_p, weights_memory_p,
                                          bias_memory_p, dst_memory_p);
    } else {
      conv_p = handler.AcquireConvolution(src_memory_p, weights_memory_p,
                                          dst_memory_p);
    }

    // push primitive to stream and wait until it's executed
    pipeline.push_back(*conv_p);
    stream(stream::kind::eager).submit(pipeline).wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(*dst_memory_p));
  }

 private:
  mkldnn::primitive_attr CreatePostOps(bool fuse_relu,
                                       bool fuse_residual_conn) const {
    mkldnn::primitive_attr conv_attr;
    mkldnn::post_ops post_operations;
    // Fusion with Elementwise layer relies on adding a sum post-operation with
    // the scale parameter. It is assumed that when fuse_residual_connection is
    // true, the output tensor contains the data coming from residual
    // connection. The result of this post_op is:
    // Output = scale * Output + Conv_Out.
    if (fuse_residual_conn) {
      post_operations.append_sum(1.0f);
    }
    // Fusion with ReLU layer is executed through the PostOps feature. Create a
    // PostOps object and configure it to execute an eltwise relu operation.
    if (fuse_relu) {
      constexpr float scale = 1.0f;
      constexpr float negative_slope = 0.0f;
      constexpr float placeholder = 0.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                     negative_slope, placeholder);
    }
    conv_attr.set_post_ops(post_operations);
    return conv_attr;
  }

  std::unique_ptr<mkldnn::convolution_forward::primitive_desc>
  ConvFwdPrimitiveDesc(const memory::desc& src, const memory::desc& weights,
                       const memory::desc& dst, const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const mkldnn::engine& engine, const bool fuse_relu,
                       const bool fuse_residual_conn,
                       mkldnn::prop_kind fwd_prop_kind) const {
    memory::dims stride_dims = {strides[0], strides[1]};
    memory::dims padding_dims = {paddings[0], paddings[1]};

    auto conv_desc = mkldnn::convolution_forward::desc(
        fwd_prop_kind, mkldnn::convolution_direct, src, weights, dst,
        stride_dims, padding_dims, padding_dims, mkldnn::padding_kind::zero);

    mkldnn::primitive_attr conv_attr =
        CreatePostOps(fuse_relu, fuse_residual_conn);

    auto p_conv_pd = new mkldnn::convolution_forward::primitive_desc(
        conv_desc, conv_attr, engine);

    return std::unique_ptr<mkldnn::convolution_forward::primitive_desc>(
        p_conv_pd);
  }

  std::unique_ptr<mkldnn::convolution_forward::primitive_desc>
  ConvFwdPrimitiveDesc(const memory::desc& src, const memory::desc& weights,
                       const memory::desc& bias, const memory::desc& dst,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const mkldnn::engine& engine, const bool fuse_relu,
                       const bool fuse_residual_conn,
                       mkldnn::prop_kind fwd_prop_kind) const {
    memory::dims stride_dims = {strides[0], strides[1]};
    memory::dims padding_dims = {paddings[0], paddings[1]};

    auto conv_desc = mkldnn::convolution_forward::desc(
        fwd_prop_kind, mkldnn::convolution_direct, src, weights, bias, dst,
        stride_dims, padding_dims, padding_dims, mkldnn::padding_kind::zero);

    mkldnn::primitive_attr conv_attr =
        CreatePostOps(fuse_relu, fuse_residual_conn);

    auto p_conv_pd = new mkldnn::convolution_forward::primitive_desc(
        conv_desc, conv_attr, engine);

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

    PADDLE_ENFORCE(input->layout() == DataLayout::kMKLDNN &&
                       input->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input tensor");
    PADDLE_ENFORCE(filter->layout() == DataLayout::kMKLDNN &&
                       filter->format() != memory::format::format_undef,
                   "Wrong layout/format set for Filter tensor");
    PADDLE_ENFORCE(output->layout() == DataLayout::kMKLDNN &&
                       output->format() != memory::format::format_undef,
                   "Wrong layout/format set for Output tensor");
    PADDLE_ENFORCE(output_grad->layout() == DataLayout::kMKLDNN &&
                       output_grad->format() != memory::format::format_undef,
                   "Wrong layout/format set for output_grad tensor");

    PADDLE_ENFORCE(
        !ctx.Attr<bool>("is_test"),
        "is_test attribute should be set to False in training phase.");

    if (!input_grad && !filter_grad) return;

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    const T* output_grad_data = output_grad->data<T>();
    T* input_grad_data = nullptr;
    T* filter_grad_data = nullptr;

    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> weights_tz =
        paddle::framework::vectorize2int(filter->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

    // Get an unique name from "argument" name of "Output" variable
    // as well as attributes of primitive to be created
    // This name will be used as key when saving info into device context
    const std::string key = platform::ConvMKLDNNHandler::GetHash(
        src_tz, weights_tz, strides, paddings, dilations, groups,
        ctx.op().Input("Output"));

    const std::string key_conv_pd = key + "@conv_pd";
    std::vector<primitive> pipeline;

    // Create user memory descriptors
    auto user_src_md = platform::MKLDNNMemDesc(
        {src_tz}, platform::MKLDNNGetDataType<T>(), input->format());
    auto user_weights_md = platform::MKLDNNMemDesc(
        {weights_tz}, platform::MKLDNNGetDataType<T>(), filter->format());
    auto user_diff_dst_md = platform::MKLDNNMemDesc(
        {dst_tz}, platform::MKLDNNGetDataType<T>(), output_grad->format());

    /* create memory descriptor for conv backward without specified format
     * ('any') which lets a primitive (conv backward in this case) choose
     * the memory format preferred for best performance
     */
    std::string data_format = ctx.Attr<std::string>("data_format");
    auto chosen_memory_format =
        platform::data_format_to_memory_format(data_format);

    auto src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto diff_src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto diff_weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto diff_dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);

    // Retrieve conv_pd from device context
    auto conv_pd =
        std::static_pointer_cast<mkldnn::convolution_forward::primitive_desc>(
            dev_ctx.GetBlob(key_conv_pd));
    PADDLE_ENFORCE(conv_pd != nullptr,
                   "Fail to find conv_pd in device context");

    // create backward convolution weights primitive descriptor
    auto conv_bwd_weights_desc = mkldnn::convolution_backward_weights::desc(
        mkldnn::convolution_direct, src_md, diff_weights_md, diff_dst_md,
        strides, paddings, paddings, mkldnn::padding_kind::zero);
    auto conv_bwd_weights_pd =
        std::make_shared<mkldnn::convolution_backward_weights::primitive_desc>(
            conv_bwd_weights_desc, mkldnn_engine, *conv_pd);

    // create backward convolution data primitive descriptor
    auto conv_bwd_data_desc = mkldnn::convolution_backward_data::desc(
        mkldnn::convolution_direct, diff_src_md, weights_md, diff_dst_md,
        strides, paddings, paddings, mkldnn::padding_kind::zero);
    auto conv_bwd_data_pd =
        std::make_shared<mkldnn::convolution_backward_data::primitive_desc>(
            conv_bwd_data_desc, mkldnn_engine, *conv_pd);

    platform::ConvMKLDNNHandler handler(conv_pd, conv_bwd_data_pd,
                                        conv_bwd_weights_pd, dev_ctx,
                                        mkldnn_engine, key);

    // create mkldnn memory from input tensors (data/weights)
    auto user_src_memory_p =
        handler.AcquireSrcMemory(user_src_md, to_void_cast<T>(input_data));
    auto user_weights_memory_p = handler.AcquireWeightsMemory(
        user_weights_md, to_void_cast<T>(filter_data));
    auto user_diff_dst_memory_p = handler.AcquireDiffDstMemory(
        user_diff_dst_md, to_void_cast<T>(output_grad_data));

    // create backward conv primitive for weights
    if (filter_grad) {
      auto src_memory_p = handler.AcquireSrcMemoryFromWeightsPrimitive(
          user_src_memory_p, pipeline);

      auto diff_dst_memory_4filter_p =
          handler.AcquireDiffDstMemoryFromWeightsPrimitive(
              user_diff_dst_memory_p, pipeline);

      const size_t size = handler.GetDiffWeightsMemorySize();
      filter_grad_data = filter_grad->mutable_data<T>(
          ctx.GetPlace(), paddle::memory::Allocator::kDefault, size);

      auto diff_weights_memory_p =
          handler.AcquireDiffWeightsMemoryFromWeightsPrimitive(
              reinterpret_cast<void*>(filter_grad_data));

      auto conv_bwd_weights_p = handler.AcquireConvolutionBackwardWeights(
          src_memory_p, diff_dst_memory_4filter_p, diff_weights_memory_p);

      // push primitive to stream and wait until it's executed
      pipeline.push_back(*conv_bwd_weights_p);

      filter_grad->set_layout(DataLayout::kMKLDNN);
      filter_grad->set_format(GetMKLDNNFormat(*diff_weights_memory_p));
    }

    if (input_grad) {
      auto weights_memory_p = handler.AcquireWeightsMemoryFromDataPrimitive(
          user_weights_memory_p, pipeline);

      auto diff_dst_memory_4data_p =
          handler.AcquireDiffDstMemoryFromDataPrimitive(user_diff_dst_memory_p,
                                                        pipeline);

      const size_t size = handler.GetDiffSourceMemorySize();
      input_grad_data = input_grad->mutable_data<T>(
          ctx.GetPlace(), paddle::memory::Allocator::kDefault, size);

      auto diff_src_memory_p = handler.AcquireDiffSrcMemoryFromDataPrimitive(
          reinterpret_cast<void*>(input_grad_data));

      auto conv_bwd_data_p = handler.AcquireConvolutionBackwardData(
          diff_dst_memory_4data_p, weights_memory_p, diff_src_memory_p);

      pipeline.push_back(*conv_bwd_data_p);

      input_grad->set_layout(DataLayout::kMKLDNN);
      input_grad->set_format(GetMKLDNNFormat(*diff_src_memory_p));
    }
    stream(stream::kind::eager).submit(pipeline).wait();
  }  // Compute()
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(conv2d, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(conv2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvMKLDNNGradOpKernel<float>);
