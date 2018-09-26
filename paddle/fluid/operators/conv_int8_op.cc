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

#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/operators/conv_int8_op.h"
#include "mkldnn.hpp"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;

class ConvMKLDNNHandler : public platform::MKLDNNHandler {
 public:
  ConvMKLDNNHandler(
      std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd,
      const platform::MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
      const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key) {
    conv_pd_ = conv_pd;
  }

  ConvMKLDNNHandler(
      std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd,
      std::shared_ptr<mkldnn::convolution_backward_data::primitive_desc>
          conv_bwd_data_pd,
      std::shared_ptr<mkldnn::convolution_backward_weights::primitive_desc>
          conv_bwd_weights_pd,
      const platform::MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
      const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        conv_pd_(conv_pd),
        conv_bwd_weights_pd_(conv_bwd_weights_pd),
        conv_bwd_data_pd_(conv_bwd_data_pd) {
    // If we are in Grad operatgor then update a key with BWD suffix to
    // distinguish from FWD memory primitives
    key_ += "-BWD";
  }

  size_t GetDstMemorySize() const {
    return conv_pd_->dst_primitive_desc().get_size();
  }

  size_t GetDiffWeightsMemorySize() const {
    return conv_bwd_weights_pd_->diff_weights_primitive_desc().get_size();
  }

  size_t GetDiffSourceMemorySize() const {
    return conv_bwd_data_pd_->diff_src_primitive_desc().get_size();
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemoryFromWeightsPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto src_pd = conv_bwd_weights_pd_->src_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(src_pd, user_pd, user_memory_p,
                               "@weights-src_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemoryFromWeightsPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto diff_dst_pd = conv_bwd_weights_pd_->diff_dst_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(diff_dst_pd, user_pd, user_memory_p,
                               "@weights-diff_dst_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffWeightsMemoryFromWeightsPrimitive(
      void* ptr) {
    return this->AcquireMemoryFromPrimitive(
        conv_bwd_weights_pd_->diff_weights_primitive_desc(), ptr,
        "@diff_weights_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemoryFromDataPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto diff_dst_pd = conv_bwd_data_pd_->diff_dst_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(diff_dst_pd, user_pd, user_memory_p,
                               "@data-diff_dst_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryFromDataPrimitive(
      const std::shared_ptr<mkldnn::memory> user_weights_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto weights_pd = conv_bwd_data_pd_->weights_primitive_desc();
    auto user_pd = user_weights_memory_p->get_primitive_desc();
    return this->AcquireMemory(weights_pd, user_pd, user_weights_memory_p,
                               "@data-weights_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffSrcMemoryFromDataPrimitive(
      void* ptr) {
    return this->AcquireMemoryFromPrimitive(
        conv_bwd_data_pd_->diff_src_primitive_desc(), ptr, "@diff_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemoryFromPrimitive(void* ptr) {
    return this->AcquireMemoryFromPrimitive(conv_pd_->dst_primitive_desc(), ptr,
                                            "@dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto src_pd = conv_pd_->src_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(src_pd, user_pd, user_memory_p, "@src_mem_p",
                               pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_weights_memory_p,
      std::vector<mkldnn::primitive>& pipeline,  // NOLINT
      bool is_persistent = false) {
    auto user_weights_pd = user_weights_memory_p->get_primitive_desc();
    auto weights_pd = conv_pd_->weights_primitive_desc();
    return this->AcquireMemory(weights_pd, user_weights_pd,
                               user_weights_memory_p, "@weights_mem_p",
                               pipeline, is_persistent);
  }

  std::shared_ptr<mkldnn::memory> AcquireBiasMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_bias_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto user_bias_pd = user_bias_memory_p->get_primitive_desc();
    auto bias_pd = conv_pd_->bias_primitive_desc();
    return this->AcquireMemory(bias_pd, user_bias_pd, user_bias_memory_p,
                               "@bias_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::convolution_forward> AcquireConvolution(
      std::shared_ptr<mkldnn::memory> src_memory_p,
      std::shared_ptr<mkldnn::memory> weights_memory_p,
      std::shared_ptr<mkldnn::memory> dst_memory_p) {
    auto prim_key = key_ + "@conv_p";
    auto conv_p = std::static_pointer_cast<mkldnn::convolution_forward>(
        dev_ctx_.GetBlob(prim_key));
    PADDLE_ENFORCE((conv_p != nullptr) || (is_reusing_ == false),
                   "Fail to find convolution primitive in device context");
    if (conv_p == nullptr) {
      conv_p = std::make_shared<mkldnn::convolution_forward>(
          *conv_pd_, *(src_memory_p), *(weights_memory_p.get()),
          *(dst_memory_p.get()));

      dev_ctx_.SetBlob(prim_key, conv_p);
    } else {
      is_reusing_ = true;
    }
    return conv_p;
  }

  std::shared_ptr<mkldnn::convolution_forward> AcquireConvolution(
      std::shared_ptr<mkldnn::memory> src_memory_p,
      std::shared_ptr<mkldnn::memory> weights_memory_p,
      std::shared_ptr<mkldnn::memory> bias_memory_p,
      std::shared_ptr<mkldnn::memory> dst_memory_p) {
    auto prim_key = key_ + "@conv_p";
    auto conv_p = std::static_pointer_cast<mkldnn::convolution_forward>(
        dev_ctx_.GetBlob(prim_key));
    PADDLE_ENFORCE((conv_p != nullptr) || (is_reusing_ == false),
                   "Fail to find convolution primitive in device context");
    if (conv_p == nullptr) {
      conv_p = std::make_shared<mkldnn::convolution_forward>(
          *conv_pd_, *(src_memory_p), *(weights_memory_p.get()),
          *(bias_memory_p.get()), *(dst_memory_p.get()));

      dev_ctx_.SetBlob(prim_key, conv_p);
    } else {
      is_reusing_ = true;
    }
    return conv_p;
  }

  std::shared_ptr<mkldnn::convolution_backward_weights>
  AcquireConvolutionBackwardWeights(
      std::shared_ptr<mkldnn::memory> src_memory_p,
      std::shared_ptr<mkldnn::memory> diff_dst_memory_p,
      std::shared_ptr<mkldnn::memory> diff_weights_memory_p) {
    auto prim_key = key_ + "@conv_bwd_weights_p";
    auto conv_bwd_weights_p =
        std::static_pointer_cast<mkldnn::convolution_backward_weights>(
            dev_ctx_.GetBlob(prim_key));
    PADDLE_ENFORCE(
        (conv_bwd_weights_p != nullptr) || (is_reusing_ == false),
        "Fail to find convolution bwd weights primitive in device context");
    if (conv_bwd_weights_p == nullptr) {
      // create backward conv primitive for weights
      conv_bwd_weights_p =
          std::make_shared<mkldnn::convolution_backward_weights>(
              *conv_bwd_weights_pd_, *src_memory_p, *diff_dst_memory_p,
              *diff_weights_memory_p);
      dev_ctx_.SetBlob(prim_key, conv_bwd_weights_p);
    } else {
      is_reusing_ = true;
    }
    return conv_bwd_weights_p;
  }

  std::shared_ptr<mkldnn::convolution_backward_data>
  AcquireConvolutionBackwardData(
      std::shared_ptr<mkldnn::memory> diff_dst_memory_p,
      std::shared_ptr<mkldnn::memory> weights_memory_p,
      std::shared_ptr<mkldnn::memory> diff_src_memory_p) {
    auto prim_key = key_ + "@conv_bwd_data_p";
    auto conv_bwd_data_p =
        std::static_pointer_cast<mkldnn::convolution_backward_data>(
            dev_ctx_.GetBlob(prim_key));
    PADDLE_ENFORCE(
        (conv_bwd_data_p != nullptr) || (is_reusing_ == false),
        "Fail to find convolution bwd data primitive in device context");
    if (conv_bwd_data_p == nullptr) {
      conv_bwd_data_p = std::make_shared<mkldnn::convolution_backward_data>(
          *conv_bwd_data_pd_, *diff_dst_memory_p, *weights_memory_p,
          *diff_src_memory_p);
      dev_ctx_.SetBlob(prim_key, conv_bwd_data_p);
    } else {
      is_reusing_ = true;
    }
    return conv_bwd_data_p;
  }

  // Generate keys for storing/retriving primitives for this operator
  // TODO(jczaja): Make hashing function more optimial
  static std::string GetHash(memory::dims& input_dims,     // NOLINT
                             memory::dims& weights_dims,   // NOLINT
                             std::vector<int>& strides,    // NOLINT
                             std::vector<int>& paddings,   // NOLINT
                             std::vector<int>& dilations,  // NOLINT
                             int groups, const std::string& suffix) {
    return dims2str(input_dims) + dims2str(weights_dims) + dims2str(strides) +
           dims2str(paddings) + dims2str(dilations) + std::to_string(groups) +
           suffix;
  }

 private:
  std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd_;
  std::shared_ptr<mkldnn::convolution_backward_weights::primitive_desc>
      conv_bwd_weights_pd_;
  std::shared_ptr<mkldnn::convolution_backward_data::primitive_desc>
      conv_bwd_data_pd_;
};

template <typename T>
class Convint8OpKernel : public paddle::framework::OpKernel<T> {
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
//for INT8
    auto* scale_in = ctx.Input<Tensor>("Scale_in");
    auto* scale_in_eltwise = ctx.Input<Tensor>("Scale_in_eltwise");
    auto* scale_weights = ctx.Input<Tensor>("Scale_weights");
    auto* scale_out = ctx.Input<Tensor>("Scale_out");

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
    bool fuse_eltwise = ctx.Attr<bool>("fuse_eltwise");
    int groups = ctx.Attr<int>("groups");

    // TODO: add support for dilation
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

//for INT8
    int count = g>1? weights_tz[1]*weights_tz[0] : weights_tz[0]; 
    T scale_in_data = *(scale_in->data<T>());
    T scale_in_eltwise_data = *(scale_in_eltwise->data<T>());
    std::vector<T> scale_weights_data(count);
    for(int i=0; i<count; i++){
        scale_weights_data[i] =*(scale_weights->data<T>());
    }
    T scale_out_data = *(scale_out->data<T>());

    std::vector<T> output_shift_scale(count);
    for(int i=0; i<count; i++){
        if(scale_weights_data[i] == 0.0)
            output_shift_scale[i] = scale_out_data;
        else 
            output_shift_scale[i] = scale_out_data / (scale_in_data * scale_weights_data[i]);
    }

    T sum_scale = scale_out_data / scale_in_eltwise_data;

    // Get unique name for storing MKLDNN primitives
    const std::string key = ConvMKLDNNHandler::GetHash(
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
        weights_tz, platform::MKLDNNGetDataType<T>(),
        (g == 1) ? chosen_memory_format : mkldnn::memory::format::goihw);
    std::vector<int> bias_tz;  // TODO(mgallus): avoid empty vector creation.
                               // Currently used whenever bias is != nullptr.
    auto dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);

    // create a conv primitive descriptor and save it for usage in backward
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd;
    if (bias) {
      bias_tz = paddle::framework::vectorize2int(bias->dims());
      auto bias_md = platform::MKLDNNMemDesc(
          bias_tz, platform::MKLDNNGetDataType<T>(), memory::format::x);
      conv_pd = ConvFwdPrimitiveDesc(src_md, weights_md, bias_md, dst_md,
                                     strides, paddings, mkldnn_engine,
                                     fuse_relu, fuse_eltwise, 
                                     output_shift_scale, sum_scale);
    } else {
      conv_pd =
          ConvFwdPrimitiveDesc(src_md, weights_md, dst_md, strides, paddings,
                               mkldnn_engine, fuse_relu, fuse_eltwise,
                               output_shift_scale, sum_scale);
    }
    // Save conv_pd/src_memory/weights_memory for backward pass
    dev_ctx.SetBlob(key_conv_pd, conv_pd);

    ConvMKLDNNHandler handler(conv_pd, dev_ctx, mkldnn_engine, key);

    // create mkldnn memory from input tensors (data/weights)
    auto user_src_memory_p =
        handler.AcquireSrcMemory(user_src_md, to_void_cast<T>(input_data));
    auto user_weights_memory_p = handler.AcquireWeightsMemory(
        user_weights_md, to_void_cast<T>(filter_data));

    T* output_data =
        output->mutable_data<T>(ctx.GetPlace(), handler.GetDstMemorySize());
    // create reorder primitive if the input format is not the preferred one
    auto src_memory_p =
        handler.AcquireSrcMemoryFromPrimitive(user_src_memory_p, pipeline);
    auto weights_memory_p = handler.AcquireWeightsMemoryFromPrimitive(
        user_weights_memory_p, pipeline, is_test);
    auto dst_memory_p =
        handler.AcquireDstMemoryFromPrimitive(to_void_cast<T>(output_data));

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
  mkldnn::primitive_attr CreatePostOps(bool fuse_relu, bool fuse_eltwise,
                         const std::vector<T> output_shift_scale, T sum_scale) const {
    mkldnn::primitive_attr conv_attr;
    mkldnn::post_ops post_operations;
    // Fusion with Elementwise layer relies on adding a sum post-operation with
    // the scale parameter. It is assumed that when fuse_eltwise is true, the
    // Output tensor contains the data coming from residual connection. The
    // result of this post_op is: Output = scale * Output + Conv_Out.
    int mask = 0;
    conv_attr.set_output_scales(mask, output_shift_scale);

    if (fuse_eltwise) {
      post_operations.append_sum(sum_scale);
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
                       const bool fuse_eltwise,
                       const std::vector<T> output_shift_scale, const T sum_scale) const {
    memory::dims stride_dims = {strides[0], strides[1]};
    memory::dims padding_dims = {paddings[0], paddings[1]};

    auto conv_desc = mkldnn::convolution_forward::desc(
        mkldnn::prop_kind::forward, mkldnn::convolution_direct, src, weights,
        dst, stride_dims, padding_dims, padding_dims,
        mkldnn::padding_kind::zero);

    mkldnn::primitive_attr conv_attr = 
        CreatePostOps(fuse_relu, fuse_eltwise, output_shift_scale, sum_scale);

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
                       const bool fuse_eltwise,
                       const std::vector<T> output_shift_scale, const T sum_scale) const {
    memory::dims stride_dims = {strides[0], strides[1]};
    memory::dims padding_dims = {paddings[0], paddings[1]};

    auto conv_desc = mkldnn::convolution_forward::desc(
        mkldnn::prop_kind::forward, mkldnn::convolution_direct, src, weights,
        bias, dst, stride_dims, padding_dims, padding_dims,
        mkldnn::padding_kind::zero);

    mkldnn::primitive_attr conv_attr = 
        CreatePostOps(fuse_relu, fuse_eltwise, output_shift_scale, sum_scale);

    auto p_conv_pd = new mkldnn::convolution_forward::primitive_desc(
        conv_desc, conv_attr, engine);

    return std::unique_ptr<mkldnn::convolution_forward::primitive_desc>(
        p_conv_pd);
  }
};

framework::OpKernelType Convint8Op::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<framework::LoDTensor>("Input")->type()),
      ctx.device_context());
}


void Convint8OpMaker::Make() {
  AddAttr<bool>("is_test", "").SetDefault(false);
  AddInput(
      "Input",
      "and W is the width of the feature.");
  AddInput("Filter",
           "(Tensor) The filter tensor of convolution operator. ");
  AddInput("Bias",
           "(Tensor) Bias to be added to each output of filter application.")
      .AsDispensable();
  AddOutput("Output",
       "The format of output tensor is also NCDHW.")
      .Reuse("Input");
  AddAttr<std::vector<int>>("strides",
                            "convolution operator.")
      .SetDefault({1, 1});
  AddAttr<std::vector<int>>("paddings",
                            "convolution operator.")
      .SetDefault({0, 0});
  AddAttr<int>(
      "groups",
      "is only connected to the second half of the input channels.")
      .SetDefault(1);
  AddAttr<std::vector<int>>("dilations",
                            "convolution operator.")
      .SetDefault({1, 1});
  AddAttr<bool>(
      "use_cudnn",
      "(bool, default false) Only used in cudnn kernel, need install cudnn")
      .SetDefault(false);
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("fuse_relu", "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("fuse_eltwise",
                "(bool, default false) Only used in mkldnn kernel. Used "
                "whenever convolution output is connected via skip connection "
                "to a previous layer.")
      .SetDefault(false);
  AddAttr<std::string>(
      "data_format",
      "the input will be transformed automatically. ")
      .SetDefault("AnyLayout");
  AddAttr<int>("workspace_size_MB",
               "better hardware. This size should be chosen carefully.")
      .SetDefault(4096);
  AddComment(R"DOC(
)DOC");
}


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

//REGISTER_OP_KERNEL(conv2d, MKLDNN, ::paddle::platform::CPUPlace,
//                   ops::Convint8OpKernel<float>);
//
//REGISTER_OP_KERNEL(conv2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
//                   ops::Convint8GradOpKernel<float>);

REGISTER_OPERATOR(conv_int8, ops::Convint8Op, ops::Convint8OpMaker, paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OP_CPU_KERNEL(conv_int8, ops::Convint8OpKernel<float>);


