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

  std::shared_ptr<mkldnn::memory> AcquireSrcMemoryFromWeightsPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {
    auto src_pd = conv_bwd_weights_pd_->src_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(src_pd, user_pd, user_memory_p,
                               "@weights-src_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemoryFromWeightsPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {
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
      std::vector<mkldnn::primitive>& pipeline) {
    auto diff_dst_pd = conv_bwd_data_pd_->diff_dst_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(diff_dst_pd, user_pd, user_memory_p,
                               "@data-diff_dst_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryFromDataPrimitive(
      const std::shared_ptr<mkldnn::memory> user_weights_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {
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
      std::vector<mkldnn::primitive>& pipeline) {
    auto src_pd = conv_pd_->src_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(src_pd, user_pd, user_memory_p, "@src_mem_p",
                               pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_weights_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {
    auto user_weights_pd = user_weights_memory_p->get_primitive_desc();
    auto weights_pd = conv_pd_->weights_primitive_desc();
    return this->AcquireMemory(weights_pd, user_weights_pd,
                               user_weights_memory_p, "@weights_mem_p",
                               pipeline);
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
  static std::string GetHash(memory::dims& input_dims,
                             memory::dims& weights_dims,
                             std::vector<int>& strides,
                             std::vector<int>& paddings,
                             std::vector<int>& dilations, int groups,
                             const std::string& suffix) {
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

    PADDLE_ENFORCE(input->layout() == DataLayout::kMKLDNN &&
                       input->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input tensor");
    PADDLE_ENFORCE(filter->layout() == DataLayout::kMKLDNN &&
                       filter->format() != memory::format::format_undef,
                   "Wrong layout/format set for Filter tensor");

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

    // Get unique name for storing MKLDNN primitives
    const std::string key = ConvMKLDNNHandler::GetHash(
        src_tz, weights_tz, strides, paddings, dilations, groups,
        ctx.op().Output("Output"));
    const std::string key_conv_pd = key + "@conv_pd";

    std::vector<primitive> pipeline;

    auto user_src_md = platform::MKLDNNMemDesc(
        {src_tz}, platform::MKLDNNGetDataType<T>(), input->format());
    auto user_weights_md = platform::MKLDNNMemDesc(
        {weights_tz}, platform::MKLDNNGetDataType<T>(), filter->format());

    /* create memory descriptor for convolution without specified format
     * ('any') which lets a primitive (convolution in this case) choose
     * the memory format preferred for best performance
     */
    auto src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), memory::format::any);
    auto weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), memory::format::any);
    auto dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T>(), memory::format::any);

    // create a conv primitive descriptor and save it for usage in backward
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd =
        ConvFwdPrimitiveDesc(src_md, weights_md, dst_md, strides, paddings,
                             mkldnn_engine);
    // Save conv_pd/src_memory/weights_memory for backward pass
    dev_ctx.SetBlob(key_conv_pd, conv_pd);

    ConvMKLDNNHandler handler(conv_pd, dev_ctx, mkldnn_engine, key);

    // create mkldnn memory from input tensors (data/weights)
    auto user_src_memory_p =
        handler.AcquireSrcMemory(user_src_md, to_void_cast<T>(input_data));
    auto user_weights_memory_p = handler.AcquireWeightsMemory(
        user_weights_md, to_void_cast<T>(filter_data));

    // create reorder primitive if the input format is not the preferred one
    auto src_memory_p =
        handler.AcquireSrcMemoryFromPrimitive(user_src_memory_p, pipeline);
    auto weights_memory_p = handler.AcquireWeightsMemoryFromPrimitive(
        user_weights_memory_p, pipeline);
    auto dst_memory_p =
        handler.AcquireDstMemoryFromPrimitive(to_void_cast<T>(output_data));

    // create convolution op primitive
    auto conv_p = handler.AcquireConvolution(src_memory_p, weights_memory_p,
                                             dst_memory_p);

    // push primitive to stream and wait until it's executed
    pipeline.push_back(*conv_p);
    stream(stream::kind::eager).submit(pipeline).wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(*dst_memory_p));
  }

 private:
  std::unique_ptr<mkldnn::convolution_forward::primitive_desc>
  ConvFwdPrimitiveDesc(const memory::desc& src, const memory::desc& weights,
                       const memory::desc& dst, const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const mkldnn::engine& engine) const {
    memory::dims stride_dims = {strides[0], strides[1]};
    memory::dims padding_dims = {paddings[0], paddings[1]};

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

    // Get an unique name from "argument" name of "Output" variable
    // This name will be used as key when saving info into device context
    const std::string key =
        ConvMKLDNNHandler::GetHash(src_tz, weights_tz, strides, paddings,
                                   dilations, groups, ctx.op().Input("Output"));

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
    auto src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), memory::format::any);
    auto diff_src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), memory::format::any);
    auto weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), memory::format::any);
    auto diff_weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), memory::format::any);
    auto diff_dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T>(), memory::format::any);

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

    ConvMKLDNNHandler handler(conv_pd, conv_bwd_data_pd, conv_bwd_weights_pd,
                              dev_ctx, mkldnn_engine, key);

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
