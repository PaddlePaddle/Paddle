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
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/framework/data_layout_transform.h"
#include <unordered_map>

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
  
  mkldnn::memory::format GetDstFormat() const {
    return static_cast<mkldnn::memory::format>(
        conv_pd_->dst_primitive_desc().desc().data.format);
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


  std::shared_ptr<mkldnn::memory> AcquireResidualDataMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_residual_data_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemoryFromResidualDataMemory(
      const std::shared_ptr<mkldnn::memory>& user_residual_memory_p,
      void* dst_ptr,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    return this->AcquireMemory(user_residual_memory_p,
                               this->AcquireDstMemoryFromPrimitive(dst_ptr),
                               "@residual_data_mem_p", pipeline);
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
      bool is_persistent = false,
      bool is_INT8 = false,
      std::vector<float> scale_data = {1.0f},
      int mask = 0) { 
    auto user_weights_pd = user_weights_memory_p->get_primitive_desc();
    auto weights_pd = conv_pd_->weights_primitive_desc();
    return this->AcquireMemory(weights_pd, user_weights_pd,
                               user_weights_memory_p, "@weights_mem_p",
                               pipeline, is_persistent,
                               is_INT8, scale_data, mask);
  }

  std::shared_ptr<mkldnn::memory> AcquireBiasMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_bias_memory_p,
      std::vector<mkldnn::primitive>& pipeline,
      bool is_persistent = false,
      bool is_INT8 = false,
      std::vector<float> scale_data = {1.0f},
      int mask = 0) {  // NOLINT
    auto user_bias_pd = user_bias_memory_p->get_primitive_desc();
    auto bias_pd = conv_pd_->bias_primitive_desc();
    return this->AcquireMemory(bias_pd, user_bias_pd, user_bias_memory_p,
                               "@bias_mem_p", pipeline, is_persistent,
                               is_INT8, scale_data, mask);
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
    const float* filter_data = filter->data<float>();

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
    const std::string key = ConvMKLDNNHandler::GetHash(
        src_tz, weights_tz, strides, paddings, dilations, groups,
        ctx.op().Output("Output"));
    const std::string key_conv_pd = key + "@conv_pd";

    bool is_INT8 = ctx.HasInput("Scale_in")? true : false;
    
    bool need_s8_to_u8 = false;
    if (fuse_residual_conn && is_INT8 && fuse_relu) {
      need_s8_to_u8 = true;
    }

    std::shared_ptr<mkldnn::convolution_forward> conv_p;
    std::shared_ptr<mkldnn::memory> src_memory_p;
    std::shared_ptr<mkldnn::memory> user_src_memory_p;
    std::shared_ptr<mkldnn::memory> dst_memory_p;
    std::vector<primitive> pipeline;
    
    auto prim_key = key + "@conv_p";
    auto dst_key = key + "@dst_mem_p";
    auto src_key = key + "@src_mem_p";
    auto user_src_key = key + "@user_src_mem_p";
    auto src_reorder_key = key + "@src_mem_p" + "reorder_p";
    conv_p = std::static_pointer_cast<mkldnn::convolution_forward>(dev_ctx.GetBlob(prim_key));
    auto src_memory_reorder_p = std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(src_reorder_key));
    src_memory_p = std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(src_key));
    if(src_memory_reorder_p){
      user_src_memory_p = std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(user_src_key));
      user_src_memory_p->set_data_handle(to_void_cast<T>(input_data));
    } else if(src_memory_p){
      src_memory_p->set_data_handle(to_void_cast<T>(input_data));
    }
    
    dst_memory_p = std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(dst_key));
    
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd;
    conv_pd = std::static_pointer_cast<mkldnn::convolution_forward::primitive_desc>(dev_ctx.GetBlob(key_conv_pd));
    std::shared_ptr<ConvMKLDNNHandler> handler;
    if(conv_pd){
      handler.reset(new ConvMKLDNNHandler(conv_pd, dev_ctx, mkldnn_engine, key));
    }
    if (!is_INT8 && dst_memory_p){
      if (fuse_residual_conn) {
        auto residual_param = ctx.Input<Tensor>("ResidualData");
        auto residual_param_data = residual_param->data<T>();
        if (residual_param->format() != handler->GetDstFormat()) {
          auto output_data =
              output->mutable_data<T>(ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, handler->GetDstMemorySize());
          auto residual_data_tz =
              paddle::framework::vectorize2int(residual_param->dims());
          auto residual_data_type =
              paddle::framework::ToMKLDNNDataType(residual_param->type());
    
          auto user_residual_md = platform::MKLDNNMemDesc(
              residual_data_tz, residual_data_type, residual_param->format());
          auto user_residual_memory_p = handler->AcquireResidualDataMemory(
              user_residual_md, to_void_cast<T>(residual_param_data));
          dst_memory_p = handler->AcquireDstMemoryFromResidualDataMemory(
              user_residual_memory_p, to_void_cast<T>(output_data), pipeline);
        } else {
          output->ShareDataWith(*residual_param);
          auto output_data = output->mutable_data<T>(ctx.GetPlace());
          dst_memory_p->set_data_handle(to_void_cast<T>(output_data));
        }
      } else {
        auto output_data =
            output->mutable_data<T>(ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, handler->GetDstMemorySize());
        dst_memory_p->set_data_handle(to_void_cast<T>(output_data)); 
      }
    } else if(is_INT8 && dst_memory_p){
      if(fuse_residual_conn) {
        auto residual_param = ctx.Input<Tensor>("ResidualData");
        auto residual_dt = paddle::framework::ToMKLDNNDataType(residual_param->type());
        output->ShareDataWith(*residual_param);
        if(residual_dt == mkldnn::memory::data_type::u8){
          uint8_t* output_data = output->mutable_data<uint8_t>(ctx.GetPlace());
          dst_memory_p->set_data_handle(to_void_cast<uint8_t>(output_data));
        } else{
          int8_t* output_data = output->mutable_data<int8_t>(ctx.GetPlace());
          dst_memory_p->set_data_handle(to_void_cast<int8_t>(output_data));
        }
      } else {
        if(fuse_relu){
          uint8_t* output_data = output->mutable_data<uint8_t>(ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, handler->GetDstMemorySize());
          dst_memory_p->set_data_handle(to_void_cast<uint8_t>(output_data));
        } else{
          int8_t* output_data = output->mutable_data<int8_t>(ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, handler->GetDstMemorySize());
          dst_memory_p->set_data_handle(to_void_cast<int8_t>(output_data));
        }
      }
    }

    if(!is_INT8){
      if(conv_p == nullptr){
        auto user_src_md = platform::MKLDNNMemDesc(
            {src_tz}, platform::MKLDNNGetDataType<T>(), input->format());
        auto user_weights_md = platform::MKLDNNMemDesc(
            {weights_tz}, platform::MKLDNNGetDataType<T>(),
            (g == 1) ? mkldnn::memory::format::oihw : mkldnn::memory::format::goihw);

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
        if (bias) {
          bias_tz = paddle::framework::vectorize2int(bias->dims());
          auto bias_md = platform::MKLDNNMemDesc(
              bias_tz, platform::MKLDNNGetDataType<T>(), memory::format::x);
          conv_pd = ConvFwdPrimitiveDesc(src_md, weights_md, bias_md, dst_md,
                                         strides, paddings, mkldnn_engine,
                                         fuse_relu, fuse_residual_conn, is_test);
        } else {
          conv_pd =
              ConvFwdPrimitiveDesc(src_md, weights_md, dst_md, strides, paddings,
                                   mkldnn_engine, fuse_relu, fuse_residual_conn, is_test);
        }
        // Save conv_pd/src_memory/weights_memory for backward pass
        dev_ctx.SetBlob(key_conv_pd, conv_pd);

        handler.reset(new ConvMKLDNNHandler(conv_pd, dev_ctx, mkldnn_engine, key));

        // create mkldnn memory from input tensors (data/weights)
        user_src_memory_p =
            handler->AcquireSrcMemory(user_src_md, to_void_cast<T>(input_data));
        auto user_weights_memory_p = handler->AcquireWeightsMemory(
            user_weights_md, to_void_cast<float>(filter_data));

        // create reorder primitive if the input format is not the preferred one
        src_memory_p =
            handler->AcquireSrcMemoryFromPrimitive(user_src_memory_p, pipeline);
        auto weights_memory_p = handler->AcquireWeightsMemoryFromPrimitive(
            user_weights_memory_p, pipeline, is_test);

        if (fuse_residual_conn) {
          auto residual_param = ctx.Input<Tensor>("ResidualData");
          auto residual_param_data = residual_param->data<T>();

          PADDLE_ENFORCE(
              residual_param_data != nullptr,
              "Provide data if you want MKLDNN conv+elementwise_add fusion");
          PADDLE_ENFORCE_EQ(output->dims(), residual_param->dims(),
                            "Output and elementwise parameter need to have the "
                            "same dimension sizes");

          if (residual_param->format() != handler->GetDstFormat()) {
            auto output_data =
                output->mutable_data<T>(ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, handler->GetDstMemorySize());
            auto residual_data_tz =
                paddle::framework::vectorize2int(residual_param->dims());
            auto residual_data_type =
                paddle::framework::ToMKLDNNDataType(residual_param->type());

            auto user_residual_md = platform::MKLDNNMemDesc(
                residual_data_tz, residual_data_type, residual_param->format());
            auto user_residual_memory_p = handler->AcquireResidualDataMemory(
                user_residual_md, to_void_cast<T>(residual_param_data));
            dst_memory_p = handler->AcquireDstMemoryFromResidualDataMemory(
                user_residual_memory_p, to_void_cast<T>(output_data), pipeline);
          } else {
            output->ShareDataWith(*residual_param);
            auto output_data = output->mutable_data<T>(ctx.GetPlace());
            dst_memory_p =
                handler->AcquireDstMemoryFromPrimitive(to_void_cast<T>(output_data));
          }
        } else {
          auto output_data =
              output->mutable_data<T>(ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, handler->GetDstMemorySize());
          dst_memory_p =
              handler->AcquireDstMemoryFromPrimitive(to_void_cast<T>(output_data));
        }

        // create convolution op primitive
        if (bias) {
          const T* bias_data = bias->data<T>();
          auto user_bias_md = platform::MKLDNNMemDesc(
              {bias_tz}, platform::MKLDNNGetDataType<T>(), memory::format::x);
          auto user_bias_memory_p =
              handler->AcquireBiasMemory(user_bias_md, to_void_cast<T>(bias_data));

          auto bias_memory_p =
              handler->AcquireBiasMemoryFromPrimitive(user_bias_memory_p, pipeline, is_test);
          conv_p = handler->AcquireConvolution(src_memory_p, weights_memory_p,
                                              bias_memory_p, dst_memory_p);
        } else {
          conv_p = handler->AcquireConvolution(src_memory_p, weights_memory_p,
                                              dst_memory_p);
        }
        // push primitive to stream and wait until it's executed
        pipeline.push_back(*conv_p);
        stream(stream::kind::eager).submit(pipeline).wait();

        output->set_layout(DataLayout::kMKLDNN);
        output->set_format(GetMKLDNNFormat(*dst_memory_p));
      } else {
        if(src_memory_reorder_p){
          pipeline.push_back(*src_memory_reorder_p);
        } 
        pipeline.push_back(*conv_p);
        stream(stream::kind::eager).submit(pipeline).wait();

        output->set_layout(DataLayout::kMKLDNN);
        output->set_format(GetMKLDNNFormat(*dst_memory_p));
      }
    } else{
      if(conv_p == nullptr){
        auto* scale_in = ctx.HasInput("Scale_in") ? ctx.Input<Tensor>("Scale_in") : nullptr;
        auto* scale_in_eltwise = ctx.HasInput("Scale_in_eltwise")? ctx.Input<Tensor>("Scale_in_eltwise") : nullptr;
        auto* scale_weights = ctx.HasInput("Scale_weights")? ctx.Input<Tensor>("Scale_weights") : nullptr;
        auto* scale_out = ctx.HasInput("Scale_out")? ctx.Input<Tensor>("Scale_out") : nullptr;

        bool is_multi_channel = (scale_weights->memory_size() > 1) ? true : false;

        static std::unordered_map<std::string, std::vector<float>> scale_map;

        bool scale_reuse = true;
        auto scale_in_key = key + "@scale_in";
        auto scale_weights_key = key + "@scale_weights";
        auto scale_out_key = key + "@scale_out";
        auto output_shift_scale_key = key + "@output_shift_scale";
        auto sum_scale_key = key + "@sum_scale";
        auto scale_in_eltwise_key = key + "@scale_in_eltwise";
        std::vector<float> scale_in_data;
        std::vector<float> scale_out_data;
        std::vector<float> scale_weights_data;
        std::vector<float> scale_in_eltwise_data;
        std::vector<float> output_shift_scale;
        std::vector<float> sum_scale = {1.0f};
        std::vector<float> none_scale = {0};

        if (GetScaleMap(scale_map, scale_in_key) == none_scale){
          scale_reuse = false;
        }

        if(!scale_reuse){
          int count = is_multi_channel? (g>1? weights_tz[1]*weights_tz[0] : weights_tz[0]) : 1; 
          scale_in_data = {*(scale_in->data<float>())};
          scale_weights_data.resize(count);
          #pragma omp parallel for if (count > 1)
          for(int i=0; i<count; i++){
            scale_weights_data[i] =*(scale_weights->data<float>() + i);
          }
          scale_out_data = {*(scale_out->data<float>())};
          output_shift_scale.resize(count);
          #pragma omp parallel for if (count > 1)
          for(int i=0; i<count; i++){
            if(scale_weights_data[i] == 0.0)
              output_shift_scale[i] = scale_out_data[0];
            else 
              output_shift_scale[i] = scale_out_data[0] / (scale_in_data[0] * scale_weights_data[i]);
          }
          if(fuse_residual_conn){
            scale_in_eltwise_data = {*(scale_in_eltwise->data<float>())};
            sum_scale[0] = scale_out_data[0] / scale_in_eltwise_data[0];
            SetScaleMap(scale_map, scale_in_eltwise_key, scale_in_eltwise_data);
          }

          //scale reuse
          SetScaleMap(scale_map, scale_in_key, scale_in_data);
          SetScaleMap(scale_map, scale_weights_key, scale_weights_data);
          SetScaleMap(scale_map, scale_out_key, scale_out_data);
          SetScaleMap(scale_map, output_shift_scale_key, output_shift_scale);
          SetScaleMap(scale_map, sum_scale_key, sum_scale);
        } else{
          scale_in_data = GetScaleMap(scale_map, scale_in_key);
          scale_out_data = GetScaleMap(scale_map, scale_out_key);
          scale_weights_data = GetScaleMap(scale_map, scale_weights_key);
          if(fuse_residual_conn){
            scale_in_eltwise_data = GetScaleMap(scale_map, scale_in_eltwise_key);
          }
          output_shift_scale = GetScaleMap(scale_map, output_shift_scale_key);
          sum_scale = GetScaleMap(scale_map, sum_scale_key); 
        }

        std::vector<primitive> pipeline;

        auto user_src_md = platform::MKLDNNMemDesc(
                {src_tz}, paddle::framework::ToMKLDNNDataType(input->type()), input->format());
        auto user_weights_md = platform::MKLDNNMemDesc(
                {weights_tz}, platform::MKLDNNGetDataType<float>(),
                (g == 1) ? mkldnn::memory::format::oihw : mkldnn::memory::format::goihw);
  
        /* create memory descriptor for convolution without specified format
         * ('any') which lets a primitive (convolution in this case) choose
         * the memory format preferred for best performance
        */
        std::string data_format = ctx.Attr<std::string>("data_format");
        auto chosen_memory_format = 
            platform::data_format_to_memory_format(data_format);
  
        auto bias_tz = paddle::framework::vectorize2int(bias->dims());

        auto src_md = platform::MKLDNNMemDesc(
            src_tz, memory::data_type::u8, chosen_memory_format);
        auto weights_md = platform::MKLDNNMemDesc(
            weights_tz, memory::data_type::s8, chosen_memory_format);

        auto dst_dt = fuse_relu?
            paddle::framework::ToMKLDNNDataType(std::type_index(typeid(unsigned char)))
            : paddle::framework::ToMKLDNNDataType(std::type_index(typeid(signed char)));

        if(fuse_residual_conn){
          auto residual = ctx.Input<Tensor>("ResidualData");
          auto residual_dt = paddle::framework::ToMKLDNNDataType(residual->type());
          if(dst_dt != residual_dt)
            dst_dt = residual_dt;
        }
        auto dst_md = platform::MKLDNNMemDesc(dst_tz, dst_dt, chosen_memory_format);

        // create a conv primitive descriptor and save it for usage in backward
        if (bias) {
          auto bias_md = platform::MKLDNNMemDesc(
              bias_tz, memory::data_type::s32, memory::format::x);
          conv_pd = ConvFwdPrimitiveDesc(src_md, weights_md, bias_md, dst_md,
                                         strides, paddings, mkldnn_engine,
                                         fuse_relu, fuse_residual_conn,
                                         output_shift_scale, sum_scale[0], is_test);
        } else {
          conv_pd =
              ConvFwdPrimitiveDesc(src_md, weights_md, dst_md, strides, paddings,
                                   mkldnn_engine, fuse_relu, fuse_residual_conn,
                                   output_shift_scale, sum_scale[0], is_test);
        }
        // Save conv_pd/src_memory/weights_memory for backward pass
        dev_ctx.SetBlob(key_conv_pd, conv_pd);

        handler.reset(new ConvMKLDNNHandler(conv_pd, dev_ctx, mkldnn_engine, key));

        // create mkldnn memory from input tensors (data/weights)
        user_src_memory_p =
            handler->AcquireSrcMemory(user_src_md, to_void_cast<T>(input_data));
        auto user_weights_memory_p = handler->AcquireWeightsMemory(
            user_weights_md, to_void_cast<float>(filter_data));

        // create reorder primitive if the input format is not the preferred one
        src_memory_p =
            handler->AcquireSrcMemoryFromPrimitive(user_src_memory_p, pipeline);
            
        std::shared_ptr<mkldnn::memory> weights_memory_p;
        int mask_reorder = is_multi_channel? ((g!= 1) ? (1<<1)+(1<<0) : 1<<0) : 0;
           weights_memory_p = handler->AcquireWeightsMemoryFromPrimitive(
           user_weights_memory_p, pipeline, is_test, is_INT8, scale_weights_data, mask_reorder);

        if(fuse_residual_conn) {
          auto residual_param = ctx.Input<Tensor>("ResidualData");
          PADDLE_ENFORCE_EQ(output->dims(), residual_param->dims(),
                "Output and elementwise parameter need to have the "
                "same dimension sizes");
          auto residual_dt = paddle::framework::ToMKLDNNDataType(residual_param->type());
          PADDLE_ENFORCE_EQ(residual_param->format(), handler->GetDstFormat(),
                "Conv input dimension and filter dimension should be the same.");
          output->ShareDataWith(*residual_param);
          if(residual_dt == mkldnn::memory::data_type::u8){
            uint8_t* output_data = output->mutable_data<uint8_t>(ctx.GetPlace());
            dst_memory_p =
                handler->AcquireDstMemoryFromPrimitive(to_void_cast<uint8_t>(output_data));
          } else{
            int8_t* output_data = output->mutable_data<int8_t>(ctx.GetPlace());
            dst_memory_p =
                handler->AcquireDstMemoryFromPrimitive(to_void_cast<int8_t>(output_data));
          }
        } else {
          if(fuse_relu){
            uint8_t* output_data = output->mutable_data<uint8_t>(ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, handler->GetDstMemorySize());
            dst_memory_p =
                handler->AcquireDstMemoryFromPrimitive(to_void_cast<uint8_t>(output_data));
          } else{
            int8_t* output_data = output->mutable_data<int8_t>(ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, handler->GetDstMemorySize());
            dst_memory_p =
                handler->AcquireDstMemoryFromPrimitive(to_void_cast<int8_t>(output_data));
          }
        }

        // create convolution op primitive
        std::vector<float> scale_bias_data;
        auto scale_bias_key = key + "@scale_bias";
        if (bias) {
          const float* bias_data = bias->data<float>();
          auto user_bias_md = platform::MKLDNNMemDesc(
              {bias_tz}, platform::MKLDNNGetDataType<float>(), memory::format::x);
          auto user_bias_memory_p =
              handler->AcquireBiasMemory(user_bias_md, to_void_cast<float>(bias_data));
          std::shared_ptr<mkldnn::memory>  bias_memory_p;
          int mask_reorder = is_multi_channel? 1<<0 : 1;
          if(!scale_reuse){
            int count = is_multi_channel? (g>1? weights_tz[1]*weights_tz[0] : weights_tz[0]) : 1;
            scale_bias_data.resize(count);
            #pragma omp parallel for if (count > 1)
            for(int i=0; i<count; i++){
              scale_bias_data[i] = scale_in_data[0] * scale_weights_data[i];
            }
            SetScaleMap(scale_map, scale_bias_key, scale_bias_data);
          } else{
            scale_bias_data = GetScaleMap(scale_map, scale_bias_key);
          }
          bias_memory_p =
              handler->AcquireBiasMemoryFromPrimitive(user_bias_memory_p, pipeline, is_test, is_INT8, scale_bias_data, mask_reorder);
          conv_p = handler->AcquireConvolution(src_memory_p, weights_memory_p,
                                              bias_memory_p, dst_memory_p);
        } else {
          conv_p = handler->AcquireConvolution(src_memory_p, weights_memory_p,
                                              dst_memory_p);
        }


          // push primitive to stream and wait until it's executed
        pipeline.push_back(*conv_p);
        stream(stream::kind::eager).submit(pipeline).wait();

        if(need_s8_to_u8){
          output->mutable_data<uint8_t>(ctx.GetPlace());
        }

        output->set_layout(DataLayout::kMKLDNN);
        output->set_format(GetMKLDNNFormat(*dst_memory_p));
      } else {
        if(src_memory_reorder_p){
          pipeline.push_back(*src_memory_reorder_p);
        }
        pipeline.push_back(*conv_p);
        stream(stream::kind::eager).submit(pipeline).wait();
      
        if (need_s8_to_u8) {
          output->mutable_data<uint8_t>(ctx.GetPlace());
        }

        output->set_layout(DataLayout::kMKLDNN);
        output->set_format(GetMKLDNNFormat(*dst_memory_p));
      }
    }
  }

 private:

    void SetScaleMap(std::unordered_map<std::string, std::vector<float>> &scale_map,
                       const std::string& name, std::vector<float> scale_data) const {
      auto it = scale_map.find(name);
      if (it == scale_map.end()) {
        scale_map[name] = scale_data;  // create new blob
      } else {
        (*it).second = scale_data;  // set data to existing blob
      }
      return;
    }

    std::vector<float> GetScaleMap(std::unordered_map<std::string, std::vector<float>> &scale_map,
         const std::string& name) const {
      auto it = scale_map.find(name);
      if (it != scale_map.end()) {
        return (*it).second;
      }
      return {0};
    }

    mkldnn::primitive_attr CreatePostOps(bool fuse_relu, bool fuse_residual_conn,
                          const std::vector<float> output_shift_scale, float sum_scale) const {
      mkldnn::primitive_attr conv_attr;
      mkldnn::post_ops post_operations;
    // Fusion with Elementwise layer relies on adding a sum post-operation with
    // the scale parameter. It is assumed that when fuse_residual_connection is
    // true, the output tensor contains the data coming from residual
    // connection. The result of this post_op is:
    // Output = scale * Output + Conv_Out.
      int mask = output_shift_scale.size() > 1 ? 1<<1 : 0;
      conv_attr.set_output_scales(mask, output_shift_scale);
      if (fuse_residual_conn) {
        post_operations.append_sum(sum_scale);
      }
      if (fuse_relu) {
        constexpr float scale = 1.0f;
        constexpr float negative_slope = 0.0f;
        constexpr float placeholder = 1.0f; //beta
        post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                       negative_slope, placeholder);
      }
      conv_attr.set_post_ops(post_operations);
      return conv_attr;
    }

      mkldnn::primitive_attr CreatePostOps(bool fuse_relu, bool fuse_residual_conn) const {

      mkldnn::primitive_attr conv_attr;
      mkldnn::post_ops post_operations;
      // Fusion with Elementwise layer relies on adding a sum post-operation with
      // the scale parameter. It is assumed that when fuse_residual_conn is true, the
      // Output tensor contains the data coming from residual connection. The
      // result of this post_op is: Output = scale * Output + Conv_Out.
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
                         const std::vector<float> output_shift_scale, const float sum_scale, bool is_test) const {
      memory::dims stride_dims = {strides[0], strides[1]};
      memory::dims padding_dims = {paddings[0], paddings[1]};

      auto propagation = is_test ? mkldnn::prop_kind::forward_scoring : mkldnn::prop_kind::forward_training;

      auto conv_desc = mkldnn::convolution_forward::desc(
          propagation, mkldnn::convolution_direct, src, weights,
          dst, stride_dims, padding_dims, padding_dims,
          mkldnn::padding_kind::zero);

      mkldnn::primitive_attr conv_attr =
          CreatePostOps(fuse_relu, fuse_residual_conn, output_shift_scale, sum_scale);

      auto p_conv_pd = new mkldnn::convolution_forward::primitive_desc(
          conv_desc, conv_attr, engine);

      return std::unique_ptr<mkldnn::convolution_forward::primitive_desc>(
          p_conv_pd);
    }

  std::unique_ptr<mkldnn::convolution_forward::primitive_desc>
    ConvFwdPrimitiveDesc(const memory::desc& src, const memory::desc& weights,
                         const memory::desc& dst, const std::vector<int>& strides,
                         const std::vector<int>& paddings,
                         const mkldnn::engine& engine, const bool fuse_relu,
                         const bool fuse_residual_conn, bool is_test=false) const{
      memory::dims stride_dims = {strides[0], strides[1]};
      memory::dims padding_dims = {paddings[0], paddings[1]};
 
      auto propagation = is_test ? mkldnn::prop_kind::forward_scoring : mkldnn::prop_kind::forward_training; //Fix propagation bug for FP32 inference.
 
      auto conv_desc = mkldnn::convolution_forward::desc(
          propagation, mkldnn::convolution_direct, src, weights,
          dst, stride_dims, padding_dims, padding_dims,
          mkldnn::padding_kind::zero);
  
      mkldnn::primitive_attr conv_attr = CreatePostOps(fuse_relu, fuse_residual_conn);
  
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
                         const std::vector<float> output_shift_scale, const float sum_scale, bool is_test) const {
      memory::dims stride_dims = {strides[0], strides[1]};
      memory::dims padding_dims = {paddings[0], paddings[1]};

      auto propagation = is_test ? mkldnn::prop_kind::forward_scoring : mkldnn::prop_kind::forward_training;

      auto conv_desc = mkldnn::convolution_forward::desc(
          propagation, mkldnn::convolution_direct, src, weights,
          bias, dst, stride_dims, padding_dims, padding_dims,
          mkldnn::padding_kind::zero);

      mkldnn::primitive_attr conv_attr = 
          CreatePostOps(fuse_relu, fuse_residual_conn, output_shift_scale, sum_scale);

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
                         const bool fuse_residual_conn, bool is_test=false) const{
      memory::dims stride_dims = {strides[0], strides[1]};
      memory::dims padding_dims = {paddings[0], paddings[1]};

      auto propagation = is_test ? mkldnn::prop_kind::forward_scoring : mkldnn::prop_kind::forward_training; //Fix propagation bug for FP32 inference.

      auto conv_desc = mkldnn::convolution_forward::desc(
          propagation, mkldnn::convolution_direct, src, weights,
          bias, dst, stride_dims, padding_dims, padding_dims,
          mkldnn::padding_kind::zero);

      mkldnn::primitive_attr conv_attr = CreatePostOps(fuse_relu, fuse_residual_conn);

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
                   ops::ConvMKLDNNOpKernel<float>,
                   ops::ConvMKLDNNOpKernel<uint8_t>,
                   ops::ConvMKLDNNOpKernel<int8_t>);

REGISTER_OP_KERNEL(conv2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvMKLDNNGradOpKernel<float>);
