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

#include "boost/optional.hpp"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using framework::DataLayout;

inline mkldnn::memory::dims GetWeightsTz(const Tensor* filter,
                                         const int groups) {
  auto iohw_weights_tz = framework::vectorize(filter->dims());
  auto weights_tz = iohw_weights_tz;

  // IOHW -> OIHW
  weights_tz[0] = iohw_weights_tz[1];
  weights_tz[1] = iohw_weights_tz[0];
  int g = std::max(groups, 1);
  platform::GetGroupConvWeightsTz(weights_tz, g);
  return weights_tz;
}

template <typename T, typename K, typename T_out>
class ConvTransposeMKLDNNHandlerT
    : public platform::MKLDNNHandlerT<T, mkldnn::deconvolution_forward> {
 public:
  ConvTransposeMKLDNNHandlerT(const framework::ExecutionContext& ctx,
                              const platform::MKLDNNDeviceContext& dev_ctx,
                              const mkldnn::engine mkldnn_engine,
                              platform::Place cpu_place, const Tensor* input,
                              const Tensor* filter, const Tensor* bias,
                              Tensor* output, const std::string& unique_name)
      : platform::MKLDNNHandlerT<T, mkldnn::deconvolution_forward>(
            dev_ctx, mkldnn_engine, cpu_place,
            platform::CreateKey(dev_ctx, framework::vectorize(input->dims()),
                                unique_name)) {
    if (!this->isCached()) {
      const bool is_test = ctx.Attr<bool>("is_test");
      PADDLE_ENFORCE_EQ(is_test, true,
                        platform::errors::InvalidArgument(
                            "ConvTransposeMKLDNN works only for inference. "
                            "The attribute \'is_test\' value should be set to "
                            "True, but got is_test=False."));

      PADDLE_ENFORCE_EQ(
          input->layout(), DataLayout::kMKLDNN,
          platform::errors::InvalidArgument(
              "Got wrong layout = %d for Input tensor.", input->layout()));
      PADDLE_ENFORCE_NE(input->format(), MKLDNNMemoryFormat::undef,
                        platform::errors::InvalidArgument(
                            "Got wrong format for Input tensor. The input "
                            "format is undefined."));

      PADDLE_ENFORCE_EQ(
          filter->layout(), DataLayout::kMKLDNN,
          platform::errors::InvalidArgument(
              "The filter tensor's laytout should be %d, but got %d.",
              DataLayout::kMKLDNN, filter->layout()));
      PADDLE_ENFORCE_NE(filter->format(), MKLDNNMemoryFormat::undef,
                        platform::errors::InvalidArgument(
                            "Got wrong formats for Filter tensor."));

      PADDLE_ENFORCE_EQ(
          input->dims().size(), 4,
          platform::errors::InvalidArgument("Input must be with 4 dimensions, "
                                            "i.e. NCHW. but got dimension =%d",
                                            input->dims().size()));
      PADDLE_ENFORCE_EQ(
          filter->dims().size(), 4,
          platform::errors::InvalidArgument("Filter must be with 4 dimensions, "
                                            "i.e. OIHW, but got dimension =%d",
                                            filter->dims().size()));

      if (bias) {
        PADDLE_ENFORCE_EQ(
            bias->layout(), DataLayout::kMKLDNN,
            platform::errors::InvalidArgument(
                "The bias tensor's laytout should be %d, but got %d.",
                DataLayout::kMKLDNN, bias->layout()));
        PADDLE_ENFORCE_NE(bias->format(), MKLDNNMemoryFormat::undef,
                          platform::errors::InvalidArgument(
                              "Got wrong format for Bias tensor."));

        PADDLE_ENFORCE_EQ(bias->dims().size(), 1,
                          platform::errors::InvalidArgument(
                              "Bias must only have 1 dimension, "
                              "i.e. X, but got dimension = %d .",
                              bias->dims().size()));
      }

      std::vector<int> strides_temp = ctx.Attr<std::vector<int>>("strides");
      mkldnn::memory::dims strides(begin(strides_temp), end(strides_temp));

      std::vector<int> paddings_temp = ctx.Attr<std::vector<int>>("paddings");
      mkldnn::memory::dims paddings(begin(paddings_temp), end(paddings_temp));

      std::vector<int> dilations_temp = ctx.Attr<std::vector<int>>("dilations");
      mkldnn::memory::dims dilations(begin(dilations_temp),
                                     end(dilations_temp));

      int groups = ctx.Attr<int>("groups");
      std::string padding_algorithm =
          ctx.Attr<std::string>("padding_algorithm");

      PADDLE_ENFORCE_EQ(
          strides.size(), 2,
          platform::errors::Unimplemented(
              "Now we only support 2d oneDNN convolution transpose op"));

      const auto& input_dims = input->dims();
      const auto data_dims =
          framework::slice_ddim(input_dims, 2, input_dims.size());
      const auto& filter_dims = filter->dims();
      const auto filter_data_dims =
          framework::slice_ddim(filter_dims, 2, filter_dims.size());

      const auto ksize = framework::vectorize(filter_data_dims);

      UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                               data_dims, strides, ksize);

      std::transform(dilations.begin(), dilations.end(), dilations.begin(),
                     [](int64_t i) { return i - 1; });

      const auto src_tz = framework::vectorize(input->dims());
      const auto weights_tz = GetWeightsTz(filter, groups);
      const auto dst_tz = framework::vectorize(output->dims());
      const auto mkldnn_paddings = platform::ToMkldnnPadding(paddings);

      /* create memory descriptor for convolution without specified format
       * ('any') which lets a primitive (convolution in this case) choose
       * the memory format preferred for best performance
       */
      const auto chosen_memory_format = MKLDNNMemoryFormat::any;
      const std::string fuse_activation =
          ctx.Attr<std::string>("fuse_activation");
      const float fuse_alpha = ctx.Attr<float>("fuse_alpha");
      const float fuse_beta = ctx.Attr<float>("fuse_beta");

      auto data_type = mkldnn::memory::data_type::f32;
      if (ctx.Attr<std::string>("mkldnn_data_type") == "bfloat16" ||
          std::is_same<T_out, platform::bfloat16>::value)
        data_type = mkldnn::memory::data_type::bf16;

      const auto src_md =
          platform::MKLDNNMemDesc(src_tz, data_type, chosen_memory_format);
      const auto weights_md =
          platform::MKLDNNMemDesc(weights_tz, data_type, chosen_memory_format);
      const auto dst_md = platform::MKLDNNMemDesc(
          dst_tz, platform::MKLDNNGetDataType<T_out>(), chosen_memory_format);

      const mkldnn::primitive_attr conv_trans_attr =
          CreatePostOps(fuse_activation, fuse_alpha, fuse_beta);
      auto fwd_prop_kind = is_test ? mkldnn::prop_kind::forward_inference
                                   : mkldnn::prop_kind::forward_training;
      if (bias) {
        std::vector<int64_t> bias_tz = framework::vectorize(bias->dims());
        const auto bias_md =
            platform::MKLDNNMemDesc(bias_tz, data_type, MKLDNNMemoryFormat::x);
        this->AcquireForwardPrimitiveDescriptor(
            conv_trans_attr, fwd_prop_kind,
            dnnl::algorithm::deconvolution_direct, src_md, weights_md, bias_md,
            dst_md, strides, dilations, mkldnn_paddings[0], mkldnn_paddings[1]);
      } else {
        this->AcquireForwardPrimitiveDescriptor(
            conv_trans_attr, fwd_prop_kind,
            dnnl::algorithm::deconvolution_direct, src_md, weights_md, dst_md,
            strides, dilations, mkldnn_paddings[0], mkldnn_paddings[1]);
      }
    }
  }

  mkldnn::primitive_attr CreatePostOps(const std::string& fuse_activation,
                                       const float& fuse_alpha,
                                       const float& fuse_beta) {
    mkldnn::primitive_attr conv_attr;
    mkldnn::post_ops post_operations;

    // Fusion with ReLU layer is executed through the PostOps feature. Create a
    // PostOps object and configure it to execute an eltwise relu operation.
    if (fuse_activation == "relu" || fuse_activation == "leaky_relu") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                     fuse_alpha, fuse_beta);
    } else if (fuse_activation == "relu6") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale,
                                     mkldnn::algorithm::eltwise_bounded_relu,
                                     fuse_alpha, fuse_beta);
    } else if (fuse_activation == "swish") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_swish,
                                     fuse_alpha, fuse_beta);
    }
    conv_attr.set_post_ops(post_operations);
    return conv_attr;
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemoryWithReorder(
      const framework::Tensor* input) {
    const T* input_data = input->data<T>();
    const std::string user_key_suffix{"@src_mem_p_user"};
    auto user_src_mem_p = this->AcquireMemory(user_key_suffix);
    if (!user_src_mem_p) {
      auto user_src_md = platform::MKLDNNMemDesc(
          framework::vectorize(input->dims()), platform::MKLDNNGetDataType<T>(),
          input->format());
      return this->AcquireMemoryWithReorder(
          user_src_md, this->fwd_pd_->src_desc(),
          platform::to_void_cast<T>(input_data), "@src_mem_p");
    } else {
      const std::string target_key_suffix{"@src_mem_p_target"};
      const auto target_src_mem_p = this->AcquireMemory(target_key_suffix);
      user_src_mem_p->set_data_handle(platform::to_void_cast<T>(input_data));
      if (user_src_mem_p != target_src_mem_p) {
        this->AcquireReorder(user_src_mem_p, target_src_mem_p, "@src_mem_p");
      }
      return target_src_mem_p;
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryWithReorder(
      const framework::Tensor* filter, const int& groups, const bool& is_test) {
    // This is workaround to make execution faster, delete
    // if statement after including md inside Tensor
    auto weights_mem_p = this->AcquireMemory("@weights_mem_p_target");
    if (is_test && weights_mem_p) {
      return weights_mem_p;
    } else {
      const K* filter_data = filter->data<K>();
      auto weights_tz = GetWeightsTz(filter, groups);
      int g = std::max(groups, 1);

      auto user_src_md = platform::MKLDNNMemDesc(
          weights_tz, platform::MKLDNNGetDataType<K>(),
          (g == 1) ? filter->format() : MKLDNNMemoryFormat::goihw);

      auto iohw_weights_tz = framework::vectorize(filter->dims());
      // Custom Reorder from IOHW to OIHW
      auto iohw2oihw_reorder =
          [&iohw_weights_tz](const K* filter_data) -> std::shared_ptr<K> {
        int o = iohw_weights_tz[1];
        int c = iohw_weights_tz[0];
        int h = iohw_weights_tz[2];
        int w = iohw_weights_tz[3];
        std::shared_ptr<K> reordered_filter_data(new K[o * c * h * w](),
                                                 std::default_delete<K[]>());
        for (int i = 0; i < c; ++i) {
          for (int j = 0; j < o; ++j) {
            int in_offset = j * h * w + i * o * h * w;
            int out_offset = j * c * h * w + i * h * w;
            std::memcpy(&(reordered_filter_data.get())[out_offset],
                        &filter_data[in_offset], h * w * sizeof(K));
          }
        }

        return reordered_filter_data;
      };

      return this->template AcquireMemoryWithReorder<K>(
          user_src_md, this->fwd_pd_->weights_desc(),
          platform::to_void_cast<K>(filter_data), "@weights_mem_p", is_test,
          iohw2oihw_reorder);
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireBiasMemoryWithReorder(
      const framework::Tensor* bias, const bool& is_test) {
    auto bias_mem_p = this->AcquireMemory("@bias_mem_p_target");
    if (is_test && bias_mem_p) {
      return bias_mem_p;
    } else {
      const K* bias_data = bias->data<K>();
      auto user_bias_md = platform::MKLDNNMemDesc(
          framework::vectorize(bias->dims()), platform::MKLDNNGetDataType<K>(),
          MKLDNNMemoryFormat::x);
      return this->AcquireMemoryWithReorder(
          user_bias_md, this->fwd_pd_->bias_desc(),
          platform::to_void_cast<K>(bias_data), "@bias_mem_p", is_test);
    }
  }
};

template <typename T, typename K>
class ConvTransposeMKLDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "Operator DNNL ConvTranspose must use CPUPlace"));
    const bool is_bfloat16 =
        ctx.Attr<std::string>("mkldnn_data_type") == "bfloat16";
    const bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    if (is_bfloat16) {
      if (force_fp32_output)
        Execute<float>(ctx);
      else
        Execute<platform::bfloat16>(ctx);
    } else {
      Execute<float>(ctx);
    }
  }

  template <typename T_out>
  void Execute(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const bool is_test = ctx.Attr<bool>("is_test");

    const auto* input = ctx.Input<Tensor>("Input");
    const auto* filter = ctx.Input<Tensor>("Filter");
    const auto* bias =
        ctx.HasInput("Bias") ? ctx.Input<Tensor>("Bias") : nullptr;
    auto* output = ctx.Output<Tensor>("Output");
    const std::string unique_name = ctx.InputName("Input") +
                                    ctx.InputName("Filter") +
                                    (bias ? ctx.InputName("Bias") : "");
    ConvTransposeMKLDNNHandlerT<T, K, T_out> handler(
        ctx, dev_ctx, mkldnn_engine, ctx.GetPlace(), input, filter, bias,
        output, unique_name);
    auto src_memory_p = handler.AcquireSrcMemoryWithReorder(input);
    auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(
        filter, ctx.Attr<int>("groups"), is_test);

    std::shared_ptr<dnnl::memory> dst_memory_p =
        handler.template AcquireDstMemory<T_out>(output);
    auto conv_p = handler.AcquireForwardPrimitive();

    std::unordered_map<int, dnnl::memory> args = {
        {MKLDNN_ARG_SRC, *src_memory_p},
        {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
        {MKLDNN_ARG_DST, *dst_memory_p}};

    if (bias) {
      auto bias_memory_p = handler.AcquireBiasMemoryWithReorder(bias, is_test);
      args.insert({MKLDNN_ARG_BIAS, *bias_memory_p});
    }
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    conv_p->execute(astream, args);
    astream.wait();
    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(platform::GetMKLDNNFormat(*dst_memory_p));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(
    conv2d_transpose, MKLDNN, ::paddle::platform::CPUPlace,
    ops::ConvTransposeMKLDNNOpKernel<float, float>,
    ops::ConvTransposeMKLDNNOpKernel<paddle::platform::bfloat16, float>);
