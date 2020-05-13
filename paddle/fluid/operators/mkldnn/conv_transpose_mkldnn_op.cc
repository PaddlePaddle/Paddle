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

template <typename T>
class ConvTransposeMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   platform::errors::InvalidArgument("It must use CPUPlace."));

    const bool is_test = ctx.Attr<bool>("is_test");
    PADDLE_ENFORCE_EQ(is_test, true,
                      platform::errors::InvalidArgument(
                          "ConvTransposeMKLDNN works only for inference. "
                          "Set is_test = True. but got is_test=False ."));

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* bias = ctx.HasInput("Bias") ? ctx.Input<Tensor>("Bias") : nullptr;
    auto* output = ctx.Output<Tensor>("Output");

    PADDLE_ENFORCE_EQ(
        input->layout(), DataLayout::kMKLDNN,
        platform::errors::InvalidArgument(
            "Got wrong layout = %d for Input tensor.", input->layout()));
    PADDLE_ENFORCE_NE(input->format(), MKLDNNMemoryFormat::undef,
                      platform::errors::InvalidArgument(
                          "Got wrong format for Input tensor."));

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
        platform::errors::InvalidArgument(
            "Input must be with 4 dimensions, i.e. NCHW. but got dimension =%d",
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

      PADDLE_ENFORCE_EQ(
          bias->dims().size(), 1,
          platform::errors::InvalidArgument("Bias must only have 1 dimension, "
                                            "i.e. X, but got dimension = %d .",
                                            bias->dims().size()));
    }

    std::vector<int> strides_temp = ctx.Attr<std::vector<int>>("strides");
    std::vector<int64_t> strides(begin(strides_temp), end(strides_temp));

    std::vector<int> paddings_temp = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int64_t> paddings(begin(paddings_temp), end(paddings_temp));

    std::vector<int> dilations_temp = ctx.Attr<std::vector<int>>("dilations");
    std::vector<int64_t> dilations(begin(dilations_temp), end(dilations_temp));

    int groups = ctx.Attr<int>("groups");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");

    auto input_dims = input->dims();
    auto data_dims = framework::slice_ddim(input_dims, 2, input_dims.size());
    auto filter_dims = filter->dims();
    auto filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());

    auto ksize = framework::vectorize(filter_data_dims);

    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             data_dims, strides, ksize);

    PADDLE_ENFORCE(
        dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1,
        "dilation in convolution is not implemented yet");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();

    auto src_tz = paddle::framework::vectorize<int64_t>(input->dims());
    auto iohw_weights_tz =
        paddle::framework::vectorize<int64_t>(filter->dims());
    auto weights_tz = iohw_weights_tz;

    // IOHW -> OIHW
    weights_tz[0] = iohw_weights_tz[1];
    weights_tz[1] = iohw_weights_tz[0];

    // Custom Reorder from IOHW to OIHW
    auto iohw2oihw_reorder =
        [&iohw_weights_tz](const T* filter_data) -> std::shared_ptr<T> {
      int o = iohw_weights_tz[1];
      int c = iohw_weights_tz[0];
      int h = iohw_weights_tz[2];
      int w = iohw_weights_tz[3];
      std::shared_ptr<T> reordered_filter_data(new T[o * c * h * w](),
                                               std::default_delete<T[]>());
      for (int i = 0; i < c; ++i) {
        for (int j = 0; j < o; ++j) {
          int in_offset = j * h * w + i * o * h * w;
          int out_offset = j * c * h * w + i * h * w;
          std::memcpy(&(reordered_filter_data.get())[out_offset],
                      &filter_data[in_offset], h * w * sizeof(T));
        }
      }

      return reordered_filter_data;
    };

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
    auto dst_tz = paddle::framework::vectorize<int64_t>(output->dims());

    // Get unique name for storing MKLDNN primitives

    const std::string key =
        platform::CreateKey(src_tz, ctx.OutputName("Output"));

    std::vector<mkldnn::primitive> pipeline;

    auto user_src_md = platform::MKLDNNMemDesc(
        {src_tz}, platform::MKLDNNGetDataType<T>(), input->format());
    auto user_weights_md = platform::MKLDNNMemDesc(
        {weights_tz}, platform::MKLDNNGetDataType<T>(),
        (g == 1) ? MKLDNNMemoryFormat::oihw : MKLDNNMemoryFormat::goihw);

    /* create memory descriptor for convolution without specified format
     * ('any') which lets a primitive (convolution in this case) choose
     * the memory format preferred for best performance
     */
    auto chosen_memory_format = MKLDNNMemoryFormat::any;
    std::string fuse_activation = ctx.Attr<std::string>("fuse_activation");
    float fuse_alpha = ctx.Attr<float>("fuse_alpha");
    float fuse_beta = ctx.Attr<float>("fuse_beta");

    auto src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    std::vector<int64_t> bias_tz;
    auto dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);

    platform::ConvTransposeMKLDNNHandler handler(dev_ctx, mkldnn_engine, key);
    // create a deconv(conv transpose) primitive descriptor and save it for
    // usage in backward
    std::shared_ptr<mkldnn::deconvolution_forward::primitive_desc>
        conv_transpose_pd;
    auto fwd_prop_kind = is_test ? mkldnn::prop_kind::forward_inference
                                 : mkldnn::prop_kind::forward_training;
    if (bias) {
      bias_tz = paddle::framework::vectorize<int64_t>(bias->dims());
      auto bias_md = platform::MKLDNNMemDesc(
          bias_tz, platform::MKLDNNGetDataType<T>(), MKLDNNMemoryFormat::x);
      conv_transpose_pd = handler.AcquireConvolutionPrimitiveDescriptor(
          src_md, weights_md, bias_md, dst_md, strides, paddings, mkldnn_engine,
          fuse_activation, fuse_alpha, fuse_beta, false, fwd_prop_kind);
    } else {
      conv_transpose_pd = handler.AcquireConvolutionPrimitiveDescriptor(
          src_md, weights_md, boost::none, dst_md, strides, paddings,
          mkldnn_engine, fuse_activation, fuse_alpha, fuse_beta, false,
          fwd_prop_kind);
    }

    // create mkldnn memory from input tensors (data/weights)
    auto user_src_memory_p = handler.AcquireSrcMemory(
        user_src_md, platform::to_void_cast<T>(input_data));
    auto user_weights_memory_p = handler.AcquireWeightsMemory(
        user_weights_md, platform::to_void_cast<T>(filter_data),
        is_test ? iohw2oihw_reorder : platform::user_function());

    // create reorder primitive if the input format is not the preferred one
    auto src_memory_p =
        handler.AcquireSrcMemoryFromPrimitive(user_src_memory_p, pipeline);
    auto weights_memory_p = handler.AcquireWeightsMemoryFromPrimitive(
        user_weights_memory_p, pipeline, is_test);

    auto output_data =
        output->mutable_data<T>(ctx.GetPlace(), handler.GetDstMemorySize());
    auto dst_memory_p = handler.AcquireDstMemoryFromPrimitive(
        platform::to_void_cast<T>(output_data));

    auto conv_p = handler.AcquireConvolution();

    mkldnn::stream astream(mkldnn_engine);
    if (bias) {
      const T* bias_data = bias->data<T>();
      auto user_bias_md = platform::MKLDNNMemDesc(
          {bias_tz}, platform::MKLDNNGetDataType<T>(), MKLDNNMemoryFormat::x);
      auto user_bias_memory_p = handler.AcquireBiasMemory(
          user_bias_md, platform::to_void_cast<T>(bias_data));

      auto bias_memory_p =
          handler.AcquireBiasMemoryFromPrimitive(user_bias_memory_p, pipeline);

      conv_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                                {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                {MKLDNN_ARG_BIAS, *bias_memory_p},
                                {MKLDNN_ARG_DST, *dst_memory_p}});
    } else {
      conv_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                                {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                {MKLDNN_ARG_DST, *dst_memory_p}});
    }
    astream.wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(platform::GetMKLDNNFormat(*dst_memory_p));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(conv2d_transpose, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvTransposeMKLDNNOpKernel<float>);
