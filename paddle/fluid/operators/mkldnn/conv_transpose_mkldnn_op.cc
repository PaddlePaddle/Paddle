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
                   "It must use CPUPlace.");

    const bool is_test = ctx.Attr<bool>("is_test");
    PADDLE_ENFORCE(
        is_test == true,
        "ConvTransposeMKLDNN works only for inference!. Set is_test = True");

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* bias = ctx.HasInput("Bias") ? ctx.Input<Tensor>("Bias") : nullptr;
    auto* output = ctx.Output<Tensor>("Output");

    PADDLE_ENFORCE(input->layout() == DataLayout::kMKLDNN &&
                       input->format() != mkldnn::memory::format::format_undef,
                   "Wrong layout/format set for Input tensor");
    PADDLE_ENFORCE(filter->layout() == DataLayout::kMKLDNN &&
                       filter->format() != mkldnn::memory::format::format_undef,
                   "Wrong layout/format set for Filter tensor");
    PADDLE_ENFORCE(input->dims().size() == 4,
                   "Input must be with 4 dimensions, i.e. NCHW");
    PADDLE_ENFORCE(filter->dims().size() == 4,
                   "Filter must be with 4 dimensions, i.e. OIHW");

    if (bias) {
      PADDLE_ENFORCE(bias->layout() == DataLayout::kMKLDNN &&
                         bias->format() != mkldnn::memory::format::format_undef,
                     "Wrong layout/format set for Bias tensor");
      PADDLE_ENFORCE(bias->dims().size() == 1,
                     "Bias must only have 1 dimension, i.e. X");
    }

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");

    // TODO(tpatejko): add support for dilation
    PADDLE_ENFORCE(
        dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1,
        "dilation in convolution is not implemented yet");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();

    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> iohw_weights_tz =
        paddle::framework::vectorize2int(filter->dims());
    std::vector<int> weights_tz = iohw_weights_tz;
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
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

    // Get unique name for storing MKLDNN primitives
    const std::string key = platform::ConvTransposeMKLDNNHandler::GetHash(
        src_tz, weights_tz, strides, paddings, dilations, groups,
        ctx.op().Output("Output"));

    std::vector<mkldnn::primitive> pipeline;

    auto user_src_md = platform::MKLDNNMemDesc(
        {src_tz}, platform::MKLDNNGetDataType<T>(), input->format());
    auto user_weights_md =
        platform::MKLDNNMemDesc({weights_tz}, platform::MKLDNNGetDataType<T>(),
                                (g == 1) ? mkldnn::memory::format::oihw
                                         : mkldnn::memory::format::goihw);

    /* create memory descriptor for convolution without specified format
     * ('any') which lets a primitive (convolution in this case) choose
     * the memory format preferred for best performance
     */
    std::string data_format = ctx.Attr<std::string>("data_format");
    auto chosen_memory_format =
        platform::data_format_to_memory_format(data_format);
    bool fuse_relu = ctx.Attr<bool>("fuse_relu");

    auto src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    std::vector<int> bias_tz;  // TODO(mgallus): avoid empty vector creation.
                               // Currently used whenever bias is != nullptr.
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
      bias_tz = paddle::framework::vectorize2int(bias->dims());
      auto bias_md = platform::MKLDNNMemDesc(
          bias_tz, platform::MKLDNNGetDataType<T>(), mkldnn::memory::format::x);
      conv_transpose_pd = handler.AcquireConvolutionPrimitiveDescriptor(
          src_md, weights_md, bias_md, dst_md, strides, paddings, mkldnn_engine,
          fuse_relu, false, fwd_prop_kind);
    } else {
      conv_transpose_pd = handler.AcquireConvolutionPrimitiveDescriptor(
          src_md, weights_md, boost::none, dst_md, strides, paddings,
          mkldnn_engine, fuse_relu, false, fwd_prop_kind);
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

    std::shared_ptr<mkldnn::memory> dst_memory_p;

    auto output_data = output->mutable_data<T>(
        ctx.GetPlace(), paddle::memory::Allocator::kDefault,
        handler.GetDstMemorySize());
    dst_memory_p = handler.AcquireDstMemoryFromPrimitive(
        platform::to_void_cast<T>(output_data));

    // create convolution op primitive
    std::shared_ptr<mkldnn::deconvolution_forward> conv_p;
    if (bias) {
      const T* bias_data = bias->data<T>();
      auto user_bias_md =
          platform::MKLDNNMemDesc({bias_tz}, platform::MKLDNNGetDataType<T>(),
                                  mkldnn::memory::format::x);
      auto user_bias_memory_p = handler.AcquireBiasMemory(
          user_bias_md, platform::to_void_cast<T>(bias_data));

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
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(platform::GetMKLDNNFormat(*dst_memory_p));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(conv2d_transpose, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvTransposeMKLDNNOpKernel<float>);
