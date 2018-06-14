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

using conv_bwd_data = mkldnn::convolution_backward_data;
using conv_bwd_weights = mkldnn::convolution_backward_weights;
using conv_fwd = mkldnn::convolution_forward;
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

    // Get unique name for index
    const std::string key = ctx.op().Output("Output");
    const std::string key_conv_pd = key + "@conv_pd";

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

    // create mkldnn memory from input tensors (data/weights)
    auto user_src_memory = memory(
        {{{src_tz}, memory::data_type::f32, input->format()}, mkldnn_engine},
        to_void_cast(input_data));
    auto user_weights_memory =
        memory({{{weights_tz}, memory::data_type::f32, filter->format()},
                mkldnn_engine},
               to_void_cast(filter_data));

    /* create memory descriptor for convolution without specified format
     * ('any') which lets a primitive (convolution in this case) choose
     * the memory format preferred for best performance
     */
    auto src_md = platform::MKLDNNMemDesc(src_tz, memory::data_type::f32,
                                          memory::format::any);
    auto weights_md = platform::MKLDNNMemDesc(
        weights_tz, memory::data_type::f32, memory::format::any);
    auto dst_md = platform::MKLDNNMemDesc(dst_tz, memory::data_type::f32,
                                          memory::format::any);

    // create a conv primitive descriptor and save it for usage in backward
    std::shared_ptr<conv_fwd::primitive_desc> conv_pd = ConvFwdPrimitiveDesc(
        src_md, weights_md, dst_md, strides, paddings, mkldnn_engine);

    // create reorder primitive if the input format is not the preferred one
    auto src_memory = user_src_memory;
    primitive reorder_src;
    bool is_src_reordered = false;
    if (memory::primitive_desc(conv_pd->src_primitive_desc()) !=
        user_src_memory.get_primitive_desc()) {
      src_memory = memory(conv_pd->src_primitive_desc());
      reorder_src = reorder(user_src_memory, src_memory);
      is_src_reordered = true;
    }
    auto weights_memory = user_weights_memory;
    primitive reorder_weights;
    bool is_weights_reordered = false;
    if (memory::primitive_desc(conv_pd->weights_primitive_desc()) !=
        user_weights_memory.get_primitive_desc()) {
      weights_memory = memory(conv_pd->weights_primitive_desc());
      reorder_weights = reorder(user_weights_memory, weights_memory);
      is_weights_reordered = true;
    }

    // create memory primitive for conv dst
    auto dst_memory = memory(conv_pd->dst_primitive_desc(), output_data);

    // create convolution op primitive
    auto conv_prim = conv_fwd(*conv_pd, src_memory, weights_memory, dst_memory);

    // push primitive to stream and wait until it's executed
    std::vector<primitive> pipeline;
    if (is_src_reordered) pipeline.push_back(reorder_src);
    if (is_weights_reordered) pipeline.push_back(reorder_weights);
    pipeline.push_back(conv_prim);
    stream(stream::kind::eager).submit(pipeline).wait();

    // Save conv_pd/src_memory/weights_memory for backward pass
    dev_ctx.SetBlob(key_conv_pd, conv_pd);

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(dst_memory));
  }

 private:
  std::unique_ptr<conv_fwd::primitive_desc> ConvFwdPrimitiveDesc(
      const memory::desc& src, const memory::desc& weights,
      const memory::desc& dst, const std::vector<int>& strides,
      const std::vector<int>& paddings, const mkldnn::engine& engine) const {
    memory::dims stride_dims = {strides[0], strides[1]};
    memory::dims padding_dims = {paddings[0], paddings[1]};

    auto conv_desc =
        conv_fwd::desc(mkldnn::prop_kind::forward, mkldnn::convolution_direct,
                       src, weights, dst, stride_dims, padding_dims,
                       padding_dims, mkldnn::padding_kind::zero);

    auto p_conv_pd = new conv_fwd::primitive_desc(conv_desc, engine);

    return std::unique_ptr<conv_fwd::primitive_desc>(p_conv_pd);
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

    // create mkldnn memory from input tensors (input/weights/output_grad)
    auto user_src_memory = memory(
        {{{src_tz}, memory::data_type::f32, input->format()}, mkldnn_engine},
        to_void_cast(input_data));
    auto user_weights_memory =
        memory({{{weights_tz}, memory::data_type::f32, filter->format()},
                mkldnn_engine},
               to_void_cast(filter_data));
    auto user_diff_dst_memory =
        memory({{{dst_tz}, memory::data_type::f32, output_grad->format()},
                mkldnn_engine},
               to_void_cast(output_grad_data));

    /* create memory descriptor for conv backward without specified format
     * ('any') which lets a primitive (conv backward in this case) choose
     * the memory format preferred for best performance
     */
    auto src_md = platform::MKLDNNMemDesc(src_tz, memory::data_type::f32,
                                          memory::format::any);
    auto diff_src_md = platform::MKLDNNMemDesc(src_tz, memory::data_type::f32,
                                               memory::format::any);
    auto weights_md = platform::MKLDNNMemDesc(
        weights_tz, memory::data_type::f32, memory::format::any);
    auto diff_weights_md = platform::MKLDNNMemDesc(
        weights_tz, memory::data_type::f32, memory::format::any);
    auto diff_dst_md = platform::MKLDNNMemDesc(dst_tz, memory::data_type::f32,
                                               memory::format::any);

    // Retrieve conv_pd from device context
    auto conv_pd = std::static_pointer_cast<conv_fwd::primitive_desc>(
        dev_ctx.GetBlob(key_conv_pd));
    PADDLE_ENFORCE(conv_pd != nullptr,
                   "Fail to find conv_pd in device context");

    // create backward conv primitive for weights
    if (filter_grad) {
      // create backward convolution primitive descriptor
      auto conv_bwd_weights_desc = conv_bwd_weights::desc(
          mkldnn::convolution_direct, src_md, diff_weights_md, diff_dst_md,
          strides, paddings, paddings, mkldnn::padding_kind::zero);
      auto conv_bwd_weights_pd = conv_bwd_weights::primitive_desc(
          conv_bwd_weights_desc, mkldnn_engine, *conv_pd);

      // create reorder primitive if the input format is not the preferred one
      auto src_memory = user_src_memory;
      primitive reorder_src;
      bool is_src_reordered = false;
      if (memory::primitive_desc(conv_bwd_weights_pd.src_primitive_desc()) !=
          user_src_memory.get_primitive_desc()) {
        src_memory = memory(conv_bwd_weights_pd.src_primitive_desc());
        reorder_src = reorder(user_src_memory, src_memory);
        is_src_reordered = true;
      }

      auto diff_dst_memory_4filter = user_diff_dst_memory;
      primitive reorder_diff_dst_4filter;
      bool is_diff_dst_reordered_4filter = false;
      if (memory::primitive_desc(
              conv_bwd_weights_pd.diff_dst_primitive_desc()) !=
          user_diff_dst_memory.get_primitive_desc()) {
        diff_dst_memory_4filter =
            memory(conv_bwd_weights_pd.diff_dst_primitive_desc());
        reorder_diff_dst_4filter =
            reorder(user_diff_dst_memory, diff_dst_memory_4filter);
        is_diff_dst_reordered_4filter = true;
      }

      // create mkldnn memory for output (i.e. diff weights)
      auto diff_weights_memory =
          memory(conv_bwd_weights_pd.diff_weights_primitive_desc(),
                 reinterpret_cast<void*>(filter_grad_data));

      // create backward conv primitive for weights
      auto conv_bwd_weights_prim =
          conv_bwd_weights(conv_bwd_weights_pd, src_memory,
                           diff_dst_memory_4filter, diff_weights_memory);

      // push primitive and execute it
      std::vector<primitive> pipeline;
      if (is_src_reordered) pipeline.push_back(reorder_src);
      if (is_diff_dst_reordered_4filter)
        pipeline.push_back(reorder_diff_dst_4filter);
      pipeline.push_back(conv_bwd_weights_prim);
      stream(stream::kind::eager).submit(pipeline).wait();

      filter_grad->set_layout(DataLayout::kMKLDNN);
      filter_grad->set_format(GetMKLDNNFormat(diff_weights_memory));
    }

    if (input_grad) {
      // create backward convolution primitive descriptor
      auto conv_bwd_data_desc = conv_bwd_data::desc(
          mkldnn::convolution_direct, diff_src_md, weights_md, diff_dst_md,
          strides, paddings, paddings, mkldnn::padding_kind::zero);
      auto conv_bwd_data_pd = conv_bwd_data::primitive_desc(
          conv_bwd_data_desc, mkldnn_engine, *conv_pd);

      // create reorder primitive if the input format is not the preferred one
      auto weights_memory = user_weights_memory;
      primitive reorder_weights;
      bool is_weights_reordered = false;
      if (memory::primitive_desc(conv_bwd_data_pd.weights_primitive_desc()) !=
          user_weights_memory.get_primitive_desc()) {
        weights_memory = memory(conv_bwd_data_pd.weights_primitive_desc());
        reorder_weights = reorder(user_weights_memory, weights_memory);
        is_weights_reordered = true;
      }

      auto diff_dst_memory_4data = user_diff_dst_memory;
      primitive reorder_diff_dst_4data;
      bool is_diff_dst_reordered_4data = false;
      if (memory::primitive_desc(conv_bwd_data_pd.diff_dst_primitive_desc()) !=
          user_diff_dst_memory.get_primitive_desc()) {
        diff_dst_memory_4data =
            memory(conv_bwd_data_pd.diff_dst_primitive_desc());
        reorder_diff_dst_4data =
            reorder(user_diff_dst_memory, diff_dst_memory_4data);
        is_diff_dst_reordered_4data = true;
      }

      // create mkldnn memory for output (i.e. diff src)
      auto diff_src_memory = memory(conv_bwd_data_pd.diff_src_primitive_desc(),
                                    reinterpret_cast<void*>(input_grad_data));

      // create backward conv primitive for data
      auto conv_bwd_data_prim =
          conv_bwd_data(conv_bwd_data_pd, diff_dst_memory_4data, weights_memory,
                        diff_src_memory);

      // push primitive and execute it
      std::vector<primitive> pipeline;
      if (is_weights_reordered) pipeline.push_back(reorder_weights);
      if (is_diff_dst_reordered_4data)
        pipeline.push_back(reorder_diff_dst_4data);
      pipeline.push_back(conv_bwd_data_prim);
      stream(stream::kind::eager).submit(pipeline).wait();

      input_grad->set_layout(DataLayout::kMKLDNN);
      input_grad->set_format(GetMKLDNNFormat(diff_src_memory));
    }
  }  // Compute()
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(conv2d, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(conv2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConvMKLDNNGradOpKernel<float>);
