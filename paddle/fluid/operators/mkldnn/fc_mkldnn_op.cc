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

#include <mkldnn/include/mkldnn_types.h>
#include <memory>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using framework::LoDTensor;
using framework::DDim;
using framework::ExecutionContext;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using mkldnn::memory;
using mkldnn::inner_product_forward;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;

template <typename T_in, typename T_w, typename T_out>
class FCPrimitiveFactory {
 public:
  explicit FCPrimitiveFactory(const mkldnn::engine& engine) : engine_(engine) {}

  void ExecuteFcPrimitive(const LoDTensor* input, const Tensor* weights,
                          const Tensor* bias, LoDTensor* output,
                          const ExecutionContext& ctx) {
    RecomputeOutputDims(ctx, input, weights, output);
    // If primitive has already been created and cached, don't create new one,
    // but update input and output data pointers and return it.
    if (fc_) {
      UpdateDataPointers(ctx, output, input);
      this->Execute();
      return;
    }
    auto src_desc = CreateMemDescriptor<T_in>(input, input->format());
    input_ = CreateMemory<T_in>(src_desc, input);

    // Since MKL-DNN doesn't support 4D column-major data formats in
    // inner_product
    // primitive, transpose the weights to be in row-major format
    weights_ = TransposeWeights(weights);
    if (src_desc.data.ndims == 4) {
      weights_ = CreateFourDimWeightsMemory(input, weights);
    }
    // If int8 data type is desired, weights are quantized to signed int8
    QuantizeWeights(ctx);

    // Choose MKLDNNMemoryFormat::any so that MKL-DNN can determine itself what
    // is the best format for output during the creation of inner product
    // primitive descriptor
    auto dst_desc = CreateMemDescriptor<T_out>(output, MKLDNNMemoryFormat::any);

    fc_ = CreateFcPrimitive(*input_, *weights_, dst_desc, bias, output, ctx);
    this->Execute();
  }

  void Execute() {
    mkldnn::stream astream(engine_);
    if (bias_) {
      fc_->execute(astream, {{MKLDNN_ARG_SRC, *input_},
                             {MKLDNN_ARG_WEIGHTS, *weights_},
                             {MKLDNN_ARG_BIAS, *bias_},
                             {MKLDNN_ARG_DST, *output_}});
    } else {
      fc_->execute(astream, {{MKLDNN_ARG_SRC, *input_},
                             {MKLDNN_ARG_WEIGHTS, *weights_},
                             {MKLDNN_ARG_DST, *output_}});
    }
    astream.wait();
  }

 private:
  void UpdateDataPointers(const ExecutionContext& ctx, Tensor* out,
                          const Tensor* in) {
    input_->set_data_handle(to_void_cast(in->data<T_in>()));
    output_->set_data_handle(out->mutable_data<T_out>(ctx.GetPlace()));
    // If the primitive exists, but the output tensor has changed its
    // variable, update its format to what has been determined in first
    // call to CreateFcPrimitive method.
    if (out->format() == MKLDNNMemoryFormat::undef) {
      auto output_format = platform::GetMKLDNNFormat(*output_);
      out->set_format((MKLDNNMemoryFormat)output_format);
    }
  }

  // Choose weight memory format based on input memory format
  MKLDNNMemoryFormat MatchWeightFormat(MKLDNNMemoryFormat fmt) {
    using format = MKLDNNMemoryFormat;
    switch (fmt) {
      case format::nChw16c:
        return format::aBcd16b;
      case format::nChw8c:
        return format::aBcd8b;
      case format::nchw:
        return format::oihw;
      case format::nhwc:
        return format::hwio;
      default:
        return format::undef;
    }
  }

  // Convert data from one data format to another
  mkldnn::memory Reorder(const memory::desc& src_desc,
                         const memory::desc& dst_desc, void* src_data) {
    auto src_mem = memory(src_desc, engine_, src_data);
    auto dst_mem = memory(dst_desc, engine_);

    auto reorder = mkldnn::reorder(src_mem, dst_mem);
    mkldnn::stream astream(engine_);
    reorder.execute(astream, src_mem, dst_mem);
    astream.wait();

    return dst_mem;
  }

  // Convert data from one data format to another and rescale it.
  // If the desired data type is (un)signed int8, quantization occurs here.
  mkldnn::memory Reorder(const memory& src_mem, const memory::desc& dst_md,
                         const std::vector<float>& scale_data) {
    mkldnn::memory dst_mem = mkldnn::memory(dst_md, engine_);
    mkldnn::primitive_attr attributes;
    // According to MKL-DNN's documentation mask determines along which
    // dimensions should the scale be applied.
    // 0 - Single scale applied to whole tensor
    // 1 - Apply Scale along a slice of each dimension which index is 1.
    //     In case of weights quantization, that dimension is output,
    //     becuase we perform per-output-channel quantization
    int mask = CreateMask(0, scale_data.size() > 1);
    attributes.set_output_scales(mask, scale_data);
    auto reorder = mkldnn::reorder({src_mem, dst_mem, attributes});

    mkldnn::stream astream(engine_);
    reorder.execute(astream,
                    {{MKLDNN_ARG_FROM, src_mem}, {MKLDNN_ARG_TO, dst_mem}});
    astream.wait();

    return dst_mem;
  }

  template <typename T>
  static mkldnn::memory::desc CreateMemDescriptor(
      const std::vector<int64_t>& dims, MKLDNNMemoryFormat format) {
    return platform::MKLDNNMemDesc(dims, platform::MKLDNNGetDataType<T>(),
                                   format);
  }

  template <typename T>
  static mkldnn::memory::desc CreateMemDescriptor(const Tensor* tensor,
                                                  MKLDNNMemoryFormat format) {
    auto dims = framework::vectorize(tensor->dims());
    return CreateMemDescriptor<T>(dims, format);
  }

  template <typename T>
  mkldnn::memory CreateMemory(const mkldnn::memory::desc& desc,
                              const Tensor* tensor) {
    return CreateMemory(desc, platform::to_void_cast<T>(tensor->data<T>()));
  }

  mkldnn::memory CreateMemory(const mkldnn::memory::desc& desc, void* data) {
    return memory(desc, engine_, data);
  }

  // Transpose weights through MKL-DNN's reorder from io to oi format.
  mkldnn::memory TransposeWeights(const Tensor* weights) {
    auto dims = framework::vectorize(weights->dims());
    std::swap(dims[0], dims[1]);  // Correct output dimensions
    auto src_desc = CreateMemDescriptor<float>(dims, MKLDNNMemoryFormat::io);
    auto dst_desc = CreateMemDescriptor<float>(dims, MKLDNNMemoryFormat::oi);
    return Reorder(src_desc, dst_desc,
                   platform::to_void_cast<float>(weights->data<float>()));
  }

  // Compute the bias scales so that its values correspond to the
  // scale of data being an output of weights and input multiplication
  std::vector<float> ComputeBiasScales(const ExecutionContext& ctx) {
    auto scale_in_data = ctx.Attr<float>("Scale_in");
    auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
    const size_t weight_scales_num = scale_weights_data.size();
    std::vector<float> bias_scales(weight_scales_num);

#pragma omp parallel for
    for (size_t i = 0; i < weight_scales_num; i++) {
      if (scale_weights_data[i] == 0.0)
        bias_scales[i] = 1.0f;
      else
        bias_scales[i] = scale_in_data * scale_weights_data[i];
    }

    return bias_scales;
  }

  // Correct output scale, to take into account scaling of input and weights
  // Since the data that comes out of input and weight multiplication is
  // scaled with its own scales, this data needs to be divided by
  // those scales to normalise them back to what their floating-point range
  // was. Then we multiply them by desired output scale we want on the output.
  std::vector<float> ComputeOutputShiftScale(const ExecutionContext& ctx) {
    auto scale_in_data = ctx.Attr<float>("Scale_in");
    auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
    // If the output will be in floats, we don't multiply by scale_out.
    auto scale_out_data = ctx.Attr<bool>("force_fp32_output")
                              ? 1.0f
                              : ctx.Attr<float>("Scale_out");
    const size_t weight_scales_num = scale_weights_data.size();
    std::vector<float> output_shift_scale(weight_scales_num);

#pragma omp parallel for
    for (size_t i = 0; i < weight_scales_num; i++) {
      if (scale_weights_data[i] == 0.0)
        output_shift_scale[i] = scale_out_data;
      else
        output_shift_scale[i] =
            scale_out_data / (scale_in_data * scale_weights_data[i]);
    }

    return output_shift_scale;
  }

  // Computing MKL-DNN's scaling mask which determines along which dimension
  // slice should the scaling be applied. For more data plase refer to:
  // https://intel.github.io/mkl-dnn/group__c__api__attributes.html
  // Section dnnl_status_t DNNL_API dnnl_primitive_attr_set_output_scales
  int CreateMask(int slice_dimension, bool is_multi_channel_quantizied) {
    return is_multi_channel_quantizied ? 1 << slice_dimension : 0;
  }

  void QuantizeWeights(const ExecutionContext& ctx) {
    auto quantized_desc = weights_->get_desc();
    quantized_desc.data.data_type =
        (mkldnn_data_type_t)platform::MKLDNNGetDataType<T_w>();
    weights_ = Reorder(*weights_, quantized_desc,
                       ctx.Attr<std::vector<float>>("Scale_weights"));
  }

  void QuantizeBias(const inner_product_forward::primitive_desc& fc_prim_desc,
                    const ExecutionContext& ctx) {
    auto bias_scales = ComputeBiasScales(ctx);
    bias_ = Reorder(*bias_, fc_prim_desc.bias_desc(), bias_scales);
  }

  // Fuse relu into FC with activation type attribute has been set to 'relu'
  mkldnn::primitive_attr CreatePostOps(const ExecutionContext& ctx) {
    mkldnn::primitive_attr attributes;
    mkldnn::post_ops post_operations;

    auto output_shift_scale = ComputeOutputShiftScale(ctx);
    int mask = CreateMask(1, output_shift_scale.size() > 1);
    attributes.set_output_scales(mask, output_shift_scale);

    if (ctx.Attr<std::string>("activation_type") == "relu") {
      constexpr float scale = 1.0f;
      constexpr float negative_slope = 0.0f;
      constexpr float placeholder = 1.0f;  // beta
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                     negative_slope, placeholder);
    }

    attributes.set_post_ops(post_operations);
    return attributes;
  }

  inner_product_forward CreateFcPrimitive(const memory& src_memory,
                                          const memory& weights_memory,
                                          const memory::desc& dst_desc,
                                          const Tensor* bias, Tensor* output,
                                          const ExecutionContext& ctx) {
    // Acquire descriptors needed for creation of inner_product primitive
    // descriptor
    const auto weights_desc = weights_memory.get_desc();
    const auto src_desc = src_memory.get_desc();
    // Based on provided attributes, create attributes used by MKL-DNN to
    // enable fused post-op activations such as 'relu'
    const auto attrs = CreatePostOps(ctx);
    // If bias exists, create inner_product primitive with or without bias
    if (bias) {
      auto bias_desc = CreateMemDescriptor<float>(bias, bias->format());
      bias_ = CreateMemory<float>(bias_desc, bias);
      // Create inner_product descriptor. At this point the format of output
      // is determined.
      auto fc_prim_desc =
          CreateFcPrimDesc(src_desc, weights_desc, bias_desc, dst_desc, attrs);
      // If int8 is desired, quantize bias into 32-bit signed int
      QuantizeBias(fc_prim_desc, ctx);

      // Based on format determined by inner_product, create output in desired
      // memory format
      output_ = CreateDstMemory(fc_prim_desc, ctx, output);

      // Return MKL-DNN primitive ready to be fed into pipeline and executed
      return inner_product_forward(fc_prim_desc);
    } else {
      auto fc_prim_desc =
          CreateFcPrimDesc(src_desc, weights_desc, dst_desc, attrs);
      output_ = CreateDstMemory(fc_prim_desc, ctx, output);
      return inner_product_forward(fc_prim_desc);
    }
  }

  mkldnn::inner_product_forward::primitive_desc CreateFcPrimDesc(
      const mkldnn::memory::desc& input_desc,
      const mkldnn::memory::desc& weights_desc,
      const mkldnn::memory::desc& bias_desc,
      const mkldnn::memory::desc& dst_desc,
      const mkldnn::primitive_attr& attrs) {
    auto fc_desc =
        inner_product_forward::desc(prop_kind::forward_scoring, input_desc,
                                    weights_desc, bias_desc, dst_desc);

    return inner_product_forward::primitive_desc(fc_desc, attrs, engine_);
  }

  mkldnn::inner_product_forward::primitive_desc CreateFcPrimDesc(
      const mkldnn::memory::desc& input_desc,
      const mkldnn::memory::desc& weights_desc,
      const mkldnn::memory::desc& dst_desc,
      const mkldnn::primitive_attr& attrs) {
    auto fc_desc = inner_product_forward::desc(prop_kind::forward, input_desc,
                                               weights_desc, dst_desc);

    return inner_product_forward::primitive_desc(fc_desc, attrs, engine_);
  }

  // Since MKL-DNN requires the number of input dimensions to be
  // equal to the number of weight dimensions, we have to convert
  // weights to 4D memory if input is 4D. It also requires that
  // all dimensions of weights and inputs agree, with an exception
  // for the batch size and number of output channels (the first dim).
  // In order to perform that we have to prepare the memory descriptor
  // by hand, as MKL-DNN's reorder does not support conversion
  // from one dimensionality to another. Hence, we set
  // the first dimension of weights to resemble number of outputs
  // and then we use the sizes of number of input channels as well
  // as image width and height for latter dimensions. Then we create
  // memories, find a format corresponding with input format and
  // perform a converion.
  mkldnn::memory CreateFourDimWeightsMemory(const Tensor* input,
                                            const Tensor* weights) {
    auto input_dims = framework::vectorize(input->dims());
    auto weight_dims = framework::vectorize(weights->dims());
    auto dims = {weight_dims[1], input_dims[1], input_dims[2], input_dims[3]};

    auto dst_format = MatchWeightFormat(input->format());
    auto src_desc = CreateMemDescriptor<float>(dims, MKLDNNMemoryFormat::oihw);
    auto dst_desc = CreateMemDescriptor<float>(dims, dst_format);

    return Reorder(src_desc, dst_desc, weights_->get_data_handle());
  }

  // Create output memory based on output tensor and inner_product
  // primitive descriptor format chosen for output
  mkldnn::memory CreateDstMemory(
      const mkldnn::inner_product_forward::primitive_desc& fc_prim_desc,
      const ExecutionContext& ctx, Tensor* output) {
    auto dst_desc = fc_prim_desc.dst_desc();
    auto buffer_size = dst_desc.get_size();
    T_out* output_data =
        output->mutable_data<T_out>(ctx.GetPlace(), buffer_size);
    memory dst_mem(dst_desc, engine_, to_void_cast<T_out>(output_data));
    output->set_format(platform::GetMKLDNNFormat(dst_mem));
    return dst_mem;
  }

  void RecomputeOutputDims(const ExecutionContext& ctx, const LoDTensor* input,
                           const Tensor* w, LoDTensor* output) {
    int in_num_col_dims = ctx.Attr<int>("in_num_col_dims");
    bool padding_weights = ctx.Attr<bool>("padding_weights");
    PADDLE_ENFORCE_EQ(padding_weights, false,
                      platform::errors::PermissionDenied(
                          "Weight padding in fc can not be used in MKLDNN."));
    std::vector<int64_t> output_dims;
    FCOutputSize(input->dims(), w->dims(), output_dims, in_num_col_dims,
                 padding_weights);
    output->Resize(framework::make_ddim(output_dims));
    output->set_lod(input->lod());
  }

 private:
  const mkldnn::engine& engine_;
  boost::optional<memory> bias_;
  boost::optional<memory> input_;
  boost::optional<memory> output_;
  boost::optional<memory> weights_;
  boost::optional<inner_product_forward> fc_;
};

// Attempt to fetch cached primitive factory based on provided parameters
// of input format, weight dimensions and output name.
// If not cached, create a new one.
template <typename T_in, typename T_w, typename T_out>
static std::shared_ptr<FCPrimitiveFactory<T_in, T_w, T_out>>
GetPrimitiveFactory(const MKLDNNDeviceContext& dev_ctx,
                    const ExecutionContext& ctx, const Tensor* input,
                    const Tensor* weights,
                    const mkldnn::engine& mkldnn_engine) {
  const std::string key = platform::CreateKey(
      platform::ThreadIDasStr(), input->format(),
      framework::vectorize<int>(weights->dims()), ctx.OutputName("Out"));

  auto prim_creator =
      std::static_pointer_cast<FCPrimitiveFactory<T_in, T_w, T_out>>(
          dev_ctx.GetBlob(key));
  if (prim_creator == nullptr) {
    prim_creator =
        std::make_shared<FCPrimitiveFactory<T_in, T_w, T_out>>(mkldnn_engine);
    dev_ctx.SetBlob(key, prim_creator);
  }

  return prim_creator;
}

// Choose appropriate primitive factory implementation based on inferred
// output type (uint8, int8 or float).
template <typename T_in, typename T_w>
static void ExecuteFc(const MKLDNNDeviceContext& dev_ctx,
                      const ExecutionContext& ctx, const LoDTensor* input,
                      const Tensor* w, const Tensor* bias, LoDTensor* output,
                      const mkldnn::engine& mkldnn_engine, bool fuse_relu,
                      bool force_fp32_output) {
  constexpr bool is_int8 =
      std::is_same<T_in, int8_t>::value || std::is_same<T_in, uint8_t>::value;
  if (!is_int8 || force_fp32_output) {
    GetPrimitiveFactory<T_in, T_w, float>(dev_ctx, ctx, input, w, mkldnn_engine)
        ->ExecuteFcPrimitive(input, w, bias, output, ctx);
  } else if (fuse_relu) {
    GetPrimitiveFactory<T_in, T_w, uint8_t>(dev_ctx, ctx, input, w,
                                            mkldnn_engine)
        ->ExecuteFcPrimitive(input, w, bias, output, ctx);
  } else {
    GetPrimitiveFactory<T_in, T_w, int8_t>(dev_ctx, ctx, input, w,
                                           mkldnn_engine)
        ->ExecuteFcPrimitive(input, w, bias, output, ctx);
  }
}

template <typename T_in, typename T_w>
class FCMKLDNNOpKernel : public framework::OpKernel<T_in> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("FC MKL-DNN must use CPUPlace."));
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto input = ctx.Input<LoDTensor>("Input");
    auto w = ctx.Input<Tensor>("W");
    auto bias = ctx.Input<Tensor>("Bias");
    auto output = ctx.Output<LoDTensor>("Out");

    bool fuse_relu = ctx.Attr<std::string>("activation_type") == "relu";
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

    ExecuteFc<T_in, T_w>(dev_ctx, ctx, input, w, bias, output, mkldnn_engine,
                         fuse_relu, force_fp32_output);

    output->set_layout(DataLayout::kMKLDNN);
  }
};
}  // namespace operators
}  // namespace paddle

// Weights of FC are by default stored using fp32, template argument of weight
// data type implies their destination data type. (What's eventually going to
// be used during computations of kernel).
namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    FP32, ops::kFCMKLDNNFP32,
                                    ops::FCMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    U8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<uint8_t, int8_t>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    S8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<int8_t, int8_t>);
