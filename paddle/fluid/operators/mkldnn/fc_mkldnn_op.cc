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

#include <memory>

#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
<<<<<<< HEAD

namespace phi {
class DenseTensor;
}  // namespace phi
=======
#include "paddle/fluid/platform/mkldnn_reuse.h"
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

namespace paddle {
namespace operators {

using dnnl::inner_product_forward;
using dnnl::memory;
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::stream;
<<<<<<< HEAD
using framework::DataLayout;
using framework::DDim;
using framework::ExecutionContext;
using framework::LoDTensor;
using framework::Tensor;
using platform::GetMKLDNNFormat;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
=======
using framework::DDim;
using framework::ExecutionContext;
using LoDTensor = phi::DenseTensor;
using phi::funcs::OneDNNGetDataType;
using phi::funcs::to_void_cast;
using platform::MKLDNNDeviceContext;
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

struct InnerProductCache {
  dnnl::inner_product_forward inner_product_p;
  dnnl::memory src_mem;
  dnnl::memory weights_mem;
  dnnl::memory bias_mem;
  dnnl::memory dst_mem;
};
template <typename T_in, typename T_w, typename T_out>
class FCMKLDNNHandler
    : public phi::funcs::OneDNNHandlerNoCachingT<T_in,
                                                 dnnl::inner_product_forward> {
 public:
<<<<<<< HEAD
  explicit FCPrimitiveFactory(const dnnl::engine& engine) : engine_(engine) {}

  void ExecuteFcPrimitive(const LoDTensor* input,
                          const Tensor* weights,
                          const Tensor* bias,
                          LoDTensor* output,
                          const MKLDNNDeviceContext& dev_ctx,
                          const ExecutionContext& ctx) {
    RecomputeOutputDims(ctx, input, weights, output);
    // If primitive has already been created and cached, don't create new one,
    // but update input and output data pointers and return it.
    if (fc_) {
      UpdateDataPointers(ctx, output, input);
      this->Execute();
      return;
    }  // Otherwise, create a new one.

    auto in_col_dims = ctx.Attr<int>("in_num_col_dims");
    PADDLE_ENFORCE_LE(
        in_col_dims,
        2,
        platform::errors::Unimplemented(
            "DNNL FC doesn't support in_num_col_dims parameter to "
            "be higher than "
            "2."));
    if (in_col_dims == 2) {
      PADDLE_ENFORCE_EQ(
          input->dims().size(),
          3,
          platform::errors::Unimplemented(
              "DNNL FC only supports in_num_col_dims equal to 2 when "
              "3 dim input is provided."));
      PADDLE_ENFORCE_EQ(
          input->format(),
          MKLDNNMemoryFormat::ncw,
          platform::errors::Unimplemented(
              "DNNL FC only supports in_num_col_dims equal to 2 when "
              "input format is equal to ncw."));
=======
  FCMKLDNNHandler(const paddle::framework::ExecutionContext& ctx,
                  const platform::MKLDNNDeviceContext& dev_ctx,
                  const phi::DenseTensor* x,
                  const phi::DenseTensor* weights,
                  const phi::DenseTensor* bias,
                  phi::DenseTensor* out,
                  const int in_num_col_dims,
                  dnnl::engine mkldnn_engine,
                  platform::Place cpu_place)
      : phi::funcs::OneDNNHandlerNoCachingT<T_in, dnnl::inner_product_forward>(
            mkldnn_engine, cpu_place),
        dev_ctx_(dev_ctx) {
    this->memory_key_ = ctx.InputName("W");

    auto x_vec_dims = phi::vectorize(x->dims());
    auto weights_vec_dims = phi::vectorize(weights->dims());

    int MB = 1;
    for (int i = 0; i < in_num_col_dims; ++i) {
      MB *= x_vec_dims[i];
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    }

    int IC = 1;
    for (size_t i = in_num_col_dims; i < x_vec_dims.size(); ++i) {
      IC *= x_vec_dims[i];
    }
<<<<<<< HEAD
    input_ = CreateMemory<T_in>(fc_prim_desc->src_desc(), input);
    // Update weights format inside of its memory
    weights_ = Reorder(
        usr_weights_desc, usr_weights_desc, weights_->get_data_handle());
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    int OC = weights_vec_dims[1];

    dnnl::memory::desc bias_md;

    auto src_md = dnnl::memory::desc(
        {MB, IC}, OneDNNGetDataType<T_in>(), dnnl::memory::format_tag::any);
    auto weights_md = dnnl::memory::desc(
        {OC, IC}, OneDNNGetDataType<T_w>(), dnnl::memory::format_tag::any);
    auto dst_md = dnnl::memory::desc(
        {MB, OC}, OneDNNGetDataType<T_out>(), dnnl::memory::format_tag::any);
    if (bias) {
      bias_md = dnnl::memory::desc({bias->numel()},
                                   OneDNNGetDataType<float>(),
                                   dnnl::memory::format_tag::a);
    }

    const auto attrs = CreateFCAttrs(ctx);

<<<<<<< HEAD
    // Return MKL-DNN primitive ready to be fed into pipeline and executed
    fc_ = inner_product_forward(*fc_prim_desc);
    this->Execute();
  }

  void Execute() {
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    if (bias_) {
      fc_->execute(astream,
                   {{DNNL_ARG_SRC, *input_},
                    {DNNL_ARG_WEIGHTS, *weights_},
                    {DNNL_ARG_BIAS, *bias_},
                    {DNNL_ARG_DST, *output_}});
    } else {
      fc_->execute(astream,
                   {{DNNL_ARG_SRC, *input_},
                    {DNNL_ARG_WEIGHTS, *weights_},
                    {DNNL_ARG_DST, *output_}});
    }
    astream.wait();
  }

 private:
  // DNNL always returns 2-dimensional data block as a result of computing
  // inner product. Hence the format 'nc' is always set for its output
  // primitive. Therefore, function SetOutputFormat is needed to choose
  // an appropriate format based on the number of input dimensions and
  // format of an input tensor.
  void SetOutputFormat(MKLDNNMemoryFormat in_format, Tensor* out) {
    int dim_num = out->dims().size();
    // In case of 2 dims, we set the only possible format, nc
    if (dim_num == 2) {
      out->set_format(MKLDNNMemoryFormat::nc);
      out->set_mem_desc({phi::vectorize(out->dims()),
                         platform::MKLDNNGetDataType<T_out>(),
                         out->format()});
      // In case of 3 dims, we generate a format that is based on number
      // of output dims and the layout of input format (nchw or nhwc).
    } else if (dim_num == 3) {
      if (in_format == MKLDNNMemoryFormat::nwc ||
          in_format == MKLDNNMemoryFormat::nhwc) {
        out->set_format(
            platform::MKLDNNFormatForSize(dim_num, MKLDNNMemoryFormat::nhwc));
      } else {
        out->set_format(
            platform::MKLDNNFormatForSize(dim_num, MKLDNNMemoryFormat::nchw));
      }
      // In any other case we overwrite the output format with the input one.
    } else {
      out->set_format(in_format);
    }
  }

  void UpdateDataPointers(const ExecutionContext& ctx,
                          Tensor* out,
                          const Tensor* in) {
    input_->set_data_handle(to_void_cast(in->data<T_in>()));
    output_->set_data_handle(out->mutable_data<T_out>(ctx.GetPlace()));
    // If the primitive exists, but the output tensor has changed its
    // variable, update its format to what has been determined in first
    // call to CreateFcPrimitive method.
    if (out->format() == MKLDNNMemoryFormat::undef) {
      SetOutputFormat(in->format(), out);
=======
    this->AcquireForwardPrimitiveDescriptor(attrs,
                                            prop_kind::forward_inference,
                                            src_md,
                                            weights_md,
                                            bias_md,
                                            dst_md);
  }

 private:
  dnnl::primitive_attr CreateFCAttrs(const ExecutionContext& ctx) {
    dnnl::primitive_attr attributes;
    dnnl::post_ops post_operations;

    float sum_scale = 1.0f;
    float activation_scale = 1.0f;
    if (phi::funcs::is_int8<T_w>()) {
      std::vector<float> output_shift_scale;
      std::tie(output_shift_scale, sum_scale, activation_scale) =
          GetOutputScales(ctx);
      int mask = CreateMask(1, output_shift_scale.size() > 1);
      attributes.set_output_scales(mask, output_shift_scale);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    }

<<<<<<< HEAD
  dnnl::inner_product_forward::primitive_desc Create2DFcPrimDescriptor(
      const LoDTensor* input,
      const Tensor* weights,
      const Tensor* bias,
      LoDTensor* output,
      const ExecutionContext& ctx) {
    auto src_desc = CreateMemDescriptor<T_in>(input, MKLDNNMemoryFormat::any);
    auto weight_dims = Get2DWeightDimsForDNNL(weights);
    auto weights_desc =
        CreateMemDescriptor<T_w>(weight_dims, MKLDNNMemoryFormat::any);
    auto bias_desc = CreateMemDescriptor<float>(bias, MKLDNNMemoryFormat::x);
    auto dst_desc = CreateMemDescriptor<T_out>(output, MKLDNNMemoryFormat::any);
    const auto attrs = CreateFCAttrs(ctx);
    return CreateFcPrimDesc(src_desc, weights_desc, bias_desc, dst_desc, attrs);
  }

  std::vector<int64_t> Get2DWeightDimsForDNNL(const Tensor* weights) {
    auto dims = phi::vectorize(weights->dims());
    std::swap(dims[0], dims[1]);  // swap input dim with output dim
    return dims;
  }

  memory::desc Create2DUserWeightsDesc() { return weights_->get_desc(); }

  dnnl::inner_product_forward::primitive_desc Create3DFcPrimDescriptor(
      const LoDTensor* input,
      const Tensor* weights,
      const Tensor* bias,
      LoDTensor* output,
      const ExecutionContext& ctx) {
    auto input_dims = phi::vectorize(input->dims());
    std::vector<int64_t> new_input_dims = {
        input_dims[0] * input_dims[1], input_dims[2], 1};
    auto src_desc =
        CreateMemDescriptor<T_in>(new_input_dims, MKLDNNMemoryFormat::any);

    auto weight_dims = Get3DWeightDimsForDNNL(weights);
    auto weights_desc =
        CreateMemDescriptor<T_w>(weight_dims, MKLDNNMemoryFormat::any);
=======
    if (ctx.HasAttr("fuse_residual_connection") &&
        ctx.Attr<bool>("fuse_residual_connection")) {
      post_operations.append_sum(sum_scale);
    }
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    // ReLU from "fc_fuse_pass"
    if (ctx.Attr<std::string>("activation_type") == "relu") {
      post_operations.append_eltwise(
          activation_scale, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    }
    platform::AppendActivation(ctx, post_operations, activation_scale);

<<<<<<< HEAD
    auto dst_dims = {input_dims[0] * input_dims[1], weight_dims[0]};
    auto dst_desc =
        CreateMemDescriptor<T_out>(dst_dims, MKLDNNMemoryFormat::any);
    const auto attrs = CreateFCAttrs(ctx);
    return CreateFcPrimDesc(src_desc, weights_desc, bias_desc, dst_desc, attrs);
  }
=======
    if (ctx.HasAttr("fused_output_scale")) {
      float scale_alpha = ctx.Attr<float>("fused_output_scale");
      post_operations.append_eltwise(
          1.0, dnnl::algorithm::eltwise_linear, scale_alpha, 0.0f);
    }
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    attributes.set_post_ops(post_operations);
    return attributes;
  }

  // Compute the bias scales so that its values correspond to the
  // scale of data being an output of weights and input multiplication
  std::vector<float> GetBiasScales(const framework::ExecutionContext& ctx) {
    if (ctx.HasAttr("Bias_scales")) {
      return ctx.Attr<std::vector<float>>("Bias_scales");
    } else {
      const float scale_in = ctx.Attr<float>("Scale_in");
      const auto& scale_weights = ctx.Attr<std::vector<float>>("Scale_weights");
      std::vector<float> bias_scales(scale_weights.size());

      for (size_t i = 0; i < bias_scales.size(); ++i) {
        if (scale_weights[i] == 0.0)
          bias_scales[i] = 1.0f;
        else
          bias_scales[i] = scale_in * scale_weights[i];
      }
      return bias_scales;
    }
  }

<<<<<<< HEAD
  dnnl::inner_product_forward::primitive_desc Create4DFcPrimDescriptor(
      const LoDTensor* input,
      const Tensor* weights,
      const Tensor* bias,
      LoDTensor* output,
      const ExecutionContext& ctx) {
    auto src_desc = CreateMemDescriptor<T_in>(input, MKLDNNMemoryFormat::any);
    // Since MKL-DNN doesn't support 4D column-major data formats in
    // inner_product primitive, transpose the weights to be in
    // row-major format
    auto dims = Get4DWeightDimsForDNNL(input, weights);
    auto weights_desc = CreateMemDescriptor<T_w>(dims, MKLDNNMemoryFormat::any);
    auto bias_desc = CreateMemDescriptor<float>(bias, MKLDNNMemoryFormat::x);
    auto dst_desc = CreateMemDescriptor<T_out>(output, MKLDNNMemoryFormat::any);
    const auto attrs = CreateFCAttrs(ctx);
    return CreateFcPrimDesc(src_desc, weights_desc, bias_desc, dst_desc, attrs);
=======
  // Correct output scale, to take into account scaling of input and weights
  // Since the data that comes out of input and weight multiplication is
  // scaled with its own scales, this data needs to be divided by
  // those scales to normalise them back to what their floating-point range
  // was. Then we multiply them by desired output scale we want on the output.
  std::tuple<std::vector<float>, float, float> GetOutputScales(
      const ExecutionContext& ctx) {
    if (ctx.HasAttr("Sum_scale")) {
      return std::make_tuple(ctx.Attr<std::vector<float>>("Output_shift_scale"),
                             ctx.Attr<float>("Sum_scale"),
                             ctx.Attr<float>("Activation_scale"));
    } else {
      auto scale_in_data = ctx.Attr<float>("Scale_in");
      auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
      bool has_activation = !ctx.Attr<std::string>("activation_type").empty();
      bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
      bool fuse_residual_conn = ctx.HasAttr("fuse_residual_connection") &&
                                ctx.Attr<bool>("fuse_residual_connection");
      auto scale_in_eltwise_data = ctx.HasAttr("Scale_in_eltwise")
                                       ? ctx.Attr<float>("Scale_in_eltwise")
                                       : 1.0f;

      // If the output will be in floats, we don't multiply by scale_out.

      float activation_scale = (!force_fp32_output && has_activation)
                                   ? ctx.Attr<float>("Scale_out")
                                   : 1.0f;
      float scale_out_data = (force_fp32_output || has_activation)
                                 ? 1.0f
                                 : ctx.Attr<float>("Scale_out");
      float sum_scale =
          fuse_residual_conn ? scale_out_data / scale_in_eltwise_data : 1.0f;
      const size_t weight_scales_num = scale_weights_data.size();

      for (size_t i = 0; i < weight_scales_num; ++i) {
        if (scale_weights_data[i] == 0.0)
          scale_weights_data[i] = scale_out_data;
        else
          scale_weights_data[i] =
              scale_out_data / (scale_in_data * scale_weights_data[i]);
      }
      return std::make_tuple(scale_weights_data, sum_scale, activation_scale);
    }
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  }

  // Computing MKL-DNN's scaling mask which determines along which dimension
  // slice should the scaling be applied. For more data plase refer to:
  // https://intel.github.io/mkl-dnn/group__c__api__attributes.html
  // Section dnnl_status_t DNNL_API dnnl_primitive_attr_set_output_scales
  int CreateMask(int slice_dimension, bool is_multi_channel_quantizied) {
    return is_multi_channel_quantizied ? 1 << slice_dimension : 0;
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorderAndAttrs(
      const dnnl::memory::desc& user_md,
      const dnnl::memory::desc& target_md,
      void* ptr,
      const dnnl::primitive_attr& attrs) {
    std::shared_ptr<dnnl::memory> target_memory_p;

    auto user_memory_p =
        std::make_shared<dnnl::memory>(user_md, this->engine_, ptr);
    target_memory_p = std::make_shared<dnnl::memory>(target_md, this->engine_);
    auto reorder_p = std::make_shared<dnnl::reorder>(
        *user_memory_p, *target_memory_p, attrs);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    {
      platform::RecordEvent record_reorder(
          "int_reorder",
          platform::TracerEventType::UserDefined,
<<<<<<< HEAD
          2,
=======
          1,
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
          platform::EventRole::kUniqueOp);
      reorder_p->execute(
          astream,
          {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
      astream.wait();
    }

    return target_memory_p;
  }

<<<<<<< HEAD
  // Convert data from one data format to another and rescale it.
  // If the desired data type is (un)signed int8, quantization occurs here.
  std::shared_ptr<dnnl::memory> ReorderWithScale(
      const std::shared_ptr<memory> src_mem,
      const memory::desc& dst_md,
      const std::vector<float>& scale_data) {
    auto dst_mem = std::make_shared<dnnl::memory>(dst_md, engine_);
    dnnl::primitive_attr attributes;
    // According to MKL-DNN's documentation mask determines along which
    // dimensions should the scale be applied.
    // 0 - Single scale applied to whole tensor
    // 1 - Apply Scale along a slice of each dimension which index is 1.
    //     In case of weights quantization, that dimension is output,
    //     becuase we perform per-output-channel quantization
    int mask = CreateMask(0, scale_data.size() > 1);
    attributes.set_output_scales(mask, scale_data);
    auto reorder = dnnl::reorder(*src_mem, *dst_mem, attributes);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    {
      platform::RecordEvent record_reorder(
          "int_reorder",
          platform::TracerEventType::UserDefined,
          2,
          platform::EventRole::kUniqueOp);
      reorder.execute(astream,
                      {{DNNL_ARG_FROM, *src_mem}, {DNNL_ARG_TO, *dst_mem}});
      astream.wait();
=======
  std::string memory_key_;
  const platform::MKLDNNDeviceContext& dev_ctx_;

 public:
  std::shared_ptr<dnnl::memory> AcquireSrcMemoryWithReorder(
      const phi::DenseTensor* x) {
    const T_in* x_data = x->data<T_in>();

    auto user_md = x->mem_desc();
    if (x->dims().size() != 2) {
      // reshape restrictions are always satisfied because in case of 3 or 4 dim
      // input, plain layout is enforced
      user_md = user_md.reshape(this->fwd_pd_->src_desc().dims());
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    }

    return this->AcquireMemoryWithReorder(
        user_md, this->fwd_pd_->src_desc(), to_void_cast<T_in>(x_data));
  }

<<<<<<< HEAD
  template <typename T>
  static dnnl::memory::desc CreateMemDescriptor(
      const std::vector<int64_t>& dims, MKLDNNMemoryFormat format) {
    return platform::MKLDNNMemDesc(
        dims, platform::MKLDNNGetDataType<T>(), format);
  }
=======
  std::shared_ptr<dnnl::memory> AcquireBiasMemoryWithReorder(
      const framework::ExecutionContext& ctx, const phi::DenseTensor* bias) {
    const float* bias_data = bias->data<float>();
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    if (phi::funcs::is_int8<T_w>() == false) {
      // for BF16/FP32 bias is 1D and has no scales, so reorder is not needed
      return this->AcquireMemoryFromPrimitive(this->fwd_pd_->bias_desc(),
                                              to_void_cast<float>(bias_data));
    } else {
      const std::string bias_key = this->memory_key_ + "@bias";
      auto memory_p = std::static_pointer_cast<dnnl::memory>(
          this->dev_ctx_.GetBlob(bias_key));

      if (!memory_p) {
        const auto& scale_data = GetBiasScales(ctx);
        dnnl::primitive_attr attrs;

        int mask = CreateMask(0, scale_data.size() > 1);
        attrs.set_output_scales(mask, scale_data);

        auto user_md = dnnl::memory::desc({bias->dims()[0]},
                                          OneDNNGetDataType<float>(),
                                          dnnl::memory::format_tag::a);

        memory_p = this->AcquireMemoryWithReorderAndAttrs(
            user_md,
            this->fwd_pd_->bias_desc(),
            to_void_cast<float>(bias_data),
            attrs);
        this->dev_ctx_.SetBlob(bias_key, memory_p);
      }
      return memory_p;
    }
  }

  std::shared_ptr<dnnl::memory> AcquireWeightsMemoryWithReorder(
      const phi::DenseTensor* weights, const std::vector<float>& scale_data) {
    const std::string weights_key = this->memory_key_ + "@weights";
    auto memory_p = std::static_pointer_cast<dnnl::memory>(
        this->dev_ctx_.GetBlob(weights_key));

    if (!memory_p) {
      const float* weights_data = weights->data<float>();
      auto weights_dims = this->fwd_pd_->weights_desc().dims();

      auto user_md = dnnl::memory::desc(weights_dims,
                                        OneDNNGetDataType<float>(),
                                        dnnl::memory::format_tag::io);

      if (phi::funcs::is_int8<T_w>()) {
        dnnl::primitive_attr attrs;
        int mask = CreateMask(0, scale_data.size() > 1);
        attrs.set_output_scales(mask, scale_data);

        memory_p = this->AcquireMemoryWithReorderAndAttrs(
            user_md,
            this->fwd_pd_->weights_desc(),
            to_void_cast<float>(weights_data),
            attrs);
      } else {
        memory_p =
            this->AcquireMemoryWithReorder(user_md,
                                           this->fwd_pd_->weights_desc(),
                                           to_void_cast<float>(weights_data));
      }

<<<<<<< HEAD
  // Create weights memory and transform to default MKL-DNN format
  std::shared_ptr<dnnl::memory> CreateWeightsMemory(const Tensor* weights) {
    auto dims = phi::vectorize(weights->dims());
    std::swap(dims[0], dims[1]);  // Correct output dimensions
    auto src_desc = CreateMemDescriptor<float>(dims, MKLDNNMemoryFormat::io);
    auto dst_desc = CreateMemDescriptor<float>(dims, MKLDNNMemoryFormat::oi);
    // Transpose weights through MKL-DNN's reorder from io to oi format.
    return Reorder(src_desc,
                   dst_desc,
                   platform::to_void_cast<float>(weights->data<float>()));
=======
      this->dev_ctx_.SetBlob(weights_key, memory_p);
    }
    return memory_p;
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  }

  std::shared_ptr<dnnl::memory> AcquireCustomDstMemory(
      const ExecutionContext& ctx, phi::DenseTensor* out) {
    if (ctx.HasAttr("fuse_residual_connection") &&
        ctx.Attr<bool>("fuse_residual_connection")) {
      auto* residual_param = ctx.Input<phi::DenseTensor>("ResidualData");

      PADDLE_ENFORCE_EQ(
          out->dims(),
          residual_param->dims(),
          platform::errors::InvalidArgument(
              "Output and elementwise parameter need to have the "
              "same dimension sizes, but got output's dimension = %d"
              " and residual param's dimension =%d .",
              out->dims().size(),
              residual_param->dims().size()));

      out->ShareDataWith(*residual_param);
    }
    return this->template AcquireDstMemory<T_out>(out);
  }  // namespace operators
};   // namespace paddle

#define IF_CHANGE_FC_TW_TYPENAME(condition, ...) \
  if (condition) {                               \
    using T_w = int8_t;                          \
    __VA_ARGS__();                               \
  } else {                                       \
    using T_w = T_in;                            \
    __VA_ARGS__();                               \
  }

template <typename T_in>
class FCMKLDNNKernel : public framework::OpKernel<T_in> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    bool fuse_relu = ctx.Attr<std::string>("activation_type") == "relu";

    IF_CHANGE_FC_TW_TYPENAME((std::is_same<T_in, uint8_t>::value), ([&] {
                               if (force_fp32_output) {
                                 this->RunKernel<float, T_w>(ctx);
                               } else if (phi::funcs::is_int8<T_in>()) {
                                 if (fuse_relu) {
                                   this->RunKernel<uint8_t, T_w>(ctx);
                                 } else {
                                   this->RunKernel<int8_t, T_w>(ctx);
                                 }
                               } else {
                                 this->RunKernel<T_in, T_w>(ctx);
                               }
                             }));
  }

  void PrepareSrcMem(const std::shared_ptr<inner_product_forward>& fc_p,
                     const std::shared_ptr<dnnl::memory>& src_mem,
                     const LoDTensor* x,
                     const dnnl::engine& engine) const {
    auto x_md = x->mem_desc().reshape(src_mem->get_desc().dims());
    if (x_md != src_mem->get_desc()) {
      dnnl::memory x_mem(x_md, engine, to_void_cast<T_in>(x->data<T_in>()));
      auto reorder_p = dnnl::reorder(x_mem, *src_mem);

      auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
      reorder_p.execute(astream, x_mem, *src_mem);
      astream.wait();
    } else {
      src_mem->set_data_handle(to_void_cast<T_in>(x->data<T_in>()));
    }
  }

<<<<<<< HEAD
  // Computing MKL-DNN's scaling mask which determines along which dimension
  // slice should the scaling be applied. For more data plase refer to:
  // https://intel.github.io/mkl-dnn/group__c__api__attributes.html
  // Section dnnl_status_t DNNL_API dnnl_primitive_attr_set_output_scales
  int CreateMask(int slice_dimension, bool is_multi_channel_quantizied) {
    return is_multi_channel_quantizied ? 1 << slice_dimension : 0;
  }

  void QuantizeWeights(const ExecutionContext& ctx, memory::desc dst) {
    weights_ = ReorderWithScale(
        weights_, dst, ctx.Attr<std::vector<float>>("Scale_weights"));
  }

  void QuantizeBias(const inner_product_forward::primitive_desc& fc_prim_desc,
                    const ExecutionContext& ctx) {
    auto bias_scales = ComputeBiasScales(ctx);
    bias_ = ReorderWithScale(bias_, fc_prim_desc.bias_desc(), bias_scales);
  }

  dnnl::primitive_attr CreateFCAttrs(const ExecutionContext& ctx) {
    dnnl::primitive_attr attributes;
    dnnl::post_ops post_operations;

    std::vector<float> output_shift_scale;
    float scale;
    std::tie(output_shift_scale, scale) = ComputeOutputShiftScale(ctx);
    int mask = CreateMask(1, output_shift_scale.size() > 1);
    attributes.set_output_scales(mask, output_shift_scale);

    float sum_scale = 1.0f;
    if (ctx.HasAttr("fuse_residual_connection") &&
        ctx.Attr<bool>("fuse_residual_connection")) {
      post_operations.append_sum(sum_scale);
    }

    if (ctx.Attr<std::string>("activation_type") == "relu") {
      constexpr float negative_slope = 0.0f;
      constexpr float placeholder = 1.0f;  // beta
      post_operations.append_eltwise(
          scale, dnnl::algorithm::eltwise_relu, negative_slope, placeholder);
    } else if (ctx.Attr<std::string>("activation_type") == "gelu") {
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(
          scale, dnnl::algorithm::eltwise_gelu, alpha, beta);
    } else if (ctx.Attr<std::string>("activation_type") == "gelu_tanh") {
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(
          scale, dnnl::algorithm::eltwise_gelu_tanh, alpha, beta);
    } else if (ctx.Attr<std::string>("activation_type") == "gelu_erf") {
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(
          scale, dnnl::algorithm::eltwise_gelu_erf, alpha, beta);
    } else if (ctx.Attr<std::string>("activation_type") == "tanh") {
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(
          scale, dnnl::algorithm::eltwise_tanh, alpha, beta);
    } else if (ctx.Attr<std::string>("activation_type") == "sigmoid") {
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(
          scale, dnnl::algorithm::eltwise_logistic, alpha, beta);
    } else if (ctx.Attr<std::string>("activation_type") == "mish") {
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(
          scale, dnnl::algorithm::eltwise_mish, alpha, beta);
    } else if (ctx.Attr<std::string>("activation_type") == "hard_swish") {
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(
          scale, dnnl::algorithm::eltwise_hardswish, alpha, beta);
=======
  template <typename T_out, typename T_w>
  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<LoDTensor>("Input");
    const auto* weights = ctx.Input<phi::DenseTensor>("W");
    const auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto out = ctx.Output<LoDTensor>("Out");

    const auto& scale_weights = ctx.Attr<std::vector<float>>("Scale_weights");

    std::shared_ptr<dnnl::inner_product_forward> fc_p;
    std::shared_ptr<dnnl::memory> src_memory_p;
    std::shared_ptr<dnnl::memory> weights_memory_p;
    std::shared_ptr<dnnl::memory> bias_memory_p;
    std::shared_ptr<dnnl::memory> dst_memory_p;

    std::string cache_key;
    cache_key.reserve(64);
    cache_key = platform::ExtendKeyWithThreadInfoIfNeeded(
        dev_ctx,
        platform::CreateKey(dev_ctx,
                            ctx.InputName("Input"),
                            ctx.InputName("W"),
                            phi::vectorize(x->dims())));

    auto inner_product_cache =
        std::static_pointer_cast<InnerProductCache>(dev_ctx.GetBlob(cache_key));

    RecomputeOutputDims(ctx, x, weights, out);

    if (inner_product_cache) {
      fc_p = std::make_shared<dnnl::inner_product_forward>(
          inner_product_cache->inner_product_p);
      src_memory_p =
          std::make_shared<dnnl::memory>(inner_product_cache->src_mem);
      PrepareSrcMem(fc_p, src_memory_p, x, mkldnn_engine);

      weights_memory_p =
          std::make_shared<dnnl::memory>(inner_product_cache->weights_mem);

      dst_memory_p =
          std::make_shared<dnnl::memory>(inner_product_cache->dst_mem);
      if (ctx.HasAttr("fuse_residual_connection") &&
          ctx.Attr<bool>("fuse_residual_connection")) {
        auto* residual_param = ctx.Input<phi::DenseTensor>("ResidualData");
        out->ShareDataWith(*residual_param);
      }
      auto out_ptr = out->mutable_data<T_out>(
          ctx.GetPlace(), dst_memory_p->get_desc().get_size());
      dst_memory_p->set_data_handle(out_ptr);

      if (bias) {
        bias_memory_p =
            std::make_shared<dnnl::memory>(inner_product_cache->bias_mem);
      }
    } else {
      auto in_col_dims = ctx.Attr<int>("in_num_col_dims");

      FCMKLDNNHandler<T_in, T_w, T_out> handler(ctx,
                                                dev_ctx,
                                                x,
                                                weights,
                                                bias,
                                                out,
                                                in_col_dims,
                                                mkldnn_engine,
                                                ctx.GetPlace());

      src_memory_p = handler.AcquireSrcMemoryWithReorder(x);
      weights_memory_p =
          handler.AcquireWeightsMemoryWithReorder(weights, scale_weights);
      dst_memory_p = handler.AcquireCustomDstMemory(ctx, out);

      if (bias) {
        bias_memory_p = handler.AcquireBiasMemoryWithReorder(ctx, bias);
      }

      fc_p = handler.AcquireForwardPrimitive();
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    }

    auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();

<<<<<<< HEAD
  dnnl::inner_product_forward::primitive_desc CreateFcPrimDesc(
      const dnnl::memory::desc& input_desc,
      const dnnl::memory::desc& weights_desc,
      const dnnl::memory::desc& bias_desc,
      const dnnl::memory::desc& dst_desc,
      const dnnl::primitive_attr& attrs) {
    auto fc_desc = inner_product_forward::desc(prop_kind::forward_scoring,
                                               input_desc,
                                               weights_desc,
                                               bias_desc,
                                               dst_desc);

    return inner_product_forward::primitive_desc(fc_desc, attrs, engine_);
  }

  // Create output memory based on output tensor and inner_product
  // primitive descriptor format chosen for output
  dnnl::memory CreateDstMemory(
      const dnnl::inner_product_forward::primitive_desc& fc_prim_desc,
      const ExecutionContext& ctx,
      Tensor* output) {
    if (ctx.HasAttr("fuse_residual_connection") &&
        ctx.Attr<bool>("fuse_residual_connection")) {
      auto* residual_param = ctx.Output<Tensor>("ResidualData");

      PADDLE_ENFORCE_EQ(
          output->dims(),
          residual_param->dims(),
          platform::errors::InvalidArgument(
              "Output and elementwise parameter need to have the "
              "same dimension sizes, but got output's dimension = %d"
              " and residual param's dimension =%d .",
              output->dims().size(),
              residual_param->dims().size()));
=======
    std::unordered_map<int, dnnl::memory> fc_args = {
        {DNNL_ARG_SRC, *src_memory_p},
        {DNNL_ARG_WEIGHTS, *weights_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    if (bias) {
      fc_args.insert({DNNL_ARG_BIAS, *bias_memory_p});
    }

    fc_p->execute(astream, fc_args);
    astream.wait();
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    if (!inner_product_cache) {
      auto ip_cache = std::make_shared<InnerProductCache>();
      ip_cache->inner_product_p = *fc_p;
      ip_cache->src_mem = *src_memory_p;
      ip_cache->weights_mem = *weights_memory_p;
      ip_cache->dst_mem = *dst_memory_p;
      if (bias) {
        ip_cache->bias_mem = *bias_memory_p;
      }
      dev_ctx.SetBlob(cache_key, ip_cache);
    }

    platform::SetOutMemDescWithLogicalLayoutFusesSupport(
        ctx,
        out,
        dst_memory_p->get_desc().reshape(phi::vectorize(out->dims())));
  }

  void RecomputeOutputDims(const ExecutionContext& ctx,
<<<<<<< HEAD
                           const LoDTensor* input,
                           const Tensor* w,
                           LoDTensor* output) {
=======
                           const LoDTensor* x,
                           const phi::DenseTensor* weights,
                           LoDTensor* out) const {
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    int in_num_col_dims = ctx.Attr<int>("in_num_col_dims");
    bool padding_weights = ctx.Attr<bool>("padding_weights");
    PADDLE_ENFORCE_EQ(padding_weights,
                      false,
                      platform::errors::PermissionDenied(
                          "Weight padding in fc can not be used in MKLDNN."));
    std::vector<int64_t> output_dims;
<<<<<<< HEAD
    FCOutputSize(input->dims(),
                 w->dims(),
=======
    FCOutputSize(x->dims(),
                 weights->dims(),
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                 output_dims,
                 in_num_col_dims,
                 padding_weights);
    out->Resize(phi::make_ddim(output_dims));
    out->set_lod(x->lod());
  }
};

<<<<<<< HEAD
// Attempt to fetch cached primitive factory based on provided parameters
// of input format, weight dimensions and output name.
// If not cached, create a new one.
template <typename T_in, typename T_w, typename T_out>
static std::shared_ptr<FCPrimitiveFactory<T_in, T_w, T_out>>
GetPrimitiveFactory(const MKLDNNDeviceContext& dev_ctx,
                    const std::string& key) {
  auto prim_creator =
      std::static_pointer_cast<FCPrimitiveFactory<T_in, T_w, T_out>>(
          dev_ctx.GetBlob(key));
  if (prim_creator == nullptr) {
    prim_creator = std::make_shared<FCPrimitiveFactory<T_in, T_w, T_out>>(
        dev_ctx.GetEngine());
    dev_ctx.SetBlob(key, prim_creator);
  }

  return prim_creator;
}

// Choose appropriate primitive factory implementation based on inferred
// output type (uint8, int8 or float).
template <typename T_in, typename T_w>
static void ExecuteFc(const ExecutionContext& ctx,
                      const LoDTensor* input,
                      const Tensor* w,
                      const Tensor* bias,
                      LoDTensor* output,
                      bool fuse_relu,
                      bool force_fp32_output) {
  auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
  std::string prim_key = platform::CreateKey(dev_ctx,
                                             input->format(),
                                             input->dims()[0],
                                             phi::vectorize<int>(w->dims()),
                                             ctx.OutputName("Out"));
  prim_key = platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, prim_key);

  constexpr bool is_int8 =
      std::is_same<T_in, int8_t>::value || std::is_same<T_in, uint8_t>::value;
  bool is_bfloat16 = std::is_same<T_in, paddle::platform::bfloat16>::value;
  if ((!is_int8 && !is_bfloat16) || force_fp32_output) {
    GetPrimitiveFactory<T_in, T_w, float>(dev_ctx, prim_key)
        ->ExecuteFcPrimitive(input, w, bias, output, dev_ctx, ctx);
  } else if (is_bfloat16) {
    GetPrimitiveFactory<T_in, T_w, platform::bfloat16>(dev_ctx, prim_key)
        ->ExecuteFcPrimitive(input, w, bias, output, dev_ctx, ctx);
  } else if (fuse_relu) {
    GetPrimitiveFactory<T_in, T_w, uint8_t>(dev_ctx, prim_key)
        ->ExecuteFcPrimitive(input, w, bias, output, dev_ctx, ctx);
  } else {
    GetPrimitiveFactory<T_in, T_w, int8_t>(dev_ctx, prim_key)
        ->ExecuteFcPrimitive(input, w, bias, output, dev_ctx, ctx);
  }
}

template <typename T_in, typename T_w>
class FCMKLDNNOpKernel : public framework::OpKernel<T_in> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()),
        true,
        platform::errors::PreconditionNotMet("FC MKL-DNN must use CPUPlace."));
    platform::MKLDNNDeviceContext::tls().log_lib_version();
    auto input = ctx.Input<LoDTensor>("Input");
    auto w = ctx.Input<Tensor>("W");
    auto bias = ctx.Input<Tensor>("Bias");
    auto output = ctx.Output<LoDTensor>("Out");

    bool fuse_relu = ctx.Attr<std::string>("activation_type") == "relu";
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

    ExecuteFc<T_in, T_w>(
        ctx, input, w, bias, output, fuse_relu, force_fp32_output);

    output->set_layout(DataLayout::kMKLDNN);
  }
};
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
}  // namespace operators
}  // namespace paddle

// Weights of FC are by default stored using fp32, template argument of weight
// data type implies their destination data type. (What's eventually going to
// be used during computations of kernel).
namespace ops = paddle::operators;
<<<<<<< HEAD
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    FP32,
                                    ops::kFCMKLDNNFP32,
                                    ops::FCMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(
    fc,
    MKLDNN,
    ::paddle::platform::CPUPlace,
    BF16,
    ops::kFCMKLDNNFP32,
    ops::FCMKLDNNOpKernel<paddle::platform::bfloat16,
                          paddle::platform::bfloat16>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    U8,
                                    ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<uint8_t, int8_t>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    S8,
                                    ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<int8_t, int8_t>);
=======

REGISTER_OP_KERNEL(fc,
                   MKLDNN,
                   ::paddle::platform::CPUPlace,
                   ops::FCMKLDNNKernel<float>,
                   ops::FCMKLDNNKernel<paddle::platform::bfloat16>,
                   ops::FCMKLDNNKernel<uint8_t>,
                   ops::FCMKLDNNKernel<int8_t>);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
