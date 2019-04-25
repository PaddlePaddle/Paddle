/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.*/

#include <mkldnn/include/mkldnn_types.h>
#include <memory>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
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

std::string CreateKey(const paddle::framework::ExecutionContext& ctx,
                      const memory::dims& input_dims,
                      const memory::dims& weights_dims,
                      const memory::data_type& src_dt, const bool fuse_relu,
                      const bool force_fp32_output, const std::string& suffix) {
  std::string key;
  key.reserve(platform::MKLDNNHandler::MaxKeyLength);
  platform::MKLDNNHandler::AppendKeyDims(&key, input_dims);
  platform::MKLDNNHandler::AppendKeyDims(&key, weights_dims);
  platform::MKLDNNHandler::AppendKey(&key, std::to_string(src_dt));
  platform::MKLDNNHandler::AppendKey(&key, std::to_string(fuse_relu));
  platform::MKLDNNHandler::AppendKey(&key, std::to_string(force_fp32_output));
  platform::MKLDNNHandler::AppendKey(&key, suffix);
  return key;
}

mkldnn::memory Reorder(const memory& src_p,
                       const memory::primitive_desc& dst_desc,
                       std::vector<mkldnn::primitive>* pipeline,
                       bool is_int8 = false, int mask = 0,
                       const std::vector<float>& scale_data = {1.0f}) {
  auto dst_mem = memory(dst_desc);
  std::shared_ptr<mkldnn::reorder> reorder_p;
  if (is_int8) {
    mkldnn::primitive_attr attri;
    attri.set_output_scales(mask, scale_data);
    auto reorder_pd = std::shared_ptr<mkldnn::reorder::primitive_desc>(
        new mkldnn::reorder::primitive_desc(src_p.get_primitive_desc(),
                                            dst_desc, attri));
    reorder_p = std::make_shared<mkldnn::reorder>(*reorder_pd, src_p, dst_mem);
  } else {
    reorder_p = std::make_shared<mkldnn::reorder>(src_p, dst_mem);
  }
  pipeline->push_back(*reorder_p);
  return dst_mem;
}

std::vector<int> GetCorrectedWeightsDims(const Tensor* weights) {
  std::vector<int> weights_dims =
      paddle::framework::vectorize2int(weights->dims());

  std::swap(weights_dims[0], weights_dims[1]);
  return weights_dims;
}

mkldnn::primitive_attr CreatePostOps(
    bool fuse_relu, const std::vector<float> output_shift_scale, int mask) {
  mkldnn::primitive_attr fc_attr;
  mkldnn::post_ops post_operations;

  fc_attr.set_output_scales(mask, output_shift_scale);

  if (fuse_relu) {
    constexpr float scale = 1.0f;
    constexpr float negative_slope = 0.0f;
    constexpr float placeholder = 1.0f;  // beta
    post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                   negative_slope, placeholder);
  }

  fc_attr.set_post_ops(post_operations);
  return fc_attr;
}

static mkldnn::memory::desc CreateMemDescriptor(const std::vector<int>& dims,
                                                mkldnn::memory::data_type dt,
                                                memory::format format) {
  return platform::MKLDNNMemDesc(dims, dt, format);
}

static mkldnn::memory::desc CreateMemDescriptor(const Tensor* tensor,
                                                mkldnn::memory::data_type dt,
                                                memory::format format) {
  auto dims = framework::vectorize2int(tensor->dims());
  return CreateMemDescriptor(dims, dt, format);
}

template <typename T, typename K>
class FCMKLDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    bool is_INT8 =
        std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
    if (!is_INT8) {
      ComputeFP32(ctx);
    } else {
      ComputeINT8(ctx);
    }
  }

  void ComputeINT8(const paddle::framework::ExecutionContext& ctx) const {
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_cpu_place(place), "It must use CPUPlace.");
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& engine = dev_ctx.GetEngine();

    auto input = ctx.Input<framework::LoDTensor>("Input");
    auto weights = ctx.Input<framework::LoDTensor>("W");
    auto bias = ctx.Input<framework::LoDTensor>("Bias");
    auto output = ctx.Output<framework::LoDTensor>("Out");

    int in_num_col_dims = ctx.Attr<int>("in_num_col_dims");
    std::vector<int64_t> output_dims;
    FCOutputSize(input->dims(), weights->dims(), output_dims, in_num_col_dims);
    output->Resize(framework::make_ddim(output_dims));
    output->set_lod(input->lod());

    bool fuse_relu = ctx.Attr<bool>("fuse_relu");
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

    std::string data_format = ctx.Attr<std::string>("data_format");

    bool is_int8 =
        std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
    auto src_dt = platform::MKLDNNGetDataType<T>();
    auto weights_dt = platform::MKLDNNGetDataType<K>();
    auto dst_dt = fuse_relu ? paddle::framework::ToMKLDNNDataType(
                                  framework::DataTypeTrait<uint8_t>::DataType)
                            : paddle::framework::ToMKLDNNDataType(
                                  framework::DataTypeTrait<int8_t>::DataType);
    if (force_fp32_output) {
      dst_dt = paddle::framework::ToMKLDNNDataType(
          framework::DataTypeTrait<float>::DataType);
    }

    auto input_data = input->data<T>();

    auto input_dims = framework::vectorize2int(input->dims());
    std::vector<int> weights_dims =
        paddle::framework::vectorize2int(weights->dims());

    std::string key = CreateKey(ctx, input_dims, weights_dims, src_dt,
                                fuse_relu, force_fp32_output,
                                ctx.op().Input("Input") + ctx.op().Input("W"));
    auto prim_key = key + "@fc_p";
    auto src_key = key + "@src_p";
    auto dst_key = key + "@dst_p";
    auto weights_key = key + "@weights_p";
    auto bias_key = key + "@bias_p";

    std::shared_ptr<inner_product_forward> fc_p;
    std::shared_ptr<memory> dst_p;
    std::shared_ptr<memory> user_weights_p;
    std::shared_ptr<memory> weights_p;
    std::shared_ptr<memory> user_bias_p;
    std::shared_ptr<memory> bias_p;

    std::vector<mkldnn::primitive> pipeline;

    fc_p = std::static_pointer_cast<inner_product_forward>(
        dev_ctx.GetBlob(prim_key));

    if (fc_p == nullptr) {
      auto scale_in_data = ctx.Attr<float>("Scale_in");
      auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
      auto scale_out_data =
          force_fp32_output ? 1.0f : ctx.Attr<float>("Scale_out");

      bool is_multi_channel = scale_weights_data.size() > 1;
      int count = is_multi_channel ? scale_weights_data.size() : 1;
      std::vector<float> output_shift_scale(count);
#pragma omp parallel for if (count > 1)
      for (int i = 0; i < count; i++) {
        if (scale_weights_data[i] == 0.0)
          output_shift_scale[i] = scale_out_data;
        else
          output_shift_scale[i] =
              scale_out_data / (scale_in_data * scale_weights_data[i]);
      }

      int fc_mask = is_multi_channel ? 1 << 1 : 0;
      mkldnn::primitive_attr fc_attr =
          CreatePostOps(fuse_relu, output_shift_scale, fc_mask);

      auto chosen_memory_format =
          platform::data_format_to_memory_format(data_format);
      auto src_desc = CreateMemDescriptor(input, src_dt, chosen_memory_format);

      if (src_desc.data.ndims == 4) {
        weights_dims = {weights_dims[1], input_dims[1], input_dims[2],
                        input_dims[3]};
      } else {
        std::swap(weights_dims[0], weights_dims[1]);
      }
      auto weights_desc = CreateMemDescriptor(
          weights_dims, memory::data_type::s8, chosen_memory_format);

      auto dst_desc = CreateMemDescriptor(output, dst_dt, chosen_memory_format);

      std::shared_ptr<mkldnn::inner_product_forward::desc> fc_desc;
      std::vector<float> scale_bias_data(count);

      if (bias) {
#pragma omp parallel for if (count > 1)
        for (int i = 0; i < count; i++) {
          if (scale_weights_data[i] == 0) {
            scale_bias_data[i] = 1.0f;
          } else {
            scale_bias_data[i] = scale_in_data * scale_weights_data[i];
          }
        }

        auto bias_desc =
            CreateMemDescriptor(bias, memory::data_type::s32,
                                memory::format::x);  // FIX ME BY ANY FMT

        fc_desc.reset(new inner_product_forward::desc(
            prop_kind::forward_scoring, src_desc, weights_desc, bias_desc,
            dst_desc));
      } else {
        fc_desc.reset(new inner_product_forward::desc(
            prop_kind::forward, src_desc, weights_desc, dst_desc));
      }

      auto fc_pd = mkldnn::inner_product_forward::primitive_desc(
          *fc_desc, fc_attr, engine);

      auto src_pd = fc_pd.src_primitive_desc();
      auto src_p =
          std::make_shared<memory>(src_pd, to_void_cast<T>(input_data));

      auto weights_data = weights->data<K>();
      auto user_weights_desc = CreateMemDescriptor(
          weights_dims, weights_dt,
          src_desc.data.ndims == 4 ? memory::format::oihw : memory::format::oi);
      auto user_weights_pd = memory::primitive_desc(user_weights_desc, engine);
      user_weights_p = std::make_shared<memory>(user_weights_pd,
                                                to_void_cast<K>(weights_data));
      auto weights_pd = fc_pd.weights_primitive_desc();
      int weights_reorder_mask = is_multi_channel ? 1 << 0 : 0;
      weights_p = std::make_shared<memory>(
          Reorder(*user_weights_p, weights_pd, &pipeline, is_int8,
                  weights_reorder_mask, scale_weights_data));

      dst_p = std::make_shared<memory>(CreateDstMemory(
          fc_pd, ctx, output, fuse_relu, force_fp32_output, is_int8));

      if (bias) {
        auto bias_data = bias->data<K>();
        auto bias_dt = platform::MKLDNNGetDataType<K>();
        auto user_bias_desc =
            CreateMemDescriptor(bias, bias_dt, memory::format::x);
        auto user_bias_pd = memory::primitive_desc(user_bias_desc, engine);
        user_bias_p =
            std::make_shared<memory>(user_bias_pd, to_void_cast<K>(bias_data));
        auto bias_pd = fc_pd.bias_primitive_desc();
        int bias_reorder_mask = is_multi_channel ? 1 << 0 : 0;
        bias_p = std::make_shared<memory>(
            Reorder(*user_bias_p, bias_pd, &pipeline, is_int8,
                    bias_reorder_mask, scale_bias_data));
        fc_p = std::make_shared<inner_product_forward>(
            fc_pd, *src_p, *weights_p, *bias_p, *dst_p);
      } else {
        fc_p = std::make_shared<inner_product_forward>(fc_pd, *src_p,
                                                       *weights_p, *dst_p);
      }
      dev_ctx.SetBlob(src_key, src_p);
      dev_ctx.SetBlob(dst_key, dst_p);
      dev_ctx.SetBlob(weights_key, weights_p);
      dev_ctx.SetBlob(bias_key, bias_p);
      dev_ctx.SetBlob(prim_key, fc_p);
    } else {
      auto src_p = std::static_pointer_cast<memory>(dev_ctx.GetBlob(src_key));
      src_p->set_data_handle(to_void_cast<T>(input_data));

      dst_p = std::static_pointer_cast<memory>(dev_ctx.GetBlob(dst_key));
      if (force_fp32_output) {
        dst_p->set_data_handle(output->mutable_data<float>(place));
      } else if (fuse_relu) {
        dst_p->set_data_handle(output->mutable_data<uint8_t>(place));
      } else {
        dst_p->set_data_handle(output->mutable_data<int8_t>(place));
      }
    }
    pipeline.push_back(*fc_p);

    stream(stream::kind::eager).submit(pipeline).wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(*dst_p));
  }

  void ComputeFP32(const paddle::framework::ExecutionContext& ctx) const {
    // Fake.
  }

 private:
  mkldnn::memory CreateDstMemory(
      const mkldnn::inner_product_forward::primitive_desc& fc_prim_desc,
      const ExecutionContext& ctx, Tensor* output, bool fuse_relu = false,
      bool force_fp32_output = false, bool is_int8 = false) const {
    auto dst_pd = fc_prim_desc.dst_primitive_desc();
    auto buffer_size = dst_pd.get_size();
    if (is_int8) {
      if (force_fp32_output) {
        float* output_data = output->mutable_data<float>(
            ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, buffer_size);
        return memory(dst_pd, to_void_cast<float>(output_data));
      } else if (fuse_relu) {
        uint8_t* output_data = output->mutable_data<uint8_t>(
            ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, buffer_size);
        return memory(dst_pd, to_void_cast<uint8_t>(output_data));
      } else {
        T* output_data = output->mutable_data<T>(
            ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, buffer_size);
        return memory(dst_pd, to_void_cast<T>(output_data));
      }
    }
    T* output_data = output->mutable_data<T>(
        ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, buffer_size);
    return memory(dst_pd, to_void_cast<T>(output_data));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    FP32, ops::kFCMKLDNNFP32,
                                    ops::FCMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    U8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<uint8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    S8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<int8_t, float>);
