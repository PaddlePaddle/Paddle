// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include "paddle/common/errors.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/expect.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi::fusion {

using phi::OneDNNContext;
using phi::funcs::OneDNNGetDataType;
using phi::funcs::to_void_cast;

struct InnerProductCache {
  dnnl::inner_product_forward inner_product_p;
  dnnl::memory src_mem;
  dnnl::memory weights_mem;
  dnnl::memory bias_mem;
  dnnl::memory dst_mem;
  dnnl::memory src_scales_mem;
  dnnl::memory wei_scales_mem;
  dnnl::memory dst_scales_mem;
};

std::tuple<std::vector<float>,
           std::vector<float>,
           std::vector<float>,
           std::vector<float>>
GetDNNLScales(const float scale_in,
              const float scale_out,
              const std::vector<float>& scale_weights) {
  std::vector<float> dnnl_src_scales = {1.f / scale_in};
  size_t count = scale_weights.size();
  std::vector<float> dnnl_wei_scales(count);
#pragma omp parallel for if (count > 50)
  for (size_t i = 0; i < count; i++) {
    dnnl_wei_scales[i] = 1.f / scale_weights[i];
  }
  std::vector<float> dnnl_psum_scales = {1.f};
  std::vector<float> dnnl_dst_scales = {1.f / scale_out};

  return std::make_tuple(
      dnnl_src_scales, dnnl_wei_scales, dnnl_psum_scales, dnnl_dst_scales);
}

template <typename T_in, typename T_w, typename T_out>
class FCOneDNNHandler
    : public phi::funcs::OneDNNHandlerNoCachingT<T_in,
                                                 dnnl::inner_product_forward> {
 public:
  FCOneDNNHandler(const OneDNNContext& dev_ctx,
                  const phi::DenseTensor* x,
                  const phi::DenseTensor* weights,
                  const phi::DenseTensor* bias,
                  phi::DenseTensor* out UNUSED,
                  const float scale_in,
                  const float scale_out,
                  const std::vector<float>& scale_weights,
                  const bool force_fp32_output,
                  const int in_num_col_dims,
                  const std::string& activation_type,
                  dnnl::engine onednn_engine,
                  phi::Place cpu_place)
      : phi::funcs::OneDNNHandlerNoCachingT<T_in, dnnl::inner_product_forward>(
            onednn_engine, cpu_place),
        dev_ctx_(dev_ctx) {
    this->memory_key_ = dev_ctx.GetInputsName("W")[0];

    auto x_vec_dims = common::vectorize(x->dims());
    auto weights_vec_dims = common::vectorize(weights->dims());

    int MB = 1;
    for (int i = 0; i < in_num_col_dims; ++i) {
      MB *= x_vec_dims[i];
    }

    int IC = 1;
    for (size_t i = in_num_col_dims; i < x_vec_dims.size(); ++i) {
      IC *= x_vec_dims[i];
    }

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

    const auto attrs = CreateFCAttrs(dev_ctx,
                                     scale_in,
                                     scale_out,
                                     scale_weights,
                                     force_fp32_output,
                                     activation_type);

    this->AcquireForwardPrimitiveDescriptor(attrs,
                                            dnnl::prop_kind::forward_inference,
                                            src_md,
                                            weights_md,
                                            bias_md,
                                            dst_md);
  }

 private:
  dnnl::primitive_attr CreateFCAttrs(const OneDNNContext& dev_ctx,
                                     const float scale_in,
                                     const float scale_out,
                                     const std::vector<float>& scale_weights,
                                     const bool force_fp32_output,
                                     const std::string& activation_type) {
    dnnl::primitive_attr attributes;
    dnnl::post_ops post_operations;

    float activation_scale = 1.0f;
    if (phi::funcs::is_int8<T_w>()) {
      std::vector<float> src_scales, wei_scales, psum_scales, dst_scales;
      std::tie(src_scales, wei_scales, psum_scales, dst_scales) =
          GetDNNLScales(scale_in, scale_out, scale_weights);

      attributes.set_scales_mask(DNNL_ARG_SRC, 0);

      dnnl::memory::desc src_scales_md(
          {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
      src_scales_mem_ = dnnl::memory(src_scales_md, this->engine_);
      memcpy(src_scales_mem_.get_data_handle(),
             src_scales.data(),
             src_scales.size() * sizeof(float));

      int mask = wei_scales.size() > 1 ? 1 : 0;
      attributes.set_scales_mask(DNNL_ARG_WEIGHTS, mask);

      dnnl::memory::desc wei_scales_md(
          {static_cast<int64_t>(wei_scales.size())},
          dnnl::memory::data_type::f32,
          dnnl::memory::format_tag::x);
      wei_scales_mem_ = dnnl::memory(wei_scales_md, this->engine_);
      memcpy(wei_scales_mem_.get_data_handle(),
             wei_scales.data(),
             wei_scales.size() * sizeof(float));

      if (!force_fp32_output) {
        attributes.set_scales_mask(DNNL_ARG_DST, 0);

        dnnl::memory::desc dst_scales_md(
            {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
        dst_scales_mem_ = dnnl::memory(dst_scales_md, this->engine_);
        memcpy(dst_scales_mem_.get_data_handle(),
               dst_scales.data(),
               dst_scales.size() * sizeof(float));
      }
    }

    // ReLU from "fc_fuse_pass"
    if (activation_type == "relu") {
      post_operations.append_eltwise(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    }
    AppendActivation(dev_ctx, post_operations, activation_scale);

    if (dev_ctx.HasDnnAttr("fused_output_scale")) {
      float scale_alpha =
          PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("fused_output_scale"));
      post_operations.append_eltwise(
          dnnl::algorithm::eltwise_linear, scale_alpha, 0.0f);
    }

    attributes.set_post_ops(post_operations);
    return attributes;
  }

  void AppendActivation(const OneDNNContext& dev_ctx,
                        dnnl::post_ops& post_ops,  // NOLINT
                        float activation_scale = 1.0f) {
    const auto invalid_attribute =
        dev_ctx.HasDnnAttr("fuse_activation")
            ? PADDLE_GET_CONST(std::string,
                               dev_ctx.GetDnnAttr("fuse_activation"))
                  .empty()
            : true;
    if (invalid_attribute) return;
    const auto fuse_activation =
        PADDLE_GET_CONST(std::string, dev_ctx.GetDnnAttr("fuse_activation"));
    const auto fuse_alpha =
        dev_ctx.HasDnnAttr("fuse_alpha")
            ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("fuse_alpha"))
            : 0.0f;
    const auto fuse_beta =
        dev_ctx.HasDnnAttr("fuse_beta")
            ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("fuse_beta"))
            : 0.0f;
    const auto activation_map = phi::funcs::OneDNNActivationMap();
    const auto& activation_type = activation_map.find(fuse_activation);

    PADDLE_ENFORCE_NE(
        activation_type,
        activation_map.end(),
        common::errors::InvalidArgument(
            "Activation '%s' not found in oneDNN algorithms mapper",
            fuse_activation));

    post_ops.append_eltwise(activation_type->second, fuse_alpha, fuse_beta);
    post_ops.append_eltwise(
        dnnl::algorithm::eltwise_linear, activation_scale, 0.0f);
  }

  // Computing oneDNN's scaling mask which determines along which dimension
  // slice should the scaling be applied.
  int CreateMask(int slice_dimension, bool is_multi_channel_quantizied) {
    return is_multi_channel_quantizied ? 1 << slice_dimension : 0;
  }

  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorderAndAttrs(
      const dnnl::memory::desc& user_md,
      const dnnl::memory::desc& target_md,
      void* ptr,
      const dnnl::primitive_attr& attrs,
      const std::vector<float>& scale_data) {
    std::shared_ptr<dnnl::memory> target_memory_p;

    auto user_memory_p =
        std::make_shared<dnnl::memory>(user_md, this->engine_, ptr);
    target_memory_p = std::make_shared<dnnl::memory>(target_md, this->engine_);
    auto reorder_p = std::make_shared<dnnl::reorder>(
        *user_memory_p, *target_memory_p, attrs);

    auto scales_md =
        dnnl::memory::desc({static_cast<int64_t>(scale_data.size())},
                           dnnl::memory::data_type::f32,
                           dnnl::memory::format_tag::x);
    auto scale_mem =
        dnnl::memory(scales_md,
                     this->engine_,
                     phi::funcs::to_void_cast<float>(scale_data.data()));

    auto& astream = OneDNNContext::tls().get_stream();
    {
      reorder_p->execute(astream,
                         {{DNNL_ARG_FROM, *user_memory_p},
                          {DNNL_ARG_TO, *target_memory_p},
                          {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, scale_mem}});
      astream.wait();
    }

    return target_memory_p;
  }

  std::string memory_key_;
  const OneDNNContext& dev_ctx_;
  dnnl::memory src_scales_mem_;
  dnnl::memory wei_scales_mem_;
  dnnl::memory dst_scales_mem_;

 public:
  std::shared_ptr<dnnl::memory> AcquireSrcMemoryWithReorder(
      const phi::DenseTensor* x) {
    const T_in* x_data = x->data<T_in>();

    auto user_md = x->mem_desc();
    if (x->dims().size() != 2) {
      // reshape restrictions are always satisfied because in case of 3 or 4 dim
      // input, plain layout is enforced
      user_md = user_md.reshape(this->fwd_pd_->src_desc().get_dims());
    }

    return this->AcquireMemoryWithReorder(
        user_md, this->fwd_pd_->src_desc(), to_void_cast<T_in>(x_data));
  }

  std::shared_ptr<dnnl::memory> AcquireBiasMemoryWithReorder(
      const phi::DenseTensor* bias) {
    const float* bias_data = bias->data<float>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->bias_desc(),
                                            to_void_cast<float>(bias_data));
  }

  std::shared_ptr<dnnl::memory> AcquireWeightsMemoryWithReorder(
      const phi::DenseTensor* weights, const std::vector<float>& scale_data) {
    const std::string weights_base_key = this->memory_key_ + "@weights";
    std::string weights_key;
    weights_key.reserve(128);
    weights_key = phi::funcs::ExtendKeyWithThreadInfoIfNeeded(
        dev_ctx_,
        phi::funcs::CreateKey(
            dev_ctx_, weights_base_key, this->fwd_pd_->weights_desc()));
    auto memory_p = std::static_pointer_cast<dnnl::memory>(
        this->dev_ctx_.GetBlob(weights_key));

    if (!memory_p) {
      const float* weights_data = weights->data<float>();
      auto weights_dims = this->fwd_pd_->weights_desc().get_dims();

      auto user_md = dnnl::memory::desc(weights_dims,
                                        OneDNNGetDataType<float>(),
                                        dnnl::memory::format_tag::io);

      if (phi::funcs::is_int8<T_w>()) {
        dnnl::primitive_attr attrs;
        int mask = CreateMask(0, scale_data.size() > 1);
        attrs.set_scales_mask(DNNL_ARG_SRC, mask);

        memory_p = this->AcquireMemoryWithReorderAndAttrs(
            user_md,
            this->fwd_pd_->weights_desc(),
            to_void_cast<float>(weights_data),
            attrs,
            scale_data);
      } else {
        memory_p =
            this->AcquireMemoryWithReorder(user_md,
                                           this->fwd_pd_->weights_desc(),
                                           to_void_cast<float>(weights_data));
      }

      this->dev_ctx_.SetBlob(weights_key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireCustomDstMemory(phi::DenseTensor* out) {
    return this->template AcquireDstMemory<T_out>(out);
  }  // namespace operators

  void SetScalesIfNeeded(std::unordered_map<int, dnnl::memory>* args) {
    if (src_scales_mem_.get_desc().is_zero() != true) {
      args->insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_mem_});
      args->insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_mem_});
    }
    // dst scales may be empty when force fp32 output
    if (dst_scales_mem_.get(true)) {
      args->insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales_mem_});
    }
  }
};

#define IF_CHANGE_FC_TW_TYPENAME(condition, ...) \
  if (condition) {                               \
    using T_w = int8_t;                          \
    __VA_ARGS__();                               \
  } else {                                       \
    using T_w = T;                               \
    __VA_ARGS__();                               \
  }

void RecomputeOutputDims(const int in_num_col_dims,
                         const bool padding_weights,
                         const phi::DenseTensor* x,
                         const phi::DenseTensor* weights,
                         phi::DenseTensor* out) {
  PADDLE_ENFORCE_EQ(padding_weights,
                    false,
                    common::errors::PermissionDenied(
                        "Weight padding in fc can not be used in oneDNN."));
  std::vector<int64_t> output_dims;
  phi::funcs::FCOutputSize(x->dims(),
                           weights->dims(),
                           output_dims,
                           in_num_col_dims,
                           padding_weights);
  out->Resize(common::make_ddim(output_dims));
  out->set_lod(x->lod());
}

template <typename T>
void PrepareSrcMem(const std::shared_ptr<dnnl::inner_product_forward>& fc_p
                       UNUSED,
                   const std::shared_ptr<dnnl::memory>& src_mem,
                   const phi::DenseTensor* x,
                   const dnnl::engine& engine) {
  auto x_md = x->mem_desc().reshape(src_mem->get_desc().get_dims());
  if (x_md != src_mem->get_desc()) {
    dnnl::memory x_mem(x_md, engine, to_void_cast<T>(x->data<T>()));
    auto reorder_p = dnnl::reorder(x_mem, *src_mem);

    auto& astream = OneDNNContext::tls().get_stream();
    reorder_p.execute(astream, x_mem, *src_mem);
    astream.wait();
  } else {
    src_mem->set_data_handle(to_void_cast<T>(x->data<T>()));
  }
}

template <typename T, typename T_out, typename T_w>
void RunKernel(const phi::OneDNNContext& dev_ctx,
               const DenseTensor& input,
               const DenseTensor& w,
               const paddle::optional<DenseTensor>& bias,
               const int in_num_col_dims,
               const std::string& activation_type,
               const bool use_mkldnn,
               const bool padding_weights,
               const bool use_quantizer,
               const std::string& mkldnn_data_type,
               const float scale_in,
               const std::vector<float>& scale_weights,
               const float scale_out,
               const bool force_fp32_output,
               DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  std::shared_ptr<dnnl::inner_product_forward> fc_p;
  std::shared_ptr<dnnl::memory> src_memory_p;
  std::shared_ptr<dnnl::memory> weights_memory_p;
  std::shared_ptr<dnnl::memory> bias_memory_p;
  std::shared_ptr<dnnl::memory> dst_memory_p;

  std::string cache_key;
  cache_key.reserve(64);
  cache_key = phi::funcs::ExtendKeyWithThreadInfoIfNeeded(
      dev_ctx,
      phi::funcs::CreateKey(dev_ctx,
                            dev_ctx.GetInputsName("Input")[0],
                            dev_ctx.GetInputsName("W")[0],
                            common::vectorize(input.dims()),
                            common::vectorize(w.dims())));

  auto inner_product_cache =
      std::static_pointer_cast<InnerProductCache>(dev_ctx.GetBlob(cache_key));

  RecomputeOutputDims(in_num_col_dims, padding_weights, &input, &w, out);

  std::unordered_map<int, dnnl::memory> fc_args;

  if (inner_product_cache) {
    fc_p = std::make_shared<dnnl::inner_product_forward>(
        inner_product_cache->inner_product_p);
    src_memory_p = std::make_shared<dnnl::memory>(inner_product_cache->src_mem);
    PrepareSrcMem<T>(fc_p, src_memory_p, &input, onednn_engine);

    weights_memory_p =
        std::make_shared<dnnl::memory>(inner_product_cache->weights_mem);

    dst_memory_p = std::make_shared<dnnl::memory>(inner_product_cache->dst_mem);

    auto out_ptr =
        dev_ctx.template Alloc<T_out>(out, dst_memory_p->get_desc().get_size());

    dst_memory_p->set_data_handle(out_ptr);

    fc_args.insert({DNNL_ARG_SRC, *src_memory_p});
    fc_args.insert({DNNL_ARG_WEIGHTS, *weights_memory_p});
    fc_args.insert({DNNL_ARG_DST, *dst_memory_p});

    if (bias) {
      bias_memory_p =
          std::make_shared<dnnl::memory>(inner_product_cache->bias_mem);
      fc_args.insert({DNNL_ARG_BIAS, *bias_memory_p});
    }

    if (inner_product_cache->src_scales_mem.get(true)) {
      fc_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                      inner_product_cache->src_scales_mem});
      fc_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                      inner_product_cache->wei_scales_mem});
    }
    if (inner_product_cache->dst_scales_mem.get(true)) {
      fc_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                      inner_product_cache->dst_scales_mem});
    }
  } else {
    // here
    FCOneDNNHandler<T, T_w, T_out> handler(dev_ctx,
                                           &input,
                                           &w,
                                           bias.get_ptr(),
                                           out,
                                           scale_in,
                                           scale_out,
                                           scale_weights,
                                           force_fp32_output,
                                           in_num_col_dims,
                                           activation_type,
                                           onednn_engine,
                                           dev_ctx.GetPlace());

    src_memory_p = handler.AcquireSrcMemoryWithReorder(&input);
    weights_memory_p =
        handler.AcquireWeightsMemoryWithReorder(&w, scale_weights);
    dst_memory_p = handler.AcquireCustomDstMemory(out);
    fc_args.insert({DNNL_ARG_SRC, *src_memory_p});
    fc_args.insert({DNNL_ARG_WEIGHTS, *weights_memory_p});
    fc_args.insert({DNNL_ARG_DST, *dst_memory_p});

    if (bias) {
      bias_memory_p = handler.AcquireBiasMemoryWithReorder(bias.get_ptr());
      fc_args.insert({DNNL_ARG_BIAS, *bias_memory_p});
    }

    if (phi::funcs::is_int8<T>()) {
      handler.SetScalesIfNeeded(&fc_args);
    }

    fc_p = handler.AcquireForwardPrimitive();
  }

  auto& astream = OneDNNContext::tls().get_stream();
  fc_p->execute(astream, fc_args);
  astream.wait();

  if (!inner_product_cache) {
    auto ip_cache = std::make_shared<InnerProductCache>();
    ip_cache->inner_product_p = *fc_p;
    ip_cache->src_mem = *src_memory_p;
    ip_cache->weights_mem = *weights_memory_p;
    ip_cache->dst_mem = *dst_memory_p;
    if (bias) {
      ip_cache->bias_mem = *bias_memory_p;
    }
    if (fc_args.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC)) {
      ip_cache->src_scales_mem =
          fc_args.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
      ip_cache->wei_scales_mem =
          fc_args.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    }

    if (fc_args.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST)) {
      ip_cache->dst_scales_mem =
          fc_args.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    }

    dev_ctx.SetBlob(cache_key, ip_cache);
  }

  const auto out_md =
      dst_memory_p->get_desc().reshape(common::vectorize(out->dims()));

  std::vector<int> reshape2_shape = {};
  if (dev_ctx.HasDnnAttr("fused_reshape2_shape")) {
    reshape2_shape = PADDLE_GET_CONST(
        std::vector<int>, dev_ctx.GetDnnAttr("fused_reshape2_shape"));
  }
  if (!reshape2_shape.empty()) {
    phi::funcs::SetOutMemDescWithReshape2FuseSupport(
        reshape2_shape, out, out_md);
  } else {
    out->set_mem_desc(out_md);
  }
}

template <typename T, typename Context>
void FCKernel(const Context& dev_ctx,
              const DenseTensor& input,
              const DenseTensor& w,
              const paddle::optional<DenseTensor>& bias,
              const int in_num_col_dims,
              const std::string& activation_type,
              const bool padding_weights,
              DenseTensor* out) {
  const bool use_mkldnn =
      dev_ctx.HasDnnAttr("use_mkldnn")
          ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("use_mkldnn"))
          : false;
  const bool use_quantizer =
      dev_ctx.HasDnnAttr("use_quantizer")
          ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("use_quantizer"))
          : false;
  const std::string mkldnn_data_type =
      dev_ctx.HasDnnAttr("mkldnn_data_type")
          ? PADDLE_GET_CONST(std::string,
                             dev_ctx.GetDnnAttr("mkldnn_data_type"))
          : "float32";
  const float scale_in =
      dev_ctx.HasDnnAttr("Scale_in")
          ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Scale_in"))
          : 1.0f;
  std::vector<float> tmp_scale_weights = {1.0f};
  const std::vector<float> scale_weights =
      dev_ctx.HasDnnAttr("Scale_weights")
          ? PADDLE_GET_CONST(std::vector<float>,
                             dev_ctx.GetDnnAttr("Scale_weights"))
          : tmp_scale_weights;
  const float scale_out =
      dev_ctx.HasDnnAttr("Scale_out")
          ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Scale_out"))
          : 1.0f;
  const bool force_fp32_output =
      dev_ctx.HasDnnAttr("force_fp32_output")
          ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("force_fp32_output"))
          : false;
  std::vector<std::string> mkldnn_data_type_list = {
      "float32", "int8", "bfloat16"};
  PADDLE_ENFORCE_EQ(std::find(mkldnn_data_type_list.begin(),
                              mkldnn_data_type_list.end(),
                              mkldnn_data_type) != mkldnn_data_type_list.end(),
                    true,
                    common::errors::InvalidArgument(
                        "The mkldnn_data_type should be [float32, "
                        "int8, bfloat16], but found %s.",
                        mkldnn_data_type.c_str()));
  auto in_dims = input.dims();
  if (use_mkldnn) {
    PADDLE_ENFORCE_EQ(
        in_dims.size() >= 2 && in_dims.size() <= 4,
        true,
        common::errors::Unimplemented(
            "The Input of fc is expected to be a 2-D, 3-D or 4-D tensor when "
            "use_mkldnn is set. But received the number of Input's "
            "dimensions is %d, Input's shape is %s.",
            in_dims.size(),
            in_dims));
  }
  bool fuse_relu = activation_type == "relu";
  IF_CHANGE_FC_TW_TYPENAME((std::is_same<T, uint8_t>::value), ([&] {
                             if (force_fp32_output) {  // NOLINT
                               RunKernel<T, float, T_w>(dev_ctx,
                                                        input,
                                                        w,
                                                        bias,
                                                        in_num_col_dims,
                                                        activation_type,
                                                        use_mkldnn,
                                                        padding_weights,
                                                        use_quantizer,
                                                        mkldnn_data_type,
                                                        scale_in,
                                                        scale_weights,
                                                        scale_out,
                                                        force_fp32_output,
                                                        out);
                             } else if (phi::funcs::is_int8<T>()) {
                               if (fuse_relu) {
                                 RunKernel<T, uint8_t, T_w>(dev_ctx,
                                                            input,
                                                            w,
                                                            bias,
                                                            in_num_col_dims,
                                                            activation_type,
                                                            use_mkldnn,
                                                            padding_weights,
                                                            use_quantizer,
                                                            mkldnn_data_type,
                                                            scale_in,
                                                            scale_weights,
                                                            scale_out,
                                                            force_fp32_output,
                                                            out);
                               } else {
                                 RunKernel<T, int8_t, T_w>(dev_ctx,
                                                           input,
                                                           w,
                                                           bias,
                                                           in_num_col_dims,
                                                           activation_type,
                                                           use_mkldnn,
                                                           padding_weights,
                                                           use_quantizer,
                                                           mkldnn_data_type,
                                                           scale_in,
                                                           scale_weights,
                                                           scale_out,
                                                           force_fp32_output,
                                                           out);
                               }
                             } else {
                               RunKernel<T, T, T_w>(dev_ctx,
                                                    input,
                                                    w,
                                                    bias,
                                                    in_num_col_dims,
                                                    activation_type,
                                                    use_mkldnn,
                                                    padding_weights,
                                                    use_quantizer,
                                                    mkldnn_data_type,
                                                    scale_in,
                                                    scale_weights,
                                                    scale_out,
                                                    force_fp32_output,
                                                    out);
                             }
                           }));
}

}  // namespace phi::fusion

PD_REGISTER_KERNEL(fc,
                   OneDNN,
                   ONEDNN,
                   phi::fusion::FCKernel,
                   float,
                   phi::dtype::bfloat16,
                   uint8_t,
                   int8_t) {}
