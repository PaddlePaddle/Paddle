/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/pool_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"

namespace paddle {
namespace platform {

using framework::DataLayout;

using user_function = std::function<std::shared_ptr<float>(const float*)>;
using memory = dnnl::memory;

template <typename T,
          typename TForward,
          typename TBackward = mkldnn_dummy_primitive,
          typename TBackward_params = mkldnn_dummy_primitive>
using MKLDNNHandlerT =
    phi::funcs::OneDNNHandlerT<T, TForward, TBackward, TBackward_params>;

template <typename T,
          typename TForward,
          typename TBackward = mkldnn_dummy_primitive,
          typename TBackward_params = mkldnn_dummy_primitive>
using MKLDNNHandlerNoCachingT = phi::funcs::
    OneDNNHandlerNoCachingT<T, TForward, TBackward, TBackward_params>;

template <typename T>
using ReductionMKLDNNHandler = phi::funcs::ReductionOneDNNHandler<T>;

template <typename T>
using BroadcastDataMKLDNNHandler = phi::funcs::BroadcastDataOneDNNHandler<T>;

template <typename T>
using BinaryMKLDNNHandler = phi::funcs::BinaryOneDNNHandler<T>;

static void AppendActivation(const framework::ExecutionContext& ctx,
                             dnnl::post_ops& post_ops,  // NOLINT
                             float activation_scale = 1.0f) {
  const auto invalid_attribute =
      ctx.HasAttr("fuse_activation")
          ? ctx.Attr<std::string>("fuse_activation").empty()
          : true;
  if (invalid_attribute) return;

  const auto fuse_activation = ctx.Attr<std::string>("fuse_activation");
  const auto fuse_alpha =
      ctx.HasAttr("fuse_alpha") ? ctx.Attr<float>("fuse_alpha") : 0.0f;
  const auto fuse_beta =
      ctx.HasAttr("fuse_beta") ? ctx.Attr<float>("fuse_beta") : 0.0f;

  if (fuse_activation == "hard_sigmoid") {
    post_ops.append_eltwise(activation_scale,
                            dnnl::algorithm::eltwise_linear,
                            fuse_alpha,
                            fuse_beta);
    post_ops.append_eltwise(
        activation_scale, dnnl::algorithm::eltwise_clip, 0.0f, 1.0f);
  } else {
    const std::unordered_map<std::string, dnnl::algorithm> activation_map = {
        {"abs", dnnl::algorithm::eltwise_abs},
        {"clip", dnnl::algorithm::eltwise_clip},
        {"gelu", dnnl::algorithm::eltwise_gelu_erf},
        {"gelu_erf", dnnl::algorithm::eltwise_gelu_erf},
        {"gelu_tanh", dnnl::algorithm::eltwise_gelu_tanh},
        {"hard_swish", dnnl::algorithm::eltwise_hardswish},
        {"leaky_relu", dnnl::algorithm::eltwise_relu},
        {"mish", dnnl::algorithm::eltwise_mish},
        {"relu", dnnl::algorithm::eltwise_relu},
        {"relu6", dnnl::algorithm::eltwise_bounded_relu},
        {"sigmoid", dnnl::algorithm::eltwise_logistic},
        {"sqrt", dnnl::algorithm::eltwise_sqrt},
        {"swish", dnnl::algorithm::eltwise_swish},
        {"tanh", dnnl::algorithm::eltwise_tanh}};

    const auto& activation_type = activation_map.find(fuse_activation);

    PADDLE_ENFORCE_NE(
        activation_type,
        activation_map.end(),
        platform::errors::InvalidArgument(
            "Activation '%s' not found in oneDNN algorithms mapper",
            fuse_activation));

    post_ops.append_eltwise(
        activation_scale, activation_type->second, fuse_alpha, fuse_beta);
  }
}

template <typename T>
constexpr bool IsInt8() {
  return std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
}

template <typename T>
constexpr bool IsBfloat16() {
  return std::is_same<T, paddle::platform::bfloat16>::value;
}

template <typename XT, typename YT, typename OT>
class MatMulV2MKLDNNHandler
    : public paddle::platform::MKLDNNHandlerNoCachingT<XT, dnnl::matmul> {
 public:
  MatMulV2MKLDNNHandler(const framework::ExecutionContext& ctx,
                        const dnnl::engine engine,
                        paddle::platform::Place cpu_place,
                        const std::vector<int64_t>& x_org_dims,
                        bool trans_x,
                        const std::vector<int64_t>& y_org_dims,
                        bool trans_y,
                        bool is_output_fused,
                        const std::vector<int64_t>& x_strides_override,
                        const std::vector<int64_t>& y_strides_override)
      : paddle::platform::MKLDNNHandlerNoCachingT<XT, dnnl::matmul>(engine,
                                                                    cpu_place) {
    // M X K * K X N
    std::vector<int64_t> x_dims(x_org_dims);
    std::vector<int64_t> y_dims(y_org_dims);

    const int MB_idx = x_dims.size() - 3;
    const int H_idx = x_dims.size() - 2;
    const int W_idx = x_dims.size() - 1;

    if (trans_x) std::swap(x_dims[H_idx], x_dims[W_idx]);
    if (trans_y) std::swap(y_dims[H_idx], y_dims[W_idx]);

    const memory::dim M = x_dims[H_idx];
    const memory::dim K = x_dims[W_idx];
    const memory::dim N = y_dims[W_idx];

    std::vector<int64_t> x_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> y_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> out_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> out_ddims(x_dims.size() - 3, 1);

    x_strides.reserve(x_dims.size());
    y_strides.reserve(x_dims.size());
    out_strides.reserve(x_dims.size());

    if (!x_strides_override.empty()) {
      x_strides = x_strides_override;
    } else {
      if (!trans_x) {
        x_strides.insert(x_strides.end(), {M * K, K, 1});
      } else {
        x_strides.insert(x_strides.end(), {M * K, 1, M});
      }
    }

    if (!y_strides_override.empty()) {
      y_strides = y_strides_override;
    } else {
      if (!trans_y) {
        y_strides.insert(y_strides.end(), {N * K, N, 1});
      } else {
        y_strides.insert(y_strides.end(), {N * K, 1, K});
      }
    }

    out_strides.insert(out_strides.end(), {M * N, N, 1});
    out_ddims.insert(out_ddims.end(),
                     {std::max(x_dims[MB_idx], y_dims[MB_idx]), M, N});

    for (int i = x_dims.size() - 4; i >= 0; --i) {
      out_ddims[i] = std::max(x_dims[i], y_dims[i]);
      if (x_strides_override.empty()) {
        x_strides[i] = x_dims[i + 1] * x_strides[i + 1];
      }
      if (y_strides_override.empty()) {
        y_strides[i] = y_dims[i + 1] * y_strides[i + 1];
      }
      out_strides[i] = out_ddims[i + 1] * out_strides[i + 1];
    }

    // TODO(jczaja): Why not for int8??
    if (!IsInt8<OT>() && is_output_fused) {
      out_strides = FakeTransposeStrides(out_ddims);
    }

    auto x_md = memory::desc(x_dims, MKLDNNGetDataType<XT>(), x_strides);
    auto y_md = memory::desc(y_dims, MKLDNNGetDataType<YT>(), y_strides);
    auto out_md = memory::desc(out_ddims, MKLDNNGetDataType<OT>(), out_strides);

    const dnnl::primitive_attr matmul_attrs = CreateMatmulAttrs(ctx);

    this->AcquireForwardPrimitiveDescriptor(matmul_attrs, x_md, y_md, out_md);
  }

  float ComputeOutputScale(const framework::ExecutionContext& ctx) {
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;
    if (ctx.HasAttr("Scale_x") && ctx.HasAttr("Scale_y") &&
        ctx.HasAttr("Scale_out")) {
      float scale_x = ctx.Attr<float>("Scale_x");
      float scale_y = ctx.Attr<float>("Scale_y");
      bool force_fp32_out = ctx.HasAttr("force_fp32_output")
                                ? ctx.Attr<bool>("force_fp32_output")
                                : false;
      float scale_out = force_fp32_out ? 1.f : ctx.Attr<float>("Scale_out");
      alpha *= scale_out / (scale_x * scale_y);
    }
    return alpha;
  }

  dnnl::primitive_attr CreateMatmulAttrs(
      const framework::ExecutionContext& ctx) {
    dnnl::primitive_attr matmul_attrs;
    dnnl::post_ops post_operations;

    float scale_out = ComputeOutputScale(ctx);
    if (scale_out != 1.0f) {
      matmul_attrs.set_output_scales(0, {scale_out});
    }

    if (ctx.HasInput("ResidualData")) {
      auto* residual_data = ctx.Input<phi::DenseTensor>("ResidualData");
      auto residual_data_tz = phi::vectorize(residual_data->dims());
      auto residual_data_md = memory::desc(residual_data_tz,
                                           MKLDNNGetDataType<OT>(),
                                           dnnl::memory::format_tag::any);
      post_operations.append_binary(dnnl::algorithm::binary_add,
                                    residual_data_md);
      if (ctx.HasAttr("Scale_in_eltwise")) {
        float sum_scale = scale_out / ctx.Attr<float>("Scale_in_eltwise");
        post_operations.append_sum(sum_scale);
      }
    }

    AppendActivation(ctx, post_operations);

    matmul_attrs.set_post_ops(post_operations);
    return matmul_attrs;
  }

  std::vector<int64_t> FakeTransposeStrides(
      const std::vector<int64_t>& matmul_out_dims) const {
    // fuse matmul_v2 + transpose + reshape guarantees that output is 4D and
    // transpose axis are: {0, 2, 1, 3}
    std::vector<int64_t> transpose_axis = {0, 2, 1, 3};
    std::vector<int64_t> fake_strides(transpose_axis.size());
    int ndims = static_cast<int>(transpose_axis.size());

    int total_stride = 1;

    for (int i = ndims - 1; i >= 0; --i) {
      fake_strides[transpose_axis[i]] = total_stride;
      total_stride *= matmul_out_dims[transpose_axis[i]];
    }

    return fake_strides;
  }

  std::shared_ptr<memory> AcquireWeightsMemory(const phi::DenseTensor* input) {
    const YT* input_data = input->data<YT>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->weights_desc(),
                                            to_void_cast<YT>(input_data));
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(phi::DenseTensor* output) {
    // We cannot use base AcquireDstMemory as it makes an allocation request
    // base on DST memory primitive size. This is fine in general, but in MatMul
    // we have primitive that covers only one batch of Data and then shift
    // pointer for every new batch. Hence phi::DenseTensor size is bigger that
    // dst memory primitive size. So would we request less memory that is there
    // and it triggers an assertion.  So as there is no 'any' format here we can
    // leave default size of phi::DenseTensor as computed in ComputeInferShape
    OT* ptr = output->mutable_data<OT>(this->place_);
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(), ptr);
  }
};

static std::unordered_map<std::string, std::string> GetAttributeMap(
    std::string act_type) {
  std::unordered_map<std::string, std::string> attr_map;
  if (act_type == "swish") {
    attr_map.emplace("beta", "fuse_alpha");
  } else if (act_type == "relu6") {
    attr_map.emplace("threshold", "fuse_alpha");
  } else if (act_type == "hard_sigmoid") {
    attr_map.emplace("slope", "fuse_alpha");
    attr_map.emplace("offset", "fuse_beta");
  } else if (act_type == "clip") {
    attr_map.emplace("min", "fuse_alpha");
    attr_map.emplace("max", "fuse_beta");
  } else {
    attr_map.emplace("alpha", "fuse_alpha");
    attr_map.emplace("beta", "fuse_beta");
  }
  return attr_map;
}

static std::vector<std::string> GetSupportedActivations() {
  return std::vector<std::string>{"abs",
                                  "clip",
                                  "gelu",
                                  "hard_sigmoid",
                                  "hard_swish",
                                  "leaky_relu",
                                  "mish",
                                  "relu",
                                  "relu6",
                                  "sigmoid",
                                  "sqrt",
                                  "swish",
                                  "tanh"};
}

class ReorderMKLDNNHandler {
 public:
  ReorderMKLDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                       framework::proto::VarType::Type vtype,
                       dnnl::memory::data_type dtype,
                       dnnl::engine engine)
      : dims_(dims),
        vtype_(vtype),
        vtype_dst_(vtype),
        dtype_(dtype),
        dtype_dst_(dtype),
        engine_(engine) {}

  ReorderMKLDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                       framework::proto::VarType::Type vtype,
                       dnnl::memory::data_type dtype,
                       framework::proto::VarType::Type vtype_dst,
                       dnnl::memory::data_type dtype_dst,
                       dnnl::engine engine)
      : dims_(dims),
        vtype_(vtype),
        vtype_dst_(vtype_dst),
        dtype_(dtype),
        dtype_dst_(dtype_dst),
        engine_(engine) {}

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const dnnl::memory::desc& md,
                                                 void* ptr) {
    return std::make_shared<dnnl::memory>(md, engine_, ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const MKLDNNMemoryFormat& fmt,
                                                 void* ptr) {
    auto md = dnnl::memory::desc(dims_, dtype_, fmt);
    return std::make_shared<dnnl::memory>(md, engine_, ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireSubmemory(
      const std::vector<int64_t>& dims,
      const std::vector<int64_t>& offset,
      const std::shared_ptr<dnnl::memory>& mem_p) {
    auto sub_md = mem_p->get_desc().submemory_desc(dims, {offset});
    auto sub_mem_p = std::make_shared<dnnl::memory>(
        sub_md, engine_, mem_p->get_data_handle());
    return sub_mem_p;
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(phi::DenseTensor* output,
                                                 const MKLDNNMemoryFormat& fmt,
                                                 platform::Place place) {
    auto dst_md = platform::MKLDNNMemDesc(dims_, dtype_dst_, fmt);
    auto dst_data = output->mutable_data(
        place, framework::TransToPhiDataType(vtype_dst_), dst_md.get_size());
    return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(
      phi::DenseTensor* output,
      const dnnl::memory::desc& src_md,
      platform::Place place) {
    if (vtype_dst_ == vtype_) {
      auto dst_data = output->mutable_data(
          place, framework::TransToPhiDataType(vtype_dst_), src_md.get_size());
      return std::make_shared<dnnl::memory>(src_md, engine_, dst_data);
    } else {
      auto dst_md = src_md;
      dst_md.data.data_type = static_cast<dnnl_data_type_t>(dtype_dst_);
      auto dst_data = output->mutable_data(
          place, framework::TransToPhiDataType(vtype_dst_), dst_md.get_size());
      return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
    }
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(
      phi::DenseTensor* output,
      const std::vector<int64_t>& dims,
      const MKLDNNMemoryFormat& fmt,
      platform::Place place) {
    auto dst_md = platform::MKLDNNMemDesc(dims, dtype_dst_, fmt);
    auto dst_data = output->mutable_data(
        place, framework::TransToPhiDataType(vtype_dst_), dst_md.get_size());
    return std::make_shared<dnnl::memory>(dst_md, engine_, dst_data);
  }

  std::shared_ptr<dnnl::reorder> AcquireReorder(
      std::shared_ptr<dnnl::memory> dst_memory_p,
      std::shared_ptr<dnnl::memory> src_memory_p) {
    return std::make_shared<dnnl::reorder>(*(src_memory_p), *(dst_memory_p));
  }

  std::shared_ptr<dnnl::reorder> AcquireReorder(
      std::shared_ptr<dnnl::memory> dst_memory_p,
      std::shared_ptr<dnnl::memory> src_memory_p,
      const dnnl::primitive_attr& attrs) {
    return std::make_shared<dnnl::reorder>(
        *(src_memory_p), *(dst_memory_p), attrs);
  }

 private:
  std::vector<int64_t> dims_;
  framework::proto::VarType::Type vtype_, vtype_dst_;
  dnnl::memory::data_type dtype_, dtype_dst_;
  dnnl::engine engine_;
};
}  // namespace platform
}  // namespace paddle
