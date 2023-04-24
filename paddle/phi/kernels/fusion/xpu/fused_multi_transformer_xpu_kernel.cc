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

#include "paddle/phi/backends/xpu/enforce_xpu.h"

#include "glog/logging.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/memcpy_kernel.h"
#ifdef PADDLE_WITH_XPU_XFT
#include "models/fused_multi_transformer_op.h"
namespace xft = baidu::xpu::xft;
#endif

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedMultiTransformerXpuKernel(
    const Context& ctx,
    const DenseTensor& xx,
    const std::vector<const DenseTensor*>& ln_scale,
    const std::vector<const DenseTensor*>& ln_bias,
    const std::vector<const DenseTensor*>& qkvw,
    const std::vector<const DenseTensor*>& qkvw_max,
    const std::vector<const DenseTensor*>& qkv_bias,
    const std::vector<const DenseTensor*>& out_linear_w,
    const std::vector<const DenseTensor*>& out_linear_wmax,
    const std::vector<const DenseTensor*>& out_linear_bias,
    const std::vector<const DenseTensor*>& ffn_ln_scale,
    const std::vector<const DenseTensor*>& ffn_ln_bias,
    const std::vector<const DenseTensor*>& ffn1_weight,
    const std::vector<const DenseTensor*>& ffn1_weight_max,
    const std::vector<const DenseTensor*>& ffn1_bias,
    const std::vector<const DenseTensor*>& ffn2_weight,
    const std::vector<const DenseTensor*>& ffn2_weight_max,
    const std::vector<const DenseTensor*>& ffn2_bias,
    const paddle::optional<std::vector<const DenseTensor*>>& cache_kv,
    const paddle::optional<std::vector<const DenseTensor*>>& pre_caches,
    const paddle::optional<DenseTensor>& rotary_pos_emb,
    const paddle::optional<DenseTensor>& time_step,
    const paddle::optional<DenseTensor>& seq_lengths,
    const paddle::optional<DenseTensor>& src_mask,
    const paddle::optional<DenseTensor>& gather_index,
    bool pre_layer_norm,
    int rotary_emb_dims,
    float epsilon,
    float dropout_rate,
    bool is_test,
    const std::string& dropout_implementation,
    const std::string& act_method,
    bool trans_qkvw,
    int ring_id,
    int gather_axis,
    DenseTensor* out,
    std::vector<DenseTensor*> cache_kv_out) {
#ifdef PADDLE_WITH_XPU_XFT
  using XPUTypeT = typename XPUTypeTrait<T>::Type;

  PADDLE_ENFORCE_EQ(pre_layer_norm,
                    true,
                    phi::errors::PreconditionNotMet(
                        "Only support pre_layer_norm = true at now."));
  PADDLE_ENFORCE_EQ(
      seq_lengths.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("seq_lengths not support at now."));
  PADDLE_ENFORCE_EQ(
      rotary_pos_emb.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("rotary_pos_emb not support at now."));
  PADDLE_ENFORCE_EQ(
      pre_caches.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("pre_caches not support at now."));
  PADDLE_ENFORCE_NE(
      src_mask.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("src_mask should not be nullptr."));
  PADDLE_ENFORCE_EQ(trans_qkvw,
                    true,
                    phi::errors::PreconditionNotMet(
                        "Only support trans_qkvw == true at now."));

  const auto x_dims = xx.dims();
  int seq_len = x_dims[1];
  const auto qkv_w_dims = qkvw[0]->dims();
  int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
  int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];

  int time_step_value = -1;
  if (time_step) {
    PADDLE_ENFORCE_EQ(time_step.get_ptr()->place(),
                      phi::CPUPlace(),
                      phi::errors::PreconditionNotMet(
                          "The place of input(time_step) must be CPUPlace."));
    // cache_seq_len
    time_step_value = time_step.get_ptr()->data<int>()[0];
    PADDLE_ENFORCE_GT(
        time_step_value,
        0,
        phi::errors::PreconditionNotMet(
            "The value of time_step must > 0, but now is %d", time_step_value));
    PADDLE_ENFORCE_EQ(
        seq_len,
        1,
        phi::errors::PreconditionNotMet(
            "In decode stage, the seq_len of input must be 1, but now is %d",
            seq_len));
  }

  XPUTypeT* x_data = reinterpret_cast<XPUTypeT*>(const_cast<T*>(xx.data<T>()));
  XPUTypeT* src_mask_data = reinterpret_cast<XPUTypeT*>(
      const_cast<T*>(src_mask.get_ptr()->data<T>()));
  auto* out_data = reinterpret_cast<XPUTypeT*>(ctx.template Alloc<T>(out));
  auto src_mask_dims = src_mask.get_ptr()->dims();
  auto out_dims = out->dims();
  auto xft_x = xft::xftTensor<XPUTypeT, 3>(
      x_data, std::array<int64_t, 3>{x_dims[0], x_dims[1], x_dims[2]});
  // TODO(mayang02): xft support mask.dtype = float16
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  float* src_mask_fp32_data =
      RAII_GUARD.alloc<float>(src_mask.get_ptr()->numel());
  int r = xpu::cast<XPUTypeT, float>(ctx.x_context(),
                                     src_mask_data,
                                     src_mask_fp32_data,
                                     src_mask.get_ptr()->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu::cast");
  auto xft_src_mask =
      xft::xftTensor<float, 4>(src_mask_fp32_data,
                               std::array<int64_t, 4>{src_mask_dims[0],
                                                      src_mask_dims[1],
                                                      src_mask_dims[2],
                                                      src_mask_dims[3]});
  auto xft_out = xft::xftTensor<XPUTypeT, 3>(
      out_data, std::array<int64_t, 3>{out_dims[0], out_dims[1], out_dims[2]});

  typedef int16_t TW;
  std::vector<xft::xftVec<float>> xft_ln_scale;
  std::vector<xft::xftVec<float>> xft_ln_bias;
  std::vector<xft::xftMat<TW>> xft_qkvw;
  std::vector<xft::xftVec<float>> xft_qkv_bias;
  std::vector<xft::xftMat<TW>> xft_out_linear_w;
  std::vector<xft::xftVec<float>> xft_out_linear_bias;
  std::vector<xft::xftVec<float>> xft_ffn_ln_scale;
  std::vector<xft::xftVec<float>> xft_ffn_ln_bias;
  std::vector<xft::xftMat<TW>> xft_ffn1_w;
  std::vector<xft::xftVec<float>> xft_ffn1_bias;
  std::vector<xft::xftMat<TW>> xft_ffn2_w;
  std::vector<xft::xftVec<float>> xft_ffn2_bias;
  std::vector<xft::xftTensor<XPUTypeT, 5>> xft_cache_kv;
  std::vector<xft::xftTensor<XPUTypeT, 5>> xft_cache_kv_out;

  // Create a temporary Tensor to store the gather output of cache_kv
  auto gather_index_t = gather_index.get_ptr();
  auto cache_kv_dims = cache_kv.get_ptr()->at(0)->dims();
  auto cache_kv_gather_dims = cache_kv_dims;
  phi::DenseTensor cache_kv_gather_tensor;
  if (gather_index_t) {
    MetaTensor cache_kv_gather_meta(&cache_kv_gather_tensor);
    phi::GatherInferMeta(*cache_kv.get_ptr()->at(0),
                         *gather_index_t,
                         Scalar(gather_axis),
                         &cache_kv_gather_meta);
    cache_kv_gather_dims = cache_kv_gather_meta.dims();
    ctx.template Alloc<T>(&cache_kv_gather_tensor);
  }

  int layers = qkvw.size();
  for (int i = 0; i < layers; ++i) {
    // step1. layer_norm
    xft_ln_scale.emplace_back(const_cast<float*>(ln_scale[i]->data<float>()),
                              std::array<int64_t, 1>{ln_scale[i]->dims()[0]});
    xft_ln_bias.emplace_back(const_cast<float*>(ln_bias[i]->data<float>()),
                             std::array<int64_t, 1>{ln_bias[i]->dims()[0]});
    // step2. qkv
    auto qkvw_dims = qkvw[i]->dims();
    xft_qkvw.emplace_back(
        const_cast<TW*>(qkvw[i]->data<TW>()),
        const_cast<float*>(qkvw_max[i]->data<float>()),
        std::array<int64_t, 2>{qkvw_dims[0] * qkvw_dims[1] * qkvw_dims[2],
                               qkvw_dims[3]});
    auto qkvb_dims = qkv_bias[i]->dims();
    xft_qkv_bias.emplace_back(
        const_cast<float*>(qkv_bias[i]->data<float>()),
        std::array<int64_t, 1>{qkvb_dims[0] * qkvb_dims[1] * qkvb_dims[2]});
    // attn out
    auto outw_dims = out_linear_w[i]->dims();
    xft_out_linear_w.emplace_back(
        const_cast<TW*>(out_linear_w[i]->data<TW>()),
        const_cast<float*>(out_linear_wmax[i]->data<float>()),
        std::array<int64_t, 2>{outw_dims[0], outw_dims[1]});
    xft_out_linear_bias.emplace_back(
        const_cast<float*>(out_linear_bias[i]->data<float>()),
        std::array<int64_t, 1>{out_linear_bias[i]->dims()[0]});
    // ffn ln
    xft_ffn_ln_scale.emplace_back(
        const_cast<float*>(ffn_ln_scale[i]->data<float>()),
        std::array<int64_t, 1>{ffn_ln_scale[i]->dims()[0]});
    xft_ffn_ln_bias.emplace_back(
        const_cast<float*>(ffn_ln_bias[i]->data<float>()),
        std::array<int64_t, 1>{ffn_ln_bias[i]->dims()[0]});
    // ffn1
    auto ffn1w_dims = ffn1_weight[i]->dims();
    xft_ffn1_w.emplace_back(
        const_cast<TW*>(ffn1_weight[i]->data<TW>()),
        const_cast<float*>(ffn1_weight_max[i]->data<float>()),
        std::array<int64_t, 2>{ffn1w_dims[0], ffn1w_dims[1]});
    xft_ffn1_bias.emplace_back(const_cast<float*>(ffn1_bias[i]->data<float>()),
                               std::array<int64_t, 1>{ffn1_bias[i]->dims()[0]});
    // ffn2
    auto ffn2w_dims = ffn2_weight[i]->dims();
    xft_ffn2_w.emplace_back(
        const_cast<TW*>(ffn2_weight[i]->data<TW>()),
        const_cast<float*>(ffn2_weight_max[i]->data<float>()),
        std::array<int64_t, 2>{ffn2w_dims[0], ffn2w_dims[1]});
    xft_ffn2_bias.emplace_back(const_cast<float*>(ffn2_bias[i]->data<float>()),
                               std::array<int64_t, 1>{ffn2_bias[i]->dims()[0]});
    // cache kv in
    auto cache_kv_data = reinterpret_cast<XPUTypeT*>(
        const_cast<T*>(cache_kv.get_ptr()->at(i)->data<T>()));
    if (gather_index_t) {
      const auto& index_type = gather_index_t->dtype();
      if (index_type == DataType::INT32) {
        r = xpu::gather<XPUTypeT, int32_t>(
            ctx.x_context(),
            cache_kv_data,
            gather_index_t->data<int32_t>(),
            reinterpret_cast<XPUTypeT*>(cache_kv_gather_tensor.data<T>()),
            phi::vectorize<int32_t>(cache_kv_dims),
            gather_index_t->dims().size() == 0 ? 1 : gather_index_t->dims()[0],
            gather_axis);
      } else {
        r = xpu::gather<XPUTypeT, int64_t>(
            ctx.x_context(),
            cache_kv_data,
            gather_index_t->data<int64_t>(),
            reinterpret_cast<XPUTypeT*>(cache_kv_gather_tensor.data<T>()),
            phi::vectorize<int32_t>(cache_kv_dims),
            gather_index_t->dims().size() == 0 ? 1 : gather_index_t->dims()[0],
            gather_axis);
      }
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu::gather");
      cache_kv_out[i]->ResizeAndAllocate(cache_kv_gather_dims);
      r = xpu::copy<XPUTypeT>(
          ctx.x_context(),
          reinterpret_cast<XPUTypeT*>(cache_kv_gather_tensor.data<T>()),
          reinterpret_cast<XPUTypeT*>(ctx.template Alloc<T>(cache_kv_out[i])),
          cache_kv_out[i]->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu::copy");
    }
    cache_kv_data = reinterpret_cast<XPUTypeT*>(
        const_cast<T*>(cache_kv.get_ptr()->at(i)->data<T>()));
    xft_cache_kv.emplace_back(cache_kv_data,
                              std::array<int64_t, 5>{cache_kv_gather_dims[0],
                                                     cache_kv_gather_dims[1],
                                                     cache_kv_gather_dims[2],
                                                     cache_kv_gather_dims[3],
                                                     cache_kv_gather_dims[4]});
    // cache kv out direct use cache_kv_data
    xft_cache_kv_out.emplace_back(
        cache_kv_data,
        std::array<int64_t, 5>{cache_kv_gather_dims[0],
                               cache_kv_gather_dims[1],
                               cache_kv_gather_dims[2],
                               cache_kv_gather_dims[3],
                               cache_kv_gather_dims[4]});
  }
  xft::NlpParam param;
  param.num_layer = layers;
  param.n_head = num_head;
  param.size_per_head = dim_head;
  param.hidden_act = act_method;
  param.is_fuse_qkv = true;
  r = xft::fused_multi_transformer<XPUTypeT, TW, int16_t>(ctx.x_context(),
                                                          xft_x,
                                                          xft_cache_kv,
                                                          xft_src_mask,
                                                          xft_ln_scale,
                                                          xft_ln_bias,
                                                          xft_qkvw,
                                                          xft_qkv_bias,
                                                          xft_out_linear_w,
                                                          xft_out_linear_bias,
                                                          xft_ffn_ln_scale,
                                                          xft_ffn_ln_bias,
                                                          xft_ffn1_w,
                                                          xft_ffn1_bias,
                                                          xft_ffn2_w,
                                                          xft_ffn2_bias,
                                                          param,
                                                          time_step_value,
                                                          &xft_out,
                                                          xft_cache_kv_out);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "xft::fused_multi_transformer");
#else
  LOG(FATAL) << "fused_multi_transformer_xpu is not supported since it's not "
                "compiled with XPU_XFT";
#endif
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_multi_transformer_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMultiTransformerXpuKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(20).SetBackend(phi::Backend::CPU);
}
