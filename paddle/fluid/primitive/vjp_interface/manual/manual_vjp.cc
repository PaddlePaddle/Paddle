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

// Auto Generated, DO NOT EDIT!

#include "paddle/fluid/primitive/vjp_interface/manual/manual_vjp.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/prim/utils/static/static_global_utils.h"
#include "paddle/fluid/primitive/backend/backend.h"
#include "paddle/fluid/primitive/base/lazy_tensor.h"
#include "paddle/fluid/primitive/decomp_rule/decomp_vjp/details.h"
#include "paddle/fluid/primitive/decomp_utils/decomp_utils.h"
#include "paddle/pir/include/core/operation.h"

namespace paddle::primitive {

std::vector<std::vector<paddle::Tensor>> add_n_vjp(
    const std::vector<paddle::Tensor>& x,
    const Tensor& out_grad,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg : stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::add_n_grad<LazyTensor>(x, out_grad);
  vjp_res[0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> reshape_vjp(
    const Tensor& xshape,
    const Tensor& out_grad,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg : stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "reshape_grad";
  auto need_skip =
      paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(
          op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() &&
      !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call Pir Decomposed backward op reshape_grad";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr;

    details::reshape_grad<LazyTensor>(xshape, out_grad, x_grad);
  } else {
    auto op_res = backend::reshape_grad<LazyTensor>(xshape, out_grad);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fused_attention_vjp(
    const Tensor& out_grad,
    const Tensor& x,
    const Tensor& qkv_weight,
    const paddle::optional<Tensor>& qkv_bias,
    const paddle::optional<Tensor>& qkv_bias_out,
    const paddle::optional<Tensor>& src_mask,
    const paddle::optional<Tensor>& src_mask_out,
    const Tensor& out_linear_weight,
    const paddle::optional<Tensor>& out_linear_bias,
    const paddle::optional<Tensor>& ln_scale,
    const paddle::optional<Tensor>& ln_bias,
    const paddle::optional<Tensor>& ln_scale_2,
    const paddle::optional<Tensor>& ln_bias_2,
    const paddle::optional<Tensor>& ln_out,
    const paddle::optional<Tensor>& ln_mean,
    const paddle::optional<Tensor>& ln_var,
    const paddle::optional<Tensor>& ln_mean_2,
    const paddle::optional<Tensor>& ln_var_2,
    const paddle::optional<Tensor>& bias_dropout_residual_out,
    const Tensor& qkv_out,
    const Tensor& transpose_out_2,
    const Tensor& qk_out,
    const Tensor& qktv_out,
    const Tensor& softmax_out,
    const Tensor& attn_dropout_mask_out,
    const Tensor& attn_dropout_out,
    const Tensor& fmha_out,
    const Tensor& out_linear_out,
    const Tensor& dropout_mask_out,
    int num_heads,
    bool transpose_qkv_wb,
    bool pre_layer_norm,
    float epsilon,
    float attn_dropout_rate,
    bool is_test,
    bool attn_dropout_fix_seed,
    int attn_dropout_seed,
    const std::string& attn_dropout_implementation,
    float dropout_rate,
    bool dropout_fix_seed,
    int dropout_seed,
    const std::string& dropout_implementation,
    float ln_epsilon,
    bool add_residual,
    int ring_id,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg : stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res =
      backend::fused_attention_grad<LazyTensor>(out_grad,
                                                x,
                                                qkv_weight,
                                                qkv_bias,
                                                qkv_bias_out,
                                                src_mask,
                                                src_mask_out,
                                                out_linear_weight,
                                                out_linear_bias,
                                                ln_scale,
                                                ln_bias,
                                                ln_scale_2,
                                                ln_bias_2,
                                                ln_out,
                                                ln_mean,
                                                ln_var,
                                                ln_mean_2,
                                                ln_var_2,
                                                bias_dropout_residual_out,
                                                qkv_out,
                                                transpose_out_2,
                                                qk_out,
                                                qktv_out,
                                                softmax_out,
                                                attn_dropout_mask_out,
                                                attn_dropout_out,
                                                fmha_out,
                                                out_linear_out,
                                                dropout_mask_out,
                                                num_heads,
                                                transpose_qkv_wb,
                                                pre_layer_norm,
                                                epsilon,
                                                attn_dropout_rate,
                                                is_test,
                                                attn_dropout_fix_seed,
                                                attn_dropout_seed,
                                                attn_dropout_implementation,
                                                dropout_rate,
                                                dropout_fix_seed,
                                                dropout_seed,
                                                dropout_implementation,
                                                ln_epsilon,
                                                add_residual,
                                                ring_id);
  // x_grad
  vjp_res[0][0] = std::get<8>(op_res);
  // ln_scale_grad
  vjp_res[1][0] = std::get<4>(op_res).get();
  // ln_bias_grad
  vjp_res[2][0] = std::get<5>(op_res).get();
  // qkv_weight_grad
  vjp_res[3][0] = std::get<9>(op_res);
  // qkv_bias_grad
  vjp_res[4][0] = std::get<0>(op_res).get();
  // out_linear_weight_grad
  vjp_res[5][0] = std::get<10>(op_res);
  // out_linear_bias_grad
  vjp_res[6][0] = std::get<3>(op_res).get();
  // ln_scale_2_grad
  vjp_res[7][0] = std::get<6>(op_res).get();
  // ln_bias_2_grad
  vjp_res[8][0] = std::get<7>(op_res).get();
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fused_gemm_epilogue_vjp(
    const Tensor& x,
    const Tensor& y,
    const paddle::optional<Tensor>& reserve_space,
    const Tensor& out_grad,
    bool trans_x,
    bool trans_y,
    const std::string& activation,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg : stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fused_gemm_epilogue_grad<LazyTensor>(
      x, y, reserve_space, out_grad, trans_x, trans_y, activation);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

}  // namespace paddle::primitive
