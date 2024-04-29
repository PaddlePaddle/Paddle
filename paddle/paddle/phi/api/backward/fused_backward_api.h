#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {


PADDLE_API void fused_bias_dropout_residual_layer_norm_grad(const Tensor& x, const Tensor& residual, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& ln_scale, const paddle::optional<Tensor>& ln_bias, const Tensor& ln_mean, const Tensor& ln_variance, const Tensor& bias_dropout_residual_out, const Tensor& dropout_mask_out, const Tensor& y_grad, float dropout_rate, bool is_test, bool dropout_fix_seed, int dropout_seed, const std::string& dropout_implementation, float ln_epsilon, Tensor* x_grad, Tensor* residual_grad, Tensor* bias_grad, Tensor* ln_scale_grad, Tensor* ln_bias_grad);

PADDLE_API void fused_dot_product_attention_grad(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& softmax_out, const Tensor& rng_state, const Tensor& mask, const Tensor& out_grad, float scaling_factor, float dropout_probability, bool is_causal_masking, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad);

PADDLE_API void fused_dropout_add_grad(const Tensor& seed_offset, const Tensor& out_grad, const Scalar& p, bool is_test, const std::string& mode, bool fix_seed, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void fused_rotary_position_embedding_grad(const paddle::optional<Tensor>& sin, const paddle::optional<Tensor>& cos, const paddle::optional<Tensor>& position_ids, const Tensor& out_q_grad, const paddle::optional<Tensor>& out_k_grad, const paddle::optional<Tensor>& out_v_grad, bool use_neox_rotary_style, bool time_major, float rotary_emb_base, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad);


}  // namespace experimental
}  // namespace paddle
