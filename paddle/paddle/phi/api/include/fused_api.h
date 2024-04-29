#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {


PADDLE_API std::tuple<Tensor, Tensor&, Tensor&, Tensor&> block_multihead_attention_(Tensor& qkv, Tensor& key_cache, Tensor& value_cache, const Tensor& seq_lens_encoder, const Tensor& seq_lens_decoder, const Tensor& seq_lens_this_time, const Tensor& padding_offsets, const Tensor& cum_offsets, const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, const Tensor& block_tables, const paddle::optional<Tensor>& pre_key_cache, const paddle::optional<Tensor>& pre_value_cache, const paddle::optional<Tensor>& rope_emb, const paddle::optional<Tensor>& mask, const paddle::optional<Tensor>& tgt_mask, const paddle::optional<Tensor>& cache_k_quant_scales, const paddle::optional<Tensor>& cache_v_quant_scales, const paddle::optional<Tensor>& cache_k_dequant_scales, const paddle::optional<Tensor>& cache_v_dequant_scales, const paddle::optional<Tensor>& qkv_out_scale, const paddle::optional<Tensor>& qkv_bias, const paddle::optional<Tensor>& out_shift, const paddle::optional<Tensor>& out_smooth, int max_seq_len, int block_size, bool use_neox_style, bool dynamic_cachekv_quant = false, int quant_round_type = 1, float quant_max_bound = 127.0, float quant_min_bound = -127.0, float out_scale = -1, const std::string& compute_dtype = "default");

PADDLE_API Tensor fused_bias_act(const Tensor& x, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& dequant_scales, const paddle::optional<Tensor>& shift, const paddle::optional<Tensor>& smooth, const std::string& act_method = "gelu", const std::string& compute_dtype = "default", float quant_scale = -1, int quant_round_type = 1, float quant_max_bound = 127.0, float quant_min_bound = -127.0);

PADDLE_API Tensor fused_bias_dropout_residual_layer_norm(const Tensor& x, const Tensor& residual, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& ln_scale, const paddle::optional<Tensor>& ln_bias, float dropout_rate = 0.5f, bool is_test = false, bool dropout_fix_seed = true, int dropout_seed = true, const std::string& dropout_implementation = "downgrade_in_infer", float ln_epsilon = 1e-5);

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> fused_bias_dropout_residual_layer_norm_intermediate(const Tensor& x, const Tensor& residual, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& ln_scale, const paddle::optional<Tensor>& ln_bias, float dropout_rate = 0.5f, bool is_test = false, bool dropout_fix_seed = true, int dropout_seed = true, const std::string& dropout_implementation = "downgrade_in_infer", float ln_epsilon = 1e-5);

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor> fused_bias_residual_layernorm(const Tensor& x, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& residual, const paddle::optional<Tensor>& norm_weight, const paddle::optional<Tensor>& norm_bias, float epsilon, float residual_alpha, int begin_norm_axis, float quant_scale, int quant_round_type, float quant_max_bound, float quant_min_bound);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> fused_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& mask, float scaling_factor, float dropout_probability, bool is_training = false, bool is_causal_masking = false);

PADDLE_API std::tuple<Tensor, Tensor> fused_dropout_add(const Tensor& x, const Tensor& y, const paddle::optional<Tensor>& seed_tensor, const Scalar& p, bool is_test, const std::string& mode, int seed = 0, bool fix_seed = false);

PADDLE_API std::tuple<Tensor, Tensor> fused_linear_param_grad_add(const Tensor& x, const Tensor& dout, const paddle::optional<Tensor>& dweight, const paddle::optional<Tensor>& dbias, bool multi_precision = true, bool has_bias = true);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> fused_rotary_position_embedding(const Tensor& q, const paddle::optional<Tensor>& k, const paddle::optional<Tensor>& v, const paddle::optional<Tensor>& sin, const paddle::optional<Tensor>& cos, const paddle::optional<Tensor>& position_ids, bool use_neox_rotary_style = true, bool time_major = false, float rotary_emb_base = 10000.0);

PADDLE_API Tensor variable_length_memory_efficient_attention(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& seq_lens, const Tensor& kv_seq_lens, const paddle::optional<Tensor>& mask, float scale, bool causal, int pre_cache_length);


}  // namespace experimental
}  // namespace paddle
