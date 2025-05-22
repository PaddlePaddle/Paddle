#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {


PADDLE_API void fused_bias_dropout_residual_layer_norm_grad(const Tensor& x, const Tensor& residual, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& ln_scale, const paddle::optional<Tensor>& ln_bias, const Tensor& ln_mean, const Tensor& ln_variance, const Tensor& bias_dropout_residual_out, const Tensor& dropout_mask_out, const Tensor& y_grad, float dropout_rate, bool is_test, bool dropout_fix_seed, int dropout_seed, const std::string& dropout_implementation, float ln_epsilon, Tensor* x_grad, Tensor* residual_grad, Tensor* bias_grad, Tensor* ln_scale_grad, Tensor* ln_bias_grad);

PADDLE_API void fused_dot_product_attention_grad(const Tensor& q, const Tensor& k, const Tensor& v, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& cu_seqlen_q, const paddle::optional<Tensor>& cu_seqlen_kv, const Tensor& out, const Tensor& softmax_out, const Tensor& rng_state, const Tensor& out_grad, float scaling_factor, float dropout_probability, const std::string& mask_type_str, const std::string& bias_type_str, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad, Tensor* bias_grad);

PADDLE_API void fused_dropout_add_grad(const Tensor& seed_offset, const Tensor& out_grad, const Scalar& p, bool is_test, const std::string& mode, bool fix_seed, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void fused_rotary_position_embedding_grad(const paddle::optional<Tensor>& sin, const paddle::optional<Tensor>& cos, const paddle::optional<Tensor>& position_ids, const Tensor& out_q_grad, const paddle::optional<Tensor>& out_k_grad, const paddle::optional<Tensor>& out_v_grad, bool use_neox_rotary_style, bool time_major, float rotary_emb_base, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad);

PADDLE_API void resnet_basic_block_grad(const Tensor& x, const Tensor& filter1, const Tensor& conv1, const Tensor& scale1, const Tensor& bias1, const Tensor& saved_mean1, const Tensor& saved_invstd1, const Tensor& filter2, const Tensor& conv2, const Tensor& conv2_input, const Tensor& scale2, const Tensor& bias2, const Tensor& saved_mean2, const Tensor& saved_invstd2, const paddle::optional<Tensor>& filter3, const paddle::optional<Tensor>& conv3, const paddle::optional<Tensor>& scale3, const paddle::optional<Tensor>& bias3, const paddle::optional<Tensor>& saved_mean3, const paddle::optional<Tensor>& saved_invstd3, const Tensor& max_input1, const Tensor& max_filter1, const Tensor& max_input2, const Tensor& max_filter2, const Tensor& max_input3, const Tensor& max_filter3, const Tensor& out, const Tensor& out_grad, int stride1, int stride2, int stride3, int padding1, int padding2, int padding3, int dilation1, int dilation2, int dilation3, int group, float momentum, float epsilon, const std::string& data_format, bool has_shortcut, bool use_global_stats, bool is_test, bool trainable_statistics, const std::string& act_type, bool find_conv_input_max, Tensor* x_grad, Tensor* filter1_grad, Tensor* scale1_grad, Tensor* bias1_grad, Tensor* filter2_grad, Tensor* scale2_grad, Tensor* bias2_grad, Tensor* filter3_grad, Tensor* scale3_grad, Tensor* bias3_grad);

PADDLE_API void resnet_unit_grad(const Tensor& x, const Tensor& filter_x, const Tensor& conv_x, const Tensor& scale_x, const Tensor& bias_x, const Tensor& saved_mean_x, const Tensor& saved_invstd_x, const paddle::optional<Tensor>& z, const paddle::optional<Tensor>& filter_z, const paddle::optional<Tensor>& conv_z, const paddle::optional<Tensor>& scale_z, const paddle::optional<Tensor>& bias_z, const paddle::optional<Tensor>& saved_mean_z, const paddle::optional<Tensor>& saved_invstd_z, const Tensor& out, const Tensor& bit_mask, const Tensor& out_grad, int stride, int stride_z, int padding, int dilation, int group, float momentum, float epsilon, const std::string& data_format, bool fuse_add, bool has_shortcut, bool use_global_stats, bool is_test, bool use_addto, const std::string& act_type, Tensor* x_grad, Tensor* filter_x_grad, Tensor* scale_x_grad, Tensor* bias_x_grad, Tensor* z_grad, Tensor* filter_z_grad, Tensor* scale_z_grad, Tensor* bias_z_grad);


}  // namespace experimental
}  // namespace paddle
