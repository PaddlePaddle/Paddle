#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {


PADDLE_API void abs_double_grad(const Tensor& x, const Tensor& grad_x_grad, Tensor* grad_out_grad);

PADDLE_API void abs_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void abs_triple_grad(const Tensor& x, const Tensor& grad_out_grad_grad, Tensor* grad_x_grad_grad);

PADDLE_API void acos_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void acosh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void addmm_grad(const Tensor& input, const Tensor& x, const Tensor& y, const Tensor& out_grad, float alpha, float beta, Tensor* input_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void affine_grid_grad(const Tensor& input, const Tensor& output_grad, const IntArray& output_shape, bool align_corners, Tensor* input_grad);

PADDLE_API void angle_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void argsort_grad(const Tensor& indices, const Tensor& x, const Tensor& out_grad, int axis, bool descending, Tensor* x_grad);

PADDLE_API void as_strided_grad(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& dims, const std::vector<int64_t>& stride, int64_t offset, Tensor* input_grad);

PADDLE_API void asin_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void asinh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void atan2_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void atan_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void atanh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void bce_loss_grad(const Tensor& input, const Tensor& label, const Tensor& out_grad, Tensor* input_grad);

PADDLE_API void bicubic_interp_grad(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_format, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, Tensor* x_grad);

PADDLE_API void bilinear_grad(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad, Tensor* weight_grad, Tensor* bias_grad);

PADDLE_API void bilinear_interp_grad(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_format, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, Tensor* x_grad);

PADDLE_API void bmm_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void broadcast_tensors_grad(const std::vector<Tensor>& input, const std::vector<Tensor>& out_grad, std::vector<Tensor*> input_grad);

PADDLE_API void ceil_grad(const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void celu_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha, Tensor* x_grad, Tensor* grad_out_grad);

PADDLE_API void celu_grad(const Tensor& x, const Tensor& out_grad, float alpha, Tensor* x_grad);

PADDLE_API void cholesky_grad(const Tensor& out, const Tensor& out_grad, bool upper, Tensor* x_grad);

PADDLE_API void cholesky_solve_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, bool upper, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void clip_double_grad(const Tensor& x, const Tensor& grad_x_grad, const Scalar& min, const Scalar& max, Tensor* grad_out_grad);

PADDLE_API void clip_grad(const Tensor& x, const Tensor& out_grad, const Scalar& min, const Scalar& max, Tensor* x_grad);

PADDLE_API void complex_grad(const Tensor& real, const Tensor& imag, const Tensor& out_grad, Tensor* real_grad, Tensor* imag_grad);

PADDLE_API void concat_grad(const std::vector<Tensor>& x, const Tensor& out_grad, const Scalar& axis, std::vector<Tensor*> x_grad);

PADDLE_API void conv2d_grad(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, const std::vector<int>& dilations, int groups, const std::string& data_format, Tensor* input_grad, Tensor* filter_grad);

PADDLE_API void conv2d_grad_grad(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, const std::vector<int>& dilations, int groups, const std::string& data_format, Tensor* input_grad, Tensor* filter_grad, Tensor* grad_out_grad);

PADDLE_API void conv3d_double_grad(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, Tensor* input_grad, Tensor* filter_grad, Tensor* grad_out_grad);

PADDLE_API void conv3d_grad(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, Tensor* input_grad, Tensor* filter_grad);

PADDLE_API void conv3d_transpose_grad(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::vector<int>& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, Tensor* x_grad, Tensor* filter_grad);

PADDLE_API void copysign_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void cos_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, Tensor* x_grad, Tensor* grad_out_grad);

PADDLE_API void cos_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void cos_triple_grad(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad, Tensor* x_grad, Tensor* grad_out_forward_grad, Tensor* grad_x_grad_forward_grad);

PADDLE_API void cosh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void crop_grad(const Tensor& x, const Tensor& out_grad, const IntArray& offsets, Tensor* x_grad);

PADDLE_API void cross_entropy_with_softmax_grad(const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis, Tensor* input_grad);

PADDLE_API void cross_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void cummax_grad(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, DataType dtype, Tensor* x_grad);

PADDLE_API void cummin_grad(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, DataType dtype, Tensor* x_grad);

PADDLE_API void cumprod_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, int dim, Tensor* x_grad);

PADDLE_API void cumsum_grad(const Tensor& x, const Tensor& out_grad, const Scalar& axis, bool flatten, bool exclusive, bool reverse, Tensor* x_grad);

PADDLE_API void depthwise_conv2d_double_grad(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, Tensor* input_grad, Tensor* filter_grad, Tensor* grad_out_grad);

PADDLE_API void depthwise_conv2d_grad(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, Tensor* input_grad, Tensor* filter_grad);

PADDLE_API void det_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void diag_grad(const Tensor& x, const Tensor& out_grad, int offset, Tensor* x_grad);

PADDLE_API void diagonal_grad(const Tensor& x, const Tensor& out_grad, int offset, int axis1, int axis2, Tensor* x_grad);

PADDLE_API void digamma_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void dist_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, float p, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void dot_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void eig_grad(const Tensor& out_w, const Tensor& out_v, const Tensor& out_w_grad, const Tensor& out_v_grad, Tensor* x_grad);

PADDLE_API void eigh_grad(const Tensor& out_w, const Tensor& out_v, const Tensor& out_w_grad, const Tensor& out_v_grad, Tensor* x_grad);

PADDLE_API void eigvalsh_grad(const Tensor& eigenvectors, const Tensor& eigenvalues_grad, const std::string& uplo, bool is_test, Tensor* x_grad);

PADDLE_API void elu_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha, Tensor* x_grad, Tensor* grad_out_grad);

PADDLE_API void elu_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, float alpha, Tensor* x_grad);

PADDLE_API void erf_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void erfinv_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void exp_double_grad(const Tensor& out, const Tensor& grad_out, const Tensor& grad_x_grad, Tensor* out_grad, Tensor* grad_out_grad);

PADDLE_API void exp_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void expand_as_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int>& target_shape, Tensor* x_grad);

PADDLE_API void expand_grad(const Tensor& x, const Tensor& out_grad, const IntArray& shape, Tensor* x_grad);

PADDLE_API void expm1_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void fft_c2c_grad(const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, Tensor* x_grad);

PADDLE_API void fft_c2r_grad(const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, int64_t last_dim_size, Tensor* x_grad);

PADDLE_API void fft_r2c_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, bool onesided, Tensor* x_grad);

PADDLE_API void fill_diagonal_grad(const Tensor& out_grad, float value, int offset, bool wrap, Tensor* x_grad);

PADDLE_API void fill_diagonal_tensor_grad(const Tensor& out_grad, int64_t offset, int dim1, int dim2, Tensor* x_grad);

PADDLE_API void fill_grad(const Tensor& out_grad, const Scalar& value, Tensor* x_grad);

PADDLE_API void flash_attn_grad(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& softmax_lse, const Tensor& seed_offset, const paddle::optional<Tensor>& attn_mask, const Tensor& out_grad, float dropout, bool causal, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad);

PADDLE_API void flash_attn_unpadded_grad(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, const Tensor& out, const Tensor& softmax_lse, const Tensor& seed_offset, const paddle::optional<Tensor>& attn_mask, const Tensor& out_grad, int64_t max_seqlen_q, int64_t max_seqlen_k, float scale, float dropout, bool causal, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad);

PADDLE_API void flash_attn_with_sparse_mask_grad(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& attn_mask_start_row_indices, const Tensor& out, const Tensor& softmax_lse, const Tensor& seed_offset, const Tensor& out_grad, float dropout, bool causal, int attn_mask_start_row, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad);

PADDLE_API void flatten_grad(const Tensor& xshape, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void floor_grad(const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void fmax_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void fmin_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void fold_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int>& output_sizes, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, Tensor* x_grad);

PADDLE_API void fractional_max_pool2d_grad(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<int>& output_size, const std::vector<int>& kernel_size, float random_u, bool return_mask, Tensor* x_grad);

PADDLE_API void fractional_max_pool3d_grad(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<int>& output_size, const std::vector<int>& kernel_size, float random_u, bool return_mask, Tensor* x_grad);

PADDLE_API void frame_grad(const Tensor& x, const Tensor& out_grad, int frame_length, int hop_length, int axis, Tensor* x_grad);

PADDLE_API void gammaincc_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* y_grad);

PADDLE_API void gammaln_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void gather_grad(const Tensor& x, const Tensor& index, const Tensor& out_grad, const Scalar& axis, Tensor* x_grad);

PADDLE_API void gather_nd_grad(const Tensor& x, const Tensor& index, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void gaussian_inplace_grad(const Tensor& out_grad, float mean, float std, int seed, Tensor* x_grad);

PADDLE_API void gelu_grad(const Tensor& x, const Tensor& out_grad, bool approximate, Tensor* x_grad);

PADDLE_API void grid_sample_grad(const Tensor& x, const Tensor& grid, const Tensor& out_grad, const std::string& mode, const std::string& padding_mode, bool align_corners, Tensor* x_grad, Tensor* grid_grad);

PADDLE_API void group_norm_grad(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& y, const Tensor& mean, const Tensor& variance, const Tensor& y_grad, float epsilon, int groups, const std::string& data_format, Tensor* x_grad, Tensor* scale_grad, Tensor* bias_grad);

PADDLE_API void gumbel_softmax_grad(const Tensor& out, const Tensor& out_grad, int axis, Tensor* x_grad);

PADDLE_API void hardshrink_grad(const Tensor& x, const Tensor& out_grad, float threshold, Tensor* x_grad);

PADDLE_API void hardsigmoid_grad(const Tensor& out, const Tensor& out_grad, float slope, float offset, Tensor* x_grad);

PADDLE_API void hardtanh_grad(const Tensor& x, const Tensor& out_grad, float t_min, float t_max, Tensor* x_grad);

PADDLE_API void heaviside_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void huber_loss_grad(const Tensor& residual, const Tensor& out_grad, float delta, Tensor* input_grad, Tensor* label_grad);

PADDLE_API void i0_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void i0e_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void i1_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void i1e_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void identity_loss_grad(const Tensor& x, const Tensor& out_grad, int reduction, Tensor* x_grad);

PADDLE_API void imag_grad(const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void index_add_grad(const Tensor& index, const Tensor& add_value, const Tensor& out_grad, int axis, Tensor* x_grad, Tensor* add_value_grad);

PADDLE_API void index_put_grad(const Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, const Tensor& out_grad, bool accumulate, Tensor* x_grad, Tensor* value_grad);

PADDLE_API void index_sample_grad(const Tensor& x, const Tensor& index, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void index_select_grad(const Tensor& x, const Tensor& index, const Tensor& out_grad, int axis, Tensor* x_grad);

PADDLE_API void index_select_strided_grad(const Tensor& x, const Tensor& out_grad, int64_t index, int axis, Tensor* x_grad);

PADDLE_API void instance_norm_double_grad(const Tensor& x, const paddle::optional<Tensor>& fwd_scale, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& grad_y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_scale_grad, const paddle::optional<Tensor>& grad_bias_grad, float epsilon, Tensor* x_grad, Tensor* fwd_scale_grad, Tensor* grad_y_grad);

PADDLE_API void instance_norm_grad(const Tensor& x, const paddle::optional<Tensor>& scale, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& y_grad, float epsilon, Tensor* x_grad, Tensor* scale_grad, Tensor* bias_grad);

PADDLE_API void inverse_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void kldiv_loss_grad(const Tensor& x, const Tensor& label, const Tensor& out_grad, const std::string& reduction, Tensor* x_grad);

PADDLE_API void kron_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void kthvalue_grad(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int k, int axis, bool keepdim, Tensor* x_grad);

PADDLE_API void label_smooth_grad(const Tensor& out_grad, float epsilon, Tensor* label_grad);

PADDLE_API void layer_norm_grad(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& mean, const Tensor& variance, const Tensor& out_grad, float epsilon, int begin_norm_axis, Tensor* x_grad, Tensor* scale_grad, Tensor* bias_grad);

PADDLE_API void leaky_relu_double_grad(const Tensor& x, const Tensor& grad_x_grad, float negative_slope, Tensor* grad_out_grad);

PADDLE_API void leaky_relu_grad(const Tensor& x, const Tensor& out_grad, float negative_slope, Tensor* x_grad);

PADDLE_API void lerp_grad(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& out, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void lgamma_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void linear_interp_grad(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_format, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, Tensor* x_grad);

PADDLE_API void log10_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void log1p_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void log2_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void log_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, Tensor* x_grad, Tensor* grad_out_grad);

PADDLE_API void log_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void log_loss_grad(const Tensor& input, const Tensor& label, const Tensor& out_grad, float epsilon, Tensor* input_grad);

PADDLE_API void log_softmax_grad(const Tensor& out, const Tensor& out_grad, int axis, Tensor* x_grad);

PADDLE_API void logcumsumexp_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, int axis, bool flatten, bool exclusive, bool reverse, Tensor* x_grad);

PADDLE_API void logit_grad(const Tensor& x, const Tensor& out_grad, float eps, Tensor* x_grad);

PADDLE_API void logsigmoid_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void lu_grad(const Tensor& x, const Tensor& out, const Tensor& pivots, const Tensor& out_grad, bool pivot, Tensor* x_grad);

PADDLE_API void lu_unpack_grad(const Tensor& x, const Tensor& y, const Tensor& l, const Tensor& u, const Tensor& pmat, const Tensor& l_grad, const Tensor& u_grad, bool unpack_ludata, bool unpack_pivots, Tensor* x_grad);

PADDLE_API void margin_cross_entropy_grad(const Tensor& logits, const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool return_softmax, int ring_id, int rank, int nranks, float margin1, float margin2, float margin3, float scale, Tensor* logits_grad);

PADDLE_API void masked_select_grad(const Tensor& x, const Tensor& mask, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void matrix_power_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, int n, Tensor* x_grad);

PADDLE_API void max_pool2d_with_index_grad(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive, Tensor* x_grad);

PADDLE_API void max_pool3d_with_index_grad(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive, Tensor* x_grad);

PADDLE_API void maxout_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, int groups, int axis, Tensor* x_grad);

PADDLE_API void mean_all_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void memory_efficient_attention_grad(const Tensor& query, const Tensor& key, const Tensor& value, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& cu_seqlens_q, const paddle::optional<Tensor>& cu_seqlens_k, const Tensor& output, const Tensor& logsumexp, const Tensor& seed_and_offset, const Tensor& output_grad, const Scalar& max_seqlen_q, const Scalar& max_seqlen_k, bool causal, double dropout_p, float scale, Tensor* query_grad, Tensor* key_grad, Tensor* value_grad, Tensor* bias_grad);

PADDLE_API void meshgrid_grad(const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs_grad, std::vector<Tensor*> inputs_grad);

PADDLE_API void mode_grad(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, bool keepdim, Tensor* x_grad);

PADDLE_API void multi_dot_grad(const std::vector<Tensor>& x, const Tensor& out_grad, std::vector<Tensor*> x_grad);

PADDLE_API void multiplex_grad(const std::vector<Tensor>& inputs, const Tensor& index, const Tensor& out_grad, std::vector<Tensor*> inputs_grad);

PADDLE_API void mv_grad(const Tensor& x, const Tensor& vec, const Tensor& out_grad, Tensor* x_grad, Tensor* vec_grad);

PADDLE_API void nanmedian_grad(const Tensor& x, const Tensor& medians, const Tensor& out_grad, const IntArray& axis, bool keepdim, const std::string& mode, Tensor* x_grad);

PADDLE_API void nearest_interp_grad(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_format, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, Tensor* x_grad);

PADDLE_API void nll_loss_grad(const Tensor& input, const Tensor& label, const paddle::optional<Tensor>& weight, const Tensor& total_weight, const Tensor& out_grad, int64_t ignore_index, const std::string& reduction, Tensor* input_grad);

PADDLE_API void overlap_add_grad(const Tensor& x, const Tensor& out_grad, int hop_length, int axis, Tensor* x_grad);

PADDLE_API void p_norm_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, float porder, int axis, float epsilon, bool keepdim, bool asvector, Tensor* x_grad);

PADDLE_API void pad3d_double_grad(const Tensor& grad_x_grad, const IntArray& paddings, const std::string& mode, float pad_value, const std::string& data_format, Tensor* grad_out_grad);

PADDLE_API void pad3d_grad(const Tensor& x, const Tensor& out_grad, const IntArray& paddings, const std::string& mode, float pad_value, const std::string& data_format, Tensor* x_grad);

PADDLE_API void pixel_shuffle_grad(const Tensor& out_grad, int upscale_factor, const std::string& data_format, Tensor* x_grad);

PADDLE_API void pixel_unshuffle_grad(const Tensor& out_grad, int downscale_factor, const std::string& data_format, Tensor* x_grad);

PADDLE_API void poisson_grad(const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void polygamma_grad(const Tensor& x, const Tensor& out_grad, int n, Tensor* x_grad);

PADDLE_API void pow_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const Scalar& y, Tensor* x_grad, Tensor* grad_out_grad);

PADDLE_API void pow_grad(const Tensor& x, const Tensor& out_grad, const Scalar& y, Tensor* x_grad);

PADDLE_API void pow_triple_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_grad_x, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_grad_out_grad, const Scalar& y, Tensor* x_grad, Tensor* grad_out_grad, Tensor* grad_grad_x_grad);

PADDLE_API void prelu_grad(const Tensor& x, const Tensor& alpha, const Tensor& out_grad, const std::string& data_format, const std::string& mode, Tensor* x_grad, Tensor* alpha_grad);

PADDLE_API void psroi_pool_grad(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& out_grad, int pooled_height, int pooled_width, int output_channels, float spatial_scale, Tensor* x_grad);

PADDLE_API void put_along_axis_grad(const Tensor& arr, const Tensor& indices, const Tensor& values, const Tensor& out, const Tensor& out_grad, int axis, const std::string& reduce, bool include_self, Tensor* arr_grad, Tensor* values_grad);

PADDLE_API void qr_grad(const Tensor& x, const Tensor& q, const Tensor& r, const Tensor& q_grad, const Tensor& r_grad, const std::string& mode, Tensor* x_grad);

PADDLE_API void real_grad(const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void reciprocal_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void relu6_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void relu_double_grad(const Tensor& out, const Tensor& grad_x_grad, Tensor* grad_out_grad);

PADDLE_API void relu_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void renorm_grad(const Tensor& x, const Tensor& out_grad, float p, int axis, float max_norm, Tensor* x_grad);

PADDLE_API void rms_norm_grad(const Tensor& x, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& residual, const Tensor& norm_weight, const paddle::optional<Tensor>& norm_bias, const Tensor& inv_var, const Tensor& out_grad, float epsilon, int begin_norm_axis, float quant_scale, Tensor* x_grad, Tensor* norm_weight_grad);

PADDLE_API void roi_align_grad(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& out_grad, int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned, Tensor* x_grad);

PADDLE_API void roi_pool_grad(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& arg_max, const Tensor& out_grad, int pooled_height, int pooled_width, float spatial_scale, Tensor* x_grad);

PADDLE_API void roll_grad(const Tensor& x, const Tensor& out_grad, const IntArray& shifts, const std::vector<int64_t>& axis, Tensor* x_grad);

PADDLE_API void round_grad(const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void rsqrt_double_grad(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad, Tensor* out_grad, Tensor* grad_out_grad);

PADDLE_API void rsqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void scatter_grad(const Tensor& index, const Tensor& updates, const Tensor& out_grad, bool overwrite, Tensor* x_grad, Tensor* updates_grad);

PADDLE_API void scatter_nd_add_grad(const Tensor& index, const Tensor& updates, const Tensor& out_grad, Tensor* x_grad, Tensor* updates_grad);

PADDLE_API void segment_pool_grad(const Tensor& x, const Tensor& segment_ids, const Tensor& out, const paddle::optional<Tensor>& summed_ids, const Tensor& out_grad, const std::string& pooltype, Tensor* x_grad);

PADDLE_API void selu_grad(const Tensor& out, const Tensor& out_grad, float scale, float alpha, Tensor* x_grad);

PADDLE_API void send_u_recv_grad(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const paddle::optional<Tensor>& out, const paddle::optional<Tensor>& dst_count, const Tensor& out_grad, const std::string& reduce_op, Tensor* x_grad);

PADDLE_API void send_ue_recv_grad(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const paddle::optional<Tensor>& out, const paddle::optional<Tensor>& dst_count, const Tensor& out_grad, const std::string& message_op, const std::string& reduce_op, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void send_uv_grad(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const Tensor& out_grad, const std::string& message_op, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void sigmoid_cross_entropy_with_logits_grad(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, const Tensor& out_grad, bool normalize, int ignore_index, Tensor* x_grad);

PADDLE_API void sigmoid_double_grad(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_x_grad, Tensor* out_grad, Tensor* fwd_grad_out_grad);

PADDLE_API void sigmoid_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void sigmoid_triple_grad(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_grad_x, const Tensor& grad_out_grad, const paddle::optional<Tensor>& grad_grad_out_grad, Tensor* out_grad, Tensor* fwd_grad_out_grad, Tensor* grad_grad_x_grad);

PADDLE_API void silu_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void sin_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, Tensor* x_grad, Tensor* grad_out_grad);

PADDLE_API void sin_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void sin_triple_grad(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad, Tensor* x_grad, Tensor* grad_out_forward_grad, Tensor* grad_x_grad_forward_grad);

PADDLE_API void sinh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void slogdet_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void softplus_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float beta, float threshold, Tensor* x_grad, Tensor* grad_out_grad);

PADDLE_API void softplus_grad(const Tensor& x, const Tensor& out_grad, float beta, float threshold, Tensor* x_grad);

PADDLE_API void softshrink_grad(const Tensor& x, const Tensor& out_grad, float threshold, Tensor* x_grad);

PADDLE_API void softsign_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void solve_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void spectral_norm_grad(const Tensor& weight, const Tensor& u, const Tensor& v, const Tensor& out_grad, int dim, int power_iters, float eps, Tensor* weight_grad);

PADDLE_API void sqrt_double_grad(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad, Tensor* out_grad, Tensor* grad_out_grad);

PADDLE_API void sqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void square_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, Tensor* x_grad, Tensor* grad_out_grad);

PADDLE_API void square_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void squared_l2_norm_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void squeeze_grad(const Tensor& xshape, const Tensor& out_grad, const IntArray& axis, Tensor* x_grad);

PADDLE_API void stack_grad(const std::vector<Tensor>& x, const Tensor& out_grad, int axis, std::vector<Tensor*> x_grad);

PADDLE_API void stanh_grad(const Tensor& x, const Tensor& out_grad, float scale_a, float scale_b, Tensor* x_grad);

PADDLE_API void svd_grad(const Tensor& x, const Tensor& u, const Tensor& vh, const Tensor& s, const paddle::optional<Tensor>& u_grad, const paddle::optional<Tensor>& vh_grad, const paddle::optional<Tensor>& s_grad, bool full_matrices, Tensor* x_grad);

PADDLE_API void swiglu_grad(const Tensor& x, const paddle::optional<Tensor>& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void take_along_axis_grad(const Tensor& arr, const Tensor& indices, const Tensor& out_grad, int axis, Tensor* arr_grad);

PADDLE_API void tan_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void tanh_double_grad(const Tensor& out, const Tensor& grad_out, const Tensor& grad_x_grad, Tensor* out_grad, Tensor* grad_out_grad);

PADDLE_API void tanh_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void tanh_shrink_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void tanh_triple_grad(const Tensor& out, const Tensor& grad_out_forward, const Tensor& grad_x_grad_forward, const paddle::optional<Tensor>& grad_out_new_grad, const paddle::optional<Tensor>& grad_out_grad_grad, Tensor* out_grad, Tensor* grad_out_forward_grad, Tensor* grad_x_grad_forward_grad);

PADDLE_API void temporal_shift_grad(const Tensor& out_grad, int seg_num, float shift_ratio, const std::string& data_format, Tensor* x_grad);

PADDLE_API void tensor_unfold_grad(const Tensor& input, const Tensor& out_grad, int64_t axis, int64_t size, int64_t step, Tensor* input_grad);

PADDLE_API void thresholded_relu_grad(const Tensor& x, const Tensor& out_grad, float threshold, Tensor* x_grad);

PADDLE_API void topk_grad(const Tensor& x, const Tensor& indices, const Tensor& out_grad, const Scalar& k, int axis, bool largest, bool sorted, Tensor* x_grad);

PADDLE_API void trace_grad(const Tensor& x, const Tensor& out_grad, int offset, int axis1, int axis2, Tensor* x_grad);

PADDLE_API void triangular_solve_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, bool upper, bool transpose, bool unitriangular, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void trilinear_interp_grad(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_format, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, Tensor* x_grad);

PADDLE_API void trunc_grad(const Tensor& out_grad, Tensor* input_grad);

PADDLE_API void unfold_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, Tensor* x_grad);

PADDLE_API void uniform_inplace_grad(const Tensor& out_grad, float min, float max, int seed, int diag_num, int diag_step, float diag_val, Tensor* x_grad);

PADDLE_API void unsqueeze_grad(const Tensor& xshape, const Tensor& out_grad, const IntArray& axis, Tensor* x_grad);

PADDLE_API void unstack_grad(const std::vector<Tensor>& out_grad, int axis, Tensor* x_grad);

PADDLE_API void view_dtype_grad(const Tensor& input, const Tensor& out_grad, DataType dtype, Tensor* input_grad);

PADDLE_API void view_shape_grad(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& dims, Tensor* input_grad);

PADDLE_API void warpctc_grad(const Tensor& logits, const paddle::optional<Tensor>& logits_length, const Tensor& warpctcgrad, const Tensor& loss_grad, int blank, bool norm_by_times, Tensor* logits_grad);

PADDLE_API void warprnnt_grad(const Tensor& input, const Tensor& input_lengths, const Tensor& warprnntgrad, const Tensor& loss_grad, int blank, float fastemit_lambda, Tensor* input_grad);

PADDLE_API void weight_only_linear_grad(const Tensor& x, const Tensor& weight, const paddle::optional<Tensor>& bias, const Tensor& weight_scale, const Tensor& out_grad, const std::string& weight_dtype, int arch, int group_size, Tensor* x_grad);

PADDLE_API void where_grad(const Tensor& condition, const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void yolo_loss_grad(const Tensor& x, const Tensor& gt_box, const Tensor& gt_label, const paddle::optional<Tensor>& gt_score, const Tensor& objectness_mask, const Tensor& gt_match_mask, const Tensor& loss_grad, const std::vector<int>& anchors, const std::vector<int>& anchor_mask, int class_num, float ignore_thresh, int downsample_ratio, bool use_label_smooth, float scale_x_y, Tensor* x_grad, Tensor* gt_box_grad, Tensor* gt_label_grad, Tensor* gt_score_grad);

PADDLE_API void silu_double_grad(const Tensor& x, const Tensor& out, const Tensor& grad_out, const Tensor& grad_x_grad, Tensor* x_grad, Tensor* grad_out_grad);

PADDLE_API void unpool3d_grad(const Tensor& x, const Tensor& indices, const Tensor& out, const Tensor& out_grad, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_size, const std::string& data_format, Tensor* x_grad);

PADDLE_API void add_double_grad(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, Tensor* grad_out_grad);

PADDLE_API void add_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void add_triple_grad(const Tensor& grad_grad_x, const Tensor& grad_grad_y, const Tensor& grad_grad_out_grad, int axis, Tensor* grad_grad_x_grad, Tensor* grad_grad_y_grad);

PADDLE_API void amax_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all, Tensor* x_grad);

PADDLE_API void amin_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all, Tensor* x_grad);

PADDLE_API void assign_out__grad(const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void batch_norm_double_grad(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& out_mean, const paddle::optional<Tensor>& out_variance, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_scale_grad, const paddle::optional<Tensor>& grad_bias_grad, float momentum, float epsilon, const std::string& data_format, bool is_test, bool use_global_stats, bool trainable_statistics, Tensor* x_grad, Tensor* scale_grad, Tensor* grad_out_grad);

PADDLE_API void batch_norm_grad(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& mean_out, const paddle::optional<Tensor>& variance_out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_format, bool is_test, bool use_global_stats, bool trainable_statistics, Tensor* x_grad, Tensor* scale_grad, Tensor* bias_grad);

PADDLE_API void c_embedding_grad(const Tensor& weight, const Tensor& x, const Tensor& out_grad, int64_t start_index, Tensor* weight_grad);

PADDLE_API void channel_shuffle_grad(const Tensor& out_grad, int groups, const std::string& data_format, Tensor* x_grad);

PADDLE_API void conv2d_transpose_double_grad(const Tensor& x, const Tensor& filter, const Tensor& grad_out, const Tensor& grad_x_grad, const Tensor& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, Tensor* x_grad, Tensor* filter_grad, Tensor* grad_out_grad);

PADDLE_API void conv2d_transpose_grad(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, Tensor* x_grad, Tensor* filter_grad);

PADDLE_API void deformable_conv_grad(const Tensor& x, const Tensor& offset, const Tensor& filter, const paddle::optional<Tensor>& mask, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, int deformable_groups, int groups, int im2col_step, Tensor* x_grad, Tensor* offset_grad, Tensor* filter_grad, Tensor* mask_grad);

PADDLE_API void depthwise_conv2d_transpose_grad(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, Tensor* x_grad, Tensor* filter_grad);

PADDLE_API void divide_double_grad(const Tensor& y, const Tensor& out, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, Tensor* y_grad, Tensor* out_grad, Tensor* grad_out_grad);

PADDLE_API void divide_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void dropout_grad(const Tensor& mask, const Tensor& out_grad, const Scalar& p, bool is_test, const std::string& mode, Tensor* x_grad);

PADDLE_API void einsum_grad(const std::vector<Tensor>& x_shape, const std::vector<Tensor>& inner_cache, const Tensor& out_grad, const std::string& equation, std::vector<Tensor*> x_grad);

PADDLE_API void elementwise_pow_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void embedding_grad(const Tensor& x, const Tensor& weight, const Tensor& out_grad, int64_t padding_idx, bool sparse, Tensor* weight_grad);

PADDLE_API void frobenius_norm_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& axis, bool keep_dim, bool reduce_all, Tensor* x_grad);

PADDLE_API void fused_batch_norm_act_grad(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& act_type, Tensor* x_grad, Tensor* scale_grad, Tensor* bias_grad);

PADDLE_API void fused_bn_add_activation_grad(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& act_type, Tensor* x_grad, Tensor* z_grad, Tensor* scale_grad, Tensor* bias_grad);

PADDLE_API void fused_softmax_mask_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void fused_softmax_mask_upper_triangle_grad(const Tensor& Out, const Tensor& Out_grad, Tensor* X_grad);

PADDLE_API void hardswish_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void hsigmoid_loss_grad(const Tensor& x, const Tensor& w, const Tensor& label, const paddle::optional<Tensor>& path, const paddle::optional<Tensor>& code, const paddle::optional<Tensor>& bias, const Tensor& pre_out, const Tensor& out_grad, int num_classes, bool is_sparse, Tensor* x_grad, Tensor* w_grad, Tensor* bias_grad);

PADDLE_API void logsumexp_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all, Tensor* x_grad);

PADDLE_API void matmul_double_grad(const Tensor& x, const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, bool transpose_x, bool transpose_y, Tensor* x_grad, Tensor* y_grad, Tensor* grad_out_grad);

PADDLE_API void matmul_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, bool transpose_x, bool transpose_y, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void max_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all, Tensor* x_grad);

PADDLE_API void maximum_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void mean_grad(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all, Tensor* x_grad);

PADDLE_API void min_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all, Tensor* x_grad);

PADDLE_API void minimum_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void mish_grad(const Tensor& x, const Tensor& out_grad, float lambda, Tensor* x_grad);

PADDLE_API void multiply_double_grad(const Tensor& x, const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, Tensor* x_grad, Tensor* y_grad, Tensor* grad_out_grad);

PADDLE_API void multiply_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void multiply_triple_grad(const Tensor& x, const Tensor& y, const Tensor& fwd_grad_out, const paddle::optional<Tensor>& fwd_grad_grad_x, const paddle::optional<Tensor>& fwd_grad_grad_y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, const paddle::optional<Tensor>& grad_grad_out_grad, int axis, Tensor* x_grad, Tensor* y_grad, Tensor* fwd_grad_out_grad, Tensor* fwd_grad_grad_x_grad, Tensor* fwd_grad_grad_y_grad);

PADDLE_API void norm_grad(const Tensor& x, const Tensor& norm, const Tensor& out_grad, int axis, float epsilon, bool is_test, Tensor* x_grad);

PADDLE_API void pad_double_grad(const Tensor& grad_x_grad, const std::vector<int>& paddings, const Scalar& pad_value, Tensor* grad_out_grad);

PADDLE_API void pad_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int>& paddings, const Scalar& pad_value, Tensor* x_grad);

PADDLE_API void pool2d_double_grad(const Tensor& x, const Tensor& grad_x_grad, const IntArray& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm, Tensor* grad_out_grad);

PADDLE_API void pool2d_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm, Tensor* x_grad);

PADDLE_API void pool3d_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm, Tensor* x_grad);

PADDLE_API void prod_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& dims, bool keep_dim, bool reduce_all, Tensor* x_grad);

PADDLE_API void repeat_interleave_grad(const Tensor& x, const Tensor& out_grad, int repeats, int axis, Tensor* x_grad);

PADDLE_API void repeat_interleave_with_tensor_index_grad(const Tensor& x, const Tensor& repeats, const Tensor& out_grad, int axis, Tensor* x_grad);

PADDLE_API void reshape_double_grad(const Tensor& grad_out, const Tensor& grad_x_grad, Tensor* grad_out_grad);

PADDLE_API void reshape_grad(const Tensor& xshape, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void rnn_grad(const Tensor& x, const std::vector<Tensor>& pre_state, const std::vector<Tensor>& weight_list, const paddle::optional<Tensor>& sequence_length, const Tensor& out, const Tensor& dropout_state_out, const Tensor& reserve, const Tensor& out_grad, const std::vector<Tensor>& state_grad, float dropout_prob, bool is_bidirec, int input_size, int hidden_size, int num_layers, const std::string& mode, int seed, bool is_test, Tensor* x_grad, std::vector<Tensor*> pre_state_grad, std::vector<Tensor*> weight_list_grad);

PADDLE_API void rrelu_grad(const Tensor& x, const Tensor& noise, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void set_value_grad(const Tensor& out_grad, const IntArray& starts, const IntArray& ends, const IntArray& steps, const std::vector<int64_t>& axes, const std::vector<int64_t>& decrease_axes, const std::vector<int64_t>& none_axes, Tensor* x_grad);

PADDLE_API void set_value_with_tensor_grad(const Tensor& values, const Tensor& out_grad, const IntArray& starts, const IntArray& ends, const IntArray& steps, const std::vector<int64_t>& axes, const std::vector<int64_t>& decrease_axes, const std::vector<int64_t>& none_axes, Tensor* x_grad, Tensor* values_grad);

PADDLE_API void slice_grad(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& axes, const IntArray& starts, const IntArray& ends, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis, Tensor* input_grad);

PADDLE_API void softmax_grad(const Tensor& out, const Tensor& out_grad, int axis, Tensor* x_grad);

PADDLE_API void strided_slice_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int>& axes, const IntArray& starts, const IntArray& ends, const IntArray& strides, Tensor* x_grad);

PADDLE_API void subtract_double_grad(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, Tensor* grad_out_grad);

PADDLE_API void subtract_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, Tensor* x_grad, Tensor* y_grad);

PADDLE_API void sum_grad(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all, Tensor* x_grad);

PADDLE_API void swish_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad);

PADDLE_API void sync_batch_norm_grad(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_format, bool is_test, bool use_global_stats, bool trainable_statistics, Tensor* x_grad, Tensor* scale_grad, Tensor* bias_grad);

PADDLE_API void tile_grad(const Tensor& x, const Tensor& out_grad, const IntArray& repeat_times, Tensor* x_grad);

PADDLE_API void trans_layout_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int>& perm, Tensor* x_grad);

PADDLE_API void transpose_grad(const Tensor& out_grad, const std::vector<int>& perm, Tensor* x_grad);

PADDLE_API void tril_grad(const Tensor& out_grad, int diagonal, Tensor* x_grad);

PADDLE_API void triu_grad(const Tensor& out_grad, int diagonal, Tensor* x_grad);

PADDLE_API void disable_check_model_nan_inf_grad(const Tensor& out_grad, int unsetflag, Tensor* x_grad);

PADDLE_API void enable_check_model_nan_inf_grad(const Tensor& out_grad, int unsetflag, Tensor* x_grad);

PADDLE_API void fused_gemm_epilogue_grad(const Tensor& x, const Tensor& y, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, bool trans_x, bool trans_y, const std::string& activation, Tensor* x_grad, Tensor* y_grad, Tensor* bias_grad);

PADDLE_API void maximum_double_grad(const Tensor& x, const Tensor& y, const Tensor& grad_x_grad, const Tensor& grad_y_grad, Tensor* grad_out_grad);

PADDLE_API void minimum_double_grad(const Tensor& x, const Tensor& y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, Tensor* grad_out_grad);

PADDLE_API void unpool_grad(const Tensor& x, const Tensor& indices, const Tensor& out, const Tensor& out_grad, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const IntArray& output_size, const std::string& data_format, Tensor* x_grad);


}  // namespace experimental
}  // namespace paddle
