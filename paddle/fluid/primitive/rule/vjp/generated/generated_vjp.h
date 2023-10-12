// Auto Generated, DO NOT EDIT!

#pragma once

#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/pir/core/value.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"

namespace paddle {
namespace primitive {

using IntArray = paddle::experimental::IntArray;

std::vector<std::vector<paddle::Tensor>> abs_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> acos_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> acosh_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> addmm_vjp(const Tensor& input, const Tensor& x, const Tensor& y, const Tensor& out_grad, float alpha, float beta, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> affine_grid_vjp(const Tensor& input, const Tensor& output_grad, const Tensor& output_shape_, bool align_corners, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> angle_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> argsort_vjp(const Tensor& indices, const Tensor& x, const Tensor& out_grad, int axis, bool descending, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> as_complex_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> as_real_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> as_strided_vjp(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& dims, const std::vector<int64_t>& stride, int64_t offset, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> asin_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> asinh_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> atan_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> atan2_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> atanh_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> bce_loss_vjp(const Tensor& input, const Tensor& label, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> bicubic_interp_vjp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> bilinear_vjp(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> bilinear_interp_vjp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> bmm_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> broadcast_tensors_vjp(const std::vector<Tensor>& input, const std::vector<Tensor>& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> ceil_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> celu_vjp(const Tensor& x, const Tensor& out_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cholesky_vjp(const Tensor& out, const Tensor& out_grad, bool upper, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cholesky_solve_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, bool upper, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> clip_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& min_, const Tensor& max_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> complex_vjp(const Tensor& real, const Tensor& imag, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> concat_vjp(const std::vector<Tensor>& x, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> conj_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> conv2d_vjp(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, const std::vector<int>& dilations, int groups, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> conv3d_vjp(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> conv3d_transpose_vjp(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::vector<int>& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cos_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cosh_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> crop_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& offsets_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cross_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cross_entropy_with_softmax_vjp(const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cummax_vjp(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, int dtype, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cummin_vjp(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, int dtype, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cumprod_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, int dim, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cumsum_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& axis_, bool flatten, bool exclusive, bool reverse, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> depthwise_conv2d_vjp(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> det_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> diag_vjp(const Tensor& x, const Tensor& out_grad, int offset, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> diagonal_vjp(const Tensor& x, const Tensor& out_grad, int offset, int axis1, int axis2, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> digamma_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> dist_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, float p, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> dot_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> eig_vjp(const Tensor& out_w, const Tensor& out_v, const Tensor& out_w_grad, const Tensor& out_v_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> eigh_vjp(const Tensor& out_w, const Tensor& out_v, const Tensor& out_w_grad, const Tensor& out_v_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> eigvalsh_vjp(const Tensor& eigenvectors, const Tensor& eigenvalues_grad, const std::string& uplo, bool is_test, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> elu_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> erf_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> erfinv_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> exp_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> expand_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& shape_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> expand_as_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<int>& target_shape, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> expm1_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fft_c2c_vjp(const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fft_c2r_vjp(const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, int64_t last_dim_size, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fft_r2c_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, bool onesided, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fill_vjp(const Tensor& out_grad, const Tensor& value_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fill_diagonal_vjp(const Tensor& out_grad, float value, int offset, bool wrap, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fill_diagonal_tensor_vjp(const Tensor& out_grad, int64_t offset, int dim1, int dim2, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> flash_attn_vjp(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& softmax_lse, const Tensor& seed_offset, const paddle::optional<Tensor>& attn_mask, const Tensor& out_grad, float dropout, bool causal, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> flash_attn_unpadded_vjp(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, const Tensor& out, const Tensor& softmax_lse, const Tensor& seed_offset, const paddle::optional<Tensor>& attn_mask, const Tensor& out_grad, int64_t max_seqlen_q, int64_t max_seqlen_k, float scale, float dropout, bool causal, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> flatten_vjp(const Tensor& xshape, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> flip_vjp(const Tensor& out_grad, const std::vector<int>& axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> floor_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fmax_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fmin_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fold_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<int>& output_sizes, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> frame_vjp(const Tensor& x, const Tensor& out_grad, int frame_length, int hop_length, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> gather_vjp(const Tensor& x, const Tensor& index, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> gather_nd_vjp(const Tensor& x, const Tensor& index, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> gaussian_inplace_vjp(const Tensor& out_grad, float mean, float std, int seed, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> gelu_vjp(const Tensor& x, const Tensor& out_grad, bool approximate, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> grid_sample_vjp(const Tensor& x, const Tensor& grid, const Tensor& out_grad, const std::string& mode, const std::string& padding_mode, bool align_corners, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> group_norm_vjp(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& y, const Tensor& mean, const Tensor& variance, const Tensor& y_grad, float epsilon, int groups, const std::string& data_layout, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> gumbel_softmax_vjp(const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> hardshrink_vjp(const Tensor& x, const Tensor& out_grad, float threshold, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> hardsigmoid_vjp(const Tensor& out, const Tensor& out_grad, float slope, float offset, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> hardtanh_vjp(const Tensor& x, const Tensor& out_grad, float t_min, float t_max, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> heaviside_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> huber_loss_vjp(const Tensor& residual, const Tensor& out_grad, float delta, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> i0_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> i0e_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> i1_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> i1e_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> imag_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> index_add_vjp(const Tensor& index, const Tensor& add_value, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> index_put_vjp(const Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, const Tensor& out_grad, bool accumulate, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> index_sample_vjp(const Tensor& x, const Tensor& index, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> index_select_vjp(const Tensor& x, const Tensor& index, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> index_select_strided_vjp(const Tensor& x, const Tensor& out_grad, int64_t index, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> instance_norm_vjp(const Tensor& x, const paddle::optional<Tensor>& scale, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& y_grad, float epsilon, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> inverse_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> kldiv_loss_vjp(const Tensor& x, const Tensor& label, const Tensor& out_grad, const std::string& reduction, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> kron_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> kthvalue_vjp(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int k, int axis, bool keepdim, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> label_smooth_vjp(const Tensor& out_grad, float epsilon, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> layer_norm_vjp(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& mean, const Tensor& variance, const Tensor& out_grad, float epsilon, int begin_norm_axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> leaky_relu_vjp(const Tensor& x, const Tensor& out_grad, float negative_slope, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> lerp_vjp(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> lgamma_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> linear_interp_vjp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> log_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> log10_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> log1p_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> log2_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> log_loss_vjp(const Tensor& input, const Tensor& label, const Tensor& out_grad, float epsilon, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> log_softmax_vjp(const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> logcumsumexp_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, int axis, bool flatten, bool exclusive, bool reverse, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> logit_vjp(const Tensor& x, const Tensor& out_grad, float eps, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> logsigmoid_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> lu_vjp(const Tensor& x, const Tensor& out, const Tensor& pivots, const Tensor& out_grad, bool pivot, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> lu_unpack_vjp(const Tensor& x, const Tensor& y, const Tensor& l, const Tensor& u, const Tensor& pmat, const Tensor& l_grad, const Tensor& u_grad, bool unpack_ludata, bool unpack_pivots, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> margin_cross_entropy_vjp(const Tensor& logits, const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool return_softmax, int ring_id, int rank, int nranks, float margin1, float margin2, float margin3, float scale, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> masked_select_vjp(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> matrix_power_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, int n, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> max_pool2d_with_index_vjp(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> max_pool3d_with_index_vjp(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> maxout_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, int groups, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> mean_all_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> memory_efficient_attention_vjp(const Tensor& query, const Tensor& key, const Tensor& value, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& cu_seqlens_q, const paddle::optional<Tensor>& cu_seqlens_k, const Tensor& output, const Tensor& logsumexp, const Tensor& seed_and_offset, const Tensor& output_grad, const Scalar& max_seqlen_q, const Scalar& max_seqlen_k, bool causal, double dropout_p, float scale, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> meshgrid_vjp(const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> mode_vjp(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, bool keepdim, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> multi_dot_vjp(const std::vector<Tensor>& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> multiplex_vjp(const std::vector<Tensor>& inputs, const Tensor& index, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> mv_vjp(const Tensor& x, const Tensor& vec, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> nanmedian_vjp(const Tensor& x, const Tensor& medians, const Tensor& out_grad, const IntArray& axis, bool keepdim, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> nearest_interp_vjp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> nll_loss_vjp(const Tensor& input, const Tensor& label, const paddle::optional<Tensor>& weight, const Tensor& total_weight, const Tensor& out_grad, int64_t ignore_index, const std::string& reduction, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> overlap_add_vjp(const Tensor& x, const Tensor& out_grad, int hop_length, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> p_norm_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, float porder, int axis, float epsilon, bool keepdim, bool asvector, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pad3d_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& paddings_, const std::string& mode, float pad_value, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pixel_shuffle_vjp(const Tensor& out_grad, int upscale_factor, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pixel_unshuffle_vjp(const Tensor& out_grad, int downscale_factor, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> poisson_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> polygamma_vjp(const Tensor& x, const Tensor& out_grad, int n, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pow_vjp(const Tensor& x, const Tensor& out_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> prelu_vjp(const Tensor& x, const Tensor& alpha, const Tensor& out_grad, const std::string& data_format, const std::string& mode, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> psroi_pool_vjp(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& out_grad, int pooled_height, int pooled_width, int output_channels, float spatial_scale, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> put_along_axis_vjp(const Tensor& arr, const Tensor& indices, const Tensor& out_grad, int axis, const std::string& reduce, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> qr_vjp(const Tensor& x, const Tensor& q, const Tensor& r, const Tensor& q_grad, const Tensor& r_grad, const std::string& mode, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> real_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> reciprocal_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> relu_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> relu6_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> renorm_vjp(const Tensor& x, const Tensor& out_grad, float p, int axis, float max_norm, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> reverse_vjp(const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> roi_align_vjp(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& out_grad, int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> roi_pool_vjp(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& arg_max, const Tensor& out_grad, int pooled_height, int pooled_width, float spatial_scale, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> roll_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& shifts_, const std::vector<int64_t>& axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> round_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> rsqrt_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> scale_vjp(const Tensor& out_grad, const Tensor& scale_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> scatter_vjp(const Tensor& index, const Tensor& updates, const Tensor& out_grad, bool overwrite, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> scatter_nd_add_vjp(const Tensor& index, const Tensor& updates, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> segment_pool_vjp(const Tensor& x, const Tensor& segment_ids, const Tensor& out, const paddle::optional<Tensor>& summed_ids, const Tensor& out_grad, const std::string& pooltype, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> selu_vjp(const Tensor& out, const Tensor& out_grad, float scale, float alpha, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> send_u_recv_vjp(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const paddle::optional<Tensor>& out, const paddle::optional<Tensor>& dst_count, const Tensor& out_grad, const std::string& reduce_op, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> send_ue_recv_vjp(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const paddle::optional<Tensor>& out, const paddle::optional<Tensor>& dst_count, const Tensor& out_grad, const std::string& message_op, const std::string& reduce_op, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> send_uv_vjp(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const Tensor& out_grad, const std::string& message_op, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sigmoid_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sigmoid_cross_entropy_with_logits_vjp(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, const Tensor& out_grad, bool normalize, int ignore_index, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sign_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> silu_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sin_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sinh_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> slogdet_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> softplus_vjp(const Tensor& x, const Tensor& out_grad, float beta, float threshold, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> softshrink_vjp(const Tensor& x, const Tensor& out_grad, float threshold, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> softsign_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> solve_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> spectral_norm_vjp(const Tensor& weight, const Tensor& u, const Tensor& v, const Tensor& out_grad, int dim, int power_iters, float eps, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sqrt_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> square_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> squared_l2_norm_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> squeeze_vjp(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> stack_vjp(const std::vector<Tensor>& x, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> stanh_vjp(const Tensor& x, const Tensor& out_grad, float scale_a, float scale_b, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> svd_vjp(const Tensor& x, const Tensor& u, const Tensor& vh, const Tensor& s, const paddle::optional<Tensor>& u_grad, const paddle::optional<Tensor>& vh_grad, const paddle::optional<Tensor>& s_grad, bool full_matrices, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> take_along_axis_vjp(const Tensor& arr, const Tensor& indices, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tan_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tanh_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tanh_shrink_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> temporal_shift_vjp(const Tensor& out_grad, int seg_num, float shift_ratio, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tensor_unfold_vjp(const Tensor& input, const Tensor& out_grad, int64_t axis, int64_t size, int64_t step, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> thresholded_relu_vjp(const Tensor& x, const Tensor& out_grad, float threshold, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> topk_vjp(const Tensor& x, const Tensor& indices, const Tensor& out_grad, const Tensor& k_, int axis, bool largest, bool sorted, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> trace_vjp(const Tensor& x, const Tensor& out_grad, int offset, int axis1, int axis2, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> triangular_solve_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, bool upper, bool transpose, bool unitriangular, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> trilinear_interp_vjp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> trunc_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> unbind_vjp(const std::vector<Tensor>& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> unfold_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> uniform_inplace_vjp(const Tensor& out_grad, float min, float max, int seed, int diag_num, int diag_step, float diag_val, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> unpool3d_vjp(const Tensor& x, const Tensor& indices, const Tensor& out, const Tensor& out_grad, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_size, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> unsqueeze_vjp(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> unstack_vjp(const std::vector<Tensor>& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> view_dtype_vjp(const Tensor& input, const Tensor& out_grad, DataType dtype, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> view_shape_vjp(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& dims, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> warpctc_vjp(const Tensor& logits, const paddle::optional<Tensor>& logits_length, const Tensor& warpctcgrad, const Tensor& loss_grad, int blank, bool norm_by_times, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> warprnnt_vjp(const Tensor& input, const Tensor& input_lengths, const Tensor& warprnntgrad, const Tensor& loss_grad, int blank, float fastemit_lambda, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> weight_only_linear_vjp(const Tensor& x, const Tensor& weight, const paddle::optional<Tensor>& bias, const Tensor& weight_scale, const Tensor& out_grad, const std::string& weight_dtype, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> where_vjp(const Tensor& condition, const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> yolo_loss_vjp(const Tensor& x, const Tensor& gt_box, const Tensor& gt_label, const paddle::optional<Tensor>& gt_score, const Tensor& objectness_mask, const Tensor& gt_match_mask, const Tensor& loss_grad, const std::vector<int>& anchors, const std::vector<int>& anchor_mask, int class_num, float ignore_thresh, int downsample_ratio, bool use_label_smooth, float scale_x_y, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> add_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> amax_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> amin_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> assign_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> assign_out__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> batch_norm_vjp(const Tensor& x, const Tensor& scale, const Tensor& bias, const paddle::optional<Tensor>& mean_out, const paddle::optional<Tensor>& variance_out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> c_embedding_vjp(const Tensor& weight, const Tensor& x, const Tensor& out_grad, int64_t start_index, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cast_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> channel_shuffle_vjp(const Tensor& out_grad, int groups, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> conv2d_transpose_vjp(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> deformable_conv_vjp(const Tensor& x, const Tensor& offset, const Tensor& filter, const paddle::optional<Tensor>& mask, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, int deformable_groups, int groups, int im2col_step, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> depthwise_conv2d_transpose_vjp(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> disable_check_model_nan_inf_vjp(const Tensor& out_grad, int unsetflag, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> divide_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> dropout_vjp(const Tensor& mask, const Tensor& out_grad, const Scalar& p, bool is_test, const std::string& mode, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> einsum_vjp(const std::vector<Tensor>& x_shape, const std::vector<Tensor>& inner_cache, const Tensor& out_grad, const std::string& equation, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> elementwise_pow_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> embedding_vjp(const Tensor& x, const Tensor& weight, const Tensor& out_grad, int64_t padding_idx, bool sparse, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> enable_check_model_nan_inf_vjp(const Tensor& out_grad, int unsetflag, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> exponential__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> frobenius_norm_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keep_dim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fused_batch_norm_act_vjp(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& act_type, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fused_bn_add_activation_vjp(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& act_type, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fused_softmax_mask_upper_triangle_vjp(const Tensor& Out, const Tensor& Out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> hardswish_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> hsigmoid_loss_vjp(const Tensor& x, const Tensor& w, const Tensor& label, const paddle::optional<Tensor>& path, const paddle::optional<Tensor>& code, const paddle::optional<Tensor>& bias, const Tensor& pre_out, const Tensor& out_grad, int num_classes, bool is_sparse, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> logsumexp_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> matmul_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, bool transpose_x, bool transpose_y, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> max_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& axis_, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> maximum_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> mean_vjp(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> min_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& axis_, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> minimum_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> mish_vjp(const Tensor& x, const Tensor& out_grad, float threshold, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> multiply_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> norm_vjp(const Tensor& x, const Tensor& norm, const Tensor& out_grad, int axis, float epsilon, bool is_test, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pad_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<int>& paddings, const Scalar& pad_value, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pool2d_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& kernel_size_, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pool3d_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> prod_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& dims_, bool keep_dim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> repeat_interleave_vjp(const Tensor& x, const Tensor& out_grad, int repeats, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> repeat_interleave_with_tensor_index_vjp(const Tensor& x, const Tensor& repeats, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> rnn_vjp(const Tensor& x, const std::vector<Tensor>& pre_state, const std::vector<Tensor>& weight_list, const paddle::optional<Tensor>& sequence_length, const Tensor& out, const Tensor& dropout_state_out, const Tensor& reserve, const Tensor& out_grad, const std::vector<Tensor>& state_grad, float dropout_prob, bool is_bidirec, int input_size, int hidden_size, int num_layers, const std::string& mode, int seed, bool is_test, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> rrelu_vjp(const Tensor& x, const Tensor& noise, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> slice_vjp(const Tensor& input, const Tensor& out_grad, const Tensor& starts_, const Tensor& ends_, const std::vector<int64_t>& axes, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> softmax_vjp(const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> split_vjp(const std::vector<Tensor>& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> split_with_num_vjp(const std::vector<Tensor>& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> strided_slice_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& starts_, const Tensor& ends_, const Tensor& strides_, const std::vector<int>& axes, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> subtract_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sum_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& axis_, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> swish_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sync_batch_norm__vjp(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tile_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& repeat_times_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> trans_layout_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<int>& perm, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> transpose_vjp(const Tensor& out_grad, const std::vector<int>& perm, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tril_vjp(const Tensor& out_grad, int diagonal, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> triu_vjp(const Tensor& out_grad, int diagonal, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> unpool_vjp(const Tensor& x, const Tensor& indices, const Tensor& out, const Tensor& out_grad, const Tensor& output_size_, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> abs_grad_vjp(const Tensor& x, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> celu_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> clip_grad_vjp(const Tensor& x, const Tensor& grad_x_grad, const Tensor& min_, const Tensor& max_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> concat_grad_vjp(const std::vector<Tensor>& grad_x_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> conv2d_grad_vjp(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, const std::vector<int>& dilations, int groups, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> conv3d_grad_vjp(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cos_double_grad_vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cos_grad_vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> depthwise_conv2d_grad_vjp(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> elu_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> expand_grad_vjp(const Tensor& grad_x_grad, const Tensor& shape_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> instance_norm_grad_vjp(const Tensor& x, const paddle::optional<Tensor>& fwd_scale, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& grad_y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_scale_grad, const paddle::optional<Tensor>& grad_bias_grad, float epsilon, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> leaky_relu_grad_vjp(const Tensor& x, const Tensor& grad_x_grad, float negative_slope, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> log_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pad3d_grad_vjp(const Tensor& grad_x_grad, const Tensor& paddings_, const std::string& mode, float pad_value, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pow_double_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_grad_x, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_grad_out_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pow_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> relu_grad_vjp(const Tensor& out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> rsqrt_grad_vjp(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sigmoid_double_grad_vjp(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_grad_x, const Tensor& grad_out_grad, const paddle::optional<Tensor>& grad_grad_out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sigmoid_grad_vjp(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sin_double_grad_vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sin_grad_vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> softplus_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float beta, float threshold, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sqrt_grad_vjp(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> square_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> squeeze_grad_vjp(const Tensor& grad_x_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tanh_double_grad_vjp(const Tensor& out, const Tensor& grad_out_forward, const Tensor& grad_x_grad_forward, const paddle::optional<Tensor>& grad_out_new_grad, const paddle::optional<Tensor>& grad_out_grad_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tanh_grad_vjp(const Tensor& out, const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> unsqueeze_grad_vjp(const Tensor& grad_x_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> add_double_grad_vjp(const Tensor& grad_grad_x, const Tensor& grad_grad_y, const Tensor& grad_grad_out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> add_grad_vjp(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> batch_norm_grad_vjp(const Tensor& x, const Tensor& scale, const paddle::optional<Tensor>& out_mean, const paddle::optional<Tensor>& out_variance, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_scale_grad, const paddle::optional<Tensor>& grad_bias_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> conv2d_transpose_grad_vjp(const Tensor& x, const Tensor& filter, const Tensor& grad_out, const Tensor& grad_x_grad, const Tensor& grad_filter_grad, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> divide_grad_vjp(const Tensor& y, const Tensor& out, const Tensor& grad_x, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> matmul_grad_vjp(const Tensor& x, const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, bool transpose_x, bool transpose_y, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> mean_grad_vjp(const Tensor& grad_x_grad, const IntArray& axis, bool keepdim, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> multiply_double_grad_vjp(const Tensor& x, const Tensor& y, const Tensor& fwd_grad_out, const paddle::optional<Tensor>& fwd_grad_grad_x, const paddle::optional<Tensor>& fwd_grad_grad_y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, const paddle::optional<Tensor>& grad_grad_out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> multiply_grad_vjp(const Tensor& x, const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pad_grad_vjp(const Tensor& grad_x_grad, const std::vector<int>& paddings, const Scalar& pad_value, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pool2d_grad_vjp(const Tensor& x, const Tensor& grad_x_grad, const Tensor& kernel_size_, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> reshape_grad_vjp(const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> slice_grad_vjp(const Tensor& grad_input_grad, const Tensor& starts_, const Tensor& ends_, const std::vector<int64_t>& axes, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> subtract_grad_vjp(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sum_grad_vjp(const Tensor& grad_x_grad, const Tensor& axis_, bool keepdim, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tile_grad_vjp(const Tensor& grad_x_grad, const Tensor& repeat_times_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> transpose_grad_vjp(const Tensor& grad_x_grad, const std::vector<int>& perm, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> abs__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> acos__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> acosh__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> addmm__vjp(const Tensor& input, const Tensor& x, const Tensor& y, const Tensor& out_grad, float alpha, float beta, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> asin__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> asinh__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> atan__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> atanh__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> bce_loss__vjp(const Tensor& input, const Tensor& label, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> ceil__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> clip__vjp(const Tensor& x, const Tensor& out_grad, const Tensor& min_, const Tensor& max_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cos__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cosh__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cross_entropy_with_softmax__vjp(const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cumprod__vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, int dim, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cumsum__vjp(const Tensor& x, const Tensor& out_grad, const Tensor& axis_, bool flatten, bool exclusive, bool reverse, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> digamma__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> elu__vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> erf__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> erfinv__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> exp__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> expm1__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fill__vjp(const Tensor& out_grad, const Tensor& value_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fill_diagonal__vjp(const Tensor& out_grad, float value, int offset, bool wrap, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fill_diagonal_tensor__vjp(const Tensor& out_grad, int64_t offset, int dim1, int dim2, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> flatten__vjp(const Tensor& xshape, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> floor__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> gaussian_inplace__vjp(const Tensor& out_grad, float mean, float std, int seed, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> hardtanh__vjp(const Tensor& x, const Tensor& out_grad, float t_min, float t_max, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> i0__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> index_add__vjp(const Tensor& index, const Tensor& add_value, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> index_put__vjp(const Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, const Tensor& out_grad, bool accumulate, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> leaky_relu__vjp(const Tensor& x, const Tensor& out_grad, float negative_slope, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> lerp__vjp(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> lgamma__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> log__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> log10__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> log1p__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> log2__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> logit__vjp(const Tensor& x, const Tensor& out_grad, float eps, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> lu__vjp(const Tensor& x, const Tensor& out, const Tensor& pivots, const Tensor& out_grad, bool pivot, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> polygamma__vjp(const Tensor& x, const Tensor& out_grad, int n, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pow__vjp(const Tensor& x, const Tensor& out_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> put_along_axis__vjp(const Tensor& arr, const Tensor& indices, const Tensor& out_grad, int axis, const std::string& reduce, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> reciprocal__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> relu__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> renorm__vjp(const Tensor& x, const Tensor& out_grad, float p, int axis, float max_norm, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> round__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> rsqrt__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> scale__vjp(const Tensor& out_grad, const Tensor& scale_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> scatter__vjp(const Tensor& index, const Tensor& updates, const Tensor& out_grad, bool overwrite, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sigmoid__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sigmoid_cross_entropy_with_logits__vjp(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, const Tensor& out_grad, bool normalize, int ignore_index, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sin__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sinh__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sqrt__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> squeeze__vjp(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tan__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tanh__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> thresholded_relu__vjp(const Tensor& x, const Tensor& out_grad, float threshold, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> trunc__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> uniform_inplace__vjp(const Tensor& out_grad, float min, float max, int seed, int diag_num, int diag_step, float diag_val, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> unsqueeze__vjp(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> where__vjp(const Tensor& condition, const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> add__vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> assign__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cast__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> divide__vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> multiply__vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> softmax__vjp(const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> subtract__vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> transpose__vjp(const Tensor& out_grad, const std::vector<int>& perm, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tril__vjp(const Tensor& out_grad, int diagonal, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> triu__vjp(const Tensor& out_grad, int diagonal, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> celu_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> clip_grad__vjp(const Tensor& x, const Tensor& grad_x_grad, const Tensor& min_, const Tensor& max_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cos_double_grad__vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cos_grad__vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> elu_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> leaky_relu_grad__vjp(const Tensor& x, const Tensor& grad_x_grad, float negative_slope, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> log_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pow_double_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_grad_x, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_grad_out_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pow_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> relu_grad__vjp(const Tensor& out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> rsqrt_grad__vjp(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sigmoid_double_grad__vjp(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_grad_x, const Tensor& grad_out_grad, const paddle::optional<Tensor>& grad_grad_out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sigmoid_grad__vjp(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sin_double_grad__vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sin_grad__vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> softplus_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float beta, float threshold, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sqrt_grad__vjp(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> square_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> squeeze_grad__vjp(const Tensor& grad_x_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tanh_double_grad__vjp(const Tensor& out, const Tensor& grad_out_forward, const Tensor& grad_x_grad_forward, const paddle::optional<Tensor>& grad_out_new_grad, const paddle::optional<Tensor>& grad_out_grad_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tanh_grad__vjp(const Tensor& out, const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> unsqueeze_grad__vjp(const Tensor& grad_x_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> add_double_grad__vjp(const Tensor& grad_grad_x, const Tensor& grad_grad_y, const Tensor& grad_grad_out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> add_grad__vjp(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> multiply_double_grad__vjp(const Tensor& x, const Tensor& y, const Tensor& fwd_grad_out, const paddle::optional<Tensor>& fwd_grad_grad_x, const paddle::optional<Tensor>& fwd_grad_grad_y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, const paddle::optional<Tensor>& grad_grad_out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> reshape_grad__vjp(const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> subtract_grad__vjp(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


}  // namespace primitive
}  // namespace paddle
