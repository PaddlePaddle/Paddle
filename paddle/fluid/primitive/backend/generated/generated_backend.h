// Auto Generated, DO NOT EDIT!

#pragma once

#include <string>
#include <vector>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/utils/optional.h"


namespace paddle {
namespace primitive {
namespace backend {

using Tensor = paddle::Tensor;
using Scalar = phi::Scalar;
using IntArray = paddle::experimental::IntArray;
using DataType = phi::DataType;

template <typename T>
Tensor abs(const Tensor& x);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> accuracy(const Tensor& x, const Tensor& indices, const Tensor& label);

template <typename T>
Tensor acos(const Tensor& x);

template <typename T>
Tensor acosh(const Tensor& x);

template <typename T>
std::tuple<Tensor, Tensor, const paddle::optional<Tensor>> adagrad_(const Tensor& param, const Tensor& grad, const Tensor& moment, const Tensor& learning_rate, const paddle::optional<Tensor>& master_param, float epsilon = 1.0e-6f, bool multi_precision = false);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, const paddle::optional<Tensor>> adam_(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment1, const Tensor& moment2, const Tensor& beta1_pow, const Tensor& beta2_pow, const paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Tensor& beta1_, const Tensor& beta2_, const Tensor& epsilon_, bool lazy_mode = false, int64_t min_row_size_to_use_multithread = 1000, bool multi_precision = false, bool use_global_beta_pow = false);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, const paddle::optional<Tensor>> adam_(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment1, const Tensor& moment2, const Tensor& beta1_pow, const Tensor& beta2_pow, const paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Scalar& beta1 = 0.9f, const Scalar& beta2 = 0.999f, const Scalar& epsilon = 1.0e-8f, bool lazy_mode = false, int64_t min_row_size_to_use_multithread = 1000, bool multi_precision = false, bool use_global_beta_pow = false);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, const paddle::optional<Tensor>> adamax_(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment, const Tensor& inf_norm, const Tensor& beta1_pow, const paddle::optional<Tensor>& master_param, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1.0e-8f, bool multi_precision = false);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, const paddle::optional<Tensor>> adamw_(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment1, const Tensor& moment2, const Tensor& beta1_pow, const Tensor& beta2_pow, const paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Tensor& beta1_, const Tensor& beta2_, const Tensor& epsilon_, float lr_ratio = 1.0f, float coeff = 0.01f, bool with_decay = false, bool lazy_mode = false, int64_t min_row_size_to_use_multithread = 1000, bool multi_precision = false, bool use_global_beta_pow = false);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, const paddle::optional<Tensor>> adamw_(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment1, const Tensor& moment2, const Tensor& beta1_pow, const Tensor& beta2_pow, const paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Scalar& beta1 = 0.9f, const Scalar& beta2 = 0.999f, const Scalar& epsilon = 1.0e-8f, float lr_ratio = 1.0f, float coeff = 0.01f, bool with_decay = false, bool lazy_mode = false, int64_t min_row_size_to_use_multithread = 1000, bool multi_precision = false, bool use_global_beta_pow = false);

template <typename T>
Tensor addmm(const Tensor& input, const Tensor& x, const Tensor& y, float beta = 1.0, float alpha = 1.0);

template <typename T>
Tensor affine_grid(const Tensor& input, const Tensor& output_shape_, bool align_corners = true);

template <typename T>
Tensor affine_grid(const Tensor& input, const IntArray& output_shape = {}, bool align_corners = true);

template <typename T>
Tensor angle(const Tensor& x);

template <typename T>
Tensor argmax(const Tensor& x, const Tensor& axis_, bool keepdims = false, bool flatten = false, int dtype = 3);

template <typename T>
Tensor argmax(const Tensor& x, const Scalar& axis, bool keepdims = false, bool flatten = false, int dtype = 3);

template <typename T>
Tensor argmin(const Tensor& x, const Tensor& axis_, bool keepdims = false, bool flatten = false, int dtype = 3);

template <typename T>
Tensor argmin(const Tensor& x, const Scalar& axis, bool keepdims = false, bool flatten = false, int dtype = 3);

template <typename T>
std::tuple<Tensor, Tensor> argsort(const Tensor& x, int axis = -1, bool descending = false);

template <typename T>
Tensor as_complex(const Tensor& x);

template <typename T>
Tensor as_real(const Tensor& x);

template <typename T>
Tensor as_strided(const Tensor& input, const std::vector<int64_t>& dims = {}, const std::vector<int64_t>& stride = {}, int64_t offset = 0);

template <typename T>
Tensor asin(const Tensor& x);

template <typename T>
Tensor asinh(const Tensor& x);

template <typename T>
Tensor atan(const Tensor& x);

template <typename T>
Tensor atan2(const Tensor& x, const Tensor& y);

template <typename T>
Tensor atanh(const Tensor& x);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> auc(const Tensor& x, const Tensor& label, const Tensor& stat_pos, const Tensor& stat_neg, const paddle::optional<Tensor>& ins_tag_weight, const std::string& curve = "ROC", int num_thresholds = (2 << 12) - 1, int slide_steps = 1);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> average_accumulates_(const Tensor& param, const Tensor& in_sum_1, const Tensor& in_sum_2, const Tensor& in_sum_3, const Tensor& in_num_accumulates, const Tensor& in_old_num_accumulates, const Tensor& in_num_updates, float average_window = 0, int64_t max_average_window = INT64_MAX, int64_t min_average_window = 10000L);

template <typename T>
Tensor bce_loss(const Tensor& input, const Tensor& label);

template <typename T>
Tensor bernoulli(const Tensor& x);

template <typename T>
Tensor bicubic_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout = "NCHW", int out_d = 0, int out_h = 0, int out_w = 0, const std::vector<float>& scale = {}, const std::string& interp_method = "bilinear", bool align_corners = true, int align_mode = 1);

template <typename T>
Tensor bilinear(const Tensor& x, const Tensor& y, const Tensor& weight, const paddle::optional<Tensor>& bias);

template <typename T>
Tensor bilinear_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout = "NCHW", int out_d = 0, int out_h = 0, int out_w = 0, const std::vector<float>& scale = {}, const std::string& interp_method = "bilinear", bool align_corners = true, int align_mode = 1);

template <typename T>
Tensor bincount(const Tensor& x, const paddle::optional<Tensor>& weights, const Tensor& minlength_);

template <typename T>
Tensor bincount(const Tensor& x, const paddle::optional<Tensor>& weights, const Scalar& minlength = 0);

template <typename T>
Tensor bitwise_and(const Tensor& x, const Tensor& y);

template <typename T>
Tensor bitwise_not(const Tensor& x);

template <typename T>
Tensor bitwise_or(const Tensor& x, const Tensor& y);

template <typename T>
Tensor bitwise_xor(const Tensor& x, const Tensor& y);

template <typename T>
Tensor bmm(const Tensor& x, const Tensor& y);

template <typename T>
Tensor box_coder(const Tensor& prior_box, const paddle::optional<Tensor>& prior_box_var, const Tensor& target_box, const std::string& code_type = "encode_center_size", bool box_normalized = true, int axis = 0, const std::vector<float>& variance = {});

template <typename T>
std::vector<Tensor> broadcast_tensors(const std::vector<Tensor>& input);

template <typename T>
Tensor ceil(const Tensor& x);

template <typename T>
Tensor celu(const Tensor& x, float alpha = 1.0);

template <typename T>
std::tuple<std::vector<Tensor>, Tensor> check_finite_and_unscale_(const std::vector<Tensor>& x, const Tensor& scale);

template <typename T>
std::tuple<Tensor, Tensor> check_numerics(const Tensor& tensor, const std::string& op_type = "", const std::string& var_name = "", int check_nan_inf_level = 0, int stack_height_limit = -1, const std::string& output_dir = "");

template <typename T>
Tensor cholesky(const Tensor& x, bool upper = false);

template <typename T>
Tensor cholesky_solve(const Tensor& x, const Tensor& y, bool upper = false);

template <typename T>
std::tuple<Tensor, Tensor> class_center_sample(const Tensor& label, int num_classes, int num_samples, int ring_id = 0, int rank = 0, int nranks = 1, bool fix_seed = false, int seed = 0);

template <typename T>
Tensor clip(const Tensor& x, const Tensor& min_, const Tensor& max_);

template <typename T>
Tensor clip(const Tensor& x, const Scalar& min, const Scalar& max);

template <typename T>
Tensor clip_by_norm(const Tensor& x, float max_norm);

template <typename T>
std::tuple<std::vector<Tensor>, Tensor> coalesce_tensor(const std::vector<Tensor>& input, DataType dtype, bool copy_data = false, bool set_constant = false, bool persist_output = false, float constant = 0.0, bool use_align = true, int align_size = -1, int size_of_dtype = -1, const std::vector<int64_t>& concated_shapes = {}, const std::vector<int64_t>& concated_ranks = {});

template <typename T>
Tensor complex(const Tensor& real, const Tensor& imag);

template <typename T>
Tensor concat(const std::vector<Tensor>& x, const Tensor& axis_);

template <typename T>
Tensor concat(const std::vector<Tensor>& x, const Scalar& axis = 0);

template <typename T>
Tensor conj(const Tensor& x);

template <typename T>
Tensor conv2d(const Tensor& input, const Tensor& filter, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, const std::string& padding_algorithm = "EXPLICIT", const std::vector<int>& dilations = {1, 1}, int groups = 1, const std::string& data_format = "NCHW");

template <typename T>
Tensor conv3d(const Tensor& input, const Tensor& filter, const std::vector<int>& strides = {1, 1, 1}, const std::vector<int>& paddings = {0, 0, 0}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1, 1}, const std::string& data_format = "NCDHW");

template <typename T>
Tensor conv3d_transpose(const Tensor& x, const Tensor& filter, const std::vector<int>& strides = {1, 1, 1}, const std::vector<int>& paddings = {0, 0, 0}, const std::vector<int>& output_padding = {}, const std::vector<int>& output_size = {}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1, 1}, const std::string& data_format = "NCHW");

template <typename T>
Tensor cos(const Tensor& x);

template <typename T>
Tensor cosh(const Tensor& x);

template <typename T>
Tensor crop(const Tensor& x, const Tensor& shape_, const Tensor& offsets_);

template <typename T>
Tensor crop(const Tensor& x, const IntArray& shape = {}, const IntArray& offsets = {});

template <typename T>
Tensor cross(const Tensor& x, const Tensor& y, int axis = 9);

template <typename T>
std::tuple<Tensor, Tensor> cross_entropy_with_softmax(const Tensor& input, const Tensor& label, bool soft_label = false, bool use_softmax = true, bool numeric_stable_mode = true, int ignore_index = -100, int axis = -1);

template <typename T>
std::tuple<Tensor, Tensor> cummax(const Tensor& x, int axis = -1, int dtype = 3);

template <typename T>
std::tuple<Tensor, Tensor> cummin(const Tensor& x, int axis = -1, int dtype = 3);

template <typename T>
Tensor cumprod(const Tensor& x, int dim);

template <typename T>
Tensor cumsum(const Tensor& x, const Tensor& axis_, bool flatten = false, bool exclusive = false, bool reverse = false);

template <typename T>
Tensor cumsum(const Tensor& x, const Scalar& axis = -1, bool flatten = false, bool exclusive = false, bool reverse = false);

template <typename T>
Tensor data(const std::string& name, const IntArray& shape, DataType dtype, Place place);

template <typename T>
Tensor depthwise_conv2d(const Tensor& input, const Tensor& filter, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1}, const std::string& data_format = "NCHW");

template <typename T>
Tensor det(const Tensor& x);

template <typename T>
Tensor diag(const Tensor& x, int offset = 0, float padding_value = 0.0);

template <typename T>
Tensor diag_embed(const Tensor& input, int offset = 0, int dim1 = -2, int dim2 = -1);

template <typename T>
Tensor diagonal(const Tensor& x, int offset = 0, int axis1 = 0, int axis2 = 1);

template <typename T>
Tensor digamma(const Tensor& x);

template <typename T>
Tensor dirichlet(const Tensor& alpha);

template <typename T>
Tensor dist(const Tensor& x, const Tensor& y, float p = 2.0);

template <typename T>
Tensor dot(const Tensor& x, const Tensor& y);

template <typename T>
std::tuple<Tensor, Tensor> edit_distance(const Tensor& hyps, const Tensor& refs, const paddle::optional<Tensor>& hypslength, const paddle::optional<Tensor>& refslength, bool normalized = false);

template <typename T>
std::tuple<Tensor, Tensor> eig(const Tensor& x);

template <typename T>
std::tuple<Tensor, Tensor> eigh(const Tensor& x, const std::string& UPLO = "L");

template <typename T>
Tensor eigvals(const Tensor& x);

template <typename T>
std::tuple<Tensor, Tensor> eigvalsh(const Tensor& x, const std::string& uplo = "L", bool is_test = false);

template <typename T>
Tensor elu(const Tensor& x, float alpha = 1.0f);

template <typename T>
Tensor equal_all(const Tensor& x, const Tensor& y);

template <typename T>
Tensor erf(const Tensor& x);

template <typename T>
Tensor erfinv(const Tensor& x);

template <typename T>
Tensor exp(const Tensor& x);

template <typename T>
Tensor expand(const Tensor& x, const Tensor& shape_);

template <typename T>
Tensor expand(const Tensor& x, const IntArray& shape = {});

template <typename T>
Tensor expand_as(const Tensor& x, const paddle::optional<Tensor>& y, const std::vector<int>& target_shape = {});

template <typename T>
Tensor expm1(const Tensor& x);

template <typename T>
Tensor fft_c2c(const Tensor& x, const std::vector<int64_t>& axes, const std::string& normalization, bool forward);

template <typename T>
Tensor fft_c2r(const Tensor& x, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, int64_t last_dim_size = 0L);

template <typename T>
Tensor fft_r2c(const Tensor& x, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, bool onesided);

template <typename T>
Tensor fill(const Tensor& x, const Tensor& value_);

template <typename T>
Tensor fill(const Tensor& x, const Scalar& value = 0);

template <typename T>
Tensor fill_diagonal(const Tensor& x, float value = 0, int offset = 0, bool wrap = false);

template <typename T>
Tensor fill_diagonal_tensor(const Tensor& x, const Tensor& y, int64_t offset = 0, int dim1 = 0, int dim2 = 1);

template <typename T>
std::tuple<Tensor, Tensor> flash_attn(const Tensor& q, const Tensor& k, const Tensor& v, const paddle::optional<Tensor>& fixed_seed_offset, const paddle::optional<Tensor>& attn_mask, float dropout = 0.0, bool causal = false, bool return_softmax = false, bool is_test = false, const std::string& rng_name = "");

template <typename T>
std::tuple<Tensor, Tensor> flash_attn_unpadded(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, const paddle::optional<Tensor>& fixed_seed_offset, const paddle::optional<Tensor>& attn_mask, int64_t max_seqlen_q, int64_t max_seqlen_k, float scale, float dropout = 0.0, bool causal = false, bool return_softmax = false, bool is_test = false, const std::string& rng_name = "");

template <typename T>
Tensor flatten(const Tensor& x, int start_axis = 1, int stop_axis = 1);

template <typename T>
Tensor flip(const Tensor& x, const std::vector<int>& axis);

template <typename T>
Tensor floor(const Tensor& x);

template <typename T>
Tensor fmax(const Tensor& x, const Tensor& y);

template <typename T>
Tensor fmin(const Tensor& x, const Tensor& y);

template <typename T>
Tensor fold(const Tensor& x, const std::vector<int>& output_sizes, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations);

template <typename T>
Tensor frame(const Tensor& x, int frame_length, int hop_length, int axis = -1);

template <typename T>
Tensor full_int_array(const IntArray& value, DataType dtype = DataType::FLOAT32, Place place = CPUPlace());

template <typename T>
Tensor gather(const Tensor& x, const Tensor& index, const Tensor& axis_);

template <typename T>
Tensor gather(const Tensor& x, const Tensor& index, const Scalar& axis = 0);

template <typename T>
Tensor gather_nd(const Tensor& x, const Tensor& index);

template <typename T>
Tensor gather_tree(const Tensor& ids, const Tensor& parents);

template <typename T>
Tensor gaussian_inplace(const Tensor& x, float mean = 0, float std = 1.0, int seed = 0);

template <typename T>
Tensor gelu(const Tensor& x, bool approximate = false);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> generate_proposals(const Tensor& scores, const Tensor& bbox_deltas, const Tensor& im_shape, const Tensor& anchors, const Tensor& variances, int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size, float eta, bool pixel_offset = true);

template <typename T>
Tensor grid_sample(const Tensor& x, const Tensor& grid, const std::string& mode = "bilinear", const std::string& padding_mode = "zeros", bool align_corners = true);

template <typename T>
Tensor group_norm(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon = 1e-5, int groups = -1, const std::string& data_layout = "NCHW");

template <typename T>
Tensor gumbel_softmax(const Tensor& x, float temperature = 1.0, bool hard = false, int axis = -1);

template <typename T>
Tensor hardshrink(const Tensor& x, float threshold = 0.5);

template <typename T>
Tensor hardsigmoid(const Tensor& x, float slope = 0.2, float offset = 0.5);

template <typename T>
Tensor hardtanh(const Tensor& x, float t_min = 0, float t_max = 24);

template <typename T>
Tensor heaviside(const Tensor& x, const Tensor& y);

template <typename T>
Tensor histogram(const Tensor& input, int64_t bins = 100, int min = 0, int max = 0);

template <typename T>
Tensor huber_loss(const Tensor& input, const Tensor& label, float delta);

template <typename T>
Tensor i0(const Tensor& x);

template <typename T>
Tensor i0e(const Tensor& x);

template <typename T>
Tensor i1(const Tensor& x);

template <typename T>
Tensor i1e(const Tensor& x);

template <typename T>
Tensor imag(const Tensor& x);

template <typename T>
Tensor index_add(const Tensor& x, const Tensor& index, const Tensor& add_value, int axis = 0);

template <typename T>
Tensor index_put(const Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, bool accumulate = false);

template <typename T>
Tensor index_sample(const Tensor& x, const Tensor& index);

template <typename T>
Tensor index_select(const Tensor& x, const Tensor& index, int axis = 0);

template <typename T>
Tensor index_select_strided(const Tensor& x, int64_t index, int axis = 0);

template <typename T>
Tensor instance_norm(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon = 1e-5);

template <typename T>
Tensor inverse(const Tensor& x);

template <typename T>
Tensor is_empty(const Tensor& x);

template <typename T>
Tensor isfinite(const Tensor& x);

template <typename T>
Tensor isinf(const Tensor& x);

template <typename T>
Tensor isnan(const Tensor& x);

template <typename T>
Tensor kldiv_loss(const Tensor& x, const Tensor& label, const std::string& reduction = "mean");

template <typename T>
Tensor kron(const Tensor& x, const Tensor& y);

template <typename T>
std::tuple<Tensor, Tensor> kthvalue(const Tensor& x, int k = 1, int axis = -1, bool keepdim = false);

template <typename T>
Tensor label_smooth(const Tensor& label, const paddle::optional<Tensor>& prior_dist, float epsilon = 0.0f);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, const paddle::optional<Tensor>> lamb_(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment1, const Tensor& moment2, const Tensor& beta1_pow, const Tensor& beta2_pow, const paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, float weight_decay, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1.0e-6f, bool always_adapt = false, bool multi_precision = false);

template <typename T>
Tensor layer_norm(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon = 1e-5, int begin_norm_axis = 1);

template <typename T>
Tensor leaky_relu(const Tensor& x, float negative_slope = 0.02f);

template <typename T>
Tensor lerp(const Tensor& x, const Tensor& y, const Tensor& weight);

template <typename T>
Tensor lgamma(const Tensor& x);

template <typename T>
Tensor linear_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout = "NCHW", int out_d = 0, int out_h = 0, int out_w = 0, const std::vector<float>& scale = {}, const std::string& interp_method = "bilinear", bool align_corners = true, int align_mode = 1);

template <typename T>
Tensor llm_int8_linear(const Tensor& x, const Tensor& weight, const paddle::optional<Tensor>& bias, const Tensor& weight_scale, float threshold = 6.0);

template <typename T>
Tensor log(const Tensor& x);

template <typename T>
Tensor log10(const Tensor& x);

template <typename T>
Tensor log1p(const Tensor& x);

template <typename T>
Tensor log2(const Tensor& x);

template <typename T>
Tensor log_loss(const Tensor& input, const Tensor& label, float epsilon);

template <typename T>
Tensor log_softmax(const Tensor& x, int axis = -1);

template <typename T>
Tensor logcumsumexp(const Tensor& x, int axis = -1, bool flatten = false, bool exclusive = false, bool reverse = false);

template <typename T>
Tensor logical_and(const Tensor& x, const Tensor& y);

template <typename T>
Tensor logical_not(const Tensor& x);

template <typename T>
Tensor logical_or(const Tensor& x, const Tensor& y);

template <typename T>
Tensor logical_xor(const Tensor& x, const Tensor& y);

template <typename T>
Tensor logit(const Tensor& x, float eps = 1e-6f);

template <typename T>
Tensor logsigmoid(const Tensor& x);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq(const Tensor& x, const Tensor& y, const Tensor& rcond_, const std::string& driver = "gels");

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq(const Tensor& x, const Tensor& y, const Scalar& rcond = 0.0f, const std::string& driver = "gels");

template <typename T>
std::tuple<Tensor, Tensor, Tensor> lu(const Tensor& x, bool pivot = true);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> lu_unpack(const Tensor& x, const Tensor& y, bool unpack_ludata = true, bool unpack_pivots = true);

template <typename T>
std::tuple<Tensor, Tensor> margin_cross_entropy(const Tensor& logits, const Tensor& label, bool return_softmax = false, int ring_id = 0, int rank = 0, int nranks = 1, float margin1 = 1.0f, float margin2 = 0.5f, float margin3 = 0.0f, float scale = 64.0f);

template <typename T>
std::tuple<Tensor, Tensor, const paddle::optional<Tensor>> masked_multihead_attention_(const Tensor& x, const Tensor& cache_kv, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& src_mask, const paddle::optional<Tensor>& cum_offsets, const paddle::optional<Tensor>& sequence_lengths, const paddle::optional<Tensor>& rotary_tensor, const paddle::optional<Tensor>& beam_cache_offset, const paddle::optional<Tensor>& qkv_out_scale, const paddle::optional<Tensor>& out_shift, const paddle::optional<Tensor>& out_smooth, int seq_len, int rotary_emb_dims, bool use_neox_rotary_style = false, const std::string& compute_dtype = "default", float out_scale = -1, int quant_round_type = 1, float quant_max_bound = 127.0, float quant_min_bound = -127.0);

template <typename T>
Tensor masked_select(const Tensor& x, const Tensor& mask);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> matrix_nms(const Tensor& bboxes, const Tensor& scores, float score_threshold, int nms_top_k, int keep_top_k, float post_threshold = 0., bool use_gaussian = false, float gaussian_sigma = 2., int background_label = 0, bool normalized = true);

template <typename T>
Tensor matrix_power(const Tensor& x, int n);

template <typename T>
std::tuple<Tensor, Tensor> max_pool2d_with_index(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, bool global_pooling = false, bool adaptive = false);

template <typename T>
std::tuple<Tensor, Tensor> max_pool3d_with_index(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides = {1, 1, 1}, const std::vector<int>& paddings = {0, 0, 0}, bool global_pooling = false, bool adaptive = false);

template <typename T>
Tensor maxout(const Tensor& x, int groups, int axis = 1);

template <typename T>
Tensor mean_all(const Tensor& x);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> memory_efficient_attention(const Tensor& query, const Tensor& key, const Tensor& value, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& cu_seqlens_q, const paddle::optional<Tensor>& cu_seqlens_k, const paddle::optional<Tensor>& causal_diagonal, const paddle::optional<Tensor>& seqlen_k, const Scalar& max_seqlen_q, const Scalar& max_seqlen_k, bool causal, double dropout_p, float scale, bool is_test);

template <typename T>
Tensor merge_selected_rows(const Tensor& x);

template <typename T>
std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, const paddle::optional<std::vector<Tensor>>> merged_adam_(const std::vector<Tensor>& param, const std::vector<Tensor>& grad, const std::vector<Tensor>& learning_rate, const std::vector<Tensor>& moment1, const std::vector<Tensor>& moment2, const std::vector<Tensor>& beta1_pow, const std::vector<Tensor>& beta2_pow, const paddle::optional<std::vector<Tensor>>& master_param, const Tensor& beta1_, const Tensor& beta2_, const Tensor& epsilon_, bool multi_precision = false, bool use_global_beta_pow = false);

template <typename T>
std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, const paddle::optional<std::vector<Tensor>>> merged_adam_(const std::vector<Tensor>& param, const std::vector<Tensor>& grad, const std::vector<Tensor>& learning_rate, const std::vector<Tensor>& moment1, const std::vector<Tensor>& moment2, const std::vector<Tensor>& beta1_pow, const std::vector<Tensor>& beta2_pow, const paddle::optional<std::vector<Tensor>>& master_param, const Scalar& beta1 = 0.9f, const Scalar& beta2 = 0.999f, const Scalar& epsilon = 1.0e-8f, bool multi_precision = false, bool use_global_beta_pow = false);

template <typename T>
std::tuple<std::vector<Tensor>, std::vector<Tensor>, const paddle::optional<std::vector<Tensor>>> merged_momentum_(const std::vector<Tensor>& param, const std::vector<Tensor>& grad, const std::vector<Tensor>& velocity, const std::vector<Tensor>& learning_rate, const paddle::optional<std::vector<Tensor>>& master_param, float mu, bool use_nesterov = false, const std::vector<std::string>& regularization_method = {}, const std::vector<float>& regularization_coeff = {}, bool multi_precision = false, float rescale_grad = 1.0f);

template <typename T>
std::vector<Tensor> meshgrid(const std::vector<Tensor>& inputs);

template <typename T>
std::tuple<Tensor, Tensor> mode(const Tensor& x, int axis = -1, bool keepdim = false);

template <typename T>
std::tuple<Tensor, Tensor, const paddle::optional<Tensor>> momentum_(const Tensor& param, const Tensor& grad, const Tensor& velocity, const Tensor& learning_rate, const paddle::optional<Tensor>& master_param, float mu, bool use_nesterov = false, const std::string& regularization_method = "", float regularization_coeff = 0.0f, bool multi_precision = false, float rescale_grad = 1.0f);

template <typename T>
Tensor multi_dot(const std::vector<Tensor>& x);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> multiclass_nms3(const Tensor& bboxes, const Tensor& scores, const paddle::optional<Tensor>& rois_num, float score_threshold, int nms_top_k, int keep_top_k, float nms_threshold = 0.3, bool normalized = true, float nms_eta = 1.0, int background_label = 0);

template <typename T>
Tensor multinomial(const Tensor& x, const Tensor& num_samples_, bool replacement = false);

template <typename T>
Tensor multinomial(const Tensor& x, const Scalar& num_samples = 1, bool replacement = false);

template <typename T>
Tensor multiplex(const std::vector<Tensor>& inputs, const Tensor& index);

template <typename T>
Tensor mv(const Tensor& x, const Tensor& vec);

template <typename T>
Tensor nanmedian(const Tensor& x, const IntArray& axis = {}, bool keepdim = true);

template <typename T>
Tensor nearest_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout = "NCHW", int out_d = 0, int out_h = 0, int out_w = 0, const std::vector<float>& scale = {}, const std::string& interp_method = "bilinear", bool align_corners = true, int align_mode = 1);

template <typename T>
Tensor nextafter(const Tensor& x, const Tensor& y);

template <typename T>
std::tuple<Tensor, Tensor> nll_loss(const Tensor& input, const Tensor& label, const paddle::optional<Tensor>& weight, int64_t ignore_index = -100, const std::string& reduction = "mean");

template <typename T>
Tensor nms(const Tensor& x, float threshold = 1.0f);

template <typename T>
Tensor nonzero(const Tensor& condition);

template <typename T>
Tensor npu_identity(const Tensor& x, int format = -1);

template <typename T>
Tensor numel(const Tensor& x);

template <typename T>
Tensor overlap_add(const Tensor& x, int hop_length, int axis = -1);

template <typename T>
Tensor p_norm(const Tensor& x, float porder = 2, int axis = -1, float epsilon = 1.0e-12f, bool keepdim = false, bool asvector = false);

template <typename T>
Tensor pad3d(const Tensor& x, const Tensor& paddings_, const std::string& mode = "constant", float pad_value = 0.0, const std::string& data_format = "NCDHW");

template <typename T>
Tensor pad3d(const Tensor& x, const IntArray& paddings, const std::string& mode = "constant", float pad_value = 0.0, const std::string& data_format = "NCDHW");

template <typename T>
Tensor pixel_shuffle(const Tensor& x, int upscale_factor = 1, const std::string& data_format = "NCHW");

template <typename T>
Tensor pixel_unshuffle(const Tensor& x, int downscale_factor = 1, const std::string& data_format = "NCHW");

template <typename T>
Tensor poisson(const Tensor& x);

template <typename T>
Tensor polygamma(const Tensor& x, int n);

template <typename T>
Tensor pow(const Tensor& x, const Scalar& y = 1.0f);

template <typename T>
Tensor prelu(const Tensor& x, const Tensor& alpha, const std::string& data_format = "NCHW", const std::string& mode = "all");

template <typename T>
std::tuple<Tensor, Tensor> prior_box(const Tensor& input, const Tensor& image, const std::vector<float>& min_sizes, const std::vector<float>& max_sizes = {}, const std::vector<float>& aspect_ratios = {}, const std::vector<float>& variances = {}, bool flip = true, bool clip = true, float step_w = 0.0, float step_h = 0.0, float offset = 0.5, bool min_max_aspect_ratios_order = false);

template <typename T>
Tensor psroi_pool(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height = 1, int pooled_width = 1, int output_channels = 1, float spatial_scale = 1.0);

template <typename T>
Tensor put_along_axis(const Tensor& arr, const Tensor& indices, const Tensor& values, int axis, const std::string& reduce = "assign");

template <typename T>
std::tuple<Tensor, Tensor> qr(const Tensor& x, const std::string& mode = "reduced");

template <typename T>
Tensor real(const Tensor& x);

template <typename T>
Tensor reciprocal(const Tensor& x);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> reindex_graph(const Tensor& x, const Tensor& neighbors, const Tensor& count, const paddle::optional<Tensor>& hashtable_value, const paddle::optional<Tensor>& hashtable_index);

template <typename T>
Tensor relu(const Tensor& x);

template <typename T>
Tensor relu6(const Tensor& x);

template <typename T>
Tensor renorm(const Tensor& x, float p, int axis, float max_norm);

template <typename T>
Tensor reverse(const Tensor& x, const Tensor& axis_);

template <typename T>
Tensor reverse(const Tensor& x, const IntArray& axis);

template <typename T>
std::tuple<Tensor, Tensor> rms_norm(const Tensor& x, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& residual, const Tensor& norm_weight, const paddle::optional<Tensor>& norm_bias, float epsilon, int begin_norm_axis, float quant_scale, int quant_round_type, float quant_max_bound, float quant_min_bound);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, const paddle::optional<Tensor>, const paddle::optional<Tensor>> rmsprop_(const Tensor& param, const Tensor& mean_square, const Tensor& grad, const Tensor& moment, const Tensor& learning_rate, const paddle::optional<Tensor>& mean_grad, const paddle::optional<Tensor>& master_param, float epsilon = 1.0e-10f, float decay = 0.9f, float momentum = 0.0f, bool centered = false, bool multi_precision = false);

template <typename T>
Tensor roi_align(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height = 1, int pooled_width = 1, float spatial_scale = 1.0, int sampling_ratio = -1, bool aligned = false);

template <typename T>
Tensor roi_pool(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height = 1, int pooled_width = 1, float spatial_scale = 1.0);

template <typename T>
Tensor roll(const Tensor& x, const Tensor& shifts_, const std::vector<int64_t>& axis = {});

template <typename T>
Tensor roll(const Tensor& x, const IntArray& shifts = {}, const std::vector<int64_t>& axis = {});

template <typename T>
Tensor round(const Tensor& x);

template <typename T>
Tensor rsqrt(const Tensor& x);

template <typename T>
Tensor scale(const Tensor& x, const Tensor& scale_, float bias = 0.0, bool bias_after_scale = true);

template <typename T>
Tensor scale(const Tensor& x, const Scalar& scale = 1.0, float bias = 0.0, bool bias_after_scale = true);

template <typename T>
Tensor scatter(const Tensor& x, const Tensor& index, const Tensor& updates, bool overwrite = true);

template <typename T>
Tensor scatter_nd_add(const Tensor& x, const Tensor& index, const Tensor& updates);

template <typename T>
Tensor searchsorted(const Tensor& sorted_sequence, const Tensor& values, bool out_int32 = false, bool right = false);

template <typename T>
Tensor segment_pool(const Tensor& x, const Tensor& segment_ids, const std::string& pooltype = "SUM");

template <typename T>
Tensor selu(const Tensor& x, float scale = 1.0507009873554804934193349852946, float alpha = 1.6732632423543772848170429916717);

template <typename T>
Tensor send_u_recv(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const Tensor& out_size_, const std::string& reduce_op = "SUM");

template <typename T>
Tensor send_u_recv(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const std::string& reduce_op = "SUM", const IntArray& out_size = {0});

template <typename T>
Tensor send_ue_recv(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const Tensor& out_size_, const std::string& message_op = "ADD", const std::string& reduce_op = "SUM");

template <typename T>
Tensor send_ue_recv(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const std::string& message_op = "ADD", const std::string& reduce_op = "SUM", const IntArray& out_size = {0});

template <typename T>
Tensor send_uv(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const std::string& message_op = "ADD");

template <typename T>
std::tuple<Tensor, const paddle::optional<Tensor>> sgd_(const Tensor& param, const Tensor& learning_rate, const Tensor& grad, const paddle::optional<Tensor>& master_param, bool multi_precision = false);

template <typename T>
Tensor shadow_output(const Tensor& x, const std::string& name);

template <typename T>
Tensor shape(const Tensor& input);

template <typename T>
Tensor shard_index(const Tensor& input, int index_num, int nshards, int shard_id, int ignore_value = -1);

template <typename T>
Tensor sigmoid(const Tensor& x);

template <typename T>
Tensor sigmoid_cross_entropy_with_logits(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, bool normalize = false, int ignore_index = -100);

template <typename T>
Tensor sign(const Tensor& x);

template <typename T>
Tensor silu(const Tensor& x);

template <typename T>
Tensor sin(const Tensor& x);

template <typename T>
Tensor sinh(const Tensor& x);

template <typename T>
Tensor slogdet(const Tensor& x);

template <typename T>
Tensor softplus(const Tensor& x, float beta = 1.0, float threshold = 20.0f);

template <typename T>
Tensor softshrink(const Tensor& x, float threshold = 0.5);

template <typename T>
Tensor softsign(const Tensor& x);

template <typename T>
Tensor solve(const Tensor& x, const Tensor& y);

template <typename T>
Tensor spectral_norm(const Tensor& weight, const Tensor& u, const Tensor& v, int dim = 0, int power_iters = 1, float eps = 1e-12f);

template <typename T>
Tensor sqrt(const Tensor& x);

template <typename T>
Tensor square(const Tensor& x);

template <typename T>
Tensor squared_l2_norm(const Tensor& x);

template <typename T>
Tensor squeeze(const Tensor& x, const Tensor& axis_);

template <typename T>
Tensor squeeze(const Tensor& x, const IntArray& axis = {});

template <typename T>
Tensor stack(const std::vector<Tensor>& x, int axis = 0);

template <typename T>
Tensor stanh(const Tensor& x, float scale_a = 0.67f, float scale_b = 1.7159f);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& x, bool full_matrices = false);

template <typename T>
Tensor take_along_axis(const Tensor& arr, const Tensor& indices, int axis);

template <typename T>
Tensor tan(const Tensor& x);

template <typename T>
Tensor tanh(const Tensor& x);

template <typename T>
Tensor tanh_shrink(const Tensor& x);

template <typename T>
Tensor temporal_shift(const Tensor& x, int seg_num, float shift_ratio = 0.25f, const std::string& data_format = "NCHW");

template <typename T>
Tensor tensor_unfold(const Tensor& input, int64_t axis, int64_t size, int64_t step);

template <typename T>
Tensor thresholded_relu(const Tensor& x, float threshold = 1.0);

template <typename T>
std::tuple<Tensor, Tensor> topk(const Tensor& x, const Tensor& k_, int axis = -1, bool largest = true, bool sorted = true);

template <typename T>
std::tuple<Tensor, Tensor> topk(const Tensor& x, const Scalar& k = 1, int axis = -1, bool largest = true, bool sorted = true);

template <typename T>
Tensor trace(const Tensor& x, int offset = 0, int axis1 = 0, int axis2 = 1);

template <typename T>
Tensor triangular_solve(const Tensor& x, const Tensor& y, bool upper = true, bool transpose = false, bool unitriangular = false);

template <typename T>
Tensor trilinear_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout = "NCHW", int out_d = 0, int out_h = 0, int out_w = 0, const std::vector<float>& scale = {}, const std::string& interp_method = "bilinear", bool align_corners = true, int align_mode = 1);

template <typename T>
Tensor trunc(const Tensor& input);

template <typename T>
std::vector<Tensor> unbind(const Tensor& input, int axis = 0);

template <typename T>
Tensor unfold(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations);

template <typename T>
Tensor uniform_inplace(const Tensor& x, float min = -1.0, float max = 1.0, int seed = 0, int diag_num = 0, int diag_step = 0, float diag_val = 1.0);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> unique_consecutive(const Tensor& x, bool return_inverse = false, bool return_counts = false, const std::vector<int>& axis = {}, int dtype = 5);

template <typename T>
Tensor unpool3d(const Tensor& x, const Tensor& indices, const std::vector<int>& ksize, const std::vector<int>& strides = {1,1,1}, const std::vector<int>& paddings = {0,0,0}, const std::vector<int>& output_size = {0,0,0}, const std::string& data_format = "NCDHW");

template <typename T>
Tensor unsqueeze(const Tensor& x, const Tensor& axis_);

template <typename T>
Tensor unsqueeze(const Tensor& x, const IntArray& axis = {});

template <typename T>
std::vector<Tensor> unstack(const Tensor& x, int axis = 0, int num = 0);

template <typename T>
std::tuple<std::vector<Tensor>, Tensor, Tensor, Tensor> update_loss_scaling_(const std::vector<Tensor>& x, const Tensor& found_infinite, const Tensor& prev_loss_scaling, const Tensor& in_good_steps, const Tensor& in_bad_steps, const Tensor& stop_update_, int incr_every_n_steps, int decr_every_n_nan_or_inf, float incr_ratio, float decr_ratio);

template <typename T>
std::tuple<std::vector<Tensor>, Tensor, Tensor, Tensor> update_loss_scaling_(const std::vector<Tensor>& x, const Tensor& found_infinite, const Tensor& prev_loss_scaling, const Tensor& in_good_steps, const Tensor& in_bad_steps, int incr_every_n_steps, int decr_every_n_nan_or_inf, float incr_ratio, float decr_ratio, const Scalar& stop_update = false);

template <typename T>
Tensor variable_length_memory_efficient_attention(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& seq_lens, const Tensor& kv_seq_lens, const paddle::optional<Tensor>& mask, float scale, bool causal);

template <typename T>
Tensor view_dtype(const Tensor& input, DataType dtype);

template <typename T>
Tensor view_shape(const Tensor& input, const std::vector<int64_t>& dims = {});

template <typename T>
std::tuple<Tensor, Tensor> viterbi_decode(const Tensor& potentials, const Tensor& transition_params, const Tensor& lengths, bool include_bos_eos_tag = true);

template <typename T>
Tensor warpctc(const Tensor& logits, const Tensor& label, const paddle::optional<Tensor>& logits_length, const paddle::optional<Tensor>& labels_length, int blank = 0, bool norm_by_times = false);

template <typename T>
Tensor warprnnt(const Tensor& input, const Tensor& label, const Tensor& input_lengths, const Tensor& label_lengths, int blank = 0, float fastemit_lambda = 0.0);

template <typename T>
Tensor weight_dequantize(const Tensor& x, const Tensor& scale, const std::string& algo = "weight_only_int8", DataType out_dtype = DataType::FLOAT16);

template <typename T>
Tensor weight_only_linear(const Tensor& x, const Tensor& weight, const paddle::optional<Tensor>& bias, const Tensor& weight_scale, const std::string& weight_dtype);

template <typename T>
std::tuple<Tensor, Tensor> weight_quantize(const Tensor& x, const std::string& algo = "weight_only_int8");

template <typename T>
std::tuple<Tensor, Tensor, Tensor> weighted_sample_neighbors(const Tensor& row, const Tensor& colptr, const Tensor& edge_weight, const Tensor& input_nodes, const paddle::optional<Tensor>& eids, int sample_size, bool return_eids);

template <typename T>
Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);

template <typename T>
std::tuple<Tensor, Tensor> yolo_box(const Tensor& x, const Tensor& img_size, const std::vector<int>& anchors = {}, int class_num = 1, float conf_thresh = 0.01, int downsample_ratio = 32, bool clip_bbox = true, float scale_x_y = 1.0, bool iou_aware = false, float iou_aware_factor = 0.5);

template <typename T>
Tensor yolo_loss(const Tensor& x, const Tensor& gt_box, const Tensor& gt_label, const paddle::optional<Tensor>& gt_score, const std::vector<int>& anchors = {}, const std::vector<int>& anchor_mask = {}, int class_num = 1, float ignore_thresh = 0.7, int downsample_ratio = 32, bool use_label_smooth = true, float scale_x_y = 1.0);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, const paddle::optional<Tensor>> adadelta_(const Tensor& param, const Tensor& grad, const Tensor& avg_squared_grad, const Tensor& avg_squared_update, const Tensor& learning_rate, const paddle::optional<Tensor>& master_param, float rho, float epsilon, bool multi_precision);

template <typename T>
Tensor add(const Tensor& x, const Tensor& y);

template <typename T>
Tensor add_n(const std::vector<Tensor>& inputs);

template <typename T>
Tensor all(const Tensor& x, const std::vector<int64_t>& axis = {}, bool keepdim = false);

template <typename T>
Tensor amax(const Tensor& x, const std::vector<int64_t>& axis = {}, bool keepdim = false);

template <typename T>
Tensor amin(const Tensor& x, const std::vector<int64_t>& axis = {}, bool keepdim = false);

template <typename T>
Tensor any(const Tensor& x, const std::vector<int64_t>& axis = {}, bool keepdim = false);

template <typename T>
Tensor arange(const Tensor& start, const Tensor& end, const Tensor& step, DataType dtype, Place place = {});

template <typename T>
Tensor assign(const Tensor& x);

template <typename T>
Tensor assign_out_(const Tensor& x, const Tensor& output);

template <typename T>
Tensor assign_value_(const Tensor& output, const std::vector<int>& shape, DataType dtype, const std::vector<Scalar>& values, Place place = {});

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> batch_norm(const Tensor& x, const Tensor& mean, const Tensor& variance, const Tensor& scale, const Tensor& bias, bool is_test, float momentum, float epsilon, const std::string& data_layout, bool use_global_stats, bool trainable_statistics);

template <typename T>
Tensor c_allgather(const Tensor& x, int ring_id, int nranks, bool use_calc_stream);

template <typename T>
Tensor c_allreduce_max(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

template <typename T>
Tensor c_allreduce_sum(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

template <typename T>
Tensor c_broadcast(const Tensor& x, int ring_id = 0, int root = 0, bool use_calc_stream = false);

template <typename T>
Tensor c_concat(const Tensor& x, int rank, int nranks, int ring_id, bool use_calc_stream, bool use_model_parallel);

template <typename T>
Tensor c_embedding(const Tensor& weight, const Tensor& x, int64_t start_index = 0);

template <typename T>
Tensor c_identity(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

template <typename T>
Tensor c_reduce_sum(const Tensor& x, int ring_id, int root_id, bool use_calc_stream);

template <typename T>
Tensor c_sync_calc_stream(const Tensor& x);

template <typename T>
Tensor c_sync_comm_stream(const Tensor& x);

template <typename T>
Tensor cast(const Tensor& x, DataType dtype);

template <typename T>
Tensor channel_shuffle(const Tensor& x, int groups, const std::string& data_format = "NCHW");

template <typename T>
Tensor conv2d_transpose(const Tensor& x, const Tensor& filter, const Tensor& output_size_, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, const std::vector<int>& output_padding = {}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1}, const std::string& data_format = "NCHW");

template <typename T>
Tensor conv2d_transpose(const Tensor& x, const Tensor& filter, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, const std::vector<int>& output_padding = {}, const IntArray& output_size = {}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1}, const std::string& data_format = "NCHW");

template <typename T>
Tensor decode_jpeg(const Tensor& x, const std::string& mode, Place place);

template <typename T>
Tensor deformable_conv(const Tensor& x, const Tensor& offset, const Tensor& filter, const paddle::optional<Tensor>& mask, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, int deformable_groups, int groups, int im2col_step);

template <typename T>
Tensor depthwise_conv2d_transpose(const Tensor& x, const Tensor& filter, const Tensor& output_size_, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, const std::vector<int>& output_padding = {}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1}, const std::string& data_format = "NCHW");

template <typename T>
Tensor depthwise_conv2d_transpose(const Tensor& x, const Tensor& filter, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, const std::vector<int>& output_padding = {}, const IntArray& output_size = {}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1}, const std::string& data_format = "NCHW");

template <typename T>
Tensor disable_check_model_nan_inf(const Tensor& x, int flag = 0);

template <typename T>
std::tuple<std::vector<Tensor>, std::vector<Tensor>, Tensor> distribute_fpn_proposals(const Tensor& fpn_rois, const paddle::optional<Tensor>& rois_num, int min_level, int max_level, int refer_level, int refer_scale, bool pixel_offset);

template <typename T>
Tensor divide(const Tensor& x, const Tensor& y);

template <typename T>
std::tuple<Tensor, Tensor> dropout(const Tensor& x, const paddle::optional<Tensor>& seed_tensor, const Scalar& p, bool is_test, const std::string& mode, int seed, bool fix_seed);

template <typename T>
std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> einsum(const std::vector<Tensor>& x, const std::string& equation);

template <typename T>
Tensor elementwise_pow(const Tensor& x, const Tensor& y);

template <typename T>
Tensor embedding(const Tensor& x, const Tensor& weight, int64_t padding_idx = -1, bool sparse = false);

template <typename T>
Tensor embedding_grad_dense(const Tensor& x, const Tensor& weight, const Tensor& out_grad, int64_t padding_idx = -1, bool sparse = false);

template <typename T>
Tensor empty(const Tensor& shape_, DataType dtype = DataType::FLOAT32, Place place = CPUPlace());

template <typename T>
Tensor empty(const IntArray& shape, DataType dtype = DataType::FLOAT32, Place place = CPUPlace());

template <typename T>
Tensor empty_like(const Tensor& x, DataType dtype = DataType::UNDEFINED, Place place = {});

template <typename T>
Tensor enable_check_model_nan_inf(const Tensor& x, int flag = 1);

template <typename T>
Tensor equal(const Tensor& x, const Tensor& y);

template <typename T>
Tensor exponential_(const Tensor& x, float lam);

template <typename T>
Tensor eye(const Tensor& num_rows_, const Tensor& num_columns_, DataType dtype = DataType::FLOAT32, Place place = {});

template <typename T>
Tensor eye(const Scalar& num_rows, const Scalar& num_columns, DataType dtype = DataType::FLOAT32, Place place = {});

template <typename T>
Tensor floor_divide(const Tensor& x, const Tensor& y);

template <typename T>
Tensor frobenius_norm(const Tensor& x, const std::vector<int64_t>& axis, bool keep_dim, bool reduce_all);

template <typename T>
Tensor full(const IntArray& shape, const Scalar& value, DataType dtype = DataType::FLOAT32, Place place = CPUPlace());

template <typename T>
Tensor full_(const Tensor& output, const IntArray& shape, const Scalar& value, DataType dtype = DataType::FLOAT32, Place place = CPUPlace());

template <typename T>
Tensor full_batch_size_like(const Tensor& input, const std::vector<int>& shape, DataType dtype, const Scalar& value, int input_dim_idx, int output_dim_idx, Place place = CPUPlace());

template <typename T>
Tensor full_like(const Tensor& x, const Tensor& value_, DataType dtype = DataType::UNDEFINED, Place place = {});

template <typename T>
Tensor full_like(const Tensor& x, const Scalar& value, DataType dtype = DataType::UNDEFINED, Place place = {});

template <typename T>
Tensor full_with_tensor(const Tensor& shape, const Tensor& value, DataType dtype = DataType::FLOAT32);

template <typename T>
std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, const paddle::optional<std::vector<Tensor>>> fused_adam_(const std::vector<Tensor>& params, const std::vector<Tensor>& grads, const Tensor& learning_rate, const std::vector<Tensor>& moments1, const std::vector<Tensor>& moments2, const std::vector<Tensor>& beta1_pows, const std::vector<Tensor>& beta2_pows, const paddle::optional<std::vector<Tensor>>& master_params, const paddle::optional<Tensor>& skip_update, const Scalar& beta1, const Scalar& beta2, const Scalar& epsilon, int chunk_size, float weight_decay, bool use_adamw, bool multi_precision, bool use_global_beta_pow);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> fused_batch_norm_act(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& mean, const Tensor& variance, float momentum, float epsilon, const std::string& act_type);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> fused_bn_add_activation(const Tensor& x, const Tensor& z, const Tensor& scale, const Tensor& bias, const Tensor& mean, const Tensor& variance, float momentum, float epsilon, const std::string& act_type);

template <typename T>
Tensor fused_softmax_mask_upper_triangle(const Tensor& X);

template <typename T>
Tensor gaussian(const Tensor& shape_, float mean, float std, int seed, DataType dtype, Place place = {});

template <typename T>
Tensor gaussian(const IntArray& shape, float mean, float std, int seed, DataType dtype, Place place = {});

template <typename T>
Tensor greater_equal(const Tensor& x, const Tensor& y);

template <typename T>
Tensor greater_than(const Tensor& x, const Tensor& y);

template <typename T>
Tensor hardswish(const Tensor& x);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> hsigmoid_loss(const Tensor& x, const Tensor& label, const Tensor& w, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& path, const paddle::optional<Tensor>& code, int num_classes, bool is_sparse);

template <typename T>
Tensor increment(const Tensor& x, float value = 1.0);

template <typename T>
Tensor less_equal(const Tensor& x, const Tensor& y);

template <typename T>
Tensor less_than(const Tensor& x, const Tensor& y);

template <typename T>
Tensor linspace(const Tensor& start, const Tensor& stop, const Tensor& number, DataType dtype, Place place);

template <typename T>
Tensor logspace(const Tensor& start, const Tensor& stop, const Tensor& num, const Tensor& base, DataType dtype, Place place = {});

template <typename T>
Tensor logsumexp(const Tensor& x, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all);

template <typename T>
Tensor matmul(const Tensor& x, const Tensor& y, bool transpose_x = false, bool transpose_y = false);

template <typename T>
Tensor matrix_rank(const Tensor& x, float tol, bool use_default_tol = true, bool hermitian = false);

template <typename T>
Tensor matrix_rank_tol(const Tensor& x, const Tensor& atol_tensor, bool use_default_tol = true, bool hermitian = false);

template <typename T>
Tensor max(const Tensor& x, const Tensor& axis_, bool keepdim = false);

template <typename T>
Tensor max(const Tensor& x, const IntArray& axis = {}, bool keepdim = false);

template <typename T>
Tensor maximum(const Tensor& x, const Tensor& y);

template <typename T>
Tensor mean(const Tensor& x, const IntArray& axis = {}, bool keepdim = false);

template <typename T>
Tensor memcpy_d2h(const Tensor& x, int dst_place_type);

template <typename T>
Tensor memcpy_h2d(const Tensor& x, int dst_place_type);

template <typename T>
Tensor min(const Tensor& x, const Tensor& axis_, bool keepdim = false);

template <typename T>
Tensor min(const Tensor& x, const IntArray& axis = {}, bool keepdim = false);

template <typename T>
Tensor minimum(const Tensor& x, const Tensor& y);

template <typename T>
Tensor mish(const Tensor& x, float lambda);

template <typename T>
Tensor multiply(const Tensor& x, const Tensor& y);

template <typename T>
std::tuple<Tensor, Tensor> norm(const Tensor& x, int axis, float epsilon, bool is_test);

template <typename T>
Tensor not_equal(const Tensor& x, const Tensor& y);

template <typename T>
Tensor one_hot(const Tensor& x, const Scalar& num_classes);

template <typename T>
Tensor ones(const IntArray& shape, DataType dtype = DataType::FLOAT32, Place place = CPUPlace());

template <typename T>
Tensor ones_like(const Tensor& x, DataType dtype = DataType::UNDEFINED, Place place = {});

template <typename T>
Tensor pad(const Tensor& x, const std::vector<int>& paddings, const Scalar& pad_value);

template <typename T>
Tensor pool2d(const Tensor& x, const Tensor& kernel_size_, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm);

template <typename T>
Tensor pool2d(const Tensor& x, const IntArray& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm);

template <typename T>
Tensor pool3d(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm);

template <typename T>
Tensor prod(const Tensor& x, const Tensor& dims_, bool keep_dim, bool reduce_all);

template <typename T>
Tensor prod(const Tensor& x, const IntArray& dims, bool keep_dim, bool reduce_all);

template <typename T>
Tensor randint(const Tensor& shape_, int low, int high, DataType dtype = DataType::INT64, Place place = {});

template <typename T>
Tensor randint(int low, int high, const IntArray& shape, DataType dtype = DataType::INT64, Place place = {});

template <typename T>
Tensor randperm(int n, DataType dtype, Place place = {});

template <typename T>
Tensor remainder(const Tensor& x, const Tensor& y);

template <typename T>
Tensor repeat_interleave(const Tensor& x, int repeats, int axis);

template <typename T>
Tensor repeat_interleave_with_tensor_index(const Tensor& x, const Tensor& repeats, int axis);

template <typename T>
Tensor reshape(const Tensor& x, const Tensor& shape_);

template <typename T>
Tensor reshape(const Tensor& x, const IntArray& shape);

template <typename T>
std::tuple<Tensor, Tensor, std::vector<Tensor>> rnn(const Tensor& x, const std::vector<Tensor>& pre_state, const std::vector<Tensor>& weight_list, const paddle::optional<Tensor>& sequence_length, const Tensor& dropout_state_in, float dropout_prob = 0.0, bool is_bidirec = false, int input_size = 10, int hidden_size = 100, int num_layers = 1, const std::string& mode = "RNN_TANH", int seed = 0, bool is_test = false);

template <typename T>
Tensor rrelu(const Tensor& x, float lower, float upper, bool is_test);

template <typename T>
Tensor slice(const Tensor& input, const Tensor& starts_, const Tensor& ends_, const std::vector<int64_t>& axes, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis);

template <typename T>
Tensor slice(const Tensor& input, const std::vector<int64_t>& axes, const IntArray& starts, const IntArray& ends, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis);

template <typename T>
Tensor softmax(const Tensor& x, int axis);

template <typename T>
std::vector<Tensor> split(const Tensor& x, const Tensor& sections_, const Tensor& axis_);

template <typename T>
std::vector<Tensor> split(const Tensor& x, const IntArray& sections, const Scalar& axis);

template <typename T>
std::vector<Tensor> split_with_num(const Tensor& x, const Tensor& axis_, int num);

template <typename T>
std::vector<Tensor> split_with_num(const Tensor& x, int num, const Scalar& axis);

template <typename T>
Tensor strided_slice(const Tensor& x, const Tensor& starts_, const Tensor& ends_, const Tensor& strides_, const std::vector<int>& axes);

template <typename T>
Tensor strided_slice(const Tensor& x, const std::vector<int>& axes, const IntArray& starts, const IntArray& ends, const IntArray& strides);

template <typename T>
Tensor subtract(const Tensor& x, const Tensor& y);

template <typename T>
Tensor sum(const Tensor& x, const Tensor& axis_, DataType dtype = DataType::UNDEFINED, bool keepdim = false);

template <typename T>
Tensor sum(const Tensor& x, const IntArray& axis = {}, DataType dtype = DataType::UNDEFINED, bool keepdim = false);

template <typename T>
Tensor swish(const Tensor& x);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> sync_batch_norm_(const Tensor& x, const Tensor& mean, const Tensor& variance, const Tensor& scale, const Tensor& bias, bool is_test, float momentum, float epsilon, const std::string& data_layout, bool use_global_stats, bool trainable_statistics);

template <typename T>
Tensor tile(const Tensor& x, const Tensor& repeat_times_);

template <typename T>
Tensor tile(const Tensor& x, const IntArray& repeat_times = {});

template <typename T>
Tensor trans_layout(const Tensor& x, const std::vector<int>& perm);

template <typename T>
Tensor transpose(const Tensor& x, const std::vector<int>& perm);

template <typename T>
Tensor tril(const Tensor& x, int diagonal);

template <typename T>
Tensor tril_indices(int rows, int cols, int offset, DataType dtype, Place place = {});

template <typename T>
Tensor triu(const Tensor& x, int diagonal);

template <typename T>
Tensor triu_indices(int row, int col, int offset, DataType dtype, Place place = {});

template <typename T>
Tensor truncated_gaussian_random(const std::vector<int>& shape, float mean, float std, int seed, DataType dtype = DataType::FLOAT32, Place place = {});

template <typename T>
Tensor uniform(const Tensor& shape_, const Tensor& min_, const Tensor& max_, DataType dtype, int seed, Place place = {});

template <typename T>
Tensor uniform(const IntArray& shape, DataType dtype, const Scalar& min, const Scalar& max, int seed, Place place = {});

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor> unique(const Tensor& x, bool return_index, bool return_inverse, bool return_counts, const std::vector<int>& axis, DataType dtype = DataType::INT64);

template <typename T>
Tensor unpool(const Tensor& x, const Tensor& indices, const Tensor& output_size_, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const std::string& data_format);

template <typename T>
Tensor unpool(const Tensor& x, const Tensor& indices, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const IntArray& output_size, const std::string& data_format);

template <typename T>
Tensor zeros(const IntArray& shape, DataType dtype = DataType::FLOAT32, Place place = CPUPlace());

template <typename T>
Tensor zeros_like(const Tensor& x, DataType dtype = DataType::UNDEFINED, Place place = {});

template <typename T>
Tensor abs_double_grad(const Tensor& x, const Tensor& grad_x_grad);

template <typename T>
Tensor abs_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor acos_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor acosh_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> addmm_grad(const Tensor& input, const Tensor& x, const Tensor& y, const Tensor& out_grad, float alpha, float beta);

template <typename T>
Tensor affine_grid_grad(const Tensor& input, const Tensor& output_grad, const Tensor& output_shape_, bool align_corners = true);

template <typename T>
Tensor affine_grid_grad(const Tensor& input, const Tensor& output_grad, const IntArray& output_shape, bool align_corners = true);

template <typename T>
Tensor angle_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor argsort_grad(const Tensor& indices, const Tensor& x, const Tensor& out_grad, int axis, bool descending);

template <typename T>
Tensor as_strided_grad(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& dims = {}, const std::vector<int64_t>& stride = {}, int64_t offset = 0);

template <typename T>
Tensor asin_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor asinh_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> atan2_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);

template <typename T>
Tensor atan_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor atanh_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor bce_loss_grad(const Tensor& input, const Tensor& label, const Tensor& out_grad);

template <typename T>
Tensor bicubic_interp_grad(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor> bilinear_grad(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& out_grad);

template <typename T>
Tensor bilinear_interp_grad(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode);

template <typename T>
std::tuple<Tensor, Tensor> bmm_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);

template <typename T>
std::vector<Tensor> broadcast_tensors_grad(const std::vector<Tensor>& input, const std::vector<Tensor>& out_grad);

template <typename T>
Tensor ceil_grad(const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> celu_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha);

template <typename T>
Tensor celu_grad(const Tensor& x, const Tensor& out_grad, float alpha);

template <typename T>
Tensor cholesky_grad(const Tensor& out, const Tensor& out_grad, bool upper);

template <typename T>
std::tuple<Tensor, Tensor> cholesky_solve_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, bool upper);

template <typename T>
Tensor clip_double_grad(const Tensor& x, const Tensor& grad_x_grad, const Tensor& min_, const Tensor& max_);

template <typename T>
Tensor clip_double_grad(const Tensor& x, const Tensor& grad_x_grad, const Scalar& min = 0., const Scalar& max = 0.);

template <typename T>
Tensor clip_grad(const Tensor& x, const Tensor& out_grad, const Tensor& min_, const Tensor& max_);

template <typename T>
Tensor clip_grad(const Tensor& x, const Tensor& out_grad, const Scalar& min = 0., const Scalar& max = 0.);

template <typename T>
std::tuple<Tensor, Tensor> complex_grad(const Tensor& real, const Tensor& imag, const Tensor& out_grad);

template <typename T>
std::vector<Tensor> concat_grad(const std::vector<Tensor>& x, const Tensor& out_grad, const Tensor& axis_);

template <typename T>
std::vector<Tensor> concat_grad(const std::vector<Tensor>& x, const Tensor& out_grad, const Scalar& axis = 0);

template <typename T>
std::tuple<Tensor, Tensor> conv2d_grad(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, const std::vector<int>& dilations, int groups, const std::string& data_format);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> conv2d_grad_grad(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, const std::vector<int>& dilations, int groups, const std::string& data_format);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> conv3d_double_grad(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

template <typename T>
std::tuple<Tensor, Tensor> conv3d_grad(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

template <typename T>
std::tuple<Tensor, Tensor> conv3d_transpose_grad(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::vector<int>& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

template <typename T>
std::tuple<Tensor, Tensor> cos_double_grad(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor cos_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> cos_triple_grad(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad);

template <typename T>
Tensor cosh_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor crop_grad(const Tensor& x, const Tensor& out_grad, const Tensor& offsets_);

template <typename T>
Tensor crop_grad(const Tensor& x, const Tensor& out_grad, const IntArray& offsets);

template <typename T>
Tensor cross_entropy_with_softmax_grad(const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis);

template <typename T>
std::tuple<Tensor, Tensor> cross_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis);

template <typename T>
Tensor cummax_grad(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, int dtype);

template <typename T>
Tensor cummin_grad(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, int dtype);

template <typename T>
Tensor cumprod_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, int dim);

template <typename T>
Tensor cumsum_grad(const Tensor& x, const Tensor& out_grad, const Tensor& axis_, bool flatten, bool exclusive, bool reverse);

template <typename T>
Tensor cumsum_grad(const Tensor& x, const Tensor& out_grad, const Scalar& axis, bool flatten, bool exclusive, bool reverse);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> depthwise_conv2d_double_grad(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

template <typename T>
std::tuple<Tensor, Tensor> depthwise_conv2d_grad(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

template <typename T>
Tensor det_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor diag_grad(const Tensor& x, const Tensor& out_grad, int offset);

template <typename T>
Tensor diagonal_grad(const Tensor& x, const Tensor& out_grad, int offset = 0, int axis1 = 0, int axis2 = 1);

template <typename T>
Tensor digamma_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> dist_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, float p);

template <typename T>
std::tuple<Tensor, Tensor> dot_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);

template <typename T>
Tensor eig_grad(const Tensor& out_w, const Tensor& out_v, const Tensor& out_w_grad, const Tensor& out_v_grad);

template <typename T>
Tensor eigh_grad(const Tensor& out_w, const Tensor& out_v, const Tensor& out_w_grad, const Tensor& out_v_grad);

template <typename T>
Tensor eigvalsh_grad(const Tensor& eigenvectors, const Tensor& eigenvalues_grad, const std::string& uplo, bool is_test);

template <typename T>
std::tuple<Tensor, Tensor> elu_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha);

template <typename T>
Tensor elu_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, float alpha);

template <typename T>
Tensor erf_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor erfinv_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor exp_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor expand_as_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int>& target_shape);

template <typename T>
Tensor expand_grad(const Tensor& x, const Tensor& out_grad, const Tensor& shape_);

template <typename T>
Tensor expand_grad(const Tensor& x, const Tensor& out_grad, const IntArray& shape);

template <typename T>
Tensor expm1_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor fft_c2c_grad(const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward);

template <typename T>
Tensor fft_c2r_grad(const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, int64_t last_dim_size);

template <typename T>
Tensor fft_r2c_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, bool onesided);

template <typename T>
Tensor fill_diagonal_grad(const Tensor& out_grad, float value, int offset, bool wrap);

template <typename T>
Tensor fill_diagonal_tensor_grad(const Tensor& out_grad, int64_t offset, int dim1, int dim2);

template <typename T>
Tensor fill_grad(const Tensor& out_grad, const Tensor& value_);

template <typename T>
Tensor fill_grad(const Tensor& out_grad, const Scalar& value);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> flash_attn_grad(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& softmax_lse, const Tensor& seed_offset, const paddle::optional<Tensor>& attn_mask, const Tensor& out_grad, float dropout = 0.0, bool causal = false);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> flash_attn_unpadded_grad(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, const Tensor& out, const Tensor& softmax_lse, const Tensor& seed_offset, const paddle::optional<Tensor>& attn_mask, const Tensor& out_grad, int64_t max_seqlen_q, int64_t max_seqlen_k, float scale, float dropout = 0.0, bool causal = false);

template <typename T>
Tensor flatten_grad(const Tensor& xshape, const Tensor& out_grad);

template <typename T>
Tensor floor_grad(const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> fmax_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> fmin_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);

template <typename T>
Tensor fold_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int>& output_sizes, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations);

template <typename T>
Tensor frame_grad(const Tensor& x, const Tensor& out_grad, int frame_length, int hop_length, int axis);

template <typename T>
Tensor gather_grad(const Tensor& x, const Tensor& index, const Tensor& out_grad, const Tensor& axis_);

template <typename T>
Tensor gather_grad(const Tensor& x, const Tensor& index, const Tensor& out_grad, const Scalar& axis = 0);

template <typename T>
Tensor gather_nd_grad(const Tensor& x, const Tensor& index, const Tensor& out_grad);

template <typename T>
Tensor gaussian_inplace_grad(const Tensor& out_grad, float mean = 0, float std = 1.0, int seed = 0);

template <typename T>
Tensor gelu_grad(const Tensor& x, const Tensor& out_grad, bool approximate);

template <typename T>
std::tuple<Tensor, Tensor> grid_sample_grad(const Tensor& x, const Tensor& grid, const Tensor& out_grad, const std::string& mode, const std::string& padding_mode, bool align_corners);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> group_norm_grad(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& y, const Tensor& mean, const Tensor& variance, const Tensor& y_grad, float epsilon, int groups, const std::string& data_layout);

template <typename T>
Tensor gumbel_softmax_grad(const Tensor& out, const Tensor& out_grad, int axis);

template <typename T>
Tensor hardshrink_grad(const Tensor& x, const Tensor& out_grad, float threshold);

template <typename T>
Tensor hardsigmoid_grad(const Tensor& out, const Tensor& out_grad, float slope, float offset);

template <typename T>
Tensor hardtanh_grad(const Tensor& x, const Tensor& out_grad, float t_min, float t_max);

template <typename T>
std::tuple<Tensor, Tensor> heaviside_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> huber_loss_grad(const Tensor& residual, const Tensor& out_grad, float delta);

template <typename T>
Tensor i0_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor i0e_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor i1_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor i1e_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor imag_grad(const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> index_add_grad(const Tensor& index, const Tensor& add_value, const Tensor& out_grad, int axis);

template <typename T>
std::tuple<Tensor, Tensor> index_put_grad(const Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, const Tensor& out_grad, bool accumulate = false);

template <typename T>
Tensor index_sample_grad(const Tensor& x, const Tensor& index, const Tensor& out_grad);

template <typename T>
Tensor index_select_grad(const Tensor& x, const Tensor& index, const Tensor& out_grad, int axis);

template <typename T>
Tensor index_select_strided_grad(const Tensor& x, const Tensor& out_grad, int64_t index, int axis);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> instance_norm_double_grad(const Tensor& x, const paddle::optional<Tensor>& fwd_scale, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& grad_y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_scale_grad, const paddle::optional<Tensor>& grad_bias_grad, float epsilon);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> instance_norm_grad(const Tensor& x, const paddle::optional<Tensor>& scale, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& y_grad, float epsilon = 1e-5);

template <typename T>
Tensor inverse_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor kldiv_loss_grad(const Tensor& x, const Tensor& label, const Tensor& out_grad, const std::string& reduction);

template <typename T>
std::tuple<Tensor, Tensor> kron_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);

template <typename T>
Tensor kthvalue_grad(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int k, int axis, bool keepdim);

template <typename T>
Tensor label_smooth_grad(const Tensor& out_grad, float epsilon);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> layer_norm_grad(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& mean, const Tensor& variance, const Tensor& out_grad, float epsilon = 1e-5, int begin_norm_axis = 1);

template <typename T>
Tensor leaky_relu_double_grad(const Tensor& x, const Tensor& grad_x_grad, float negative_slope);

template <typename T>
Tensor leaky_relu_grad(const Tensor& x, const Tensor& out_grad, float negative_slope);

template <typename T>
std::tuple<Tensor, Tensor> lerp_grad(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor lgamma_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor linear_interp_grad(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode);

template <typename T>
Tensor log10_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor log1p_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor log2_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> log_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor log_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor log_loss_grad(const Tensor& input, const Tensor& label, const Tensor& out_grad, float epsilon);

template <typename T>
Tensor log_softmax_grad(const Tensor& out, const Tensor& out_grad, int axis);

template <typename T>
Tensor logcumsumexp_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, int axis, bool flatten, bool exclusive, bool reverse);

template <typename T>
Tensor logit_grad(const Tensor& x, const Tensor& out_grad, float eps);

template <typename T>
Tensor logsigmoid_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor lu_grad(const Tensor& x, const Tensor& out, const Tensor& pivots, const Tensor& out_grad, bool pivot);

template <typename T>
Tensor lu_unpack_grad(const Tensor& x, const Tensor& y, const Tensor& l, const Tensor& u, const Tensor& pmat, const Tensor& l_grad, const Tensor& u_grad, bool unpack_ludata, bool unpack_pivots);

template <typename T>
Tensor margin_cross_entropy_grad(const Tensor& logits, const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool return_softmax, int ring_id, int rank, int nranks, float margin1, float margin2, float margin3, float scale);

template <typename T>
Tensor masked_select_grad(const Tensor& x, const Tensor& mask, const Tensor& out_grad);

template <typename T>
Tensor matrix_power_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, int n);

template <typename T>
Tensor max_pool2d_with_index_grad(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive);

template <typename T>
Tensor max_pool3d_with_index_grad(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive);

template <typename T>
Tensor maxout_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, int groups, int axis);

template <typename T>
Tensor mean_all_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor> memory_efficient_attention_grad(const Tensor& query, const Tensor& key, const Tensor& value, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& cu_seqlens_q, const paddle::optional<Tensor>& cu_seqlens_k, const Tensor& output, const Tensor& logsumexp, const Tensor& seed_and_offset, const Tensor& output_grad, const Scalar& max_seqlen_q, const Scalar& max_seqlen_k, bool causal, double dropout_p, float scale);

template <typename T>
std::vector<Tensor> meshgrid_grad(const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs_grad);

template <typename T>
Tensor mode_grad(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, bool keepdim);

template <typename T>
std::vector<Tensor> multi_dot_grad(const std::vector<Tensor>& x, const Tensor& out_grad);

template <typename T>
std::vector<Tensor> multiplex_grad(const std::vector<Tensor>& inputs, const Tensor& index, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> mv_grad(const Tensor& x, const Tensor& vec, const Tensor& out_grad);

template <typename T>
Tensor nanmedian_grad(const Tensor& x, const Tensor& medians, const Tensor& out_grad, const IntArray& axis, bool keepdim);

template <typename T>
Tensor nearest_interp_grad(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode);

template <typename T>
Tensor nll_loss_grad(const Tensor& input, const Tensor& label, const paddle::optional<Tensor>& weight, const Tensor& total_weight, const Tensor& out_grad, int64_t ignore_index, const std::string& reduction);

template <typename T>
Tensor overlap_add_grad(const Tensor& x, const Tensor& out_grad, int hop_length, int axis);

template <typename T>
Tensor p_norm_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, float porder, int axis, float epsilon, bool keepdim, bool asvector);

template <typename T>
Tensor pad3d_double_grad(const Tensor& grad_x_grad, const Tensor& paddings_, const std::string& mode, float pad_value, const std::string& data_format);

template <typename T>
Tensor pad3d_double_grad(const Tensor& grad_x_grad, const IntArray& paddings, const std::string& mode, float pad_value, const std::string& data_format);

template <typename T>
Tensor pad3d_grad(const Tensor& x, const Tensor& out_grad, const Tensor& paddings_, const std::string& mode, float pad_value, const std::string& data_format);

template <typename T>
Tensor pad3d_grad(const Tensor& x, const Tensor& out_grad, const IntArray& paddings, const std::string& mode, float pad_value, const std::string& data_format);

template <typename T>
Tensor pixel_shuffle_grad(const Tensor& out_grad, int upscale_factor, const std::string& data_format);

template <typename T>
Tensor pixel_unshuffle_grad(const Tensor& out_grad, int downscale_factor, const std::string& data_format);

template <typename T>
Tensor poisson_grad(const Tensor& out_grad);

template <typename T>
Tensor polygamma_grad(const Tensor& x, const Tensor& out_grad, int n);

template <typename T>
std::tuple<Tensor, Tensor> pow_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const Scalar& y);

template <typename T>
Tensor pow_grad(const Tensor& x, const Tensor& out_grad, const Scalar& y = -1);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> pow_triple_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_grad_x, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_grad_out_grad, const Scalar& y);

template <typename T>
std::tuple<Tensor, Tensor> prelu_grad(const Tensor& x, const Tensor& alpha, const Tensor& out_grad, const std::string& data_format, const std::string& mode);

template <typename T>
Tensor psroi_pool_grad(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& out_grad, int pooled_height, int pooled_width, int output_channels, float spatial_scale);

template <typename T>
std::tuple<Tensor, Tensor> put_along_axis_grad(const Tensor& arr, const Tensor& indices, const Tensor& out_grad, int axis, const std::string& reduce);

template <typename T>
Tensor qr_grad(const Tensor& x, const Tensor& q, const Tensor& r, const Tensor& q_grad, const Tensor& r_grad, const std::string& mode);

template <typename T>
Tensor real_grad(const Tensor& out_grad);

template <typename T>
Tensor reciprocal_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor relu6_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor relu_double_grad(const Tensor& out, const Tensor& grad_x_grad);

template <typename T>
Tensor relu_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor renorm_grad(const Tensor& x, const Tensor& out_grad, float p, int axis, float max_norm);

template <typename T>
Tensor roi_align_grad(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& out_grad, int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned);

template <typename T>
Tensor roi_pool_grad(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& arg_max, const Tensor& out_grad, int pooled_height, int pooled_width, float spatial_scale);

template <typename T>
Tensor roll_grad(const Tensor& x, const Tensor& out_grad, const Tensor& shifts_, const std::vector<int64_t>& axis);

template <typename T>
Tensor roll_grad(const Tensor& x, const Tensor& out_grad, const IntArray& shifts, const std::vector<int64_t>& axis);

template <typename T>
Tensor round_grad(const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> rsqrt_double_grad(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad);

template <typename T>
Tensor rsqrt_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> scatter_grad(const Tensor& index, const Tensor& updates, const Tensor& out_grad, bool overwrite);

template <typename T>
std::tuple<Tensor, Tensor> scatter_nd_add_grad(const Tensor& index, const Tensor& updates, const Tensor& out_grad);

template <typename T>
Tensor segment_pool_grad(const Tensor& x, const Tensor& segment_ids, const Tensor& out, const paddle::optional<Tensor>& summed_ids, const Tensor& out_grad, const std::string& pooltype);

template <typename T>
Tensor selu_grad(const Tensor& out, const Tensor& out_grad, float scale, float alpha);

template <typename T>
Tensor send_u_recv_grad(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const paddle::optional<Tensor>& out, const paddle::optional<Tensor>& dst_count, const Tensor& out_grad, const std::string& reduce_op = "SUM");

template <typename T>
std::tuple<Tensor, Tensor> send_ue_recv_grad(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const paddle::optional<Tensor>& out, const paddle::optional<Tensor>& dst_count, const Tensor& out_grad, const std::string& message_op, const std::string& reduce_op);

template <typename T>
std::tuple<Tensor, Tensor> send_uv_grad(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const Tensor& out_grad, const std::string& message_op = "ADD");

template <typename T>
Tensor sigmoid_cross_entropy_with_logits_grad(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, const Tensor& out_grad, bool normalize, int ignore_index);

template <typename T>
std::tuple<Tensor, Tensor> sigmoid_double_grad(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor sigmoid_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> sigmoid_triple_grad(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_grad_x, const Tensor& grad_out_grad, const paddle::optional<Tensor>& grad_grad_out_grad);

template <typename T>
Tensor silu_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> sin_double_grad(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor sin_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> sin_triple_grad(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad);

template <typename T>
Tensor sinh_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor slogdet_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> softplus_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float beta, float threshold);

template <typename T>
Tensor softplus_grad(const Tensor& x, const Tensor& out_grad, float beta, float threshold);

template <typename T>
Tensor softshrink_grad(const Tensor& x, const Tensor& out_grad, float threshold);

template <typename T>
Tensor softsign_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> solve_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor spectral_norm_grad(const Tensor& weight, const Tensor& u, const Tensor& v, const Tensor& out_grad, int dim, int power_iters, float eps);

template <typename T>
std::tuple<Tensor, Tensor> sqrt_double_grad(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad);

template <typename T>
Tensor sqrt_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> square_double_grad(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor square_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor squared_l2_norm_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor squeeze_grad(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_);

template <typename T>
Tensor squeeze_grad(const Tensor& xshape, const Tensor& out_grad, const IntArray& axis);

template <typename T>
std::vector<Tensor> stack_grad(const std::vector<Tensor>& x, const Tensor& out_grad, int axis);

template <typename T>
Tensor stanh_grad(const Tensor& x, const Tensor& out_grad, float scale_a, float scale_b);

template <typename T>
Tensor svd_grad(const Tensor& x, const Tensor& u, const Tensor& vh, const Tensor& s, const paddle::optional<Tensor>& u_grad, const paddle::optional<Tensor>& vh_grad, const paddle::optional<Tensor>& s_grad, bool full_matrices);

template <typename T>
Tensor take_along_axis_grad(const Tensor& arr, const Tensor& indices, const Tensor& out_grad, int axis);

template <typename T>
Tensor tan_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> tanh_double_grad(const Tensor& out, const Tensor& grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor tanh_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor tanh_shrink_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> tanh_triple_grad(const Tensor& out, const Tensor& grad_out_forward, const Tensor& grad_x_grad_forward, const paddle::optional<Tensor>& grad_out_new_grad, const paddle::optional<Tensor>& grad_out_grad_grad);

template <typename T>
Tensor temporal_shift_grad(const Tensor& out_grad, int seg_num, float shift_ratio, const std::string& data_format);

template <typename T>
Tensor tensor_unfold_grad(const Tensor& input, const Tensor& out_grad, int64_t axis, int64_t size, int64_t step);

template <typename T>
Tensor thresholded_relu_grad(const Tensor& x, const Tensor& out_grad, float threshold);

template <typename T>
Tensor topk_grad(const Tensor& x, const Tensor& indices, const Tensor& out_grad, const Tensor& k_, int axis, bool largest, bool sorted);

template <typename T>
Tensor topk_grad(const Tensor& x, const Tensor& indices, const Tensor& out_grad, const Scalar& k, int axis, bool largest, bool sorted);

template <typename T>
Tensor trace_grad(const Tensor& x, const Tensor& out_grad, int offset, int axis1, int axis2);

template <typename T>
std::tuple<Tensor, Tensor> triangular_solve_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, bool upper, bool transpose, bool unitriangular);

template <typename T>
Tensor trilinear_interp_grad(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode);

template <typename T>
Tensor trunc_grad(const Tensor& out_grad);

template <typename T>
Tensor unfold_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations);

template <typename T>
Tensor uniform_inplace_grad(const Tensor& out_grad, float min = -1.0, float max = 1.0, int seed = 0, int diag_num = 0, int diag_step = 0, float diag_val = 1.0);

template <typename T>
Tensor unsqueeze_grad(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_);

template <typename T>
Tensor unsqueeze_grad(const Tensor& xshape, const Tensor& out_grad, const IntArray& axis);

template <typename T>
Tensor unstack_grad(const std::vector<Tensor>& out_grad, int axis);

template <typename T>
Tensor view_dtype_grad(const Tensor& input, const Tensor& out_grad, DataType dtype);

template <typename T>
Tensor view_shape_grad(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& dims = {});

template <typename T>
Tensor warpctc_grad(const Tensor& logits, const paddle::optional<Tensor>& logits_length, const Tensor& warpctcgrad, const Tensor& loss_grad, int blank, bool norm_by_times);

template <typename T>
Tensor warprnnt_grad(const Tensor& input, const Tensor& input_lengths, const Tensor& warprnntgrad, const Tensor& loss_grad, int blank = 0, float fastemit_lambda = 0.0);

template <typename T>
Tensor weight_only_linear_grad(const Tensor& x, const Tensor& weight, const paddle::optional<Tensor>& bias, const Tensor& weight_scale, const Tensor& out_grad, const std::string& weight_dtype);

template <typename T>
std::tuple<Tensor, Tensor> where_grad(const Tensor& condition, const Tensor& x, const Tensor& y, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor> yolo_loss_grad(const Tensor& x, const Tensor& gt_box, const Tensor& gt_label, const paddle::optional<Tensor>& gt_score, const Tensor& objectness_mask, const Tensor& gt_match_mask, const Tensor& loss_grad, const std::vector<int>& anchors, const std::vector<int>& anchor_mask, int class_num, float ignore_thresh, int downsample_ratio, bool use_label_smooth, float scale_x_y);

template <typename T>
Tensor unpool3d_grad(const Tensor& x, const Tensor& indices, const Tensor& out, const Tensor& out_grad, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_size, const std::string& data_format);

template <typename T>
Tensor add_double_grad(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis = -1);

template <typename T>
std::tuple<Tensor, Tensor> add_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis = -1);

template <typename T>
std::tuple<Tensor, Tensor> add_triple_grad(const Tensor& grad_grad_x, const Tensor& grad_grad_y, const Tensor& grad_grad_out_grad, int axis = -1);

template <typename T>
Tensor amax_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis = {}, bool keepdim = false, bool reduce_all = false);

template <typename T>
Tensor amin_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis = {}, bool keepdim = false, bool reduce_all = false);

template <typename T>
Tensor assign_out__grad(const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> batch_norm_double_grad(const Tensor& x, const Tensor& scale, const paddle::optional<Tensor>& out_mean, const paddle::optional<Tensor>& out_variance, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_scale_grad, const paddle::optional<Tensor>& grad_bias_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> batch_norm_grad(const Tensor& x, const Tensor& scale, const Tensor& bias, const paddle::optional<Tensor>& mean_out, const paddle::optional<Tensor>& variance_out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics);

template <typename T>
Tensor c_embedding_grad(const Tensor& weight, const Tensor& x, const Tensor& out_grad, int64_t start_index = 0);

template <typename T>
Tensor channel_shuffle_grad(const Tensor& out_grad, int groups, const std::string& data_format = "NCHW");

template <typename T>
std::tuple<Tensor, Tensor, Tensor> conv2d_transpose_double_grad(const Tensor& x, const Tensor& filter, const Tensor& grad_out, const Tensor& grad_x_grad, const Tensor& grad_filter_grad, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> conv2d_transpose_double_grad(const Tensor& x, const Tensor& filter, const Tensor& grad_out, const Tensor& grad_x_grad, const Tensor& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

template <typename T>
std::tuple<Tensor, Tensor> conv2d_transpose_grad(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

template <typename T>
std::tuple<Tensor, Tensor> conv2d_transpose_grad(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor> deformable_conv_grad(const Tensor& x, const Tensor& offset, const Tensor& filter, const paddle::optional<Tensor>& mask, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, int deformable_groups, int groups, int im2col_step);

template <typename T>
std::tuple<Tensor, Tensor> depthwise_conv2d_transpose_grad(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

template <typename T>
std::tuple<Tensor, Tensor> depthwise_conv2d_transpose_grad(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> divide_double_grad(const Tensor& y, const Tensor& out, const Tensor& grad_x, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis = -1);

template <typename T>
std::tuple<Tensor, Tensor> divide_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis = -1);

template <typename T>
Tensor dropout_grad(const Tensor& mask, const Tensor& out_grad, const Scalar& p, bool is_test, const std::string& mode);

template <typename T>
std::vector<Tensor> einsum_grad(const std::vector<Tensor>& x_shape, const std::vector<Tensor>& inner_cache, const Tensor& out_grad, const std::string& equation);

template <typename T>
std::tuple<Tensor, Tensor> elementwise_pow_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);

template <typename T>
Tensor frobenius_norm_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keep_dim, bool reduce_all);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> fused_batch_norm_act_grad(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& act_type);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor> fused_bn_add_activation_grad(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& act_type);

template <typename T>
Tensor fused_softmax_mask_upper_triangle_grad(const Tensor& Out, const Tensor& Out_grad);

template <typename T>
Tensor hardswish_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> hsigmoid_loss_grad(const Tensor& x, const Tensor& w, const Tensor& label, const paddle::optional<Tensor>& path, const paddle::optional<Tensor>& code, const paddle::optional<Tensor>& bias, const Tensor& pre_out, const Tensor& out_grad, int num_classes, bool is_sparse);

template <typename T>
Tensor logsumexp_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> matmul_double_grad(const Tensor& x, const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, bool transpose_x = false, bool transpose_y = false);

template <typename T>
std::tuple<Tensor, Tensor> matmul_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, bool transpose_x = false, bool transpose_y = false);

template <typename T>
Tensor max_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& axis_, bool keepdim = false, bool reduce_all = false);

template <typename T>
Tensor max_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& axis = {}, bool keepdim = false, bool reduce_all = false);

template <typename T>
std::tuple<Tensor, Tensor> maximum_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);

template <typename T>
Tensor mean_grad(const Tensor& x, const Tensor& out_grad, const IntArray& axis = {}, bool keepdim = false, bool reduce_all = false);

template <typename T>
Tensor min_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& axis_, bool keepdim = false, bool reduce_all = false);

template <typename T>
Tensor min_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& axis = {}, bool keepdim = false, bool reduce_all = false);

template <typename T>
std::tuple<Tensor, Tensor> minimum_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);

template <typename T>
Tensor mish_grad(const Tensor& x, const Tensor& out_grad, float threshold);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> multiply_double_grad(const Tensor& x, const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis = -1);

template <typename T>
std::tuple<Tensor, Tensor> multiply_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis = -1);

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> multiply_triple_grad(const Tensor& x, const Tensor& y, const Tensor& fwd_grad_out, const paddle::optional<Tensor>& fwd_grad_grad_x, const paddle::optional<Tensor>& fwd_grad_grad_y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, const paddle::optional<Tensor>& grad_grad_out_grad, int axis = -1);

template <typename T>
Tensor norm_grad(const Tensor& x, const Tensor& norm, const Tensor& out_grad, int axis, float epsilon, bool is_test);

template <typename T>
Tensor pad_double_grad(const Tensor& grad_x_grad, const std::vector<int>& paddings, const Scalar& pad_value);

template <typename T>
Tensor pad_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int>& paddings, const Scalar& pad_value);

template <typename T>
Tensor pool2d_double_grad(const Tensor& x, const Tensor& grad_x_grad, const Tensor& kernel_size_, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm);

template <typename T>
Tensor pool2d_double_grad(const Tensor& x, const Tensor& grad_x_grad, const IntArray& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm);

template <typename T>
Tensor pool2d_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& kernel_size_, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm);

template <typename T>
Tensor pool2d_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm);

template <typename T>
Tensor pool3d_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm);

template <typename T>
Tensor prod_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& dims_, bool keep_dim, bool reduce_all);

template <typename T>
Tensor prod_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& dims, bool keep_dim, bool reduce_all);

template <typename T>
Tensor repeat_interleave_grad(const Tensor& x, const Tensor& out_grad, int repeats, int axis);

template <typename T>
Tensor repeat_interleave_with_tensor_index_grad(const Tensor& x, const Tensor& repeats, const Tensor& out_grad, int axis);

template <typename T>
Tensor reshape_double_grad(const Tensor& grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor reshape_grad(const Tensor& xshape, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> rnn_grad(const Tensor& x, const std::vector<Tensor>& pre_state, const std::vector<Tensor>& weight_list, const paddle::optional<Tensor>& sequence_length, const Tensor& out, const Tensor& dropout_state_out, const Tensor& reserve, const Tensor& out_grad, const std::vector<Tensor>& state_grad, float dropout_prob, bool is_bidirec, int input_size, int hidden_size, int num_layers, const std::string& mode, int seed, bool is_test);

template <typename T>
Tensor rrelu_grad(const Tensor& x, const Tensor& noise, const Tensor& out_grad);

template <typename T>
Tensor slice_grad(const Tensor& input, const Tensor& out_grad, const Tensor& starts_, const Tensor& ends_, const std::vector<int64_t>& axes, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis);

template <typename T>
Tensor slice_grad(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& axes, const IntArray& starts, const IntArray& ends, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis);

template <typename T>
Tensor softmax_grad(const Tensor& out, const Tensor& out_grad, int axis);

template <typename T>
Tensor strided_slice_grad(const Tensor& x, const Tensor& out_grad, const Tensor& starts_, const Tensor& ends_, const Tensor& strides_, const std::vector<int>& axes);

template <typename T>
Tensor strided_slice_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int>& axes, const IntArray& starts, const IntArray& ends, const IntArray& strides);

template <typename T>
Tensor subtract_double_grad(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis = -1);

template <typename T>
std::tuple<Tensor, Tensor> subtract_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis = -1);

template <typename T>
Tensor sum_grad(const Tensor& x, const Tensor& out_grad, const Tensor& axis_, bool keepdim, bool reduce_all = false);

template <typename T>
Tensor sum_grad(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all = false);

template <typename T>
Tensor swish_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> sync_batch_norm_grad(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics);

template <typename T>
Tensor tile_grad(const Tensor& x, const Tensor& out_grad, const Tensor& repeat_times_);

template <typename T>
Tensor tile_grad(const Tensor& x, const Tensor& out_grad, const IntArray& repeat_times);

template <typename T>
Tensor trans_layout_grad(const Tensor& x, const Tensor& out_grad, const std::vector<int>& perm);

template <typename T>
Tensor transpose_grad(const Tensor& out_grad, const std::vector<int>& perm);

template <typename T>
Tensor tril_grad(const Tensor& out_grad, int diagonal);

template <typename T>
Tensor triu_grad(const Tensor& out_grad, int diagonal);

template <typename T>
Tensor disable_check_model_nan_inf_grad(const Tensor& out_grad, int unsetflag = 1);

template <typename T>
Tensor enable_check_model_nan_inf_grad(const Tensor& out_grad, int unsetflag = 0);

template <typename T>
Tensor unpool_grad(const Tensor& x, const Tensor& indices, const Tensor& out, const Tensor& out_grad, const Tensor& output_size_, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const std::string& data_format);

template <typename T>
Tensor unpool_grad(const Tensor& x, const Tensor& indices, const Tensor& out, const Tensor& out_grad, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const IntArray& output_size, const std::string& data_format);

template <typename T>
Tensor abs_(const Tensor& x);

template <typename T>
Tensor acos_(const Tensor& x);

template <typename T>
Tensor acosh_(const Tensor& x);

template <typename T>
Tensor addmm_(const Tensor& input, const Tensor& x, const Tensor& y, float beta = 1.0, float alpha = 1.0);

template <typename T>
Tensor asin_(const Tensor& x);

template <typename T>
Tensor asinh_(const Tensor& x);

template <typename T>
Tensor atan_(const Tensor& x);

template <typename T>
Tensor atanh_(const Tensor& x);

template <typename T>
Tensor bce_loss_(const Tensor& input, const Tensor& label);

template <typename T>
Tensor bitwise_and_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor bitwise_not_(const Tensor& x);

template <typename T>
Tensor bitwise_or_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor bitwise_xor_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor ceil_(const Tensor& x);

template <typename T>
Tensor clip_(const Tensor& x, const Tensor& min_, const Tensor& max_);

template <typename T>
Tensor clip_(const Tensor& x, const Scalar& min, const Scalar& max);

template <typename T>
Tensor cos_(const Tensor& x);

template <typename T>
Tensor cosh_(const Tensor& x);

template <typename T>
std::tuple<Tensor, Tensor> cross_entropy_with_softmax_(const Tensor& input, const Tensor& label, bool soft_label = false, bool use_softmax = true, bool numeric_stable_mode = true, int ignore_index = -100, int axis = -1);

template <typename T>
Tensor cumprod_(const Tensor& x, int dim);

template <typename T>
Tensor cumsum_(const Tensor& x, const Tensor& axis_, bool flatten = false, bool exclusive = false, bool reverse = false);

template <typename T>
Tensor cumsum_(const Tensor& x, const Scalar& axis = -1, bool flatten = false, bool exclusive = false, bool reverse = false);

template <typename T>
Tensor digamma_(const Tensor& x);

template <typename T>
Tensor elu_(const Tensor& x, float alpha = 1.0f);

template <typename T>
Tensor erf_(const Tensor& x);

template <typename T>
Tensor erfinv_(const Tensor& x);

template <typename T>
Tensor exp_(const Tensor& x);

template <typename T>
Tensor expm1_(const Tensor& x);

template <typename T>
Tensor fill_(const Tensor& x, const Tensor& value_);

template <typename T>
Tensor fill_(const Tensor& x, const Scalar& value = 0);

template <typename T>
Tensor fill_diagonal_(const Tensor& x, float value = 0, int offset = 0, bool wrap = false);

template <typename T>
Tensor fill_diagonal_tensor_(const Tensor& x, const Tensor& y, int64_t offset = 0, int dim1 = 0, int dim2 = 1);

template <typename T>
Tensor flatten_(const Tensor& x, int start_axis = 1, int stop_axis = 1);

template <typename T>
Tensor floor_(const Tensor& x);

template <typename T>
Tensor gaussian_inplace_(const Tensor& x, float mean = 0, float std = 1.0, int seed = 0);

template <typename T>
Tensor hardtanh_(const Tensor& x, float t_min = 0, float t_max = 24);

template <typename T>
Tensor i0_(const Tensor& x);

template <typename T>
Tensor index_add_(const Tensor& x, const Tensor& index, const Tensor& add_value, int axis = 0);

template <typename T>
Tensor index_put_(const Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, bool accumulate = false);

template <typename T>
Tensor leaky_relu_(const Tensor& x, float negative_slope = 0.02f);

template <typename T>
Tensor lerp_(const Tensor& x, const Tensor& y, const Tensor& weight);

template <typename T>
Tensor lgamma_(const Tensor& x);

template <typename T>
Tensor log_(const Tensor& x);

template <typename T>
Tensor log10_(const Tensor& x);

template <typename T>
Tensor log1p_(const Tensor& x);

template <typename T>
Tensor log2_(const Tensor& x);

template <typename T>
Tensor logical_and_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor logical_not_(const Tensor& x);

template <typename T>
Tensor logical_or_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor logical_xor_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor logit_(const Tensor& x, float eps = 1e-6f);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> lu_(const Tensor& x, bool pivot = true);

template <typename T>
Tensor polygamma_(const Tensor& x, int n);

template <typename T>
Tensor pow_(const Tensor& x, const Scalar& y = 1.0f);

template <typename T>
Tensor put_along_axis_(const Tensor& arr, const Tensor& indices, const Tensor& values, int axis, const std::string& reduce = "assign");

template <typename T>
Tensor reciprocal_(const Tensor& x);

template <typename T>
Tensor relu_(const Tensor& x);

template <typename T>
Tensor renorm_(const Tensor& x, float p, int axis, float max_norm);

template <typename T>
Tensor round_(const Tensor& x);

template <typename T>
Tensor rsqrt_(const Tensor& x);

template <typename T>
Tensor scale_(const Tensor& x, const Tensor& scale_, float bias = 0.0, bool bias_after_scale = true);

template <typename T>
Tensor scale_(const Tensor& x, const Scalar& scale = 1.0, float bias = 0.0, bool bias_after_scale = true);

template <typename T>
Tensor scatter_(const Tensor& x, const Tensor& index, const Tensor& updates, bool overwrite = true);

template <typename T>
Tensor sigmoid_(const Tensor& x);

template <typename T>
Tensor sigmoid_cross_entropy_with_logits_(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, bool normalize = false, int ignore_index = -100);

template <typename T>
Tensor sin_(const Tensor& x);

template <typename T>
Tensor sinh_(const Tensor& x);

template <typename T>
Tensor sqrt_(const Tensor& x);

template <typename T>
Tensor squeeze_(const Tensor& x, const Tensor& axis_);

template <typename T>
Tensor squeeze_(const Tensor& x, const IntArray& axis = {});

template <typename T>
Tensor tan_(const Tensor& x);

template <typename T>
Tensor tanh_(const Tensor& x);

template <typename T>
Tensor thresholded_relu_(const Tensor& x, float threshold = 1.0);

template <typename T>
Tensor trunc_(const Tensor& input);

template <typename T>
Tensor uniform_inplace_(const Tensor& x, float min = -1.0, float max = 1.0, int seed = 0, int diag_num = 0, int diag_step = 0, float diag_val = 1.0);

template <typename T>
Tensor unsqueeze_(const Tensor& x, const Tensor& axis_);

template <typename T>
Tensor unsqueeze_(const Tensor& x, const IntArray& axis = {});

template <typename T>
Tensor where_(const Tensor& condition, const Tensor& x, const Tensor& y);

template <typename T>
Tensor add_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor assign_(const Tensor& x);

template <typename T>
Tensor c_allreduce_max_(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

template <typename T>
Tensor c_allreduce_sum_(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

template <typename T>
Tensor c_broadcast_(const Tensor& x, int ring_id = 0, int root = 0, bool use_calc_stream = false);

template <typename T>
Tensor c_identity_(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

template <typename T>
Tensor c_reduce_sum_(const Tensor& x, int ring_id, int root_id, bool use_calc_stream);

template <typename T>
Tensor c_sync_calc_stream_(const Tensor& x);

template <typename T>
Tensor c_sync_comm_stream_(const Tensor& x);

template <typename T>
Tensor cast_(const Tensor& x, DataType dtype);

template <typename T>
Tensor divide_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor equal_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor floor_divide_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor greater_equal_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor greater_than_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor increment_(const Tensor& x, float value = 1.0);

template <typename T>
Tensor less_equal_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor less_than_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor multiply_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor not_equal_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor remainder_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor reshape_(const Tensor& x, const Tensor& shape_);

template <typename T>
Tensor reshape_(const Tensor& x, const IntArray& shape);

template <typename T>
Tensor softmax_(const Tensor& x, int axis);

template <typename T>
Tensor subtract_(const Tensor& x, const Tensor& y);

template <typename T>
Tensor transpose_(const Tensor& x, const std::vector<int>& perm);

template <typename T>
Tensor tril_(const Tensor& x, int diagonal);

template <typename T>
Tensor triu_(const Tensor& x, int diagonal);

template <typename T>
Tensor acos_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor acosh_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor asin_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor asinh_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor atan_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor atanh_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor bce_loss_grad_(const Tensor& input, const Tensor& label, const Tensor& out_grad);

template <typename T>
Tensor ceil_grad_(const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> celu_double_grad_(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha);

template <typename T>
Tensor celu_grad_(const Tensor& x, const Tensor& out_grad, float alpha);

template <typename T>
Tensor clip_grad_(const Tensor& x, const Tensor& out_grad, const Tensor& min_, const Tensor& max_);

template <typename T>
Tensor clip_grad_(const Tensor& x, const Tensor& out_grad, const Scalar& min = 0., const Scalar& max = 0.);

template <typename T>
std::tuple<Tensor, Tensor> cos_double_grad_(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor cos_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> cos_triple_grad_(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad);

template <typename T>
Tensor cosh_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor cross_entropy_with_softmax_grad_(const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis);

template <typename T>
std::tuple<Tensor, Tensor> elu_double_grad_(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha);

template <typename T>
Tensor elu_grad_(const Tensor& x, const Tensor& out, const Tensor& out_grad, float alpha);

template <typename T>
Tensor exp_grad_(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor expm1_grad_(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor fill_diagonal_tensor_grad_(const Tensor& out_grad, int64_t offset, int dim1, int dim2);

template <typename T>
Tensor fill_grad_(const Tensor& out_grad, const Tensor& value_);

template <typename T>
Tensor fill_grad_(const Tensor& out_grad, const Scalar& value);

template <typename T>
Tensor flatten_grad_(const Tensor& xshape, const Tensor& out_grad);

template <typename T>
Tensor floor_grad_(const Tensor& out_grad);

template <typename T>
Tensor gaussian_inplace_grad_(const Tensor& out_grad, float mean = 0, float std = 1.0, int seed = 0);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> group_norm_grad_(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& y, const Tensor& mean, const Tensor& variance, const Tensor& y_grad, float epsilon, int groups, const std::string& data_layout);

template <typename T>
Tensor hardshrink_grad_(const Tensor& x, const Tensor& out_grad, float threshold);

template <typename T>
Tensor hardsigmoid_grad_(const Tensor& out, const Tensor& out_grad, float slope, float offset);

template <typename T>
Tensor hardtanh_grad_(const Tensor& x, const Tensor& out_grad, float t_min, float t_max);

template <typename T>
std::tuple<Tensor, Tensor> index_add_grad_(const Tensor& index, const Tensor& add_value, const Tensor& out_grad, int axis);

template <typename T>
Tensor leaky_relu_double_grad_(const Tensor& x, const Tensor& grad_x_grad, float negative_slope);

template <typename T>
Tensor leaky_relu_grad_(const Tensor& x, const Tensor& out_grad, float negative_slope);

template <typename T>
Tensor log10_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor log1p_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor log2_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> log_double_grad_(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor log_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor logsigmoid_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor lu_grad_(const Tensor& x, const Tensor& out, const Tensor& pivots, const Tensor& out_grad, bool pivot);

template <typename T>
Tensor margin_cross_entropy_grad_(const Tensor& logits, const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool return_softmax, int ring_id, int rank, int nranks, float margin1, float margin2, float margin3, float scale);

template <typename T>
std::tuple<Tensor, Tensor> pow_double_grad_(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const Scalar& y);

template <typename T>
Tensor pow_grad_(const Tensor& x, const Tensor& out_grad, const Scalar& y = -1);

template <typename T>
Tensor reciprocal_grad_(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor relu6_grad_(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor relu_double_grad_(const Tensor& out, const Tensor& grad_x_grad);

template <typename T>
Tensor relu_grad_(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor round_grad_(const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> rsqrt_double_grad_(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad);

template <typename T>
Tensor rsqrt_grad_(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor sigmoid_cross_entropy_with_logits_grad_(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, const Tensor& out_grad, bool normalize, int ignore_index);

template <typename T>
std::tuple<Tensor, Tensor> sigmoid_double_grad_(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor sigmoid_grad_(const Tensor& out, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> sigmoid_triple_grad_(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_grad_x, const Tensor& grad_out_grad, const paddle::optional<Tensor>& grad_grad_out_grad);

template <typename T>
Tensor silu_grad_(const Tensor& x, const Tensor& out, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> sin_double_grad_(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor sin_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> sin_triple_grad_(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad);

template <typename T>
Tensor sinh_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> softplus_double_grad_(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float beta, float threshold);

template <typename T>
Tensor softplus_grad_(const Tensor& x, const Tensor& out_grad, float beta, float threshold);

template <typename T>
Tensor softshrink_grad_(const Tensor& x, const Tensor& out_grad, float threshold);

template <typename T>
Tensor softsign_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> sqrt_double_grad_(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad);

template <typename T>
Tensor sqrt_grad_(const Tensor& out, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> square_double_grad_(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor square_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor squeeze_grad_(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_);

template <typename T>
Tensor squeeze_grad_(const Tensor& xshape, const Tensor& out_grad, const IntArray& axis);

template <typename T>
Tensor tan_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> tanh_double_grad_(const Tensor& out, const Tensor& grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor tanh_grad_(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor tanh_shrink_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> tanh_triple_grad_(const Tensor& out, const Tensor& grad_out_forward, const Tensor& grad_x_grad_forward, const paddle::optional<Tensor>& grad_out_new_grad, const paddle::optional<Tensor>& grad_out_grad_grad);

template <typename T>
Tensor thresholded_relu_grad_(const Tensor& x, const Tensor& out_grad, float threshold);

template <typename T>
Tensor uniform_inplace_grad_(const Tensor& out_grad, float min = -1.0, float max = 1.0, int seed = 0, int diag_num = 0, int diag_step = 0, float diag_val = 1.0);

template <typename T>
Tensor unsqueeze_grad_(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_);

template <typename T>
Tensor unsqueeze_grad_(const Tensor& xshape, const Tensor& out_grad, const IntArray& axis);

template <typename T>
Tensor add_double_grad_(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis = -1);

template <typename T>
std::tuple<Tensor, Tensor> add_grad_(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis = -1);

template <typename T>
std::tuple<Tensor, Tensor> add_triple_grad_(const Tensor& grad_grad_x, const Tensor& grad_grad_y, const Tensor& grad_grad_out_grad, int axis = -1);

template <typename T>
Tensor assign_out__grad_(const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> batch_norm_double_grad_(const Tensor& x, const Tensor& scale, const paddle::optional<Tensor>& out_mean, const paddle::optional<Tensor>& out_variance, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_scale_grad, const paddle::optional<Tensor>& grad_bias_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> divide_double_grad_(const Tensor& y, const Tensor& out, const Tensor& grad_x, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis = -1);

template <typename T>
Tensor hardswish_grad_(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor mish_grad_(const Tensor& x, const Tensor& out_grad, float threshold);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> multiply_double_grad_(const Tensor& x, const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis = -1);

template <typename T>
Tensor reshape_double_grad_(const Tensor& grad_out, const Tensor& grad_x_grad);

template <typename T>
Tensor reshape_grad_(const Tensor& xshape, const Tensor& out_grad);

template <typename T>
Tensor subtract_double_grad_(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis = -1);

template <typename T>
std::tuple<Tensor, Tensor> subtract_grad_(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis = -1);

template <typename T>
Tensor swish_grad_(const Tensor& x, const Tensor& out_grad);

}  // namespace backend
}  // namespace primitive
}  // namespace paddle
