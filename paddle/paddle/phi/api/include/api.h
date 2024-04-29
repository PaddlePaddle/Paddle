#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {


PADDLE_API Tensor abs(const Tensor& x);

PADDLE_API Tensor& abs_(Tensor& x);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> accuracy(const Tensor& x, const Tensor& indices, const Tensor& label);

PADDLE_API Tensor acos(const Tensor& x);

PADDLE_API Tensor& acos_(Tensor& x);

PADDLE_API Tensor acosh(const Tensor& x);

PADDLE_API Tensor& acosh_(Tensor& x);

PADDLE_API std::tuple<Tensor&, Tensor&, paddle::optional<Tensor>&> adagrad_(Tensor& param, const Tensor& grad, Tensor& moment, const Tensor& learning_rate, paddle::optional<Tensor>& master_param, float epsilon = 1.0e-6f, bool multi_precision = false);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> adam_(Tensor& param, const Tensor& grad, const Tensor& learning_rate, Tensor& moment1, Tensor& moment2, Tensor& beta1_pow, Tensor& beta2_pow, paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Scalar& beta1 = 0.9f, const Scalar& beta2 = 0.999f, const Scalar& epsilon = 1.0e-8f, bool lazy_mode = false, int64_t min_row_size_to_use_multithread = 1000, bool multi_precision = false, bool use_global_beta_pow = false);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> adamax_(Tensor& param, const Tensor& grad, const Tensor& learning_rate, Tensor& moment, Tensor& inf_norm, const Tensor& beta1_pow, paddle::optional<Tensor>& master_param, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1.0e-8f, bool multi_precision = false);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> adamw_(Tensor& param, const Tensor& grad, const Tensor& learning_rate, Tensor& moment1, Tensor& moment2, Tensor& beta1_pow, Tensor& beta2_pow, paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Scalar& beta1 = 0.9f, const Scalar& beta2 = 0.999f, const Scalar& epsilon = 1.0e-8f, float lr_ratio = 1.0f, float coeff = 0.01f, bool with_decay = false, bool lazy_mode = false, int64_t min_row_size_to_use_multithread = 1000, bool multi_precision = false, bool use_global_beta_pow = false);

PADDLE_API Tensor addmm(const Tensor& input, const Tensor& x, const Tensor& y, float beta = 1.0, float alpha = 1.0);

PADDLE_API Tensor& addmm_(Tensor& input, const Tensor& x, const Tensor& y, float beta = 1.0, float alpha = 1.0);

PADDLE_API Tensor affine_grid(const Tensor& input, const IntArray& output_shape = {}, bool align_corners = true);

PADDLE_API Tensor allclose(const Tensor& x, const Tensor& y, const Scalar& rtol = "1e-5", const Scalar& atol = "1e-8", bool equal_nan = false);

PADDLE_API Tensor angle(const Tensor& x);

PADDLE_API Tensor apply_per_channel_scale(const Tensor& x, const Tensor& scales);

PADDLE_API Tensor argmax(const Tensor& x, const Scalar& axis, bool keepdims = false, bool flatten = false, DataType dtype = DataType::INT64);

PADDLE_API Tensor argmin(const Tensor& x, const Scalar& axis, bool keepdims = false, bool flatten = false, DataType dtype = DataType::INT64);

PADDLE_API std::tuple<Tensor, Tensor> argsort(const Tensor& x, int axis = -1, bool descending = false);

PADDLE_API Tensor as_complex(const Tensor& x);

PADDLE_API Tensor as_real(const Tensor& x);

PADDLE_API Tensor as_strided(const Tensor& input, const std::vector<int64_t>& dims = {}, const std::vector<int64_t>& stride = {}, int64_t offset = 0);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> asgd_(Tensor& param, const Tensor& grad, const Tensor& learning_rate, Tensor& d, Tensor& y, const Tensor& n, paddle::optional<Tensor>& master_param, bool multi_precision = false);

PADDLE_API Tensor asin(const Tensor& x);

PADDLE_API Tensor& asin_(Tensor& x);

PADDLE_API Tensor asinh(const Tensor& x);

PADDLE_API Tensor& asinh_(Tensor& x);

PADDLE_API Tensor atan(const Tensor& x);

PADDLE_API Tensor& atan_(Tensor& x);

PADDLE_API Tensor atan2(const Tensor& x, const Tensor& y);

PADDLE_API Tensor atanh(const Tensor& x);

PADDLE_API Tensor& atanh_(Tensor& x);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> auc(const Tensor& x, const Tensor& label, const Tensor& stat_pos, const Tensor& stat_neg, const paddle::optional<Tensor>& ins_tag_weight, const std::string& curve = "ROC", int num_thresholds = (2 << 12) - 1, int slide_steps = 1);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, Tensor&> average_accumulates_(const Tensor& param, Tensor& in_sum_1, Tensor& in_sum_2, Tensor& in_sum_3, Tensor& in_num_accumulates, Tensor& in_old_num_accumulates, Tensor& in_num_updates, float average_window = 0, int64_t max_average_window = INT64_MAX, int64_t min_average_window = 10000L);

PADDLE_API Tensor bce_loss(const Tensor& input, const Tensor& label);

PADDLE_API Tensor& bce_loss_(Tensor& input, const Tensor& label);

PADDLE_API Tensor bernoulli(const Tensor& x);

PADDLE_API Tensor bicubic_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_format = "NCHW", int out_d = 0, int out_h = 0, int out_w = 0, const std::vector<float>& scale = {}, const std::string& interp_method = "bilinear", bool align_corners = true, int align_mode = 1);

PADDLE_API Tensor bilinear(const Tensor& x, const Tensor& y, const Tensor& weight, const paddle::optional<Tensor>& bias);

PADDLE_API Tensor bilinear_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_format = "NCHW", int out_d = 0, int out_h = 0, int out_w = 0, const std::vector<float>& scale = {}, const std::string& interp_method = "bilinear", bool align_corners = true, int align_mode = 1);

PADDLE_API Tensor bincount(const Tensor& x, const paddle::optional<Tensor>& weights, const Scalar& minlength = 0);

PADDLE_API Tensor binomial(const Tensor& count, const Tensor& prob);

PADDLE_API Tensor bitwise_and(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& bitwise_and_(Tensor& x, const Tensor& y);

PADDLE_API Tensor bitwise_left_shift(const Tensor& x, const Tensor& y, bool is_arithmetic = true);

PADDLE_API Tensor& bitwise_left_shift_(Tensor& x, const Tensor& y, bool is_arithmetic = true);

PADDLE_API Tensor bitwise_not(const Tensor& x);

PADDLE_API Tensor& bitwise_not_(Tensor& x);

PADDLE_API Tensor bitwise_or(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& bitwise_or_(Tensor& x, const Tensor& y);

PADDLE_API Tensor bitwise_right_shift(const Tensor& x, const Tensor& y, bool is_arithmetic = true);

PADDLE_API Tensor& bitwise_right_shift_(Tensor& x, const Tensor& y, bool is_arithmetic = true);

PADDLE_API Tensor bitwise_xor(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& bitwise_xor_(Tensor& x, const Tensor& y);

PADDLE_API Tensor bmm(const Tensor& x, const Tensor& y);

PADDLE_API Tensor box_coder(const Tensor& prior_box, const paddle::optional<Tensor>& prior_box_var, const Tensor& target_box, const std::string& code_type = "encode_center_size", bool box_normalized = true, int axis = 0, const std::vector<float>& variance = {});

PADDLE_API std::vector<Tensor> broadcast_tensors(const std::vector<Tensor>& input);

PADDLE_API Tensor ceil(const Tensor& x);

PADDLE_API Tensor& ceil_(Tensor& x);

PADDLE_API Tensor celu(const Tensor& x, float alpha = 1.0);

PADDLE_API std::tuple<std::vector<Tensor>&, Tensor> check_finite_and_unscale_(std::vector<Tensor>& x, const Tensor& scale);

PADDLE_API std::tuple<Tensor, Tensor> check_numerics(const Tensor& tensor, const std::string& op_type = "", const std::string& var_name = "", int check_nan_inf_level = 0, int stack_height_limit = -1, const std::string& output_dir = "");

PADDLE_API Tensor cholesky(const Tensor& x, bool upper = false);

PADDLE_API Tensor cholesky_solve(const Tensor& x, const Tensor& y, bool upper = false);

PADDLE_API std::tuple<Tensor, Tensor> class_center_sample(const Tensor& label, int num_classes, int num_samples, int ring_id = 0, int rank = 0, int nranks = 1, bool fix_seed = false, int seed = 0);

PADDLE_API Tensor clip(const Tensor& x, const Scalar& min, const Scalar& max);

PADDLE_API Tensor& clip_(Tensor& x, const Scalar& min, const Scalar& max);

PADDLE_API Tensor clip_by_norm(const Tensor& x, float max_norm);

PADDLE_API std::tuple<std::vector<Tensor>, Tensor> coalesce_tensor(const std::vector<Tensor>& input, DataType dtype, bool copy_data = false, bool set_constant = false, bool persist_output = false, float constant = 0.0, bool use_align = true, int align_size = -1, int size_of_dtype = -1, const std::vector<int64_t>& concated_shapes = {}, const std::vector<int64_t>& concated_ranks = {});

PADDLE_API Tensor complex(const Tensor& real, const Tensor& imag);

PADDLE_API Tensor concat(const std::vector<Tensor>& x, const Scalar& axis = 0);

PADDLE_API Tensor conj(const Tensor& x);

PADDLE_API Tensor conv2d(const Tensor& input, const Tensor& filter, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, const std::string& padding_algorithm = "EXPLICIT", const std::vector<int>& dilations = {1, 1}, int groups = 1, const std::string& data_format = "NCHW");

PADDLE_API Tensor conv3d(const Tensor& input, const Tensor& filter, const std::vector<int>& strides = {1, 1, 1}, const std::vector<int>& paddings = {0, 0, 0}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1, 1}, const std::string& data_format = "NCDHW");

PADDLE_API Tensor conv3d_transpose(const Tensor& x, const Tensor& filter, const std::vector<int>& strides = {1, 1, 1}, const std::vector<int>& paddings = {0, 0, 0}, const std::vector<int>& output_padding = {}, const std::vector<int>& output_size = {}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1, 1}, const std::string& data_format = "NCHW");

PADDLE_API Tensor copysign(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& copysign_(Tensor& x, const Tensor& y);

PADDLE_API Tensor cos(const Tensor& x);

PADDLE_API Tensor& cos_(Tensor& x);

PADDLE_API Tensor cosh(const Tensor& x);

PADDLE_API Tensor& cosh_(Tensor& x);

PADDLE_API Tensor crop(const Tensor& x, const IntArray& shape = {}, const IntArray& offsets = {});

PADDLE_API Tensor cross(const Tensor& x, const Tensor& y, int axis = 9);

PADDLE_API std::tuple<Tensor, Tensor> cross_entropy_with_softmax(const Tensor& input, const Tensor& label, bool soft_label = false, bool use_softmax = true, bool numeric_stable_mode = true, int ignore_index = -100, int axis = -1);

PADDLE_API std::tuple<Tensor&, Tensor> cross_entropy_with_softmax_(Tensor& input, const Tensor& label, bool soft_label = false, bool use_softmax = true, bool numeric_stable_mode = true, int ignore_index = -100, int axis = -1);

PADDLE_API std::tuple<Tensor, Tensor> cummax(const Tensor& x, int axis = -1, DataType dtype = DataType::INT64);

PADDLE_API std::tuple<Tensor, Tensor> cummin(const Tensor& x, int axis = -1, DataType dtype = DataType::INT64);

PADDLE_API Tensor cumprod(const Tensor& x, int dim);

PADDLE_API Tensor& cumprod_(Tensor& x, int dim);

PADDLE_API Tensor cumsum(const Tensor& x, const Scalar& axis = -1, bool flatten = false, bool exclusive = false, bool reverse = false);

PADDLE_API Tensor& cumsum_(Tensor& x, const Scalar& axis = -1, bool flatten = false, bool exclusive = false, bool reverse = false);

PADDLE_API Tensor data(const std::string& name, const IntArray& shape, DataType dtype, const Place& place);

PADDLE_API Tensor depthwise_conv2d(const Tensor& input, const Tensor& filter, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1}, const std::string& data_format = "NCHW");

PADDLE_API Tensor det(const Tensor& x);

PADDLE_API Tensor diag(const Tensor& x, int offset = 0, float padding_value = 0.0);

PADDLE_API Tensor diag_embed(const Tensor& input, int offset = 0, int dim1 = -2, int dim2 = -1);

PADDLE_API Tensor diagonal(const Tensor& x, int offset = 0, int axis1 = 0, int axis2 = 1);

PADDLE_API Tensor digamma(const Tensor& x);

PADDLE_API Tensor& digamma_(Tensor& x);

PADDLE_API Tensor dirichlet(const Tensor& alpha);

PADDLE_API Tensor dist(const Tensor& x, const Tensor& y, float p = 2.0);

PADDLE_API Tensor dot(const Tensor& x, const Tensor& y);

PADDLE_API std::tuple<Tensor, Tensor> edit_distance(const Tensor& hyps, const Tensor& refs, const paddle::optional<Tensor>& hypslength, const paddle::optional<Tensor>& refslength, bool normalized = false);

PADDLE_API std::tuple<Tensor, Tensor> eig(const Tensor& x);

PADDLE_API std::tuple<Tensor, Tensor> eigh(const Tensor& x, const std::string& UPLO = "L");

PADDLE_API Tensor eigvals(const Tensor& x);

PADDLE_API std::tuple<Tensor, Tensor> eigvalsh(const Tensor& x, const std::string& uplo = "L", bool is_test = false);

PADDLE_API Tensor elu(const Tensor& x, float alpha = 1.0f);

PADDLE_API Tensor& elu_(Tensor& x, float alpha = 1.0f);

PADDLE_API Tensor equal_all(const Tensor& x, const Tensor& y);

PADDLE_API Tensor erf(const Tensor& x);

PADDLE_API Tensor& erf_(Tensor& x);

PADDLE_API Tensor erfinv(const Tensor& x);

PADDLE_API Tensor& erfinv_(Tensor& x);

PADDLE_API Tensor exp(const Tensor& x);

PADDLE_API Tensor& exp_(Tensor& x);

PADDLE_API Tensor expand(const Tensor& x, const IntArray& shape = {});

PADDLE_API Tensor expand_as(const Tensor& x, const paddle::optional<Tensor>& y, const std::vector<int>& target_shape = {});

PADDLE_API Tensor expm1(const Tensor& x);

PADDLE_API Tensor& expm1_(Tensor& x);

PADDLE_API Tensor fft_c2c(const Tensor& x, const std::vector<int64_t>& axes, const std::string& normalization, bool forward);

PADDLE_API Tensor fft_c2r(const Tensor& x, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, int64_t last_dim_size = 0L);

PADDLE_API Tensor fft_r2c(const Tensor& x, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, bool onesided);

PADDLE_API Tensor fill(const Tensor& x, const Scalar& value = 0);

PADDLE_API Tensor& fill_(Tensor& x, const Scalar& value = 0);

PADDLE_API Tensor fill_diagonal(const Tensor& x, float value = 0, int offset = 0, bool wrap = false);

PADDLE_API Tensor& fill_diagonal_(Tensor& x, float value = 0, int offset = 0, bool wrap = false);

PADDLE_API Tensor fill_diagonal_tensor(const Tensor& x, const Tensor& y, int64_t offset = 0, int dim1 = 0, int dim2 = 1);

PADDLE_API Tensor& fill_diagonal_tensor_(Tensor& x, const Tensor& y, int64_t offset = 0, int dim1 = 0, int dim2 = 1);

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor> flash_attn(const Tensor& q, const Tensor& k, const Tensor& v, const paddle::optional<Tensor>& fixed_seed_offset, const paddle::optional<Tensor>& attn_mask, float dropout = 0.0, bool causal = false, bool return_softmax = false, bool is_test = false, const std::string& rng_name = "");

PADDLE_API std::tuple<Tensor, Tensor> flash_attn_unpadded(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, const paddle::optional<Tensor>& fixed_seed_offset, const paddle::optional<Tensor>& attn_mask, int64_t max_seqlen_q, int64_t max_seqlen_k, float scale, float dropout = 0.0, bool causal = false, bool return_softmax = false, bool is_test = false, const std::string& rng_name = "");

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor> flash_attn_with_sparse_mask(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& attn_mask_start_row_indices, const paddle::optional<Tensor>& fixed_seed_offset, float dropout = 0.0, bool causal = false, int attn_mask_start_row = 0, bool return_softmax = false, bool is_test = false, const std::string& rng_name = "");

PADDLE_API Tensor flatten(const Tensor& x, int start_axis = 1, int stop_axis = 1);

PADDLE_API Tensor& flatten_(Tensor& x, int start_axis = 1, int stop_axis = 1);

PADDLE_API Tensor flip(const Tensor& x, const std::vector<int>& axis);

PADDLE_API Tensor floor(const Tensor& x);

PADDLE_API Tensor& floor_(Tensor& x);

PADDLE_API Tensor fmax(const Tensor& x, const Tensor& y);

PADDLE_API Tensor fmin(const Tensor& x, const Tensor& y);

PADDLE_API Tensor fold(const Tensor& x, const std::vector<int>& output_sizes, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations);

PADDLE_API std::tuple<Tensor, Tensor> fractional_max_pool2d(const Tensor& x, const std::vector<int>& output_size, const std::vector<int>& kernel_size = {0, 0}, float random_u = 0.0, bool return_mask = true);

PADDLE_API std::tuple<Tensor, Tensor> fractional_max_pool3d(const Tensor& x, const std::vector<int>& output_size, const std::vector<int>& kernel_size = {0, 0, 0}, float random_u = 0.0, bool return_mask = true);

PADDLE_API Tensor frame(const Tensor& x, int frame_length, int hop_length, int axis = -1);

PADDLE_API Tensor full_int_array(const std::vector<int64_t>& value, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());

PADDLE_API Tensor gammaincc(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& gammaincc_(Tensor& x, const Tensor& y);

PADDLE_API Tensor gammaln(const Tensor& x);

PADDLE_API Tensor& gammaln_(Tensor& x);

PADDLE_API Tensor gather(const Tensor& x, const Tensor& index, const Scalar& axis = 0);

PADDLE_API Tensor gather_nd(const Tensor& x, const Tensor& index);

PADDLE_API Tensor gather_tree(const Tensor& ids, const Tensor& parents);

PADDLE_API Tensor gaussian_inplace(const Tensor& x, float mean = 0, float std = 1.0, int seed = 0);

PADDLE_API Tensor& gaussian_inplace_(Tensor& x, float mean = 0, float std = 1.0, int seed = 0);

PADDLE_API Tensor gelu(const Tensor& x, bool approximate = false);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> generate_proposals(const Tensor& scores, const Tensor& bbox_deltas, const Tensor& im_shape, const Tensor& anchors, const Tensor& variances, int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size, float eta, bool pixel_offset = true);

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> graph_khop_sampler(const Tensor& row, const Tensor& colptr, const Tensor& x, const paddle::optional<Tensor>& eids, const std::vector<int>& sample_sizes, bool return_eids);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> graph_sample_neighbors(const Tensor& row, const Tensor& colptr, const Tensor& x, const paddle::optional<Tensor>& eids, const paddle::optional<Tensor>& perm_buffer, int sample_size, bool return_eids, bool flag_perm_buffer);

PADDLE_API Tensor grid_sample(const Tensor& x, const Tensor& grid, const std::string& mode = "bilinear", const std::string& padding_mode = "zeros", bool align_corners = true);

PADDLE_API Tensor group_norm(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon = 1e-5, int groups = -1, const std::string& data_format = "NCHW");

PADDLE_API Tensor gumbel_softmax(const Tensor& x, float temperature = 1.0, bool hard = false, int axis = -1);

PADDLE_API Tensor hardshrink(const Tensor& x, float threshold = 0.5);

PADDLE_API Tensor hardsigmoid(const Tensor& x, float slope = 0.2, float offset = 0.5);

PADDLE_API Tensor hardtanh(const Tensor& x, float t_min = 0, float t_max = 24);

PADDLE_API Tensor& hardtanh_(Tensor& x, float t_min = 0, float t_max = 24);

PADDLE_API Tensor heaviside(const Tensor& x, const Tensor& y);

PADDLE_API Tensor histogram(const Tensor& input, int64_t bins = 100, int min = 0, int max = 0);

PADDLE_API Tensor huber_loss(const Tensor& input, const Tensor& label, float delta);

PADDLE_API Tensor i0(const Tensor& x);

PADDLE_API Tensor& i0_(Tensor& x);

PADDLE_API Tensor i0e(const Tensor& x);

PADDLE_API Tensor i1(const Tensor& x);

PADDLE_API Tensor i1e(const Tensor& x);

PADDLE_API Tensor identity_loss(const Tensor& x, int reduction = 1);

PADDLE_API Tensor& identity_loss_(Tensor& x, int reduction = 1);

PADDLE_API Tensor imag(const Tensor& x);

PADDLE_API Tensor index_add(const Tensor& x, const Tensor& index, const Tensor& add_value, int axis = 0);

PADDLE_API Tensor& index_add_(Tensor& x, const Tensor& index, const Tensor& add_value, int axis = 0);

PADDLE_API Tensor index_put(const Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, bool accumulate = false);

PADDLE_API Tensor& index_put_(Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, bool accumulate = false);

PADDLE_API Tensor index_sample(const Tensor& x, const Tensor& index);

PADDLE_API Tensor index_select(const Tensor& x, const Tensor& index, int axis = 0);

PADDLE_API Tensor index_select_strided(const Tensor& x, int64_t index, int axis = 0);

PADDLE_API Tensor instance_norm(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon = 1e-5);

PADDLE_API Tensor inverse(const Tensor& x);

PADDLE_API Tensor is_empty(const Tensor& x);

PADDLE_API Tensor isclose(const Tensor& x, const Tensor& y, const Scalar& rtol = 1e-5, const Scalar& atol = 1e-8, bool equal_nan = false);

PADDLE_API Tensor isfinite(const Tensor& x);

PADDLE_API Tensor isinf(const Tensor& x);

PADDLE_API Tensor isnan(const Tensor& x);

PADDLE_API Tensor kldiv_loss(const Tensor& x, const Tensor& label, const std::string& reduction = "mean");

PADDLE_API Tensor kron(const Tensor& x, const Tensor& y);

PADDLE_API std::tuple<Tensor, Tensor> kthvalue(const Tensor& x, int k = 1, int axis = -1, bool keepdim = false);

PADDLE_API Tensor label_smooth(const Tensor& label, const paddle::optional<Tensor>& prior_dist, float epsilon = 0.0f);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> lamb_(Tensor& param, const Tensor& grad, const Tensor& learning_rate, Tensor& moment1, Tensor& moment2, Tensor& beta1_pow, Tensor& beta2_pow, paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, float weight_decay, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1.0e-6f, bool always_adapt = false, bool multi_precision = false);

PADDLE_API Tensor layer_norm(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon = 1e-5, int begin_norm_axis = 1);

PADDLE_API Tensor leaky_relu(const Tensor& x, float negative_slope = 0.02f);

PADDLE_API Tensor& leaky_relu_(Tensor& x, float negative_slope = 0.02f);

PADDLE_API Tensor lerp(const Tensor& x, const Tensor& y, const Tensor& weight);

PADDLE_API Tensor& lerp_(Tensor& x, const Tensor& y, const Tensor& weight);

PADDLE_API Tensor lgamma(const Tensor& x);

PADDLE_API Tensor& lgamma_(Tensor& x);

PADDLE_API Tensor linear_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_format = "NCHW", int out_d = 0, int out_h = 0, int out_w = 0, const std::vector<float>& scale = {}, const std::string& interp_method = "bilinear", bool align_corners = true, int align_mode = 1);

PADDLE_API Tensor llm_int8_linear(const Tensor& x, const Tensor& weight, const paddle::optional<Tensor>& bias, const Tensor& weight_scale, float threshold = 6.0);

PADDLE_API Tensor log(const Tensor& x);

PADDLE_API Tensor& log_(Tensor& x);

PADDLE_API Tensor log10(const Tensor& x);

PADDLE_API Tensor& log10_(Tensor& x);

PADDLE_API Tensor log1p(const Tensor& x);

PADDLE_API Tensor& log1p_(Tensor& x);

PADDLE_API Tensor log2(const Tensor& x);

PADDLE_API Tensor& log2_(Tensor& x);

PADDLE_API Tensor log_loss(const Tensor& input, const Tensor& label, float epsilon);

PADDLE_API Tensor log_softmax(const Tensor& x, int axis = -1);

PADDLE_API Tensor logcumsumexp(const Tensor& x, int axis = -1, bool flatten = false, bool exclusive = false, bool reverse = false);

PADDLE_API Tensor logical_and(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& logical_and_(Tensor& x, const Tensor& y);

PADDLE_API Tensor logical_not(const Tensor& x);

PADDLE_API Tensor& logical_not_(Tensor& x);

PADDLE_API Tensor logical_or(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& logical_or_(Tensor& x, const Tensor& y);

PADDLE_API Tensor logical_xor(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& logical_xor_(Tensor& x, const Tensor& y);

PADDLE_API Tensor logit(const Tensor& x, float eps = 1e-6f);

PADDLE_API Tensor& logit_(Tensor& x, float eps = 1e-6f);

PADDLE_API Tensor logsigmoid(const Tensor& x);

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq(const Tensor& x, const Tensor& y, const Scalar& rcond = 0.0f, const std::string& driver = "gels");

PADDLE_API std::tuple<Tensor, Tensor, Tensor> lu(const Tensor& x, bool pivot = true);

PADDLE_API std::tuple<Tensor&, Tensor, Tensor> lu_(Tensor& x, bool pivot = true);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> lu_unpack(const Tensor& x, const Tensor& y, bool unpack_ludata = true, bool unpack_pivots = true);

PADDLE_API std::tuple<Tensor, Tensor> margin_cross_entropy(const Tensor& logits, const Tensor& label, bool return_softmax = false, int ring_id = 0, int rank = 0, int nranks = 1, float margin1 = 1.0f, float margin2 = 0.5f, float margin3 = 0.0f, float scale = 64.0f);

PADDLE_API std::tuple<Tensor, Tensor&, paddle::optional<Tensor>&> masked_multihead_attention_(const Tensor& x, Tensor& cache_kv, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& src_mask, const paddle::optional<Tensor>& cum_offsets, const paddle::optional<Tensor>& sequence_lengths, const paddle::optional<Tensor>& rotary_tensor, paddle::optional<Tensor>& beam_cache_offset, const paddle::optional<Tensor>& qkv_out_scale, const paddle::optional<Tensor>& out_shift, const paddle::optional<Tensor>& out_smooth, int seq_len, int rotary_emb_dims, bool use_neox_rotary_style = false, const std::string& compute_dtype = "default", float out_scale = -1, int quant_round_type = 1, float quant_max_bound = 127.0, float quant_min_bound = -127.0);

PADDLE_API Tensor masked_select(const Tensor& x, const Tensor& mask);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> matrix_nms(const Tensor& bboxes, const Tensor& scores, float score_threshold, int nms_top_k, int keep_top_k, float post_threshold = 0., bool use_gaussian = false, float gaussian_sigma = 2., int background_label = 0, bool normalized = true);

PADDLE_API Tensor matrix_power(const Tensor& x, int n);

PADDLE_API std::tuple<Tensor, Tensor> max_pool2d_with_index(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, bool global_pooling = false, bool adaptive = false);

PADDLE_API std::tuple<Tensor, Tensor> max_pool3d_with_index(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides = {1, 1, 1}, const std::vector<int>& paddings = {0, 0, 0}, bool global_pooling = false, bool adaptive = false);

PADDLE_API Tensor maxout(const Tensor& x, int groups, int axis = 1);

PADDLE_API Tensor mean_all(const Tensor& x);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> memory_efficient_attention(const Tensor& query, const Tensor& key, const Tensor& value, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& cu_seqlens_q, const paddle::optional<Tensor>& cu_seqlens_k, const paddle::optional<Tensor>& causal_diagonal, const paddle::optional<Tensor>& seqlen_k, const Scalar& max_seqlen_q, const Scalar& max_seqlen_k, bool causal, double dropout_p, float scale, bool is_test);

PADDLE_API Tensor merge_selected_rows(const Tensor& x);

PADDLE_API std::tuple<std::vector<Tensor>&, std::vector<Tensor>&, std::vector<Tensor>&, std::vector<Tensor>&, std::vector<Tensor>&, paddle::optional<std::vector<Tensor>>&> merged_adam_(std::vector<Tensor>& param, const std::vector<Tensor>& grad, const std::vector<Tensor>& learning_rate, std::vector<Tensor>& moment1, std::vector<Tensor>& moment2, std::vector<Tensor>& beta1_pow, std::vector<Tensor>& beta2_pow, paddle::optional<std::vector<Tensor>>& master_param, const Scalar& beta1 = 0.9f, const Scalar& beta2 = 0.999f, const Scalar& epsilon = 1.0e-8f, bool multi_precision = false, bool use_global_beta_pow = false);

PADDLE_API std::tuple<std::vector<Tensor>&, std::vector<Tensor>&, paddle::optional<std::vector<Tensor>>&> merged_momentum_(std::vector<Tensor>& param, const std::vector<Tensor>& grad, std::vector<Tensor>& velocity, const std::vector<Tensor>& learning_rate, paddle::optional<std::vector<Tensor>>& master_param, float mu, bool use_nesterov = false, const std::vector<std::string>& regularization_method = {}, const std::vector<float>& regularization_coeff = {}, bool multi_precision = false, float rescale_grad = 1.0f);

PADDLE_API std::vector<Tensor> meshgrid(const std::vector<Tensor>& inputs);

PADDLE_API std::tuple<Tensor, Tensor> mode(const Tensor& x, int axis = -1, bool keepdim = false);

PADDLE_API std::tuple<Tensor&, Tensor&, paddle::optional<Tensor>&> momentum_(Tensor& param, const Tensor& grad, Tensor& velocity, const Tensor& learning_rate, paddle::optional<Tensor>& master_param, float mu, bool use_nesterov = false, const std::string& regularization_method = "", float regularization_coeff = 0.0f, bool multi_precision = false, float rescale_grad = 1.0f);

PADDLE_API Tensor multi_dot(const std::vector<Tensor>& x);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> multiclass_nms3(const Tensor& bboxes, const Tensor& scores, const paddle::optional<Tensor>& rois_num, float score_threshold, int nms_top_k, int keep_top_k, float nms_threshold = 0.3, bool normalized = true, float nms_eta = 1.0, int background_label = 0);

PADDLE_API Tensor multinomial(const Tensor& x, const Scalar& num_samples = 1, bool replacement = false);

PADDLE_API Tensor multiplex(const std::vector<Tensor>& inputs, const Tensor& index);

PADDLE_API Tensor mv(const Tensor& x, const Tensor& vec);

PADDLE_API std::tuple<Tensor, Tensor> nanmedian(const Tensor& x, const IntArray& axis = {}, bool keepdim = true, const std::string& mode = "avg");

PADDLE_API Tensor nearest_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_format = "NCHW", int out_d = 0, int out_h = 0, int out_w = 0, const std::vector<float>& scale = {}, const std::string& interp_method = "bilinear", bool align_corners = true, int align_mode = 1);

PADDLE_API Tensor nextafter(const Tensor& x, const Tensor& y);

PADDLE_API std::tuple<Tensor, Tensor> nll_loss(const Tensor& input, const Tensor& label, const paddle::optional<Tensor>& weight, int64_t ignore_index = -100, const std::string& reduction = "mean");

PADDLE_API Tensor nms(const Tensor& x, float threshold = 1.0f);

PADDLE_API Tensor nonzero(const Tensor& condition);

PADDLE_API Tensor npu_identity(const Tensor& x, int format = -1);

PADDLE_API Tensor numel(const Tensor& x);

PADDLE_API Tensor overlap_add(const Tensor& x, int hop_length, int axis = -1);

PADDLE_API Tensor p_norm(const Tensor& x, float porder = 2, int axis = -1, float epsilon = 1.0e-12f, bool keepdim = false, bool asvector = false);

PADDLE_API Tensor pad3d(const Tensor& x, const IntArray& paddings, const std::string& mode = "constant", float pad_value = 0.0, const std::string& data_format = "NCDHW");

PADDLE_API Tensor pixel_shuffle(const Tensor& x, int upscale_factor = 1, const std::string& data_format = "NCHW");

PADDLE_API Tensor pixel_unshuffle(const Tensor& x, int downscale_factor = 1, const std::string& data_format = "NCHW");

PADDLE_API Tensor poisson(const Tensor& x);

PADDLE_API Tensor polygamma(const Tensor& x, int n);

PADDLE_API Tensor& polygamma_(Tensor& x, int n);

PADDLE_API Tensor pow(const Tensor& x, const Scalar& y = 1.0f);

PADDLE_API Tensor& pow_(Tensor& x, const Scalar& y = 1.0f);

PADDLE_API Tensor prelu(const Tensor& x, const Tensor& alpha, const std::string& data_format = "NCHW", const std::string& mode = "all");

PADDLE_API std::tuple<Tensor, Tensor> prior_box(const Tensor& input, const Tensor& image, const std::vector<float>& min_sizes, const std::vector<float>& max_sizes = {}, const std::vector<float>& aspect_ratios = {}, const std::vector<float>& variances = {}, bool flip = true, bool clip = true, float step_w = 0.0, float step_h = 0.0, float offset = 0.5, bool min_max_aspect_ratios_order = false);

PADDLE_API Tensor psroi_pool(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height = 1, int pooled_width = 1, int output_channels = 1, float spatial_scale = 1.0);

PADDLE_API Tensor put_along_axis(const Tensor& arr, const Tensor& indices, const Tensor& values, int axis, const std::string& reduce = "assign", bool include_self = true);

PADDLE_API Tensor& put_along_axis_(Tensor& arr, const Tensor& indices, const Tensor& values, int axis, const std::string& reduce = "assign", bool include_self = true);

PADDLE_API std::tuple<Tensor, Tensor> qr(const Tensor& x, const std::string& mode = "reduced");

PADDLE_API Tensor real(const Tensor& x);

PADDLE_API Tensor reciprocal(const Tensor& x);

PADDLE_API Tensor& reciprocal_(Tensor& x);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> reindex_graph(const Tensor& x, const Tensor& neighbors, const Tensor& count, const paddle::optional<Tensor>& hashtable_value, const paddle::optional<Tensor>& hashtable_index);

PADDLE_API Tensor relu(const Tensor& x);

PADDLE_API Tensor& relu_(Tensor& x);

PADDLE_API Tensor relu6(const Tensor& x);

PADDLE_API Tensor renorm(const Tensor& x, float p, int axis, float max_norm);

PADDLE_API Tensor& renorm_(Tensor& x, float p, int axis, float max_norm);

PADDLE_API Tensor reverse(const Tensor& x, const IntArray& axis);

PADDLE_API std::tuple<Tensor, Tensor> rms_norm(const Tensor& x, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& residual, const Tensor& norm_weight, const paddle::optional<Tensor>& norm_bias, float epsilon, int begin_norm_axis, float quant_scale, int quant_round_type, float quant_max_bound, float quant_min_bound);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&, paddle::optional<Tensor>&> rmsprop_(Tensor& param, Tensor& mean_square, const Tensor& grad, Tensor& moment, const Tensor& learning_rate, paddle::optional<Tensor>& mean_grad, paddle::optional<Tensor>& master_param, float epsilon = 1.0e-10f, float decay = 0.9f, float momentum = 0.0f, bool centered = false, bool multi_precision = false);

PADDLE_API Tensor roi_align(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height = 1, int pooled_width = 1, float spatial_scale = 1.0, int sampling_ratio = -1, bool aligned = false);

PADDLE_API Tensor roi_pool(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height = 1, int pooled_width = 1, float spatial_scale = 1.0);

PADDLE_API Tensor roll(const Tensor& x, const IntArray& shifts = {}, const std::vector<int64_t>& axis = {});

PADDLE_API Tensor round(const Tensor& x);

PADDLE_API Tensor& round_(Tensor& x);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> rprop_(Tensor& param, const Tensor& grad, Tensor& prev, Tensor& learning_rate, paddle::optional<Tensor>& master_param, const Tensor& learning_rate_range, const Tensor& etas, bool multi_precision = false);

PADDLE_API Tensor rsqrt(const Tensor& x);

PADDLE_API Tensor& rsqrt_(Tensor& x);

PADDLE_API Tensor scale(const Tensor& x, const Scalar& scale = 1.0, const Scalar& bias = 0.0, bool bias_after_scale = true);

PADDLE_API Tensor& scale_(Tensor& x, const Scalar& scale = 1.0, const Scalar& bias = 0.0, bool bias_after_scale = true);

PADDLE_API Tensor scatter(const Tensor& x, const Tensor& index, const Tensor& updates, bool overwrite = true);

PADDLE_API Tensor& scatter_(Tensor& x, const Tensor& index, const Tensor& updates, bool overwrite = true);

PADDLE_API Tensor scatter_nd_add(const Tensor& x, const Tensor& index, const Tensor& updates);

PADDLE_API Tensor searchsorted(const Tensor& sorted_sequence, const Tensor& values, bool out_int32 = false, bool right = false);

PADDLE_API Tensor segment_pool(const Tensor& x, const Tensor& segment_ids, const std::string& pooltype = "SUM");

PADDLE_API Tensor selu(const Tensor& x, float scale = 1.0507009873554804934193349852946, float alpha = 1.6732632423543772848170429916717);

PADDLE_API Tensor send_u_recv(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const std::string& reduce_op = "SUM", const IntArray& out_size = {0});

PADDLE_API Tensor send_ue_recv(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const std::string& message_op = "ADD", const std::string& reduce_op = "SUM", const IntArray& out_size = {0});

PADDLE_API Tensor send_uv(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const std::string& message_op = "ADD");

PADDLE_API std::tuple<Tensor&, paddle::optional<Tensor>&> sgd_(Tensor& param, const Tensor& learning_rate, const Tensor& grad, paddle::optional<Tensor>& master_param, bool multi_precision = false);

PADDLE_API Tensor shape(const Tensor& input);

PADDLE_API Tensor shard_index(const Tensor& input, int index_num, int nshards, int shard_id, int ignore_value = -1);

PADDLE_API Tensor sigmoid(const Tensor& x);

PADDLE_API Tensor& sigmoid_(Tensor& x);

PADDLE_API Tensor sigmoid_cross_entropy_with_logits(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, bool normalize = false, int ignore_index = -100);

PADDLE_API Tensor& sigmoid_cross_entropy_with_logits_(Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, bool normalize = false, int ignore_index = -100);

PADDLE_API Tensor sign(const Tensor& x);

PADDLE_API Tensor silu(const Tensor& x);

PADDLE_API Tensor sin(const Tensor& x);

PADDLE_API Tensor& sin_(Tensor& x);

PADDLE_API Tensor sinh(const Tensor& x);

PADDLE_API Tensor& sinh_(Tensor& x);

PADDLE_API Tensor slogdet(const Tensor& x);

PADDLE_API Tensor softplus(const Tensor& x, float beta = 1.0, float threshold = 20.0f);

PADDLE_API Tensor softshrink(const Tensor& x, float threshold = 0.5);

PADDLE_API Tensor softsign(const Tensor& x);

PADDLE_API Tensor solve(const Tensor& x, const Tensor& y);

PADDLE_API Tensor spectral_norm(const Tensor& weight, const Tensor& u, const Tensor& v, int dim = 0, int power_iters = 1, float eps = 1e-12f);

PADDLE_API Tensor sqrt(const Tensor& x);

PADDLE_API Tensor& sqrt_(Tensor& x);

PADDLE_API Tensor square(const Tensor& x);

PADDLE_API Tensor squared_l2_norm(const Tensor& x);

PADDLE_API Tensor squeeze(const Tensor& x, const IntArray& axis = {});

PADDLE_API Tensor& squeeze_(Tensor& x, const IntArray& axis = {});

PADDLE_API Tensor stack(const std::vector<Tensor>& x, int axis = 0);

PADDLE_API Tensor standard_gamma(const Tensor& x);

PADDLE_API Tensor stanh(const Tensor& x, float scale_a = 0.67f, float scale_b = 1.7159f);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& x, bool full_matrices = false);

PADDLE_API Tensor swiglu(const Tensor& x, const paddle::optional<Tensor>& y);

PADDLE_API Tensor take_along_axis(const Tensor& arr, const Tensor& indices, int axis);

PADDLE_API Tensor tan(const Tensor& x);

PADDLE_API Tensor& tan_(Tensor& x);

PADDLE_API Tensor tanh(const Tensor& x);

PADDLE_API Tensor& tanh_(Tensor& x);

PADDLE_API Tensor tanh_shrink(const Tensor& x);

PADDLE_API Tensor temporal_shift(const Tensor& x, int seg_num, float shift_ratio = 0.25f, const std::string& data_format = "NCHW");

PADDLE_API Tensor tensor_unfold(const Tensor& input, int64_t axis, int64_t size, int64_t step);

PADDLE_API Tensor thresholded_relu(const Tensor& x, float threshold = 1.0);

PADDLE_API Tensor& thresholded_relu_(Tensor& x, float threshold = 1.0);

PADDLE_API std::tuple<Tensor, Tensor> top_p_sampling(const Tensor& x, const Tensor& ps, const paddle::optional<Tensor>& threshold, int seed = -1);

PADDLE_API std::tuple<Tensor, Tensor> topk(const Tensor& x, const Scalar& k = 1, int axis = -1, bool largest = true, bool sorted = true);

PADDLE_API Tensor trace(const Tensor& x, int offset = 0, int axis1 = 0, int axis2 = 1);

PADDLE_API Tensor triangular_solve(const Tensor& x, const Tensor& y, bool upper = true, bool transpose = false, bool unitriangular = false);

PADDLE_API Tensor trilinear_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_format = "NCHW", int out_d = 0, int out_h = 0, int out_w = 0, const std::vector<float>& scale = {}, const std::string& interp_method = "bilinear", bool align_corners = true, int align_mode = 1);

PADDLE_API Tensor trunc(const Tensor& input);

PADDLE_API Tensor& trunc_(Tensor& input);

PADDLE_API std::vector<Tensor> unbind(const Tensor& input, int axis = 0);

PADDLE_API Tensor unfold(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations);

PADDLE_API Tensor uniform_inplace(const Tensor& x, float min = -1.0, float max = 1.0, int seed = 0, int diag_num = 0, int diag_step = 0, float diag_val = 1.0);

PADDLE_API Tensor& uniform_inplace_(Tensor& x, float min = -1.0, float max = 1.0, int seed = 0, int diag_num = 0, int diag_step = 0, float diag_val = 1.0);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> unique_consecutive(const Tensor& x, bool return_inverse = false, bool return_counts = false, const std::vector<int>& axis = {}, DataType dtype = DataType::FLOAT32);

PADDLE_API Tensor unpool3d(const Tensor& x, const Tensor& indices, const std::vector<int>& ksize, const std::vector<int>& strides = {1,1,1}, const std::vector<int>& paddings = {0,0,0}, const std::vector<int>& output_size = {0,0,0}, const std::string& data_format = "NCDHW");

PADDLE_API Tensor unsqueeze(const Tensor& x, const IntArray& axis = {});

PADDLE_API Tensor& unsqueeze_(Tensor& x, const IntArray& axis = {});

PADDLE_API std::vector<Tensor> unstack(const Tensor& x, int axis = 0, int num = 0);

PADDLE_API std::tuple<std::vector<Tensor>&, Tensor&, Tensor&, Tensor&> update_loss_scaling_(std::vector<Tensor>& x, const Tensor& found_infinite, Tensor& prev_loss_scaling, Tensor& in_good_steps, Tensor& in_bad_steps, int incr_every_n_steps, int decr_every_n_nan_or_inf, float incr_ratio, float decr_ratio, const Scalar& stop_update = false);

PADDLE_API Tensor view_dtype(const Tensor& input, DataType dtype);

PADDLE_API Tensor view_shape(const Tensor& input, const std::vector<int64_t>& dims = {});

PADDLE_API std::tuple<Tensor, Tensor> viterbi_decode(const Tensor& potentials, const Tensor& transition_params, const Tensor& lengths, bool include_bos_eos_tag = true);

PADDLE_API Tensor warpctc(const Tensor& logits, const Tensor& label, const paddle::optional<Tensor>& logits_length, const paddle::optional<Tensor>& labels_length, int blank = 0, bool norm_by_times = false);

PADDLE_API Tensor warprnnt(const Tensor& input, const Tensor& label, const Tensor& input_lengths, const Tensor& label_lengths, int blank = 0, float fastemit_lambda = 0.0);

PADDLE_API Tensor weight_dequantize(const Tensor& x, const Tensor& scale, const std::string& algo = "weight_only_int8", DataType out_dtype = DataType::FLOAT16, int group_size = -1);

PADDLE_API Tensor weight_only_linear(const Tensor& x, const Tensor& weight, const paddle::optional<Tensor>& bias, const Tensor& weight_scale, const std::string& weight_dtype, int arch = 80, int group_size = -1);

PADDLE_API std::tuple<Tensor, Tensor> weight_quantize(const Tensor& x, const std::string& algo = "weight_only_int8", int arch = 80, int group_size = -1);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> weighted_sample_neighbors(const Tensor& row, const Tensor& colptr, const Tensor& edge_weight, const Tensor& input_nodes, const paddle::optional<Tensor>& eids, int sample_size, bool return_eids);

PADDLE_API Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);

PADDLE_API Tensor& where_(const Tensor& condition, Tensor& x, const Tensor& y);

PADDLE_API std::tuple<Tensor, Tensor> yolo_box(const Tensor& x, const Tensor& img_size, const std::vector<int>& anchors = {}, int class_num = 1, float conf_thresh = 0.01, int downsample_ratio = 32, bool clip_bbox = true, float scale_x_y = 1.0, bool iou_aware = false, float iou_aware_factor = 0.5);

PADDLE_API Tensor yolo_loss(const Tensor& x, const Tensor& gt_box, const Tensor& gt_label, const paddle::optional<Tensor>& gt_score, const std::vector<int>& anchors = {}, const std::vector<int>& anchor_mask = {}, int class_num = 1, float ignore_thresh = 0.7, int downsample_ratio = 32, bool use_label_smooth = true, float scale_x_y = 1.0);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> adadelta_(Tensor& param, const Tensor& grad, Tensor& avg_squared_grad, Tensor& avg_squared_update, const Tensor& learning_rate, paddle::optional<Tensor>& master_param, float rho, float epsilon, bool multi_precision);

PADDLE_API Tensor add(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& add_(Tensor& x, const Tensor& y);

PADDLE_API Tensor add_n(const std::vector<Tensor>& inputs);

PADDLE_API Tensor all(const Tensor& x, const std::vector<int64_t>& axis = {}, bool keepdim = false);

PADDLE_API Tensor amax(const Tensor& x, const std::vector<int64_t>& axis = {}, bool keepdim = false);

PADDLE_API Tensor amin(const Tensor& x, const std::vector<int64_t>& axis = {}, bool keepdim = false);

PADDLE_API Tensor any(const Tensor& x, const std::vector<int64_t>& axis = {}, bool keepdim = false);

PADDLE_API Tensor arange(const Tensor& start, const Tensor& end, const Tensor& step, DataType dtype, const Place& place = {});

PADDLE_API Tensor assign(const Tensor& x);

PADDLE_API Tensor& assign_(Tensor& x);

PADDLE_API Tensor& assign_out_(const Tensor& x, Tensor& output);

PADDLE_API Tensor& assign_value_(Tensor& output, const std::vector<int>& shape, DataType dtype, const std::vector<phi::Scalar>& values, const Place& place = {});

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> batch_norm(const Tensor& x, const Tensor& mean, const Tensor& variance, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, bool is_test, float momentum, float epsilon, const std::string& data_format, bool use_global_stats, bool trainable_statistics);

PADDLE_API Tensor c_allgather(const Tensor& x, int ring_id, int nranks, bool use_calc_stream);

PADDLE_API Tensor c_allreduce_max(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

PADDLE_API Tensor& c_allreduce_max_(Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

PADDLE_API Tensor c_allreduce_min(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

PADDLE_API Tensor& c_allreduce_min_(Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

PADDLE_API Tensor c_allreduce_prod(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

PADDLE_API Tensor& c_allreduce_prod_(Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

PADDLE_API Tensor c_allreduce_sum(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

PADDLE_API Tensor& c_allreduce_sum_(Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

PADDLE_API Tensor c_broadcast(const Tensor& x, int ring_id = 0, int root = 0, bool use_calc_stream = false);

PADDLE_API Tensor& c_broadcast_(Tensor& x, int ring_id = 0, int root = 0, bool use_calc_stream = false);

PADDLE_API Tensor c_concat(const Tensor& x, int rank, int nranks, int ring_id, bool use_calc_stream, bool use_model_parallel);

PADDLE_API Tensor c_embedding(const Tensor& weight, const Tensor& x, int64_t start_index = 0, int64_t vocab_size = -1);

PADDLE_API Tensor c_identity(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

PADDLE_API Tensor& c_identity_(Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel);

PADDLE_API Tensor c_reduce_sum(const Tensor& x, int ring_id, int root_id, bool use_calc_stream);

PADDLE_API Tensor& c_reduce_sum_(Tensor& x, int ring_id, int root_id, bool use_calc_stream);

PADDLE_API Tensor c_sync_calc_stream(const Tensor& x);

PADDLE_API Tensor& c_sync_calc_stream_(Tensor& x);

PADDLE_API Tensor c_sync_comm_stream(const Tensor& x, int ring_id);

PADDLE_API Tensor& c_sync_comm_stream_(Tensor& x, int ring_id);

PADDLE_API Tensor cast(const Tensor& x, DataType dtype);

PADDLE_API Tensor& cast_(Tensor& x, DataType dtype);

PADDLE_API Tensor channel_shuffle(const Tensor& x, int groups, const std::string& data_format = "NCHW");

PADDLE_API Tensor conv2d_transpose(const Tensor& x, const Tensor& filter, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, const std::vector<int>& output_padding = {}, const IntArray& output_size = {}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1}, const std::string& data_format = "NCHW");

PADDLE_API Tensor conv2d_transpose_bias(const Tensor& x, const Tensor& filter, const Tensor& bias, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, const std::vector<int>& output_padding = {}, const IntArray& output_size = {}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1}, const std::string& data_format = "NCHW");

PADDLE_API Tensor copy_to(const Tensor& x, const Place& place, bool blocking);

PADDLE_API Tensor decode_jpeg(const Tensor& x, const std::string& mode, const Place& place);

PADDLE_API Tensor deformable_conv(const Tensor& x, const Tensor& offset, const Tensor& filter, const paddle::optional<Tensor>& mask, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, int deformable_groups, int groups, int im2col_step);

PADDLE_API Tensor depthwise_conv2d_transpose(const Tensor& x, const Tensor& filter, const std::vector<int>& strides = {1, 1}, const std::vector<int>& paddings = {0, 0}, const std::vector<int>& output_padding = {}, const IntArray& output_size = {}, const std::string& padding_algorithm = "EXPLICIT", int groups = 1, const std::vector<int>& dilations = {1, 1}, const std::string& data_format = "NCHW");

PADDLE_API Tensor disable_check_model_nan_inf(const Tensor& x, int flag = 0);

PADDLE_API std::tuple<std::vector<Tensor>, std::vector<Tensor>, Tensor> distribute_fpn_proposals(const Tensor& fpn_rois, const paddle::optional<Tensor>& rois_num, int min_level, int max_level, int refer_level, int refer_scale, bool pixel_offset);

PADDLE_API Tensor divide(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& divide_(Tensor& x, const Tensor& y);

PADDLE_API Tensor dropout(const Tensor& x, const paddle::optional<Tensor>& seed_tensor, const Scalar& p, bool is_test, const std::string& mode, int seed, bool fix_seed);

PADDLE_API std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> einsum(const std::vector<Tensor>& x, const std::string& equation);

PADDLE_API Tensor elementwise_pow(const Tensor& x, const Tensor& y);

PADDLE_API Tensor embedding(const Tensor& x, const Tensor& weight, int64_t padding_idx = -1, bool sparse = false);

PADDLE_API Tensor embedding_grad_dense(const Tensor& x, const Tensor& weight, const Tensor& out_grad, int64_t padding_idx = -1, bool sparse = false);

PADDLE_API Tensor empty(const IntArray& shape, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());

PADDLE_API Tensor empty_like(const Tensor& x, DataType dtype = DataType::UNDEFINED, const Place& place = {});

PADDLE_API Tensor enable_check_model_nan_inf(const Tensor& x, int flag = 1);

PADDLE_API Tensor equal(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& equal_(Tensor& x, const Tensor& y);

PADDLE_API Tensor& exponential_(Tensor& x, float lam);

PADDLE_API Tensor eye(const Scalar& num_rows, const Scalar& num_columns, DataType dtype = DataType::FLOAT32, const Place& place = {});

PADDLE_API Tensor floor_divide(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& floor_divide_(Tensor& x, const Tensor& y);

PADDLE_API Tensor frobenius_norm(const Tensor& x, const IntArray& axis, bool keep_dim, bool reduce_all);

PADDLE_API Tensor full(const IntArray& shape, const Scalar& value, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());

PADDLE_API Tensor& full_(Tensor& output, const IntArray& shape, const Scalar& value, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());

PADDLE_API Tensor full_batch_size_like(const Tensor& input, const std::vector<int>& shape, DataType dtype, const Scalar& value, int input_dim_idx, int output_dim_idx, const Place& place = CPUPlace());

PADDLE_API Tensor full_like(const Tensor& x, const Scalar& value, DataType dtype = DataType::UNDEFINED, const Place& place = {});

PADDLE_API Tensor full_with_tensor(const Tensor& value, const IntArray& shape, DataType dtype = DataType::FLOAT32);

PADDLE_API std::tuple<std::vector<Tensor>&, std::vector<Tensor>&, std::vector<Tensor>&, std::vector<Tensor>&, std::vector<Tensor>&, paddle::optional<std::vector<Tensor>>&> fused_adam_(std::vector<Tensor>& params, const std::vector<Tensor>& grads, const Tensor& learning_rate, std::vector<Tensor>& moments1, std::vector<Tensor>& moments2, std::vector<Tensor>& beta1_pows, std::vector<Tensor>& beta2_pows, paddle::optional<std::vector<Tensor>>& master_params, const paddle::optional<Tensor>& skip_update, const Scalar& beta1, const Scalar& beta2, const Scalar& epsilon, int chunk_size, float weight_decay, bool use_adamw, bool multi_precision, bool use_global_beta_pow);

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> fused_batch_norm_act(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& mean, const Tensor& variance, float momentum, float epsilon, const std::string& act_type);

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> fused_bn_add_activation(const Tensor& x, const Tensor& z, const Tensor& scale, const Tensor& bias, const Tensor& mean, const Tensor& variance, float momentum, float epsilon, const std::string& act_type);

PADDLE_API std::tuple<Tensor, Tensor> fused_gemm_epilogue(const Tensor& x, const Tensor& y, const Tensor& bias, bool trans_x, bool trans_y, const std::string& activation);

PADDLE_API std::tuple<std::vector<Tensor>, Tensor> fused_multi_transformer(const Tensor& x, const std::vector<Tensor>& ln_scales, const std::vector<Tensor>& ln_biases, const std::vector<Tensor>& qkv_weights, const paddle::optional<std::vector<Tensor>>& qkv_biases, const paddle::optional<std::vector<Tensor>>& cache_kvs, const paddle::optional<std::vector<Tensor>>& pre_caches, const paddle::optional<Tensor>& rotary_tensor, const paddle::optional<Tensor>& time_step, const paddle::optional<Tensor>& seq_lengths, const paddle::optional<Tensor>& src_mask, const std::vector<Tensor>& out_linear_weights, const paddle::optional<std::vector<Tensor>>& out_linear_biases, const std::vector<Tensor>& ffn_ln_scales, const std::vector<Tensor>& ffn_ln_biases, const std::vector<Tensor>& ffn1_weights, const paddle::optional<std::vector<Tensor>>& ffn1_biases, const std::vector<Tensor>& ffn2_weights, const paddle::optional<std::vector<Tensor>>& ffn2_biases, bool pre_layer_norm = true, float epsilon = 1e-5, float dropout_rate = .5f, int rotary_emb_dims = 0, bool is_test = false, const std::string& dropout_implementation = "downgrade_in_infer", const std::string& act_method = "gelu", bool trans_qkvw = true, int ring_id = -1);

PADDLE_API Tensor fused_softmax_mask(const Tensor& x, const Tensor& mask);

PADDLE_API Tensor fused_softmax_mask_upper_triangle(const Tensor& X);

PADDLE_API Tensor gaussian(const IntArray& shape, float mean, float std, int seed, DataType dtype, const Place& place = {});

PADDLE_API Tensor greater_equal(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& greater_equal_(Tensor& x, const Tensor& y);

PADDLE_API Tensor greater_than(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& greater_than_(Tensor& x, const Tensor& y);

PADDLE_API Tensor hardswish(const Tensor& x);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> hsigmoid_loss(const Tensor& x, const Tensor& label, const Tensor& w, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& path, const paddle::optional<Tensor>& code, int num_classes, bool is_sparse);

PADDLE_API Tensor increment(const Tensor& x, float value = 1.0);

PADDLE_API Tensor& increment_(Tensor& x, float value = 1.0);

PADDLE_API Tensor less_equal(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& less_equal_(Tensor& x, const Tensor& y);

PADDLE_API Tensor less_than(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& less_than_(Tensor& x, const Tensor& y);

PADDLE_API Tensor linspace(const Tensor& start, const Tensor& stop, const Tensor& number, DataType dtype, const Place& place);

PADDLE_API Tensor logspace(const Tensor& start, const Tensor& stop, const Tensor& num, const Tensor& base, DataType dtype, const Place& place = {});

PADDLE_API Tensor logsumexp(const Tensor& x, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all);

PADDLE_API Tensor matmul(const Tensor& x, const Tensor& y, bool transpose_x = false, bool transpose_y = false);

PADDLE_API Tensor matrix_rank(const Tensor& x, float tol, bool use_default_tol = true, bool hermitian = false);

PADDLE_API Tensor matrix_rank_tol(const Tensor& x, const Tensor& atol_tensor, bool use_default_tol = true, bool hermitian = false);

PADDLE_API Tensor max(const Tensor& x, const IntArray& axis = {}, bool keepdim = false);

PADDLE_API Tensor maximum(const Tensor& x, const Tensor& y);

PADDLE_API Tensor mean(const Tensor& x, const IntArray& axis = {}, bool keepdim = false);

PADDLE_API Tensor memcpy_d2h(const Tensor& x, int dst_place_type);

PADDLE_API Tensor memcpy_h2d(const Tensor& x, int dst_place_type);

PADDLE_API Tensor min(const Tensor& x, const IntArray& axis = {}, bool keepdim = false);

PADDLE_API Tensor minimum(const Tensor& x, const Tensor& y);

PADDLE_API Tensor mish(const Tensor& x, float lambda);

PADDLE_API Tensor multiply(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& multiply_(Tensor& x, const Tensor& y);

PADDLE_API std::tuple<Tensor, Tensor> norm(const Tensor& x, int axis, float epsilon, bool is_test);

PADDLE_API Tensor not_equal(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& not_equal_(Tensor& x, const Tensor& y);

PADDLE_API Tensor one_hot(const Tensor& x, const Scalar& num_classes);

PADDLE_API Tensor ones(const IntArray& shape, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());

PADDLE_API Tensor ones_like(const Tensor& x, DataType dtype = DataType::UNDEFINED, const Place& place = {});

PADDLE_API Tensor pad(const Tensor& x, const std::vector<int>& paddings, const Scalar& pad_value);

PADDLE_API Tensor pool2d(const Tensor& x, const IntArray& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm);

PADDLE_API Tensor pool3d(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm);

PADDLE_API Tensor prod(const Tensor& x, const IntArray& dims, bool keep_dim, bool reduce_all);

PADDLE_API Tensor randint(int low, int high, const IntArray& shape, DataType dtype = DataType::INT64, const Place& place = {});

PADDLE_API Tensor randperm(int n, DataType dtype, const Place& place = {});

PADDLE_API Tensor read_file(const std::string& filename = "", DataType dtype = DataType::UINT8, const Place& place = CPUPlace());

PADDLE_API Tensor remainder(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& remainder_(Tensor& x, const Tensor& y);

PADDLE_API Tensor repeat_interleave(const Tensor& x, int repeats, int axis);

PADDLE_API Tensor repeat_interleave_with_tensor_index(const Tensor& x, const Tensor& repeats, int axis);

PADDLE_API Tensor reshape(const Tensor& x, const IntArray& shape);

PADDLE_API Tensor& reshape_(Tensor& x, const IntArray& shape);

PADDLE_API std::tuple<Tensor, Tensor, std::vector<Tensor>> rnn(const Tensor& x, const std::vector<Tensor>& pre_state, const std::vector<Tensor>& weight_list, const paddle::optional<Tensor>& sequence_length, const Tensor& dropout_state_in, float dropout_prob = 0.0, bool is_bidirec = false, int input_size = 10, int hidden_size = 100, int num_layers = 1, const std::string& mode = "RNN_TANH", int seed = 0, bool is_test = false);

PADDLE_API Tensor rrelu(const Tensor& x, float lower, float upper, bool is_test);

PADDLE_API Tensor sequence_mask(const Tensor& x, const Scalar& max_len, DataType out_dtype);

PADDLE_API Tensor set_value(const Tensor& x, const IntArray& starts, const IntArray& ends, const IntArray& steps, const std::vector<int64_t>& axes, const std::vector<int64_t>& decrease_axes, const std::vector<int64_t>& none_axes, const std::vector<int64_t>& shape, const std::vector<phi::Scalar>& values);

PADDLE_API Tensor& set_value_(Tensor& x, const IntArray& starts, const IntArray& ends, const IntArray& steps, const std::vector<int64_t>& axes, const std::vector<int64_t>& decrease_axes, const std::vector<int64_t>& none_axes, const std::vector<int64_t>& shape, const std::vector<phi::Scalar>& values);

PADDLE_API Tensor set_value_with_tensor(const Tensor& x, const Tensor& values, const IntArray& starts, const IntArray& ends, const IntArray& steps, const std::vector<int64_t>& axes, const std::vector<int64_t>& decrease_axes, const std::vector<int64_t>& none_axes);

PADDLE_API Tensor& set_value_with_tensor_(Tensor& x, const Tensor& values, const IntArray& starts, const IntArray& ends, const IntArray& steps, const std::vector<int64_t>& axes, const std::vector<int64_t>& decrease_axes, const std::vector<int64_t>& none_axes);

PADDLE_API Tensor slice(const Tensor& input, const std::vector<int64_t>& axes, const IntArray& starts, const IntArray& ends, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis);

PADDLE_API Tensor softmax(const Tensor& x, int axis);

PADDLE_API Tensor& softmax_(Tensor& x, int axis);

PADDLE_API std::vector<Tensor> split(const Tensor& x, const IntArray& sections, const Scalar& axis);

PADDLE_API std::vector<Tensor> split_with_num(const Tensor& x, int num, const Scalar& axis);

PADDLE_API Tensor strided_slice(const Tensor& x, const std::vector<int>& axes, const IntArray& starts, const IntArray& ends, const IntArray& strides);

PADDLE_API Tensor subtract(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& subtract_(Tensor& x, const Tensor& y);

PADDLE_API Tensor sum(const Tensor& x, const IntArray& axis = {}, DataType dtype = DataType::UNDEFINED, bool keepdim = false);

PADDLE_API Tensor swish(const Tensor& x);

PADDLE_API std::tuple<Tensor, Tensor&, Tensor&, Tensor, Tensor, Tensor> sync_batch_norm_(const Tensor& x, Tensor& mean, Tensor& variance, const Tensor& scale, const Tensor& bias, bool is_test, float momentum, float epsilon, const std::string& data_format, bool use_global_stats, bool trainable_statistics);

PADDLE_API Tensor tile(const Tensor& x, const IntArray& repeat_times = {});

PADDLE_API Tensor trans_layout(const Tensor& x, const std::vector<int>& perm);

PADDLE_API Tensor transpose(const Tensor& x, const std::vector<int>& perm);

PADDLE_API Tensor& transpose_(Tensor& x, const std::vector<int>& perm);

PADDLE_API Tensor tril(const Tensor& x, int diagonal);

PADDLE_API Tensor& tril_(Tensor& x, int diagonal);

PADDLE_API Tensor tril_indices(int rows, int cols, int offset, DataType dtype, const Place& place = {});

PADDLE_API Tensor triu(const Tensor& x, int diagonal);

PADDLE_API Tensor& triu_(Tensor& x, int diagonal);

PADDLE_API Tensor triu_indices(int row, int col, int offset, DataType dtype, const Place& place = {});

PADDLE_API Tensor truncated_gaussian_random(const std::vector<int>& shape, float mean, float std, int seed, DataType dtype = DataType::FLOAT32, const Place& place = {});

PADDLE_API Tensor uniform(const IntArray& shape, DataType dtype, const Scalar& min, const Scalar& max, int seed, const Place& place = {});

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor> unique(const Tensor& x, bool return_index, bool return_inverse, bool return_counts, const std::vector<int>& axis, DataType dtype = DataType::INT64);

PADDLE_API Tensor unpool(const Tensor& x, const Tensor& indices, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const IntArray& output_size, const std::string& data_format);

PADDLE_API Tensor zeros(const IntArray& shape, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());

PADDLE_API Tensor zeros_like(const Tensor& x, DataType dtype = DataType::UNDEFINED, const Place& place = {});


}  // namespace experimental
}  // namespace paddle
