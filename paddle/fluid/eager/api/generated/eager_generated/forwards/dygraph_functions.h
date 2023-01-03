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

#pragma once
#include "glog/logging.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/to_static/run_program_op_func.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/api/all.h"

using CPUPlace = phi::CPUPlace;

extern std::unordered_map<std::string, std::vector<std::string>>
    core_ops_args_info;
extern std::unordered_map<std::string, std::vector<std::string>>
    core_ops_args_type_info;
extern std::unordered_map<std::string, std::vector<std::string>>
    core_ops_returns_info;

paddle::experimental::Tensor acos_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor acosh_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor addmm_ad_func(
    const paddle::experimental::Tensor& input,
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    float beta = 1.0,
    float alpha = 1.0);

paddle::experimental::Tensor angle_ad_func(
    const paddle::experimental::Tensor& x);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
argsort_ad_func(const paddle::experimental::Tensor& x,
                int axis,
                bool descending);

paddle::experimental::Tensor as_complex_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor as_real_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor asin_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor asinh_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor atan_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor atan2_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor atanh_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor bernoulli_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor bmm_ad_func(const paddle::experimental::Tensor& x,
                                         const paddle::experimental::Tensor& y);

paddle::experimental::Tensor ceil_ad_func(
    const paddle::experimental::Tensor& x);
paddle::experimental::Tensor& ceil__ad_func(paddle::experimental::Tensor& x);

paddle::experimental::Tensor celu_ad_func(const paddle::experimental::Tensor& x,
                                          float alpha);

paddle::experimental::Tensor cholesky_ad_func(
    const paddle::experimental::Tensor& x, bool upper);

paddle::experimental::Tensor cholesky_solve_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    bool upper);

paddle::experimental::Tensor complex_ad_func(
    const paddle::experimental::Tensor& real,
    const paddle::experimental::Tensor& imag);

paddle::experimental::Tensor conj_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor cos_ad_func(const paddle::experimental::Tensor& x);

paddle::experimental::Tensor cosh_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor crop_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::IntArray shape,
    paddle::experimental::IntArray offsets);

paddle::experimental::Tensor cross_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    int axis = 9);

paddle::experimental::Tensor det_ad_func(const paddle::experimental::Tensor& x);

paddle::experimental::Tensor diag_ad_func(const paddle::experimental::Tensor& x,
                                          int offset,
                                          float padding_value);

paddle::experimental::Tensor diag_embed_ad_func(
    const paddle::experimental::Tensor& input,
    int offset = 0,
    int dim1 = -2,
    int dim2 = -1);

paddle::experimental::Tensor diagonal_ad_func(
    const paddle::experimental::Tensor& x, int offset, int axis1, int axis2);

paddle::experimental::Tensor digamma_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor dist_ad_func(const paddle::experimental::Tensor& x,
                                          const paddle::experimental::Tensor& y,
                                          float p);

paddle::experimental::Tensor dot_ad_func(const paddle::experimental::Tensor& x,
                                         const paddle::experimental::Tensor& y);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
eig_ad_func(const paddle::experimental::Tensor& x);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
eigh_ad_func(const paddle::experimental::Tensor& x, std::string UPLO);

paddle::experimental::Tensor eigvals_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor elu_ad_func(const paddle::experimental::Tensor& x,
                                         float alpha);
paddle::experimental::Tensor& elu__ad_func(paddle::experimental::Tensor& x,
                                           float alpha);

paddle::experimental::Tensor equal_all_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor erf_ad_func(const paddle::experimental::Tensor& x);

paddle::experimental::Tensor erfinv_ad_func(
    const paddle::experimental::Tensor& x);
paddle::experimental::Tensor& erfinv__ad_func(paddle::experimental::Tensor& x);

paddle::experimental::Tensor exp_ad_func(const paddle::experimental::Tensor& x);
paddle::experimental::Tensor& exp__ad_func(paddle::experimental::Tensor& x);

paddle::experimental::Tensor expm1_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor fft_c2c_ad_func(
    const paddle::experimental::Tensor& x,
    std::vector<int64_t> axes,
    std::string normalization,
    bool forward);

paddle::experimental::Tensor fft_c2r_ad_func(
    const paddle::experimental::Tensor& x,
    std::vector<int64_t> axes,
    std::string normalization,
    bool forward,
    int64_t last_dim_size);

paddle::experimental::Tensor fft_r2c_ad_func(
    const paddle::experimental::Tensor& x,
    std::vector<int64_t> axes,
    std::string normalization,
    bool forward,
    bool onesided);

paddle::experimental::Tensor fill_diagonal_tensor_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    int64_t offset,
    int dim1,
    int dim2);
paddle::experimental::Tensor& fill_diagonal_tensor__ad_func(
    paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    int64_t offset,
    int dim1,
    int dim2);

paddle::experimental::Tensor flip_ad_func(const paddle::experimental::Tensor& x,
                                          std::vector<int> axis);

paddle::experimental::Tensor floor_ad_func(
    const paddle::experimental::Tensor& x);
paddle::experimental::Tensor& floor__ad_func(paddle::experimental::Tensor& x);

paddle::experimental::Tensor fold_ad_func(const paddle::experimental::Tensor& x,
                                          std::vector<int> output_sizes,
                                          std::vector<int> kernel_sizes,
                                          std::vector<int> strides,
                                          std::vector<int> paddings,
                                          std::vector<int> dilations);

paddle::experimental::Tensor frame_ad_func(
    const paddle::experimental::Tensor& x,
    int frame_length,
    int hop_length,
    int axis = -1);

paddle::experimental::Tensor gather_nd_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& index);

paddle::experimental::Tensor gather_tree_ad_func(
    const paddle::experimental::Tensor& ids,
    const paddle::experimental::Tensor& parents);

paddle::experimental::Tensor gelu_ad_func(const paddle::experimental::Tensor& x,
                                          bool approximate);

paddle::experimental::Tensor grid_sample_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& grid,
    std::string mode,
    std::string padding_mode,
    bool align_corners);

paddle::experimental::Tensor gumbel_softmax_ad_func(
    const paddle::experimental::Tensor& x,
    float temperature,
    bool hard,
    int axis);

paddle::experimental::Tensor hardshrink_ad_func(
    const paddle::experimental::Tensor& x, float threshold);

paddle::experimental::Tensor hardsigmoid_ad_func(
    const paddle::experimental::Tensor& x, float slope, float offset);

paddle::experimental::Tensor histogram_ad_func(
    const paddle::experimental::Tensor& input,
    int64_t bins = 100,
    int min = 0,
    int max = 0);

paddle::experimental::Tensor index_sample_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& index);

paddle::experimental::Tensor index_select_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& index,
    int axis);

paddle::experimental::Tensor inverse_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor is_empty_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor isclose_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    paddle::experimental::Scalar rtol = "1e-5",
    paddle::experimental::Scalar atol = "1e-8",
    bool equal_nan = false);

paddle::experimental::Tensor isfinite_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor isinf_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor isnan_ad_func(
    const paddle::experimental::Tensor& x);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
kthvalue_ad_func(const paddle::experimental::Tensor& x,
                 int k,
                 int axis,
                 bool keepdim);

paddle::experimental::Tensor label_smooth_ad_func(
    const paddle::experimental::Tensor& label,
    const paddle::optional<paddle::experimental::Tensor>& prior_dist,
    float epsilon);

paddle::experimental::Tensor leaky_relu_ad_func(
    const paddle::experimental::Tensor& x, float negative_slope);

paddle::experimental::Tensor lerp_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    const paddle::experimental::Tensor& weight);
paddle::experimental::Tensor& lerp__ad_func(
    paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    const paddle::experimental::Tensor& weight);

paddle::experimental::Tensor lgamma_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor log_ad_func(const paddle::experimental::Tensor& x);

paddle::experimental::Tensor log10_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor log1p_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor log2_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor log_loss_ad_func(
    const paddle::experimental::Tensor& input,
    const paddle::experimental::Tensor& label,
    float epsilon);

paddle::experimental::Tensor logit_ad_func(
    const paddle::experimental::Tensor& x, float eps = 1e-6f);

paddle::experimental::Tensor logsigmoid_ad_func(
    const paddle::experimental::Tensor& x);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
lu_unpack_ad_func(const paddle::experimental::Tensor& x,
                  const paddle::experimental::Tensor& y,
                  bool unpack_ludata = true,
                  bool unpack_pivots = true);

paddle::experimental::Tensor masked_select_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& mask);

paddle::experimental::Tensor matrix_power_ad_func(
    const paddle::experimental::Tensor& x, int n);

paddle::experimental::Tensor maxout_ad_func(
    const paddle::experimental::Tensor& x, int groups, int axis);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
mode_ad_func(const paddle::experimental::Tensor& x,
             int axis = -1,
             bool keepdim = false);

paddle::experimental::Tensor multinomial_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::Scalar num_samples = 1,
    bool replacement = false);

paddle::experimental::Tensor mv_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& vec);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
nll_loss_ad_func(const paddle::experimental::Tensor& input,
                 const paddle::experimental::Tensor& label,
                 const paddle::optional<paddle::experimental::Tensor>& weight,
                 int64_t ignore_index = -100,
                 std::string reduction = "mean");

paddle::experimental::Tensor npu_identity_ad_func(
    const paddle::experimental::Tensor& x, int format = -1);

paddle::experimental::Tensor overlap_add_ad_func(
    const paddle::experimental::Tensor& x, int hop_length, int axis);

paddle::experimental::Tensor pixel_shuffle_ad_func(
    const paddle::experimental::Tensor& x,
    int upscale_factor = 1,
    std::string data_format = "NCHW");

paddle::experimental::Tensor poisson_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor put_along_axis_ad_func(
    const paddle::experimental::Tensor& arr,
    const paddle::experimental::Tensor& indices,
    const paddle::experimental::Tensor& value,
    int axis,
    std::string reduce = "assign");
paddle::experimental::Tensor& put_along_axis__ad_func(
    paddle::experimental::Tensor& arr,
    const paddle::experimental::Tensor& indices,
    const paddle::experimental::Tensor& value,
    int axis,
    std::string reduce = "assign");

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
qr_ad_func(const paddle::experimental::Tensor& x, std::string mode = "reduced");

paddle::experimental::Tensor reciprocal_ad_func(
    const paddle::experimental::Tensor& x);
paddle::experimental::Tensor& reciprocal__ad_func(
    paddle::experimental::Tensor& x);

paddle::experimental::Tensor relu_ad_func(
    const paddle::experimental::Tensor& x);
paddle::experimental::Tensor& relu__ad_func(paddle::experimental::Tensor& x);

paddle::experimental::Tensor renorm_ad_func(
    const paddle::experimental::Tensor& x, float p, int axis, float max_norm);

paddle::experimental::Tensor roll_ad_func(const paddle::experimental::Tensor& x,
                                          paddle::experimental::IntArray shifts,
                                          std::vector<int64_t> axis);

paddle::experimental::Tensor round_ad_func(
    const paddle::experimental::Tensor& x);
paddle::experimental::Tensor& round__ad_func(paddle::experimental::Tensor& x);

paddle::experimental::Tensor rsqrt_ad_func(
    const paddle::experimental::Tensor& x);
paddle::experimental::Tensor& rsqrt__ad_func(paddle::experimental::Tensor& x);

paddle::experimental::Tensor scatter_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& index,
    const paddle::experimental::Tensor& updates,
    bool overwrite = true);
paddle::experimental::Tensor& scatter__ad_func(
    paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& index,
    const paddle::experimental::Tensor& updates,
    bool overwrite = true);

paddle::experimental::Tensor scatter_nd_add_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& index,
    const paddle::experimental::Tensor& updates);

paddle::experimental::Tensor searchsorted_ad_func(
    const paddle::experimental::Tensor& sorted_sequence,
    const paddle::experimental::Tensor& values,
    bool out_int32 = false,
    bool right = false);

paddle::experimental::Tensor selu_ad_func(
    const paddle::experimental::Tensor& x,
    float scale = 1.0507009873554804934193349852946,
    float alpha = 1.6732632423543772848170429916717);

paddle::experimental::Tensor send_uv_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    const paddle::experimental::Tensor& src_index,
    const paddle::experimental::Tensor& dst_index,
    std::string message_op = "ADD");

paddle::experimental::Tensor shard_index_ad_func(
    const paddle::experimental::Tensor& input,
    int index_num,
    int nshards,
    int shard_id,
    int ignore_value = -1);

paddle::experimental::Tensor sigmoid_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor silu_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor sin_ad_func(const paddle::experimental::Tensor& x);

paddle::experimental::Tensor sinh_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor softplus_ad_func(
    const paddle::experimental::Tensor& x, float beta, float threshold);

paddle::experimental::Tensor softshrink_ad_func(
    const paddle::experimental::Tensor& x, float threshold);

paddle::experimental::Tensor softsign_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor solve_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor sqrt_ad_func(
    const paddle::experimental::Tensor& x);
paddle::experimental::Tensor& sqrt__ad_func(paddle::experimental::Tensor& x);

paddle::experimental::Tensor square_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor squeeze_ad_func(
    const paddle::experimental::Tensor& x, paddle::experimental::IntArray axis);
paddle::experimental::Tensor& squeeze__ad_func(
    paddle::experimental::Tensor& x, paddle::experimental::IntArray axis);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
svd_ad_func(const paddle::experimental::Tensor& x, bool full_matrices = false);

paddle::experimental::Tensor take_along_axis_ad_func(
    const paddle::experimental::Tensor& arr,
    const paddle::experimental::Tensor& indices,
    int axis);

paddle::experimental::Tensor tan_ad_func(const paddle::experimental::Tensor& x);

paddle::experimental::Tensor tanh_ad_func(
    const paddle::experimental::Tensor& x);
paddle::experimental::Tensor& tanh__ad_func(paddle::experimental::Tensor& x);

paddle::experimental::Tensor tanh_shrink_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor thresholded_relu_ad_func(
    const paddle::experimental::Tensor& x, float threshold);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
topk_ad_func(const paddle::experimental::Tensor& x,
             paddle::experimental::Scalar k,
             int axis = -1,
             bool largest = true,
             bool sorted = true);

paddle::experimental::Tensor trace_ad_func(
    const paddle::experimental::Tensor& x, int offset, int axis1, int axis2);

paddle::experimental::Tensor trunc_ad_func(
    const paddle::experimental::Tensor& input);

paddle::experimental::Tensor unfold_ad_func(
    const paddle::experimental::Tensor& x,
    std::vector<int> kernel_sizes,
    std::vector<int> strides,
    std::vector<int> paddings,
    std::vector<int> dilations);

paddle::experimental::Tensor unsqueeze_ad_func(
    const paddle::experimental::Tensor& x, paddle::experimental::IntArray axes);
paddle::experimental::Tensor& unsqueeze__ad_func(
    paddle::experimental::Tensor& x, paddle::experimental::IntArray axes);

std::vector<paddle::experimental::Tensor> unstack_ad_func(
    const paddle::experimental::Tensor& x, int axis = 0, int num = 0);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
viterbi_decode_ad_func(const paddle::experimental::Tensor& potentials,
                       const paddle::experimental::Tensor& transition_params,
                       const paddle::experimental::Tensor& lengths,
                       bool include_bos_eos_tag = true);

paddle::experimental::Tensor warprnnt_ad_func(
    const paddle::experimental::Tensor& input,
    const paddle::experimental::Tensor& label,
    const paddle::experimental::Tensor& input_lengths,
    const paddle::experimental::Tensor& label_lengths,
    int blank = 0,
    float fastemit_lambda = 0.0);

paddle::experimental::Tensor where_ad_func(
    const paddle::experimental::Tensor& condition,
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor abs_ad_func(const paddle::experimental::Tensor& x);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
accuracy_ad_func(const paddle::experimental::Tensor& x,
                 const paddle::experimental::Tensor& indices,
                 const paddle::experimental::Tensor& label);

std::tuple<paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&>
adadelta__ad_func(paddle::experimental::Tensor& param,
                  const paddle::experimental::Tensor& grad,
                  paddle::experimental::Tensor& avg_squared_grad,
                  paddle::experimental::Tensor& avg_squared_update,
                  float rho,
                  float epsilon);

std::tuple<paddle::experimental::Tensor&, paddle::experimental::Tensor&>
adagrad__ad_func(paddle::experimental::Tensor& param,
                 const paddle::experimental::Tensor& grad,
                 paddle::experimental::Tensor& moment,
                 const paddle::experimental::Tensor& learning_rate,
                 float epsilon);

std::tuple<paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::optional<paddle::experimental::Tensor>&>
adam__ad_func(paddle::experimental::Tensor& param,
              const paddle::experimental::Tensor& grad,
              const paddle::experimental::Tensor& learning_rate,
              paddle::experimental::Tensor& moment1,
              paddle::experimental::Tensor& moment2,
              paddle::experimental::Tensor& beta1_pow,
              paddle::experimental::Tensor& beta2_pow,
              paddle::optional<paddle::experimental::Tensor>& master_param,
              const paddle::optional<paddle::experimental::Tensor>& skip_update,
              paddle::experimental::Scalar beta1,
              paddle::experimental::Scalar beta2,
              paddle::experimental::Scalar epsilon,
              bool lazy_mode,
              int64_t min_row_size_to_use_multithread,
              bool multi_precision,
              bool use_global_beta_pow);

std::tuple<paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&>
adamax__ad_func(paddle::experimental::Tensor& param,
                const paddle::experimental::Tensor& grad,
                const paddle::experimental::Tensor& learning_rate,
                paddle::experimental::Tensor& moment,
                paddle::experimental::Tensor& inf_norm,
                const paddle::experimental::Tensor& beta1_pow,
                float beta1,
                float beta2,
                float epsilon);

std::tuple<paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::optional<paddle::experimental::Tensor>&>
adamw__ad_func(
    paddle::experimental::Tensor& param,
    const paddle::experimental::Tensor& grad,
    const paddle::experimental::Tensor& learning_rate,
    paddle::experimental::Tensor& moment1,
    paddle::experimental::Tensor& moment2,
    paddle::experimental::Tensor& beta1_pow,
    paddle::experimental::Tensor& beta2_pow,
    paddle::optional<paddle::experimental::Tensor>& master_param,
    const paddle::optional<paddle::experimental::Tensor>& skip_update,
    paddle::experimental::Scalar beta1,
    paddle::experimental::Scalar beta2,
    paddle::experimental::Scalar epsilon,
    float lr_ratio,
    float coeff,
    bool with_decay,
    bool lazy_mode,
    int64_t min_row_size_to_use_multithread,
    bool multi_precision,
    bool use_global_beta_pow);

paddle::experimental::Tensor add_ad_func(const paddle::experimental::Tensor& x,
                                         const paddle::experimental::Tensor& y);
paddle::experimental::Tensor& add__ad_func(
    paddle::experimental::Tensor& x, const paddle::experimental::Tensor& y);

paddle::experimental::Tensor affine_grid_ad_func(
    const paddle::experimental::Tensor& input,
    paddle::experimental::IntArray outputShape,
    bool align_corners = true);

paddle::experimental::Tensor all_ad_func(const paddle::experimental::Tensor& x,
                                         std::vector<int64_t> axis = {},
                                         bool keepdim = false);

paddle::experimental::Tensor allclose_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    paddle::experimental::Scalar rtol,
    paddle::experimental::Scalar atol,
    bool equal_nan);

paddle::experimental::Tensor amax_ad_func(const paddle::experimental::Tensor& x,
                                          std::vector<int64_t> axis = {},
                                          bool keepdim = false);

paddle::experimental::Tensor amin_ad_func(const paddle::experimental::Tensor& x,
                                          std::vector<int64_t> axis = {},
                                          bool keepdim = false);

paddle::experimental::Tensor any_ad_func(const paddle::experimental::Tensor& x,
                                         std::vector<int64_t> axis = {},
                                         bool keepdim = false);

paddle::experimental::Tensor arange_ad_func(
    const paddle::experimental::Tensor& start,
    const paddle::experimental::Tensor& end,
    const paddle::experimental::Tensor& step,
    paddle::experimental::DataType dtype,
    paddle::Place place = {});

paddle::experimental::Tensor argmax_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::Scalar axis,
    bool keepdims,
    bool flatten,
    int dtype);

paddle::experimental::Tensor argmin_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::Scalar axis,
    bool keepdims,
    bool flatten,
    int dtype);

paddle::experimental::Tensor assign_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor& assign_out__ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::Tensor& output);

paddle::experimental::Tensor& assign_value__ad_func(
    paddle::experimental::Tensor& output,
    std::vector<int> shape,
    paddle::experimental::DataType dtype,
    std::vector<phi::Scalar> values,
    paddle::Place place = {});

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
auc_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& label,
    const paddle::experimental::Tensor& stat_pos,
    const paddle::experimental::Tensor& stat_neg,
    const paddle::optional<paddle::experimental::Tensor>& ins_tag_weight,
    std::string curve,
    int num_thresholds,
    int slide_steps);

std::tuple<paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&>
average_accumulates__ad_func(
    const paddle::experimental::Tensor& param,
    paddle::experimental::Tensor& in_sum_1,
    paddle::experimental::Tensor& in_sum_2,
    paddle::experimental::Tensor& in_sum_3,
    paddle::experimental::Tensor& in_num_accumulates,
    paddle::experimental::Tensor& in_old_num_accumulates,
    paddle::experimental::Tensor& in_num_updates,
    float average_window,
    int64_t max_average_window,
    int64_t min_average_window);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
batch_norm_ad_func(const paddle::experimental::Tensor& x,
                   const paddle::experimental::Tensor& mean,
                   const paddle::experimental::Tensor& variance,
                   const paddle::experimental::Tensor& scale,
                   const paddle::experimental::Tensor& bias,
                   bool is_test,
                   float momentum,
                   float epsilon,
                   std::string data_layout,
                   bool use_global_stats,
                   bool trainable_statistics);

paddle::experimental::Tensor bce_loss_ad_func(
    const paddle::experimental::Tensor& input,
    const paddle::experimental::Tensor& label);

paddle::experimental::Tensor bicubic_interp_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::optional<paddle::experimental::Tensor>& out_size,
    const paddle::optional<std::vector<paddle::experimental::Tensor>>&
        size_tensor,
    const paddle::optional<paddle::experimental::Tensor>& scale_tensor,
    std::string data_layout,
    int out_d,
    int out_h,
    int out_w,
    std::vector<float> scale,
    std::string interp_method,
    bool align_corners,
    int align_mode);

paddle::experimental::Tensor bilinear_interp_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::optional<paddle::experimental::Tensor>& out_size,
    const paddle::optional<std::vector<paddle::experimental::Tensor>>&
        size_tensor,
    const paddle::optional<paddle::experimental::Tensor>& scale_tensor,
    std::string data_layout,
    int out_d,
    int out_h,
    int out_w,
    std::vector<float> scale,
    std::string interp_method,
    bool align_corners,
    int align_mode);

paddle::experimental::Tensor bilinear_tensor_product_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    const paddle::experimental::Tensor& weight,
    const paddle::optional<paddle::experimental::Tensor>& bias);

paddle::experimental::Tensor bincount_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::optional<paddle::experimental::Tensor>& weights,
    paddle::experimental::Scalar minlength = 0);

paddle::experimental::Tensor bitwise_and_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor bitwise_not_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor bitwise_or_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor bitwise_xor_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor box_coder_ad_func(
    const paddle::experimental::Tensor& prior_box,
    const paddle::optional<paddle::experimental::Tensor>& prior_box_var,
    const paddle::experimental::Tensor& target_box,
    std::string code_type,
    bool box_normalized,
    int axis,
    std::vector<float> variance);

std::vector<paddle::experimental::Tensor> broadcast_tensors_ad_func(
    const std::vector<paddle::experimental::Tensor>& input);

paddle::experimental::Tensor cast_ad_func(const paddle::experimental::Tensor& x,
                                          paddle::experimental::DataType dtype);

std::tuple<std::vector<paddle::experimental::Tensor>&,
           paddle::experimental::Tensor&>
check_finite_and_unscale__ad_func(
    std::vector<paddle::experimental::Tensor>& x,
    const paddle::experimental::Tensor& scale,
    paddle::experimental::Tensor& input_found_infinite);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
class_center_sample_ad_func(const paddle::experimental::Tensor& label,
                            int num_classes,
                            int num_samples,
                            int ring_id,
                            int rank,
                            int nranks,
                            bool fix_seed,
                            int seed);

paddle::experimental::Tensor clip_ad_func(const paddle::experimental::Tensor& x,
                                          paddle::experimental::Scalar min,
                                          paddle::experimental::Scalar max);
paddle::experimental::Tensor& clip__ad_func(paddle::experimental::Tensor& x,
                                            paddle::experimental::Scalar min,
                                            paddle::experimental::Scalar max);

paddle::experimental::Tensor clip_by_norm_ad_func(
    const paddle::experimental::Tensor& x, float max_norm);

std::tuple<std::vector<paddle::experimental::Tensor>,
           paddle::experimental::Tensor>
coalesce_tensor_ad_func(const std::vector<paddle::experimental::Tensor>& input,
                        paddle::experimental::DataType dtype,
                        bool copy_data = false,
                        bool set_constant = false,
                        bool persist_output = false,
                        float constant = 0.0,
                        bool use_align = true,
                        int align_size = -1,
                        int size_of_dtype = -1,
                        std::vector<int64_t> concated_shapes = {},
                        std::vector<int64_t> concated_ranks = {});

paddle::experimental::Tensor concat_ad_func(
    const std::vector<paddle::experimental::Tensor>& x,
    paddle::experimental::Scalar axis);

paddle::experimental::Tensor conv2d_transpose_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& filter,
    std::vector<int> strides,
    std::vector<int> paddings,
    std::vector<int> output_padding,
    paddle::experimental::IntArray output_size,
    std::string padding_algorithm,
    int groups,
    std::vector<int> dilations,
    std::string data_format);

paddle::experimental::Tensor conv3d_ad_func(
    const paddle::experimental::Tensor& input,
    const paddle::experimental::Tensor& filter,
    std::vector<int> strides,
    std::vector<int> paddings,
    std::string padding_algorithm,
    int groups,
    std::vector<int> dilations,
    std::string data_format);

paddle::experimental::Tensor conv3d_transpose_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& filter,
    std::vector<int> strides,
    std::vector<int> paddings,
    std::vector<int> output_padding,
    std::vector<int> output_size,
    std::string padding_algorithm,
    int groups,
    std::vector<int> dilations,
    std::string data_format);

paddle::experimental::Tensor copy_to_ad_func(
    const paddle::experimental::Tensor& x, paddle::Place place, bool blocking);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
cross_entropy_with_softmax_ad_func(const paddle::experimental::Tensor& input,
                                   const paddle::experimental::Tensor& label,
                                   bool soft_label,
                                   bool use_softmax,
                                   bool numeric_stable_mode,
                                   int ignore_index,
                                   int axis);

paddle::experimental::Tensor cumprod_ad_func(
    const paddle::experimental::Tensor& x, int dim);

paddle::experimental::Tensor cumsum_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::Scalar axis,
    bool flatten,
    bool exclusive,
    bool reverse);

paddle::experimental::Tensor decode_jpeg_ad_func(
    const paddle::experimental::Tensor& x,
    std::string mode,
    paddle::Place place);

paddle::experimental::Tensor deformable_conv_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& offset,
    const paddle::experimental::Tensor& filter,
    const paddle::optional<paddle::experimental::Tensor>& mask,
    std::vector<int> strides,
    std::vector<int> paddings,
    std::vector<int> dilations,
    int deformable_groups,
    int groups,
    int im2col_step);

paddle::experimental::Tensor depthwise_conv2d_ad_func(
    const paddle::experimental::Tensor& input,
    const paddle::experimental::Tensor& filter,
    std::vector<int> strides,
    std::vector<int> paddings,
    std::string padding_algorithm,
    int groups,
    std::vector<int> dilations,
    std::string data_format);

paddle::experimental::Tensor depthwise_conv2d_transpose_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& filter,
    std::vector<int> strides,
    std::vector<int> paddings,
    std::vector<int> output_padding,
    paddle::experimental::IntArray output_size,
    std::string padding_algorithm,
    int groups,
    std::vector<int> dilations,
    std::string data_format);

paddle::experimental::Tensor dirichlet_ad_func(
    const paddle::experimental::Tensor& alpha);

std::tuple<std::vector<paddle::experimental::Tensor>,
           std::vector<paddle::experimental::Tensor>,
           paddle::experimental::Tensor>
distribute_fpn_proposals_ad_func(
    const paddle::experimental::Tensor& fpn_rois,
    const paddle::optional<paddle::experimental::Tensor>& rois_num,
    int min_level,
    int max_level,
    int refer_level,
    int refer_scale,
    bool pixel_offset);

paddle::experimental::Tensor divide_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
dropout_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::optional<paddle::experimental::Tensor>& seed_tensor,
    paddle::experimental::Scalar p,
    bool is_test,
    std::string mode,
    int seed,
    bool fix_seed);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
edit_distance_ad_func(
    const paddle::experimental::Tensor& hyps,
    const paddle::experimental::Tensor& refs,
    const paddle::optional<paddle::experimental::Tensor>& hypslength,
    const paddle::optional<paddle::experimental::Tensor>& refslength,
    bool normalized = false);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
eigvalsh_ad_func(const paddle::experimental::Tensor& x,
                 std::string uplo,
                 bool is_test);

std::tuple<paddle::experimental::Tensor,
           std::vector<paddle::experimental::Tensor>,
           std::vector<paddle::experimental::Tensor>>
einsum_ad_func(const std::vector<paddle::experimental::Tensor>& x,
               std::string equation);

paddle::experimental::Tensor elementwise_pow_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor embedding_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& weight,
    int64_t padding_idx = -1,
    bool sparse = false);

paddle::experimental::Tensor empty_ad_func(
    paddle::experimental::IntArray shape,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::FLOAT32,
    paddle::Place place = CPUPlace());

paddle::experimental::Tensor empty_like_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::UNDEFINED,
    paddle::Place place = {});

paddle::experimental::Tensor equal_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor expand_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::IntArray shape);

paddle::experimental::Tensor expand_as_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::optional<paddle::experimental::Tensor>& y,
    std::vector<int> target_shape);

paddle::experimental::Tensor& exponential__ad_func(
    paddle::experimental::Tensor& x, float lam);

paddle::experimental::Tensor eye_ad_func(
    paddle::experimental::Scalar num_rows,
    paddle::experimental::Scalar num_columns,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::FLOAT32,
    paddle::Place place = {});

paddle::experimental::Tensor fill_ad_func(const paddle::experimental::Tensor& x,
                                          paddle::experimental::Scalar value);
paddle::experimental::Tensor& fill__ad_func(paddle::experimental::Tensor& x,
                                            paddle::experimental::Scalar value);

paddle::experimental::Tensor fill_diagonal_ad_func(
    const paddle::experimental::Tensor& x, float value, int offset, bool wrap);
paddle::experimental::Tensor& fill_diagonal__ad_func(
    paddle::experimental::Tensor& x, float value, int offset, bool wrap);

paddle::experimental::Tensor flatten_ad_func(
    const paddle::experimental::Tensor& x, int start_axis, int stop_axis);
paddle::experimental::Tensor& flatten__ad_func(paddle::experimental::Tensor& x,
                                               int start_axis,
                                               int stop_axis);

paddle::experimental::Tensor floor_divide_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor fmax_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor fmin_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor frobenius_norm_ad_func(
    const paddle::experimental::Tensor& x,
    std::vector<int64_t> axis,
    bool keep_dim,
    bool reduce_all);

paddle::experimental::Tensor full_ad_func(
    paddle::experimental::IntArray shape,
    paddle::experimental::Scalar value,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::FLOAT32,
    paddle::Place place = CPUPlace());

paddle::experimental::Tensor& full__ad_func(
    paddle::experimental::Tensor& output,
    paddle::experimental::IntArray shape,
    paddle::experimental::Scalar value,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::FLOAT32,
    paddle::Place place = CPUPlace());

paddle::experimental::Tensor full_batch_size_like_ad_func(
    const paddle::experimental::Tensor& input,
    std::vector<int> shape,
    paddle::experimental::DataType dtype,
    paddle::experimental::Scalar value,
    int input_dim_idx,
    int output_dim_idx,
    paddle::Place place = CPUPlace());

paddle::experimental::Tensor full_like_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::Scalar value,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::UNDEFINED,
    paddle::Place place = {});

paddle::experimental::Tensor gather_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& index,
    paddle::experimental::Scalar axis = 0);

paddle::experimental::Tensor gaussian_ad_func(
    paddle::experimental::IntArray shape,
    float mean,
    float std,
    int seed,
    paddle::experimental::DataType dtype,
    paddle::Place place = {});

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
generate_proposals_ad_func(const paddle::experimental::Tensor& scores,
                           const paddle::experimental::Tensor& bbox_deltas,
                           const paddle::experimental::Tensor& im_shape,
                           const paddle::experimental::Tensor& anchors,
                           const paddle::experimental::Tensor& variances,
                           int pre_nms_top_n,
                           int post_nms_top_n,
                           float nms_thresh,
                           float min_size,
                           float eta,
                           bool pixel_offset = true);

paddle::experimental::Tensor greater_equal_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor greater_than_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor group_norm_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::optional<paddle::experimental::Tensor>& scale,
    const paddle::optional<paddle::experimental::Tensor>& bias,
    float epsilon,
    int groups,
    std::string data_layout);

paddle::experimental::Tensor hardswish_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor hardtanh_ad_func(
    const paddle::experimental::Tensor& x, float t_min, float t_max);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
hsigmoid_loss_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& label,
    const paddle::experimental::Tensor& w,
    const paddle::optional<paddle::experimental::Tensor>& bias,
    const paddle::optional<paddle::experimental::Tensor>& path,
    const paddle::optional<paddle::experimental::Tensor>& code,
    int num_classes,
    bool remote_prefetch,
    bool is_sparse);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
huber_loss_ad_func(const paddle::experimental::Tensor& input,
                   const paddle::experimental::Tensor& label,
                   float delta);

paddle::experimental::Tensor imag_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor increment_ad_func(
    const paddle::experimental::Tensor& x, float value = 1.0);
paddle::experimental::Tensor& increment__ad_func(
    paddle::experimental::Tensor& x, float value = 1.0);

paddle::experimental::Tensor index_add_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& index,
    const paddle::experimental::Tensor& add_value,
    int axis);
paddle::experimental::Tensor& index_add__ad_func(
    paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& index,
    const paddle::experimental::Tensor& add_value,
    int axis);

paddle::experimental::Tensor instance_norm_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::optional<paddle::experimental::Tensor>& scale,
    const paddle::optional<paddle::experimental::Tensor>& bias,
    float epsilon);

paddle::experimental::Tensor kldiv_loss_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& label,
    std::string reduction);

paddle::experimental::Tensor kron_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

std::tuple<paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::optional<paddle::experimental::Tensor>&>
lamb__ad_func(paddle::experimental::Tensor& param,
              const paddle::experimental::Tensor& grad,
              const paddle::experimental::Tensor& learning_rate,
              paddle::experimental::Tensor& moment1,
              paddle::experimental::Tensor& moment2,
              paddle::experimental::Tensor& beta1_pow,
              paddle::experimental::Tensor& beta2_pow,
              paddle::optional<paddle::experimental::Tensor>& master_param,
              const paddle::optional<paddle::experimental::Tensor>& skip_update,
              float weight_decay,
              float beta1,
              float beta2,
              float epsilon,
              bool multi_precision);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
layer_norm_ad_func(const paddle::experimental::Tensor& x,
                   const paddle::optional<paddle::experimental::Tensor>& scale,
                   const paddle::optional<paddle::experimental::Tensor>& bias,
                   float epsilon,
                   int begin_norm_axis);

paddle::experimental::Tensor less_equal_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor less_than_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor linear_interp_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::optional<paddle::experimental::Tensor>& out_size,
    const paddle::optional<std::vector<paddle::experimental::Tensor>>&
        size_tensor,
    const paddle::optional<paddle::experimental::Tensor>& scale_tensor,
    std::string data_layout,
    int out_d,
    int out_h,
    int out_w,
    std::vector<float> scale,
    std::string interp_method,
    bool align_corners,
    int align_mode);

paddle::experimental::Tensor linspace_ad_func(
    const paddle::experimental::Tensor& start,
    const paddle::experimental::Tensor& stop,
    const paddle::experimental::Tensor& number,
    paddle::experimental::DataType dtype,
    paddle::Place place);

paddle::experimental::Tensor log_softmax_ad_func(
    const paddle::experimental::Tensor& x, int axis);

paddle::experimental::Tensor logcumsumexp_ad_func(
    const paddle::experimental::Tensor& x,
    int axis,
    bool flatten,
    bool exclusive,
    bool reverse);

paddle::experimental::Tensor logical_and_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor logical_not_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor logical_or_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor logical_xor_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor logsumexp_ad_func(
    const paddle::experimental::Tensor& x,
    std::vector<int64_t> axis,
    bool keepdim,
    bool reduce_all);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
lstsq_ad_func(const paddle::experimental::Tensor& x,
              const paddle::experimental::Tensor& y,
              paddle::experimental::Scalar rcond,
              std::string driver);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
lu_ad_func(const paddle::experimental::Tensor& x, bool pivot);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
margin_cross_entropy_ad_func(const paddle::experimental::Tensor& logits,
                             const paddle::experimental::Tensor& label,
                             bool return_softmax,
                             int ring_id,
                             int rank,
                             int nranks,
                             float margin1,
                             float margin2,
                             float margin3,
                             float scale);

paddle::experimental::Tensor matmul_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    bool transpose_x = false,
    bool transpose_y = false);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
matrix_nms_ad_func(const paddle::experimental::Tensor& bboxes,
                   const paddle::experimental::Tensor& scores,
                   float score_threshold,
                   int nms_top_k,
                   int keep_top_k,
                   float post_threshold = 0.,
                   bool use_gaussian = false,
                   float gaussian_sigma = 2.0,
                   int background_label = 0,
                   bool normalized = true);

paddle::experimental::Tensor matrix_rank_ad_func(
    const paddle::experimental::Tensor& x,
    float tol,
    bool hermitian = false,
    bool use_default_tol = true);

paddle::experimental::Tensor matrix_rank_tol_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& atol_tensor,
    bool use_default_tol = true,
    bool hermitian = false);

paddle::experimental::Tensor max_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::IntArray axis = {},
    bool keepdim = false);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
max_pool2d_with_index_ad_func(const paddle::experimental::Tensor& x,
                              std::vector<int> kernel_size,
                              std::vector<int> strides,
                              std::vector<int> paddings,
                              bool global_pooling,
                              bool adaptive);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
max_pool3d_with_index_ad_func(const paddle::experimental::Tensor& x,
                              std::vector<int> kernel_size,
                              std::vector<int> strides,
                              std::vector<int> paddings,
                              bool global_pooling,
                              bool adaptive);

paddle::experimental::Tensor maximum_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor mean_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::IntArray axis = {},
    bool keepdim = false);

paddle::experimental::Tensor mean_all_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor merge_selected_rows_ad_func(
    const paddle::experimental::Tensor& x);

std::tuple<std::vector<paddle::experimental::Tensor>&,
           std::vector<paddle::experimental::Tensor>&,
           std::vector<paddle::experimental::Tensor>&,
           std::vector<paddle::experimental::Tensor>&,
           std::vector<paddle::experimental::Tensor>&,
           paddle::optional<std::vector<paddle::experimental::Tensor>>&>
merged_adam__ad_func(
    std::vector<paddle::experimental::Tensor>& param,
    const std::vector<paddle::experimental::Tensor>& grad,
    const std::vector<paddle::experimental::Tensor>& learning_rate,
    std::vector<paddle::experimental::Tensor>& moment1,
    std::vector<paddle::experimental::Tensor>& moment2,
    std::vector<paddle::experimental::Tensor>& beta1_pow,
    std::vector<paddle::experimental::Tensor>& beta2_pow,
    paddle::optional<std::vector<paddle::experimental::Tensor>>& master_param,
    paddle::experimental::Scalar beta1,
    paddle::experimental::Scalar beta2,
    paddle::experimental::Scalar epsilon,
    bool multi_precision,
    bool use_global_beta_pow);

std::tuple<std::vector<paddle::experimental::Tensor>&,
           std::vector<paddle::experimental::Tensor>&,
           paddle::optional<std::vector<paddle::experimental::Tensor>>&>
merged_momentum__ad_func(
    std::vector<paddle::experimental::Tensor>& param,
    const std::vector<paddle::experimental::Tensor>& grad,
    std::vector<paddle::experimental::Tensor>& velocity,
    const std::vector<paddle::experimental::Tensor>& learning_rate,
    paddle::optional<std::vector<paddle::experimental::Tensor>>& master_param,
    float mu,
    bool use_nesterov = false,
    std::vector<std::string> regularization_method = {},
    std::vector<float> regularization_coeff = {},
    bool multi_precision = false,
    float rescale_grad = 1.0f);

std::vector<paddle::experimental::Tensor> meshgrid_ad_func(
    const std::vector<paddle::experimental::Tensor>& inputs);

paddle::experimental::Tensor min_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::IntArray axis = {},
    bool keepdim = false);

paddle::experimental::Tensor minimum_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor mish_ad_func(const paddle::experimental::Tensor& x,
                                          float threshold);

std::tuple<paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::optional<paddle::experimental::Tensor>&>
momentum__ad_func(paddle::experimental::Tensor& param,
                  const paddle::experimental::Tensor& grad,
                  paddle::experimental::Tensor& velocity,
                  const paddle::experimental::Tensor& learning_rate,
                  paddle::optional<paddle::experimental::Tensor>& master_param,
                  float mu,
                  bool use_nesterov = false,
                  std::string regularization_method = "",
                  float regularization_coeff = 0.0,
                  bool multi_precision = false,
                  float rescale_grad = 1.0f);

paddle::experimental::Tensor multi_dot_ad_func(
    const std::vector<paddle::experimental::Tensor>& x);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
multiclass_nms3_ad_func(
    const paddle::experimental::Tensor& bboxes,
    const paddle::experimental::Tensor& scores,
    const paddle::optional<paddle::experimental::Tensor>& rois_num,
    float score_threshold,
    int nms_top_k,
    int keep_top_k,
    float nms_threshold = 0.3,
    bool normalized = true,
    float nms_eta = 1.0,
    int background_label = 0);

paddle::experimental::Tensor multiplex_ad_func(
    const std::vector<paddle::experimental::Tensor>& inputs,
    const paddle::experimental::Tensor& index);

paddle::experimental::Tensor multiply_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor nearest_interp_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::optional<paddle::experimental::Tensor>& out_size,
    const paddle::optional<std::vector<paddle::experimental::Tensor>>&
        size_tensor,
    const paddle::optional<paddle::experimental::Tensor>& scale_tensor,
    std::string data_layout,
    int out_d,
    int out_h,
    int out_w,
    std::vector<float> scale,
    std::string interp_method,
    bool align_corners,
    int align_mode);

paddle::experimental::Tensor nms_ad_func(const paddle::experimental::Tensor& x,
                                         float threshold);

paddle::experimental::Tensor nonzero_ad_func(
    const paddle::experimental::Tensor& condition);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
norm_ad_func(const paddle::experimental::Tensor& x,
             int axis,
             float epsilon,
             bool is_test);

paddle::experimental::Tensor not_equal_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor numel_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor one_hot_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::Scalar num_classes);

paddle::experimental::Tensor ones_ad_func(
    paddle::experimental::IntArray shape,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::FLOAT32,
    paddle::Place place = CPUPlace());

paddle::experimental::Tensor ones_like_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::UNDEFINED,
    paddle::Place place = {});

paddle::experimental::Tensor p_norm_ad_func(
    const paddle::experimental::Tensor& x,
    float porder,
    int axis,
    float epsilon,
    bool keepdim,
    bool asvector = false);

paddle::experimental::Tensor pad_ad_func(
    const paddle::experimental::Tensor& x,
    std::vector<int> paddings,
    paddle::experimental::Scalar pad_value);

paddle::experimental::Tensor pad3d_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::IntArray paddings,
    std::string mode,
    float pad_value,
    std::string data_format);

paddle::experimental::Tensor pool2d_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::IntArray kernel_size,
    std::vector<int> strides,
    std::vector<int> paddings,
    bool ceil_mode,
    bool exclusive,
    std::string data_format,
    std::string pooling_type,
    bool global_pooling,
    bool adaptive,
    std::string padding_algorithm);

paddle::experimental::Tensor pool3d_ad_func(
    const paddle::experimental::Tensor& x,
    std::vector<int> kernel_size,
    std::vector<int> strides,
    std::vector<int> paddings,
    bool ceil_mode,
    bool exclusive,
    std::string data_format,
    std::string pooling_type,
    bool global_pooling,
    bool adaptive,
    std::string padding_algorithm);

paddle::experimental::Tensor pow_ad_func(const paddle::experimental::Tensor& x,
                                         paddle::experimental::Scalar y);

paddle::experimental::Tensor prelu_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& alpha,
    std::string data_format,
    std::string mode);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
prior_box_ad_func(const paddle::experimental::Tensor& input,
                  const paddle::experimental::Tensor& image,
                  std::vector<float> min_sizes,
                  std::vector<float> aspect_ratios,
                  std::vector<float> variances,
                  std::vector<float> max_sizes = {},
                  bool flip = true,
                  bool clip = true,
                  float step_w = 0.0,
                  float step_h = 0.0,
                  float offset = 0.5,
                  bool min_max_aspect_ratios_order = false);

paddle::experimental::Tensor prod_ad_func(const paddle::experimental::Tensor& x,
                                          paddle::experimental::IntArray dims,
                                          bool keep_dim,
                                          bool reduce_all);

paddle::experimental::Tensor psroi_pool_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& boxes,
    const paddle::optional<paddle::experimental::Tensor>& boxes_num,
    int pooled_height,
    int pooled_width,
    int output_channels,
    float spatial_scale);

paddle::experimental::Tensor randint_ad_func(
    int low,
    int high,
    paddle::experimental::IntArray shape,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::INT64,
    paddle::Place place = {});

paddle::experimental::Tensor randperm_ad_func(
    int n, paddle::experimental::DataType dtype, paddle::Place place = {});

paddle::experimental::Tensor real_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor relu6_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor remainder_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);
paddle::experimental::Tensor& remainder__ad_func(
    paddle::experimental::Tensor& x, const paddle::experimental::Tensor& y);

paddle::experimental::Tensor repeat_interleave_ad_func(
    const paddle::experimental::Tensor& x, int repeats, int axis);

paddle::experimental::Tensor repeat_interleave_with_tensor_index_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& repeats,
    int axis);

paddle::experimental::Tensor reshape_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::IntArray shape);
paddle::experimental::Tensor& reshape__ad_func(
    paddle::experimental::Tensor& x, paddle::experimental::IntArray shape);

paddle::experimental::Tensor reverse_ad_func(
    const paddle::experimental::Tensor& x, paddle::experimental::IntArray axis);

std::tuple<paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::optional<paddle::experimental::Tensor>&>
rmsprop__ad_func(paddle::experimental::Tensor& param,
                 paddle::experimental::Tensor& mean_square,
                 const paddle::experimental::Tensor& grad,
                 paddle::experimental::Tensor& moment,
                 const paddle::experimental::Tensor& learning_rate,
                 paddle::optional<paddle::experimental::Tensor>& mean_grad,
                 float epsilon,
                 float decay,
                 float momentum,
                 bool centered);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           std::vector<paddle::experimental::Tensor>>
rnn_ad_func(
    const paddle::experimental::Tensor& x,
    const std::vector<paddle::experimental::Tensor>& pre_state,
    const std::vector<paddle::experimental::Tensor>& weight_list,
    const paddle::optional<paddle::experimental::Tensor>& sequence_length,
    const paddle::experimental::Tensor& dropout_state_in,
    float dropout_prob,
    bool is_bidirec,
    int input_size,
    int hidden_size,
    int num_layers,
    std::string mode,
    int seed,
    bool is_test);

paddle::experimental::Tensor roi_align_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& boxes,
    const paddle::optional<paddle::experimental::Tensor>& boxes_num,
    int pooled_height,
    int pooled_width,
    float spatial_scale,
    int sampling_ratio,
    bool aligned);

paddle::experimental::Tensor roi_pool_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& boxes,
    const paddle::optional<paddle::experimental::Tensor>& boxes_num,
    int pooled_height,
    int pooled_width,
    float spatial_scale);

paddle::experimental::Tensor scale_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::Scalar scale,
    float bias,
    bool bias_after_scale);
paddle::experimental::Tensor& scale__ad_func(paddle::experimental::Tensor& x,
                                             paddle::experimental::Scalar scale,
                                             float bias,
                                             bool bias_after_scale);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
segment_pool_ad_func(const paddle::experimental::Tensor& x,
                     const paddle::experimental::Tensor& segment_ids,
                     std::string pooltype);

paddle::experimental::Tensor send_u_recv_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& src_index,
    const paddle::experimental::Tensor& dst_index,
    std::string reduce_op = "SUM",
    paddle::experimental::IntArray out_size = {0});

paddle::experimental::Tensor send_ue_recv_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    const paddle::experimental::Tensor& src_index,
    const paddle::experimental::Tensor& dst_index,
    std::string message_op,
    std::string reduce_op,
    paddle::experimental::IntArray out_size);

std::tuple<paddle::experimental::Tensor&,
           paddle::optional<paddle::experimental::Tensor>&>
sgd__ad_func(paddle::experimental::Tensor& param,
             const paddle::experimental::Tensor& learning_rate,
             const paddle::experimental::Tensor& grad,
             paddle::optional<paddle::experimental::Tensor>& master_param,
             bool multi_precision);

paddle::experimental::Tensor shape_ad_func(
    const paddle::experimental::Tensor& input);

paddle::experimental::Tensor sigmoid_cross_entropy_with_logits_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& label,
    bool normalize,
    int ignore_index);

paddle::experimental::Tensor sign_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor slice_ad_func(
    const paddle::experimental::Tensor& input,
    std::vector<int64_t> axes,
    paddle::experimental::IntArray starts,
    paddle::experimental::IntArray ends,
    std::vector<int64_t> infer_flags,
    std::vector<int64_t> decrease_axis);

paddle::experimental::Tensor slogdet_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor softmax_ad_func(
    const paddle::experimental::Tensor& x, int axis);
paddle::experimental::Tensor& softmax__ad_func(paddle::experimental::Tensor& x,
                                               int axis);

paddle::experimental::Tensor spectral_norm_ad_func(
    const paddle::experimental::Tensor& weight,
    const paddle::experimental::Tensor& u,
    const paddle::experimental::Tensor& v,
    int dim,
    int power_iters,
    float eps);

std::vector<paddle::experimental::Tensor> split_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::IntArray num_or_sections,
    paddle::experimental::Scalar axis);

std::vector<paddle::experimental::Tensor> split_with_num_ad_func(
    const paddle::experimental::Tensor& x,
    int num,
    paddle::experimental::Scalar axis);

paddle::experimental::Tensor squared_l2_norm_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor stack_ad_func(
    const std::vector<paddle::experimental::Tensor>& x, int axis);

paddle::experimental::Tensor strided_slice_ad_func(
    const paddle::experimental::Tensor& x,
    std::vector<int> axes,
    paddle::experimental::IntArray starts,
    paddle::experimental::IntArray ends,
    paddle::experimental::IntArray strides);

paddle::experimental::Tensor subtract_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);
paddle::experimental::Tensor& subtract__ad_func(
    paddle::experimental::Tensor& x, const paddle::experimental::Tensor& y);

paddle::experimental::Tensor sum_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::IntArray axis = {},
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::UNDEFINED,
    bool keepdim = false);

paddle::experimental::Tensor swish_ad_func(
    const paddle::experimental::Tensor& x);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
sync_batch_norm__ad_func(const paddle::experimental::Tensor& x,
                         paddle::experimental::Tensor& mean,
                         paddle::experimental::Tensor& variance,
                         const paddle::experimental::Tensor& scale,
                         const paddle::experimental::Tensor& bias,
                         bool is_test,
                         float momentum,
                         float epsilon,
                         std::string data_layout,
                         bool use_global_stats,
                         bool trainable_statistics);

paddle::experimental::Tensor temporal_shift_ad_func(
    const paddle::experimental::Tensor& x,
    int seg_num,
    float shift_ratio,
    std::string data_format_str);

paddle::experimental::Tensor tile_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::IntArray repeat_times);

paddle::experimental::Tensor transpose_ad_func(
    const paddle::experimental::Tensor& x, std::vector<int> perm);

paddle::experimental::Tensor triangular_solve_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    bool upper,
    bool tranpose,
    bool unitriangular);

paddle::experimental::Tensor tril_ad_func(const paddle::experimental::Tensor& x,
                                          int diagonal);

paddle::experimental::Tensor tril_indices_ad_func(
    int rows,
    int cols,
    int offset,
    paddle::experimental::DataType dtype,
    paddle::Place place = {});

paddle::experimental::Tensor trilinear_interp_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::optional<paddle::experimental::Tensor>& out_size,
    const paddle::optional<std::vector<paddle::experimental::Tensor>>&
        size_tensor,
    const paddle::optional<paddle::experimental::Tensor>& scale_tensor,
    std::string data_layout,
    int out_d,
    int out_h,
    int out_w,
    std::vector<float> scale,
    std::string interp_method,
    bool align_corners,
    int align_mode);

paddle::experimental::Tensor triu_ad_func(const paddle::experimental::Tensor& x,
                                          int diagonal);

paddle::experimental::Tensor triu_indices_ad_func(
    int row,
    int col,
    int offset,
    paddle::experimental::DataType dtype,
    paddle::Place place = {});

paddle::experimental::Tensor truncated_gaussian_random_ad_func(
    std::vector<int> shape,
    float mean,
    float std,
    int seed,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::FLOAT32,
    paddle::Place place = {});

std::vector<paddle::experimental::Tensor> unbind_ad_func(
    const paddle::experimental::Tensor& input, int axis);

paddle::experimental::Tensor uniform_ad_func(
    paddle::experimental::IntArray shape,
    paddle::experimental::DataType dtype,
    paddle::experimental::Scalar min,
    paddle::experimental::Scalar max,
    int seed,
    paddle::Place place = {});

paddle::experimental::Tensor uniform_inplace_ad_func(
    const paddle::experimental::Tensor& x,
    float min,
    float max,
    int seed,
    int diag_num,
    int diag_step,
    float diag_val);
paddle::experimental::Tensor& uniform_inplace__ad_func(
    paddle::experimental::Tensor& x,
    float min,
    float max,
    int seed,
    int diag_num,
    int diag_step,
    float diag_val);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
unique_ad_func(const paddle::experimental::Tensor& x,
               bool return_index,
               bool return_inverse,
               bool return_counts,
               std::vector<int> axis,
               paddle::experimental::DataType dtype =
                   paddle::experimental::DataType::INT64);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
unique_consecutive_ad_func(const paddle::experimental::Tensor& x,
                           bool return_inverse,
                           bool return_counts,
                           std::vector<int> axis,
                           int dtype);

paddle::experimental::Tensor unpool_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& indices,
    std::vector<int> ksize,
    std::vector<int> strides,
    std::vector<int> padding,
    paddle::experimental::IntArray output_size,
    std::string data_format);

paddle::experimental::Tensor unpool3d_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& indices,
    std::vector<int> ksize,
    std::vector<int> strides,
    std::vector<int> padding,
    std::vector<int> output_size,
    std::string data_format);

std::tuple<std::vector<paddle::experimental::Tensor>&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&>
update_loss_scaling__ad_func(std::vector<paddle::experimental::Tensor>& x,
                             const paddle::experimental::Tensor& found_infinite,
                             paddle::experimental::Tensor& prev_loss_scaling,
                             paddle::experimental::Tensor& in_good_steps,
                             paddle::experimental::Tensor& in_bad_steps,
                             int incr_every_n_steps,
                             int decr_every_n_nan_or_inf,
                             float incr_ratio,
                             float decr_ratio,
                             paddle::experimental::Scalar stop_update);

paddle::experimental::Tensor warpctc_ad_func(
    const paddle::experimental::Tensor& logits,
    const paddle::experimental::Tensor& label,
    const paddle::optional<paddle::experimental::Tensor>& logits_length,
    const paddle::optional<paddle::experimental::Tensor>& labels_length,
    int blank,
    bool norm_by_times);

std::tuple<paddle::experimental::Tensor, paddle::experimental::Tensor>
yolo_box_ad_func(const paddle::experimental::Tensor& x,
                 const paddle::experimental::Tensor& img_size,
                 std::vector<int> anchors,
                 int class_num,
                 float conf_thresh,
                 int downsample_ratio,
                 bool clip_bbox,
                 float scale_x_y = 1.0,
                 bool iou_aware = false,
                 float iou_aware_factor = 0.5);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
yolo_loss_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& gt_box,
    const paddle::experimental::Tensor& gt_label,
    const paddle::optional<paddle::experimental::Tensor>& gt_score,
    std::vector<int> anchors,
    std::vector<int> anchor_mask,
    int class_num,
    float ignore_thresh,
    int downsample_ratio,
    bool use_label_smooth = true,
    float scale_x_y = 1.0);

paddle::experimental::Tensor zeros_ad_func(
    paddle::experimental::IntArray shape,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::FLOAT32,
    paddle::Place place = CPUPlace());

paddle::experimental::Tensor zeros_like_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::UNDEFINED,
    paddle::Place place = {});

namespace sparse {
paddle::experimental::Tensor abs_ad_func(const paddle::experimental::Tensor& x);

paddle::experimental::Tensor acos_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor acosh_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor add_ad_func(const paddle::experimental::Tensor& x,
                                         const paddle::experimental::Tensor& y);

paddle::experimental::Tensor asin_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor asinh_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor atan_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor atanh_ad_func(
    const paddle::experimental::Tensor& x);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
batch_norm__ad_func(const paddle::experimental::Tensor& x,
                    paddle::experimental::Tensor& mean,
                    paddle::experimental::Tensor& variance,
                    const paddle::experimental::Tensor& scale,
                    const paddle::experimental::Tensor& bias,
                    bool is_test,
                    float momentum,
                    float epsilon,
                    std::string data_layout,
                    bool use_global_stats,
                    bool trainable_statistics);

paddle::experimental::Tensor cast_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::DataType index_dtype,
    paddle::experimental::DataType value_dtype);

paddle::experimental::Tensor conv3d_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& kernel,
    std::vector<int> paddings,
    std::vector<int> dilations,
    std::vector<int> strides,
    int groups,
    bool subm,
    std::string key);

paddle::experimental::Tensor divide_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor divide_scalar_ad_func(
    const paddle::experimental::Tensor& x, float scalar);

paddle::experimental::Tensor expm1_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor leaky_relu_ad_func(
    const paddle::experimental::Tensor& x, float alpha);

paddle::experimental::Tensor log1p_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor multiply_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor pow_ad_func(const paddle::experimental::Tensor& x,
                                         float factor);

paddle::experimental::Tensor relu_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor relu6_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor reshape_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::IntArray shape);

paddle::experimental::Tensor scale_ad_func(
    const paddle::experimental::Tensor& x,
    float scale,
    float bias,
    bool bias_after_scale);

paddle::experimental::Tensor sin_ad_func(const paddle::experimental::Tensor& x);

paddle::experimental::Tensor sinh_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor softmax_ad_func(
    const paddle::experimental::Tensor& x, int axis = -1);

paddle::experimental::Tensor sparse_coo_tensor_ad_func(
    const paddle::experimental::Tensor& values,
    const paddle::experimental::Tensor& indices,
    std::vector<int64_t> shape);

paddle::experimental::Tensor sqrt_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor square_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor subtract_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor&,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
sync_batch_norm__ad_func(const paddle::experimental::Tensor& x,
                         paddle::experimental::Tensor& mean,
                         paddle::experimental::Tensor& variance,
                         const paddle::experimental::Tensor& scale,
                         const paddle::experimental::Tensor& bias,
                         bool is_test,
                         float momentum,
                         float epsilon,
                         std::string data_layout,
                         bool use_global_stats,
                         bool trainable_statistics);

paddle::experimental::Tensor tan_ad_func(const paddle::experimental::Tensor& x);

paddle::experimental::Tensor tanh_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor to_dense_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor to_sparse_coo_ad_func(
    const paddle::experimental::Tensor& x, int64_t sparse_dim);

paddle::experimental::Tensor to_sparse_csr_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor transpose_ad_func(
    const paddle::experimental::Tensor& x, std::vector<int> perm);

paddle::experimental::Tensor values_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor addmm_ad_func(
    const paddle::experimental::Tensor& input,
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    float beta = 1.0,
    float alpha = 1.0);

paddle::experimental::Tensor coalesce_ad_func(
    const paddle::experimental::Tensor& x);

paddle::experimental::Tensor full_like_ad_func(
    const paddle::experimental::Tensor& x,
    paddle::experimental::Scalar value,
    paddle::experimental::DataType dtype =
        paddle::experimental::DataType::UNDEFINED);

paddle::experimental::Tensor fused_attention_ad_func(
    const paddle::experimental::Tensor& query,
    const paddle::experimental::Tensor& key,
    const paddle::experimental::Tensor& value,
    const paddle::experimental::Tensor& sparse_mask,
    const paddle::optional<paddle::experimental::Tensor>& key_padding_mask,
    const paddle::optional<paddle::experimental::Tensor>& attn_mask);

paddle::experimental::Tensor masked_matmul_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y,
    const paddle::experimental::Tensor& mask);

paddle::experimental::Tensor matmul_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& y);

paddle::experimental::Tensor maxpool_ad_func(
    const paddle::experimental::Tensor& x,
    std::vector<int> kernel_sizes,
    std::vector<int> paddings,
    std::vector<int> dilations,
    std::vector<int> strides);

paddle::experimental::Tensor mv_ad_func(
    const paddle::experimental::Tensor& x,
    const paddle::experimental::Tensor& vec);

}  // namespace sparse

namespace strings {
paddle::experimental::Tensor empty_ad_func(paddle::experimental::IntArray shape,
                                           paddle::Place place = CPUPlace());

paddle::experimental::Tensor empty_like_ad_func(
    const paddle::experimental::Tensor& x, paddle::Place place = {});

paddle::experimental::Tensor lower_ad_func(
    const paddle::experimental::Tensor& x, bool use_utf8_encoding);

paddle::experimental::Tensor upper_ad_func(
    const paddle::experimental::Tensor& x, bool use_utf8_encoding);

}  // namespace strings
