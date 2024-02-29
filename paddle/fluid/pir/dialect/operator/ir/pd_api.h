

#pragma once

#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/manual_api.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/pir/core/value.h"
#include "paddle/utils/optional.h"

namespace paddle {

namespace dialect {

pir::OpResult abs(const pir::Value& x);

pir::OpResult abs_(const pir::Value& x);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> accuracy(
    const pir::Value& x, const pir::Value& indices, const pir::Value& label);

pir::OpResult acos(const pir::Value& x);

pir::OpResult acos_(const pir::Value& x);

pir::OpResult acosh(const pir::Value& x);

pir::OpResult acosh_(const pir::Value& x);

std::tuple<pir::OpResult, pir::OpResult, paddle::optional<pir::OpResult>>
adagrad_(const pir::Value& param,
         const pir::Value& grad,
         const pir::Value& moment,
         const pir::Value& learning_rate,
         const paddle::optional<pir::Value>& master_param,
         float epsilon = 1.0e-6f,
         bool multi_precision = false);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
adam_(const pir::Value& param,
      const pir::Value& grad,
      const pir::Value& learning_rate,
      const pir::Value& moment1,
      const pir::Value& moment2,
      const pir::Value& beta1_pow,
      const pir::Value& beta2_pow,
      const paddle::optional<pir::Value>& master_param,
      const paddle::optional<pir::Value>& skip_update,
      float beta1 = 0.9f,
      float beta2 = 0.999f,
      float epsilon = 1.0e-8f,
      bool lazy_mode = false,
      int64_t min_row_size_to_use_multithread = 1000,
      bool multi_precision = false,
      bool use_global_beta_pow = false);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
adam_(const pir::Value& param,
      const pir::Value& grad,
      const pir::Value& learning_rate,
      const pir::Value& moment1,
      const pir::Value& moment2,
      const pir::Value& beta1_pow,
      const pir::Value& beta2_pow,
      const paddle::optional<pir::Value>& master_param,
      const paddle::optional<pir::Value>& skip_update,
      pir::Value beta1,
      pir::Value beta2,
      pir::Value epsilon,
      bool lazy_mode = false,
      int64_t min_row_size_to_use_multithread = 1000,
      bool multi_precision = false,
      bool use_global_beta_pow = false);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
adamax_(const pir::Value& param,
        const pir::Value& grad,
        const pir::Value& learning_rate,
        const pir::Value& moment,
        const pir::Value& inf_norm,
        const pir::Value& beta1_pow,
        const paddle::optional<pir::Value>& master_param,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1.0e-8f,
        bool multi_precision = false);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
adamw_(const pir::Value& param,
       const pir::Value& grad,
       const pir::Value& learning_rate,
       const pir::Value& moment1,
       const pir::Value& moment2,
       const pir::Value& beta1_pow,
       const pir::Value& beta2_pow,
       const paddle::optional<pir::Value>& master_param,
       const paddle::optional<pir::Value>& skip_update,
       float beta1 = 0.9f,
       float beta2 = 0.999f,
       float epsilon = 1.0e-8f,
       float lr_ratio = 1.0f,
       float coeff = 0.01f,
       bool with_decay = false,
       bool lazy_mode = false,
       int64_t min_row_size_to_use_multithread = 1000,
       bool multi_precision = false,
       bool use_global_beta_pow = false);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
adamw_(const pir::Value& param,
       const pir::Value& grad,
       const pir::Value& learning_rate,
       const pir::Value& moment1,
       const pir::Value& moment2,
       const pir::Value& beta1_pow,
       const pir::Value& beta2_pow,
       const paddle::optional<pir::Value>& master_param,
       const paddle::optional<pir::Value>& skip_update,
       pir::Value beta1,
       pir::Value beta2,
       pir::Value epsilon,
       float lr_ratio = 1.0f,
       float coeff = 0.01f,
       bool with_decay = false,
       bool lazy_mode = false,
       int64_t min_row_size_to_use_multithread = 1000,
       bool multi_precision = false,
       bool use_global_beta_pow = false);

pir::OpResult addmm(const pir::Value& input,
                    const pir::Value& x,
                    const pir::Value& y,
                    float beta = 1.0,
                    float alpha = 1.0);

pir::OpResult addmm_(const pir::Value& input,
                     const pir::Value& x,
                     const pir::Value& y,
                     float beta = 1.0,
                     float alpha = 1.0);

pir::OpResult affine_grid(const pir::Value& input,
                          const std::vector<int64_t>& output_shape = {},
                          bool align_corners = true);

pir::OpResult affine_grid(const pir::Value& input,
                          pir::Value output_shape,
                          bool align_corners = true);

pir::OpResult affine_grid(const pir::Value& input,
                          std::vector<pir::Value> output_shape,
                          bool align_corners = true);

pir::OpResult allclose(const pir::Value& x,
                       const pir::Value& y,
                       float rtol = 1e-5,
                       float atol = 1e-8,
                       bool equal_nan = false);

pir::OpResult allclose(const pir::Value& x,
                       const pir::Value& y,
                       pir::Value rtol,
                       pir::Value atol,
                       bool equal_nan = false);

pir::OpResult angle(const pir::Value& x);

pir::OpResult argmax(const pir::Value& x,
                     int64_t axis,
                     bool keepdims = false,
                     bool flatten = false,
                     phi::DataType dtype = phi::DataType::INT64);

pir::OpResult argmax(const pir::Value& x,
                     pir::Value axis,
                     bool keepdims = false,
                     bool flatten = false,
                     phi::DataType dtype = phi::DataType::INT64);

pir::OpResult argmin(const pir::Value& x,
                     int64_t axis,
                     bool keepdims = false,
                     bool flatten = false,
                     phi::DataType dtype = phi::DataType::INT64);

pir::OpResult argmin(const pir::Value& x,
                     pir::Value axis,
                     bool keepdims = false,
                     bool flatten = false,
                     phi::DataType dtype = phi::DataType::INT64);

std::tuple<pir::OpResult, pir::OpResult> argsort(const pir::Value& x,
                                                 int axis = -1,
                                                 bool descending = false);

pir::OpResult as_complex(const pir::Value& x);

pir::OpResult as_real(const pir::Value& x);

pir::OpResult as_strided(const pir::Value& input,
                         const std::vector<int64_t>& dims = {},
                         const std::vector<int64_t>& stride = {},
                         int64_t offset = 0);

pir::OpResult asin(const pir::Value& x);

pir::OpResult asin_(const pir::Value& x);

pir::OpResult asinh(const pir::Value& x);

pir::OpResult asinh_(const pir::Value& x);

pir::OpResult atan(const pir::Value& x);

pir::OpResult atan_(const pir::Value& x);

pir::OpResult atan2(const pir::Value& x, const pir::Value& y);

pir::OpResult atanh(const pir::Value& x);

pir::OpResult atanh_(const pir::Value& x);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> auc(
    const pir::Value& x,
    const pir::Value& label,
    const pir::Value& stat_pos,
    const pir::Value& stat_neg,
    const paddle::optional<pir::Value>& ins_tag_weight,
    const std::string& curve = "ROC",
    int num_thresholds = (2 << 12) - 1,
    int slide_steps = 1);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
average_accumulates_(const pir::Value& param,
                     const pir::Value& in_sum_1,
                     const pir::Value& in_sum_2,
                     const pir::Value& in_sum_3,
                     const pir::Value& in_num_accumulates,
                     const pir::Value& in_old_num_accumulates,
                     const pir::Value& in_num_updates,
                     float average_window = 0,
                     int64_t max_average_window = INT64_MAX,
                     int64_t min_average_window = 10000L);

pir::OpResult bce_loss(const pir::Value& input, const pir::Value& label);

pir::OpResult bce_loss_(const pir::Value& input, const pir::Value& label);

pir::OpResult bernoulli(const pir::Value& x);

pir::OpResult bicubic_interp(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const std::string& data_layout = "NCHW",
    int out_d = 0,
    int out_h = 0,
    int out_w = 0,
    const std::vector<float>& scale = {},
    const std::string& interp_method = "bilinear",
    bool align_corners = true,
    int align_mode = 1);

pir::OpResult bilinear(const pir::Value& x,
                       const pir::Value& y,
                       const pir::Value& weight,
                       const paddle::optional<pir::Value>& bias);

pir::OpResult bilinear_interp(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const std::string& data_layout = "NCHW",
    int out_d = 0,
    int out_h = 0,
    int out_w = 0,
    const std::vector<float>& scale = {},
    const std::string& interp_method = "bilinear",
    bool align_corners = true,
    int align_mode = 1);

pir::OpResult bincount(const pir::Value& x,
                       const paddle::optional<pir::Value>& weights,
                       int minlength = 0);

pir::OpResult bincount(const pir::Value& x,
                       const paddle::optional<pir::Value>& weights,
                       pir::Value minlength);

pir::OpResult binomial(const pir::Value& count, const pir::Value& prob);

pir::OpResult bitwise_and(const pir::Value& x, const pir::Value& y);

pir::OpResult bitwise_and_(const pir::Value& x, const pir::Value& y);

pir::OpResult bitwise_not(const pir::Value& x);

pir::OpResult bitwise_not_(const pir::Value& x);

pir::OpResult bitwise_or(const pir::Value& x, const pir::Value& y);

pir::OpResult bitwise_or_(const pir::Value& x, const pir::Value& y);

pir::OpResult bitwise_xor(const pir::Value& x, const pir::Value& y);

pir::OpResult bitwise_xor_(const pir::Value& x, const pir::Value& y);

pir::OpResult bmm(const pir::Value& x, const pir::Value& y);

pir::OpResult box_coder(const pir::Value& prior_box,
                        const paddle::optional<pir::Value>& prior_box_var,
                        const pir::Value& target_box,
                        const std::string& code_type = "encode_center_size",
                        bool box_normalized = true,
                        int axis = 0,
                        const std::vector<float>& variance = {});

std::vector<pir::OpResult> broadcast_tensors(
    const std::vector<pir::Value>& input);

pir::OpResult ceil(const pir::Value& x);

pir::OpResult ceil_(const pir::Value& x);

pir::OpResult celu(const pir::Value& x, float alpha = 1.0);

std::tuple<std::vector<pir::OpResult>, pir::OpResult> check_finite_and_unscale_(
    const std::vector<pir::Value>& x, const pir::Value& scale);

std::tuple<pir::OpResult, pir::OpResult> check_numerics(
    const pir::Value& tensor,
    const std::string& op_type = "",
    const std::string& var_name = "",
    int check_nan_inf_level = 0,
    int stack_height_limit = -1,
    const std::string& output_dir = "");

pir::OpResult cholesky(const pir::Value& x, bool upper = false);

pir::OpResult cholesky_solve(const pir::Value& x,
                             const pir::Value& y,
                             bool upper = false);

std::tuple<pir::OpResult, pir::OpResult> class_center_sample(
    const pir::Value& label,
    int num_classes,
    int num_samples,
    int ring_id = 0,
    int rank = 0,
    int nranks = 1,
    bool fix_seed = false,
    int seed = 0);

pir::OpResult clip(const pir::Value& x, float min, float max);

pir::OpResult clip(const pir::Value& x, pir::Value min, pir::Value max);

pir::OpResult clip_(const pir::Value& x, float min, float max);

pir::OpResult clip_(const pir::Value& x, pir::Value min, pir::Value max);

pir::OpResult clip_by_norm(const pir::Value& x, float max_norm);

std::tuple<std::vector<pir::OpResult>, pir::OpResult> coalesce_tensor(
    const std::vector<pir::Value>& input,
    phi::DataType dtype,
    bool copy_data = false,
    bool set_constant = false,
    bool persist_output = false,
    float constant = 0.0,
    bool use_align = true,
    int align_size = -1,
    int size_of_dtype = -1,
    const std::vector<int64_t>& concated_shapes = {},
    const std::vector<int64_t>& concated_ranks = {});

pir::OpResult complex(const pir::Value& real, const pir::Value& imag);

pir::OpResult concat(const std::vector<pir::Value>& x, int axis = 0);

pir::OpResult concat(const std::vector<pir::Value>& x, pir::Value axis);

pir::OpResult conj(const pir::Value& x);

pir::OpResult conv2d(const pir::Value& input,
                     const pir::Value& filter,
                     const std::vector<int>& strides = {1, 1},
                     const std::vector<int>& paddings = {0, 0},
                     const std::string& padding_algorithm = "EXPLICIT",
                     const std::vector<int>& dilations = {1, 1},
                     int groups = 1,
                     const std::string& data_format = "NCHW");

pir::OpResult conv3d(const pir::Value& input,
                     const pir::Value& filter,
                     const std::vector<int>& strides = {1, 1, 1},
                     const std::vector<int>& paddings = {0, 0, 0},
                     const std::string& padding_algorithm = "EXPLICIT",
                     int groups = 1,
                     const std::vector<int>& dilations = {1, 1, 1},
                     const std::string& data_format = "NCDHW");

pir::OpResult conv3d_transpose(
    const pir::Value& x,
    const pir::Value& filter,
    const std::vector<int>& strides = {1, 1, 1},
    const std::vector<int>& paddings = {0, 0, 0},
    const std::vector<int>& output_padding = {},
    const std::vector<int>& output_size = {},
    const std::string& padding_algorithm = "EXPLICIT",
    int groups = 1,
    const std::vector<int>& dilations = {1, 1, 1},
    const std::string& data_format = "NCHW");

pir::OpResult cos(const pir::Value& x);

pir::OpResult cos_(const pir::Value& x);

pir::OpResult cosh(const pir::Value& x);

pir::OpResult cosh_(const pir::Value& x);

pir::OpResult crop(const pir::Value& x,
                   const std::vector<int64_t>& shape = {},
                   const std::vector<int64_t>& offsets = {});

pir::OpResult crop(const pir::Value& x, pir::Value shape, pir::Value offsets);

pir::OpResult crop(const pir::Value& x,
                   std::vector<pir::Value> shape,
                   std::vector<pir::Value> offsets);

pir::OpResult cross(const pir::Value& x, const pir::Value& y, int axis = 9);

std::tuple<pir::OpResult, pir::OpResult> cross_entropy_with_softmax(
    const pir::Value& input,
    const pir::Value& label,
    bool soft_label = false,
    bool use_softmax = true,
    bool numeric_stable_mode = true,
    int ignore_index = -100,
    int axis = -1);

std::tuple<pir::OpResult, pir::OpResult> cross_entropy_with_softmax_(
    const pir::Value& input,
    const pir::Value& label,
    bool soft_label = false,
    bool use_softmax = true,
    bool numeric_stable_mode = true,
    int ignore_index = -100,
    int axis = -1);

std::tuple<pir::OpResult, pir::OpResult> cummax(
    const pir::Value& x,
    int axis = -1,
    phi::DataType dtype = phi::DataType::INT64);

std::tuple<pir::OpResult, pir::OpResult> cummin(
    const pir::Value& x,
    int axis = -1,
    phi::DataType dtype = phi::DataType::INT64);

pir::OpResult cumprod(const pir::Value& x, int dim);

pir::OpResult cumprod_(const pir::Value& x, int dim);

pir::OpResult cumsum(const pir::Value& x,
                     int axis = -1,
                     bool flatten = false,
                     bool exclusive = false,
                     bool reverse = false);

pir::OpResult cumsum(const pir::Value& x,
                     pir::Value axis,
                     bool flatten = false,
                     bool exclusive = false,
                     bool reverse = false);

pir::OpResult cumsum_(const pir::Value& x,
                      int axis = -1,
                      bool flatten = false,
                      bool exclusive = false,
                      bool reverse = false);

pir::OpResult cumsum_(const pir::Value& x,
                      pir::Value axis,
                      bool flatten = false,
                      bool exclusive = false,
                      bool reverse = false);

pir::OpResult data(const std::string& name,
                   const std::vector<int64_t>& shape,
                   phi::DataType dtype,
                   const Place& place);

pir::OpResult depthwise_conv2d(
    const pir::Value& input,
    const pir::Value& filter,
    const std::vector<int>& strides = {1, 1},
    const std::vector<int>& paddings = {0, 0},
    const std::string& padding_algorithm = "EXPLICIT",
    int groups = 1,
    const std::vector<int>& dilations = {1, 1},
    const std::string& data_format = "NCHW");

pir::OpResult det(const pir::Value& x);

pir::OpResult diag(const pir::Value& x,
                   int offset = 0,
                   float padding_value = 0.0);

pir::OpResult diag_embed(const pir::Value& input,
                         int offset = 0,
                         int dim1 = -2,
                         int dim2 = -1);

pir::OpResult diagonal(const pir::Value& x,
                       int offset = 0,
                       int axis1 = 0,
                       int axis2 = 1);

pir::OpResult digamma(const pir::Value& x);

pir::OpResult digamma_(const pir::Value& x);

pir::OpResult dirichlet(const pir::Value& alpha);

pir::OpResult dist(const pir::Value& x, const pir::Value& y, float p = 2.0);

pir::OpResult dot(const pir::Value& x, const pir::Value& y);

std::tuple<pir::OpResult, pir::OpResult> edit_distance(
    const pir::Value& hyps,
    const pir::Value& refs,
    const paddle::optional<pir::Value>& hypslength,
    const paddle::optional<pir::Value>& refslength,
    bool normalized = false);

std::tuple<pir::OpResult, pir::OpResult> eig(const pir::Value& x);

std::tuple<pir::OpResult, pir::OpResult> eigh(const pir::Value& x,
                                              const std::string& UPLO = "L");

pir::OpResult eigvals(const pir::Value& x);

std::tuple<pir::OpResult, pir::OpResult> eigvalsh(const pir::Value& x,
                                                  const std::string& uplo = "L",
                                                  bool is_test = false);

pir::OpResult elu(const pir::Value& x, float alpha = 1.0f);

pir::OpResult elu_(const pir::Value& x, float alpha = 1.0f);

pir::OpResult equal_all(const pir::Value& x, const pir::Value& y);

pir::OpResult erf(const pir::Value& x);

pir::OpResult erf_(const pir::Value& x);

pir::OpResult erfinv(const pir::Value& x);

pir::OpResult erfinv_(const pir::Value& x);

pir::OpResult exp(const pir::Value& x);

pir::OpResult exp_(const pir::Value& x);

pir::OpResult expand(const pir::Value& x,
                     const std::vector<int64_t>& shape = {});

pir::OpResult expand(const pir::Value& x, pir::Value shape);

pir::OpResult expand(const pir::Value& x, std::vector<pir::Value> shape);

pir::OpResult expand_as(const pir::Value& x,
                        const paddle::optional<pir::Value>& y,
                        const std::vector<int>& target_shape = {});

pir::OpResult expm1(const pir::Value& x);

pir::OpResult expm1_(const pir::Value& x);

pir::OpResult fft_c2c(const pir::Value& x,
                      const std::vector<int64_t>& axes,
                      const std::string& normalization,
                      bool forward);

pir::OpResult fft_c2r(const pir::Value& x,
                      const std::vector<int64_t>& axes,
                      const std::string& normalization,
                      bool forward,
                      int64_t last_dim_size = 0L);

pir::OpResult fft_r2c(const pir::Value& x,
                      const std::vector<int64_t>& axes,
                      const std::string& normalization,
                      bool forward,
                      bool onesided);

pir::OpResult fill(const pir::Value& x, float value = 0);

pir::OpResult fill(const pir::Value& x, pir::Value value);

pir::OpResult fill_(const pir::Value& x, float value = 0);

pir::OpResult fill_(const pir::Value& x, pir::Value value);

pir::OpResult fill_diagonal(const pir::Value& x,
                            float value = 0,
                            int offset = 0,
                            bool wrap = false);

pir::OpResult fill_diagonal_(const pir::Value& x,
                             float value = 0,
                             int offset = 0,
                             bool wrap = false);

pir::OpResult fill_diagonal_tensor(const pir::Value& x,
                                   const pir::Value& y,
                                   int64_t offset = 0,
                                   int dim1 = 0,
                                   int dim2 = 1);

pir::OpResult fill_diagonal_tensor_(const pir::Value& x,
                                    const pir::Value& y,
                                    int64_t offset = 0,
                                    int dim1 = 0,
                                    int dim2 = 1);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
flash_attn(const pir::Value& q,
           const pir::Value& k,
           const pir::Value& v,
           const paddle::optional<pir::Value>& fixed_seed_offset,
           const paddle::optional<pir::Value>& attn_mask,
           float dropout = 0.0,
           bool causal = false,
           bool return_softmax = false,
           bool is_test = false,
           const std::string& rng_name = "");

std::tuple<pir::OpResult, pir::OpResult> flash_attn_unpadded(
    const pir::Value& q,
    const pir::Value& k,
    const pir::Value& v,
    const pir::Value& cu_seqlens_q,
    const pir::Value& cu_seqlens_k,
    const paddle::optional<pir::Value>& fixed_seed_offset,
    const paddle::optional<pir::Value>& attn_mask,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float scale,
    float dropout = 0.0,
    bool causal = false,
    bool return_softmax = false,
    bool is_test = false,
    const std::string& rng_name = "");

pir::OpResult flatten(const pir::Value& x,
                      int start_axis = 1,
                      int stop_axis = 1);

pir::OpResult flatten_(const pir::Value& x,
                       int start_axis = 1,
                       int stop_axis = 1);

pir::OpResult flip(const pir::Value& x, const std::vector<int>& axis);

pir::OpResult floor(const pir::Value& x);

pir::OpResult floor_(const pir::Value& x);

pir::OpResult fmax(const pir::Value& x, const pir::Value& y);

pir::OpResult fmin(const pir::Value& x, const pir::Value& y);

pir::OpResult fold(const pir::Value& x,
                   const std::vector<int>& output_sizes,
                   const std::vector<int>& kernel_sizes,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   const std::vector<int>& dilations);

pir::OpResult frame(const pir::Value& x,
                    int frame_length,
                    int hop_length,
                    int axis = -1);

pir::OpResult full_int_array(const std::vector<int64_t>& value,
                             phi::DataType dtype = phi::DataType::FLOAT32,
                             const Place& place = phi::CPUPlace());

pir::OpResult gammaln(const pir::Value& x);

pir::OpResult gammaln_(const pir::Value& x);

pir::OpResult gather(const pir::Value& x,
                     const pir::Value& index,
                     int axis = 0);

pir::OpResult gather(const pir::Value& x,
                     const pir::Value& index,
                     pir::Value axis);

pir::OpResult gather_nd(const pir::Value& x, const pir::Value& index);

pir::OpResult gather_tree(const pir::Value& ids, const pir::Value& parents);

pir::OpResult gaussian_inplace(const pir::Value& x,
                               float mean = 0,
                               float std = 1.0,
                               int seed = 0);

pir::OpResult gaussian_inplace_(const pir::Value& x,
                                float mean = 0,
                                float std = 1.0,
                                int seed = 0);

pir::OpResult gelu(const pir::Value& x, bool approximate = false);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> generate_proposals(
    const pir::Value& scores,
    const pir::Value& bbox_deltas,
    const pir::Value& im_shape,
    const pir::Value& anchors,
    const pir::Value& variances,
    int pre_nms_top_n,
    int post_nms_top_n,
    float nms_thresh,
    float min_size,
    float eta,
    bool pixel_offset = true);

pir::OpResult grid_sample(const pir::Value& x,
                          const pir::Value& grid,
                          const std::string& mode = "bilinear",
                          const std::string& padding_mode = "zeros",
                          bool align_corners = true);

pir::OpResult group_norm(const pir::Value& x,
                         const paddle::optional<pir::Value>& scale,
                         const paddle::optional<pir::Value>& bias,
                         float epsilon = 1e-5,
                         int groups = -1,
                         const std::string& data_layout = "NCHW");

pir::OpResult gumbel_softmax(const pir::Value& x,
                             float temperature = 1.0,
                             bool hard = false,
                             int axis = -1);

pir::OpResult hardshrink(const pir::Value& x, float threshold = 0.5);

pir::OpResult hardsigmoid(const pir::Value& x,
                          float slope = 0.2,
                          float offset = 0.5);

pir::OpResult hardtanh(const pir::Value& x, float t_min = 0, float t_max = 24);

pir::OpResult hardtanh_(const pir::Value& x, float t_min = 0, float t_max = 24);

pir::OpResult heaviside(const pir::Value& x, const pir::Value& y);

pir::OpResult histogram(const pir::Value& input,
                        int64_t bins = 100,
                        int min = 0,
                        int max = 0);

pir::OpResult huber_loss(const pir::Value& input,
                         const pir::Value& label,
                         float delta);

pir::OpResult i0(const pir::Value& x);

pir::OpResult i0_(const pir::Value& x);

pir::OpResult i0e(const pir::Value& x);

pir::OpResult i1(const pir::Value& x);

pir::OpResult i1e(const pir::Value& x);

pir::OpResult identity_loss(const pir::Value& x, int reduction = 1);

pir::OpResult identity_loss_(const pir::Value& x, int reduction = 1);

pir::OpResult imag(const pir::Value& x);

pir::OpResult index_add(const pir::Value& x,
                        const pir::Value& index,
                        const pir::Value& add_value,
                        int axis = 0);

pir::OpResult index_add_(const pir::Value& x,
                         const pir::Value& index,
                         const pir::Value& add_value,
                         int axis = 0);

pir::OpResult index_put(const pir::Value& x,
                        const std::vector<pir::Value>& indices,
                        const pir::Value& value,
                        bool accumulate = false);

pir::OpResult index_put_(const pir::Value& x,
                         const std::vector<pir::Value>& indices,
                         const pir::Value& value,
                         bool accumulate = false);

pir::OpResult index_sample(const pir::Value& x, const pir::Value& index);

pir::OpResult index_select(const pir::Value& x,
                           const pir::Value& index,
                           int axis = 0);

pir::OpResult index_select_strided(const pir::Value& x,
                                   int64_t index,
                                   int axis = 0);

pir::OpResult instance_norm(const pir::Value& x,
                            const paddle::optional<pir::Value>& scale,
                            const paddle::optional<pir::Value>& bias,
                            float epsilon = 1e-5);

pir::OpResult inverse(const pir::Value& x);

pir::OpResult is_empty(const pir::Value& x);

pir::OpResult isclose(const pir::Value& x,
                      const pir::Value& y,
                      double rtol = 1e-5,
                      double atol = 1e-8,
                      bool equal_nan = false);

pir::OpResult isclose(const pir::Value& x,
                      const pir::Value& y,
                      pir::Value rtol,
                      pir::Value atol,
                      bool equal_nan = false);

pir::OpResult isfinite(const pir::Value& x);

pir::OpResult isinf(const pir::Value& x);

pir::OpResult isnan(const pir::Value& x);

pir::OpResult kldiv_loss(const pir::Value& x,
                         const pir::Value& label,
                         const std::string& reduction = "mean");

pir::OpResult kron(const pir::Value& x, const pir::Value& y);

std::tuple<pir::OpResult, pir::OpResult> kthvalue(const pir::Value& x,
                                                  int k = 1,
                                                  int axis = -1,
                                                  bool keepdim = false);

pir::OpResult label_smooth(const pir::Value& label,
                           const paddle::optional<pir::Value>& prior_dist,
                           float epsilon = 0.0f);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
lamb_(const pir::Value& param,
      const pir::Value& grad,
      const pir::Value& learning_rate,
      const pir::Value& moment1,
      const pir::Value& moment2,
      const pir::Value& beta1_pow,
      const pir::Value& beta2_pow,
      const paddle::optional<pir::Value>& master_param,
      const paddle::optional<pir::Value>& skip_update,
      float weight_decay,
      float beta1 = 0.9,
      float beta2 = 0.999,
      float epsilon = 1.0e-6f,
      bool always_adapt = false,
      bool multi_precision = false);

pir::OpResult layer_norm(const pir::Value& x,
                         const paddle::optional<pir::Value>& scale,
                         const paddle::optional<pir::Value>& bias,
                         float epsilon = 1e-5,
                         int begin_norm_axis = 1);

pir::OpResult leaky_relu(const pir::Value& x, float negative_slope = 0.02f);

pir::OpResult leaky_relu_(const pir::Value& x, float negative_slope = 0.02f);

pir::OpResult lerp(const pir::Value& x,
                   const pir::Value& y,
                   const pir::Value& weight);

pir::OpResult lerp_(const pir::Value& x,
                    const pir::Value& y,
                    const pir::Value& weight);

pir::OpResult lgamma(const pir::Value& x);

pir::OpResult lgamma_(const pir::Value& x);

pir::OpResult linear_interp(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const std::string& data_layout = "NCHW",
    int out_d = 0,
    int out_h = 0,
    int out_w = 0,
    const std::vector<float>& scale = {},
    const std::string& interp_method = "bilinear",
    bool align_corners = true,
    int align_mode = 1);

pir::OpResult llm_int8_linear(const pir::Value& x,
                              const pir::Value& weight,
                              const paddle::optional<pir::Value>& bias,
                              const pir::Value& weight_scale,
                              float threshold = 6.0);

pir::OpResult log(const pir::Value& x);

pir::OpResult log_(const pir::Value& x);

pir::OpResult log10(const pir::Value& x);

pir::OpResult log10_(const pir::Value& x);

pir::OpResult log1p(const pir::Value& x);

pir::OpResult log1p_(const pir::Value& x);

pir::OpResult log2(const pir::Value& x);

pir::OpResult log2_(const pir::Value& x);

pir::OpResult log_loss(const pir::Value& input,
                       const pir::Value& label,
                       float epsilon);

pir::OpResult log_softmax(const pir::Value& x, int axis = -1);

pir::OpResult logcumsumexp(const pir::Value& x,
                           int axis = -1,
                           bool flatten = false,
                           bool exclusive = false,
                           bool reverse = false);

pir::OpResult logical_and(const pir::Value& x, const pir::Value& y);

pir::OpResult logical_and_(const pir::Value& x, const pir::Value& y);

pir::OpResult logical_not(const pir::Value& x);

pir::OpResult logical_not_(const pir::Value& x);

pir::OpResult logical_or(const pir::Value& x, const pir::Value& y);

pir::OpResult logical_or_(const pir::Value& x, const pir::Value& y);

pir::OpResult logical_xor(const pir::Value& x, const pir::Value& y);

pir::OpResult logical_xor_(const pir::Value& x, const pir::Value& y);

pir::OpResult logit(const pir::Value& x, float eps = 1e-6f);

pir::OpResult logit_(const pir::Value& x, float eps = 1e-6f);

pir::OpResult logsigmoid(const pir::Value& x);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult> lstsq(
    const pir::Value& x,
    const pir::Value& y,
    float rcond = 0.0f,
    const std::string& driver = "gels");

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult> lstsq(
    const pir::Value& x,
    const pir::Value& y,
    pir::Value rcond,
    const std::string& driver = "gels");

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> lu(const pir::Value& x,
                                                           bool pivot = true);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> lu_(const pir::Value& x,
                                                            bool pivot = true);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> lu_unpack(
    const pir::Value& x,
    const pir::Value& y,
    bool unpack_ludata = true,
    bool unpack_pivots = true);

std::tuple<pir::OpResult, pir::OpResult> margin_cross_entropy(
    const pir::Value& logits,
    const pir::Value& label,
    bool return_softmax = false,
    int ring_id = 0,
    int rank = 0,
    int nranks = 1,
    float margin1 = 1.0f,
    float margin2 = 0.5f,
    float margin3 = 0.0f,
    float scale = 64.0f);

std::tuple<pir::OpResult, pir::OpResult, paddle::optional<pir::OpResult>>
masked_multihead_attention_(
    const pir::Value& x,
    const pir::Value& cache_kv,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& src_mask,
    const paddle::optional<pir::Value>& cum_offsets,
    const paddle::optional<pir::Value>& sequence_lengths,
    const paddle::optional<pir::Value>& rotary_tensor,
    const paddle::optional<pir::Value>& beam_cache_offset,
    const paddle::optional<pir::Value>& qkv_out_scale,
    const paddle::optional<pir::Value>& out_shift,
    const paddle::optional<pir::Value>& out_smooth,
    int seq_len,
    int rotary_emb_dims,
    bool use_neox_rotary_style = false,
    const std::string& compute_dtype = "default",
    float out_scale = -1,
    int quant_round_type = 1,
    float quant_max_bound = 127.0,
    float quant_min_bound = -127.0);

pir::OpResult masked_select(const pir::Value& x, const pir::Value& mask);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> matrix_nms(
    const pir::Value& bboxes,
    const pir::Value& scores,
    float score_threshold,
    int nms_top_k,
    int keep_top_k,
    float post_threshold = 0.,
    bool use_gaussian = false,
    float gaussian_sigma = 2.,
    int background_label = 0,
    bool normalized = true);

pir::OpResult matrix_power(const pir::Value& x, int n);

std::tuple<pir::OpResult, pir::OpResult> max_pool2d_with_index(
    const pir::Value& x,
    const std::vector<int>& kernel_size,
    const std::vector<int>& strides = {1, 1},
    const std::vector<int>& paddings = {0, 0},
    bool global_pooling = false,
    bool adaptive = false);

std::tuple<pir::OpResult, pir::OpResult> max_pool3d_with_index(
    const pir::Value& x,
    const std::vector<int>& kernel_size,
    const std::vector<int>& strides = {1, 1, 1},
    const std::vector<int>& paddings = {0, 0, 0},
    bool global_pooling = false,
    bool adaptive = false);

pir::OpResult maxout(const pir::Value& x, int groups, int axis = 1);

pir::OpResult mean_all(const pir::Value& x);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
memory_efficient_attention(const pir::Value& query,
                           const pir::Value& key,
                           const pir::Value& value,
                           const paddle::optional<pir::Value>& bias,
                           const paddle::optional<pir::Value>& cu_seqlens_q,
                           const paddle::optional<pir::Value>& cu_seqlens_k,
                           const paddle::optional<pir::Value>& causal_diagonal,
                           const paddle::optional<pir::Value>& seqlen_k,
                           float max_seqlen_q,
                           float max_seqlen_k,
                           bool causal,
                           double dropout_p,
                           float scale,
                           bool is_test);

pir::OpResult merge_selected_rows(const pir::Value& x);

std::tuple<std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           paddle::optional<std::vector<pir::OpResult>>>
merged_adam_(const std::vector<pir::Value>& param,
             const std::vector<pir::Value>& grad,
             const std::vector<pir::Value>& learning_rate,
             const std::vector<pir::Value>& moment1,
             const std::vector<pir::Value>& moment2,
             const std::vector<pir::Value>& beta1_pow,
             const std::vector<pir::Value>& beta2_pow,
             const paddle::optional<std::vector<pir::Value>>& master_param,
             float beta1 = 0.9f,
             float beta2 = 0.999f,
             float epsilon = 1.0e-8f,
             bool multi_precision = false,
             bool use_global_beta_pow = false);

std::tuple<std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           paddle::optional<std::vector<pir::OpResult>>>
merged_adam_(const std::vector<pir::Value>& param,
             const std::vector<pir::Value>& grad,
             const std::vector<pir::Value>& learning_rate,
             const std::vector<pir::Value>& moment1,
             const std::vector<pir::Value>& moment2,
             const std::vector<pir::Value>& beta1_pow,
             const std::vector<pir::Value>& beta2_pow,
             const paddle::optional<std::vector<pir::Value>>& master_param,
             pir::Value beta1,
             pir::Value beta2,
             pir::Value epsilon,
             bool multi_precision = false,
             bool use_global_beta_pow = false);

std::tuple<std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           paddle::optional<std::vector<pir::OpResult>>>
merged_momentum_(const std::vector<pir::Value>& param,
                 const std::vector<pir::Value>& grad,
                 const std::vector<pir::Value>& velocity,
                 const std::vector<pir::Value>& learning_rate,
                 const paddle::optional<std::vector<pir::Value>>& master_param,
                 float mu,
                 bool use_nesterov = false,
                 const std::vector<std::string>& regularization_method = {},
                 const std::vector<float>& regularization_coeff = {},
                 bool multi_precision = false,
                 float rescale_grad = 1.0f);

std::vector<pir::OpResult> meshgrid(const std::vector<pir::Value>& inputs);

std::tuple<pir::OpResult, pir::OpResult> mode(const pir::Value& x,
                                              int axis = -1,
                                              bool keepdim = false);

std::tuple<pir::OpResult, pir::OpResult, paddle::optional<pir::OpResult>>
momentum_(const pir::Value& param,
          const pir::Value& grad,
          const pir::Value& velocity,
          const pir::Value& learning_rate,
          const paddle::optional<pir::Value>& master_param,
          float mu,
          bool use_nesterov = false,
          const std::string& regularization_method = "",
          float regularization_coeff = 0.0f,
          bool multi_precision = false,
          float rescale_grad = 1.0f);

pir::OpResult multi_dot(const std::vector<pir::Value>& x);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> multiclass_nms3(
    const pir::Value& bboxes,
    const pir::Value& scores,
    const paddle::optional<pir::Value>& rois_num,
    float score_threshold,
    int nms_top_k,
    int keep_top_k,
    float nms_threshold = 0.3,
    bool normalized = true,
    float nms_eta = 1.0,
    int background_label = 0);

pir::OpResult multinomial(const pir::Value& x,
                          int num_samples = 1,
                          bool replacement = false);

pir::OpResult multinomial(const pir::Value& x,
                          pir::Value num_samples,
                          bool replacement = false);

pir::OpResult multiplex(const std::vector<pir::Value>& inputs,
                        const pir::Value& index);

pir::OpResult mv(const pir::Value& x, const pir::Value& vec);

pir::OpResult nanmedian(const pir::Value& x,
                        const std::vector<int64_t>& axis = {},
                        bool keepdim = true);

pir::OpResult nearest_interp(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const std::string& data_layout = "NCHW",
    int out_d = 0,
    int out_h = 0,
    int out_w = 0,
    const std::vector<float>& scale = {},
    const std::string& interp_method = "bilinear",
    bool align_corners = true,
    int align_mode = 1);

pir::OpResult nextafter(const pir::Value& x, const pir::Value& y);

std::tuple<pir::OpResult, pir::OpResult> nll_loss(
    const pir::Value& input,
    const pir::Value& label,
    const paddle::optional<pir::Value>& weight,
    int64_t ignore_index = -100,
    const std::string& reduction = "mean");

pir::OpResult nms(const pir::Value& x, float threshold = 1.0f);

pir::OpResult nonzero(const pir::Value& condition);

pir::OpResult npu_identity(const pir::Value& x, int format = -1);

pir::OpResult numel(const pir::Value& x);

pir::OpResult overlap_add(const pir::Value& x, int hop_length, int axis = -1);

pir::OpResult p_norm(const pir::Value& x,
                     float porder = 2,
                     int axis = -1,
                     float epsilon = 1.0e-12f,
                     bool keepdim = false,
                     bool asvector = false);

pir::OpResult pad3d(const pir::Value& x,
                    const std::vector<int64_t>& paddings,
                    const std::string& mode = "constant",
                    float pad_value = 0.0,
                    const std::string& data_format = "NCDHW");

pir::OpResult pad3d(const pir::Value& x,
                    pir::Value paddings,
                    const std::string& mode = "constant",
                    float pad_value = 0.0,
                    const std::string& data_format = "NCDHW");

pir::OpResult pad3d(const pir::Value& x,
                    std::vector<pir::Value> paddings,
                    const std::string& mode = "constant",
                    float pad_value = 0.0,
                    const std::string& data_format = "NCDHW");

pir::OpResult pixel_shuffle(const pir::Value& x,
                            int upscale_factor = 1,
                            const std::string& data_format = "NCHW");

pir::OpResult pixel_unshuffle(const pir::Value& x,
                              int downscale_factor = 1,
                              const std::string& data_format = "NCHW");

pir::OpResult poisson(const pir::Value& x);

pir::OpResult polygamma(const pir::Value& x, int n);

pir::OpResult polygamma_(const pir::Value& x, int n);

pir::OpResult pow(const pir::Value& x, float y = 1.0f);

pir::OpResult pow_(const pir::Value& x, float y = 1.0f);

pir::OpResult prelu(const pir::Value& x,
                    const pir::Value& alpha,
                    const std::string& data_format = "NCHW",
                    const std::string& mode = "all");

std::tuple<pir::OpResult, pir::OpResult> prior_box(
    const pir::Value& input,
    const pir::Value& image,
    const std::vector<float>& min_sizes,
    const std::vector<float>& max_sizes = {},
    const std::vector<float>& aspect_ratios = {},
    const std::vector<float>& variances = {},
    bool flip = true,
    bool clip = true,
    float step_w = 0.0,
    float step_h = 0.0,
    float offset = 0.5,
    bool min_max_aspect_ratios_order = false);

pir::OpResult psroi_pool(const pir::Value& x,
                         const pir::Value& boxes,
                         const paddle::optional<pir::Value>& boxes_num,
                         int pooled_height = 1,
                         int pooled_width = 1,
                         int output_channels = 1,
                         float spatial_scale = 1.0);

pir::OpResult put_along_axis(const pir::Value& arr,
                             const pir::Value& indices,
                             const pir::Value& values,
                             int axis,
                             const std::string& reduce = "assign",
                             bool include_self = true);

pir::OpResult put_along_axis_(const pir::Value& arr,
                              const pir::Value& indices,
                              const pir::Value& values,
                              int axis,
                              const std::string& reduce = "assign",
                              bool include_self = true);

std::tuple<pir::OpResult, pir::OpResult> qr(
    const pir::Value& x, const std::string& mode = "reduced");

pir::OpResult real(const pir::Value& x);

pir::OpResult reciprocal(const pir::Value& x);

pir::OpResult reciprocal_(const pir::Value& x);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> reindex_graph(
    const pir::Value& x,
    const pir::Value& neighbors,
    const pir::Value& count,
    const paddle::optional<pir::Value>& hashtable_value,
    const paddle::optional<pir::Value>& hashtable_index);

pir::OpResult relu(const pir::Value& x);

pir::OpResult relu_(const pir::Value& x);

pir::OpResult relu6(const pir::Value& x);

pir::OpResult renorm(const pir::Value& x, float p, int axis, float max_norm);

pir::OpResult renorm_(const pir::Value& x, float p, int axis, float max_norm);

pir::OpResult reverse(const pir::Value& x, const std::vector<int64_t>& axis);

pir::OpResult reverse(const pir::Value& x, pir::Value axis);

pir::OpResult reverse(const pir::Value& x, std::vector<pir::Value> axis);

std::tuple<pir::OpResult, pir::OpResult> rms_norm(
    const pir::Value& x,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& residual,
    const pir::Value& norm_weight,
    const paddle::optional<pir::Value>& norm_bias,
    float epsilon,
    int begin_norm_axis,
    float quant_scale,
    int quant_round_type,
    float quant_max_bound,
    float quant_min_bound);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>,
           paddle::optional<pir::OpResult>>
rmsprop_(const pir::Value& param,
         const pir::Value& mean_square,
         const pir::Value& grad,
         const pir::Value& moment,
         const pir::Value& learning_rate,
         const paddle::optional<pir::Value>& mean_grad,
         const paddle::optional<pir::Value>& master_param,
         float epsilon = 1.0e-10f,
         float decay = 0.9f,
         float momentum = 0.0f,
         bool centered = false,
         bool multi_precision = false);

pir::OpResult roi_align(const pir::Value& x,
                        const pir::Value& boxes,
                        const paddle::optional<pir::Value>& boxes_num,
                        int pooled_height = 1,
                        int pooled_width = 1,
                        float spatial_scale = 1.0,
                        int sampling_ratio = -1,
                        bool aligned = false);

pir::OpResult roi_pool(const pir::Value& x,
                       const pir::Value& boxes,
                       const paddle::optional<pir::Value>& boxes_num,
                       int pooled_height = 1,
                       int pooled_width = 1,
                       float spatial_scale = 1.0);

pir::OpResult roll(const pir::Value& x,
                   const std::vector<int64_t>& shifts = {},
                   const std::vector<int64_t>& axis = {});

pir::OpResult roll(const pir::Value& x,
                   pir::Value shifts,
                   const std::vector<int64_t>& axis = {});

pir::OpResult roll(const pir::Value& x,
                   std::vector<pir::Value> shifts,
                   const std::vector<int64_t>& axis = {});

pir::OpResult round(const pir::Value& x);

pir::OpResult round_(const pir::Value& x);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
rprop_(const pir::Value& param,
       const pir::Value& grad,
       const pir::Value& prev,
       const pir::Value& learning_rate,
       const paddle::optional<pir::Value>& master_param,
       const pir::Value& learning_rate_range,
       const pir::Value& etas,
       bool multi_precision = false);

pir::OpResult rsqrt(const pir::Value& x);

pir::OpResult rsqrt_(const pir::Value& x);

pir::OpResult scale(const pir::Value& x,
                    float scale = 1.0,
                    float bias = 0.0,
                    bool bias_after_scale = true);

pir::OpResult scale(const pir::Value& x,
                    pir::Value scale,
                    float bias = 0.0,
                    bool bias_after_scale = true);

pir::OpResult scale_(const pir::Value& x,
                     float scale = 1.0,
                     float bias = 0.0,
                     bool bias_after_scale = true);

pir::OpResult scale_(const pir::Value& x,
                     pir::Value scale,
                     float bias = 0.0,
                     bool bias_after_scale = true);

pir::OpResult scatter(const pir::Value& x,
                      const pir::Value& index,
                      const pir::Value& updates,
                      bool overwrite = true);

pir::OpResult scatter_(const pir::Value& x,
                       const pir::Value& index,
                       const pir::Value& updates,
                       bool overwrite = true);

pir::OpResult scatter_nd_add(const pir::Value& x,
                             const pir::Value& index,
                             const pir::Value& updates);

pir::OpResult searchsorted(const pir::Value& sorted_sequence,
                           const pir::Value& values,
                           bool out_int32 = false,
                           bool right = false);

pir::OpResult segment_pool(const pir::Value& x,
                           const pir::Value& segment_ids,
                           const std::string& pooltype = "SUM");

pir::OpResult selu(const pir::Value& x,
                   float scale = 1.0507009873554804934193349852946,
                   float alpha = 1.6732632423543772848170429916717);

pir::OpResult send_u_recv(const pir::Value& x,
                          const pir::Value& src_index,
                          const pir::Value& dst_index,
                          const std::string& reduce_op = "SUM",
                          const std::vector<int64_t>& out_size = {0});

pir::OpResult send_u_recv(const pir::Value& x,
                          const pir::Value& src_index,
                          const pir::Value& dst_index,
                          pir::Value out_size,
                          const std::string& reduce_op = "SUM");

pir::OpResult send_u_recv(const pir::Value& x,
                          const pir::Value& src_index,
                          const pir::Value& dst_index,
                          std::vector<pir::Value> out_size,
                          const std::string& reduce_op = "SUM");

pir::OpResult send_ue_recv(const pir::Value& x,
                           const pir::Value& y,
                           const pir::Value& src_index,
                           const pir::Value& dst_index,
                           const std::string& message_op = "ADD",
                           const std::string& reduce_op = "SUM",
                           const std::vector<int64_t>& out_size = {0});

pir::OpResult send_ue_recv(const pir::Value& x,
                           const pir::Value& y,
                           const pir::Value& src_index,
                           const pir::Value& dst_index,
                           pir::Value out_size,
                           const std::string& message_op = "ADD",
                           const std::string& reduce_op = "SUM");

pir::OpResult send_ue_recv(const pir::Value& x,
                           const pir::Value& y,
                           const pir::Value& src_index,
                           const pir::Value& dst_index,
                           std::vector<pir::Value> out_size,
                           const std::string& message_op = "ADD",
                           const std::string& reduce_op = "SUM");

pir::OpResult send_uv(const pir::Value& x,
                      const pir::Value& y,
                      const pir::Value& src_index,
                      const pir::Value& dst_index,
                      const std::string& message_op = "ADD");

std::tuple<pir::OpResult, paddle::optional<pir::OpResult>> sgd_(
    const pir::Value& param,
    const pir::Value& learning_rate,
    const pir::Value& grad,
    const paddle::optional<pir::Value>& master_param,
    bool multi_precision = false);

pir::OpResult shape(const pir::Value& input);

pir::OpResult shard_index(const pir::Value& input,
                          int index_num,
                          int nshards,
                          int shard_id,
                          int ignore_value = -1);

pir::OpResult sigmoid(const pir::Value& x);

pir::OpResult sigmoid_(const pir::Value& x);

pir::OpResult sigmoid_cross_entropy_with_logits(
    const pir::Value& x,
    const pir::Value& label,
    const paddle::optional<pir::Value>& pos_weight,
    bool normalize = false,
    int ignore_index = -100);

pir::OpResult sigmoid_cross_entropy_with_logits_(
    const pir::Value& x,
    const pir::Value& label,
    const paddle::optional<pir::Value>& pos_weight,
    bool normalize = false,
    int ignore_index = -100);

pir::OpResult sign(const pir::Value& x);

pir::OpResult silu(const pir::Value& x);

pir::OpResult sin(const pir::Value& x);

pir::OpResult sin_(const pir::Value& x);

pir::OpResult sinh(const pir::Value& x);

pir::OpResult sinh_(const pir::Value& x);

pir::OpResult slogdet(const pir::Value& x);

pir::OpResult softplus(const pir::Value& x,
                       float beta = 1.0,
                       float threshold = 20.0f);

pir::OpResult softshrink(const pir::Value& x, float threshold = 0.5);

pir::OpResult softsign(const pir::Value& x);

pir::OpResult solve(const pir::Value& x, const pir::Value& y);

pir::OpResult spectral_norm(const pir::Value& weight,
                            const pir::Value& u,
                            const pir::Value& v,
                            int dim = 0,
                            int power_iters = 1,
                            float eps = 1e-12f);

pir::OpResult sqrt(const pir::Value& x);

pir::OpResult sqrt_(const pir::Value& x);

pir::OpResult square(const pir::Value& x);

pir::OpResult squared_l2_norm(const pir::Value& x);

pir::OpResult squeeze(const pir::Value& x,
                      const std::vector<int64_t>& axis = {});

pir::OpResult squeeze(const pir::Value& x, pir::Value axis);

pir::OpResult squeeze(const pir::Value& x, std::vector<pir::Value> axis);

pir::OpResult squeeze_(const pir::Value& x,
                       const std::vector<int64_t>& axis = {});

pir::OpResult squeeze_(const pir::Value& x, pir::Value axis);

pir::OpResult squeeze_(const pir::Value& x, std::vector<pir::Value> axis);

pir::OpResult stack(const std::vector<pir::Value>& x, int axis = 0);

pir::OpResult standard_gamma(const pir::Value& x);

pir::OpResult stanh(const pir::Value& x,
                    float scale_a = 0.67f,
                    float scale_b = 1.7159f);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> svd(
    const pir::Value& x, bool full_matrices = false);

pir::OpResult take_along_axis(const pir::Value& arr,
                              const pir::Value& indices,
                              int axis);

pir::OpResult tan(const pir::Value& x);

pir::OpResult tan_(const pir::Value& x);

pir::OpResult tanh(const pir::Value& x);

pir::OpResult tanh_(const pir::Value& x);

pir::OpResult tanh_shrink(const pir::Value& x);

pir::OpResult temporal_shift(const pir::Value& x,
                             int seg_num,
                             float shift_ratio = 0.25f,
                             const std::string& data_format = "NCHW");

pir::OpResult tensor_unfold(const pir::Value& input,
                            int64_t axis,
                            int64_t size,
                            int64_t step);

pir::OpResult thresholded_relu(const pir::Value& x, float threshold = 1.0);

pir::OpResult thresholded_relu_(const pir::Value& x, float threshold = 1.0);

std::tuple<pir::OpResult, pir::OpResult> top_p_sampling(
    const pir::Value& x,
    const pir::Value& ps,
    const paddle::optional<pir::Value>& threshold,
    int seed = -1);

std::tuple<pir::OpResult, pir::OpResult> topk(const pir::Value& x,
                                              int k = 1,
                                              int axis = -1,
                                              bool largest = true,
                                              bool sorted = true);

std::tuple<pir::OpResult, pir::OpResult> topk(const pir::Value& x,
                                              pir::Value k,
                                              int axis = -1,
                                              bool largest = true,
                                              bool sorted = true);

pir::OpResult trace(const pir::Value& x,
                    int offset = 0,
                    int axis1 = 0,
                    int axis2 = 1);

pir::OpResult triangular_solve(const pir::Value& x,
                               const pir::Value& y,
                               bool upper = true,
                               bool transpose = false,
                               bool unitriangular = false);

pir::OpResult trilinear_interp(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const std::string& data_layout = "NCHW",
    int out_d = 0,
    int out_h = 0,
    int out_w = 0,
    const std::vector<float>& scale = {},
    const std::string& interp_method = "bilinear",
    bool align_corners = true,
    int align_mode = 1);

pir::OpResult trunc(const pir::Value& input);

pir::OpResult trunc_(const pir::Value& input);

std::vector<pir::OpResult> unbind(const pir::Value& input, int axis = 0);

pir::OpResult unfold(const pir::Value& x,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations);

pir::OpResult uniform_inplace(const pir::Value& x,
                              float min = -1.0,
                              float max = 1.0,
                              int seed = 0,
                              int diag_num = 0,
                              int diag_step = 0,
                              float diag_val = 1.0);

pir::OpResult uniform_inplace_(const pir::Value& x,
                               float min = -1.0,
                               float max = 1.0,
                               int seed = 0,
                               int diag_num = 0,
                               int diag_step = 0,
                               float diag_val = 1.0);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> unique_consecutive(
    const pir::Value& x,
    bool return_inverse = false,
    bool return_counts = false,
    const std::vector<int>& axis = {},
    phi::DataType dtype = phi::DataType::FLOAT32);

pir::OpResult unpool3d(const pir::Value& x,
                       const pir::Value& indices,
                       const std::vector<int>& ksize,
                       const std::vector<int>& strides = {1, 1, 1},
                       const std::vector<int>& paddings = {0, 0, 0},
                       const std::vector<int>& output_size = {0, 0, 0},
                       const std::string& data_format = "NCDHW");

pir::OpResult unsqueeze(const pir::Value& x,
                        const std::vector<int64_t>& axis = {});

pir::OpResult unsqueeze(const pir::Value& x, pir::Value axis);

pir::OpResult unsqueeze(const pir::Value& x, std::vector<pir::Value> axis);

pir::OpResult unsqueeze_(const pir::Value& x,
                         const std::vector<int64_t>& axis = {});

pir::OpResult unsqueeze_(const pir::Value& x, pir::Value axis);

pir::OpResult unsqueeze_(const pir::Value& x, std::vector<pir::Value> axis);

std::vector<pir::OpResult> unstack(const pir::Value& x,
                                   int axis = 0,
                                   int num = 0);

std::tuple<std::vector<pir::OpResult>,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
update_loss_scaling_(const std::vector<pir::Value>& x,
                     const pir::Value& found_infinite,
                     const pir::Value& prev_loss_scaling,
                     const pir::Value& in_good_steps,
                     const pir::Value& in_bad_steps,
                     int incr_every_n_steps,
                     int decr_every_n_nan_or_inf,
                     float incr_ratio,
                     float decr_ratio,
                     bool stop_update = false);

std::tuple<std::vector<pir::OpResult>,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
update_loss_scaling_(const std::vector<pir::Value>& x,
                     const pir::Value& found_infinite,
                     const pir::Value& prev_loss_scaling,
                     const pir::Value& in_good_steps,
                     const pir::Value& in_bad_steps,
                     pir::Value stop_update,
                     int incr_every_n_steps,
                     int decr_every_n_nan_or_inf,
                     float incr_ratio,
                     float decr_ratio);

pir::OpResult view_dtype(const pir::Value& input, phi::DataType dtype);

pir::OpResult view_shape(const pir::Value& input,
                         const std::vector<int64_t>& dims = {});

std::tuple<pir::OpResult, pir::OpResult> viterbi_decode(
    const pir::Value& potentials,
    const pir::Value& transition_params,
    const pir::Value& lengths,
    bool include_bos_eos_tag = true);

pir::OpResult warpctc(const pir::Value& logits,
                      const pir::Value& label,
                      const paddle::optional<pir::Value>& logits_length,
                      const paddle::optional<pir::Value>& labels_length,
                      int blank = 0,
                      bool norm_by_times = false);

pir::OpResult warprnnt(const pir::Value& input,
                       const pir::Value& label,
                       const pir::Value& input_lengths,
                       const pir::Value& label_lengths,
                       int blank = 0,
                       float fastemit_lambda = 0.0);

pir::OpResult weight_dequantize(
    const pir::Value& x,
    const pir::Value& scale,
    const std::string& algo = "weight_only_int8",
    phi::DataType out_dtype = phi::DataType::FLOAT16,
    int group_size = -1);

pir::OpResult weight_only_linear(const pir::Value& x,
                                 const pir::Value& weight,
                                 const paddle::optional<pir::Value>& bias,
                                 const pir::Value& weight_scale,
                                 const std::string& weight_dtype,
                                 int arch = 80,
                                 int group_size = -1);

std::tuple<pir::OpResult, pir::OpResult> weight_quantize(
    const pir::Value& x,
    const std::string& algo = "weight_only_int8",
    int arch = 80,
    int group_size = -1);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
weighted_sample_neighbors(const pir::Value& row,
                          const pir::Value& colptr,
                          const pir::Value& edge_weight,
                          const pir::Value& input_nodes,
                          const paddle::optional<pir::Value>& eids,
                          int sample_size,
                          bool return_eids);

pir::OpResult where(const pir::Value& condition,
                    const pir::Value& x,
                    const pir::Value& y);

pir::OpResult where_(const pir::Value& condition,
                     const pir::Value& x,
                     const pir::Value& y);

std::tuple<pir::OpResult, pir::OpResult> yolo_box(
    const pir::Value& x,
    const pir::Value& img_size,
    const std::vector<int>& anchors = {},
    int class_num = 1,
    float conf_thresh = 0.01,
    int downsample_ratio = 32,
    bool clip_bbox = true,
    float scale_x_y = 1.0,
    bool iou_aware = false,
    float iou_aware_factor = 0.5);

pir::OpResult yolo_loss(const pir::Value& x,
                        const pir::Value& gt_box,
                        const pir::Value& gt_label,
                        const paddle::optional<pir::Value>& gt_score,
                        const std::vector<int>& anchors = {},
                        const std::vector<int>& anchor_mask = {},
                        int class_num = 1,
                        float ignore_thresh = 0.7,
                        int downsample_ratio = 32,
                        bool use_label_smooth = true,
                        float scale_x_y = 1.0);

pir::OpResult abs_double_grad(const pir::Value& x,
                              const pir::Value& grad_x_grad);

pir::OpResult abs_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult acos_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult acos_grad_(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult acosh_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult acosh_grad_(const pir::Value& x, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> addmm_grad(
    const pir::Value& input,
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out_grad,
    float alpha,
    float beta);

pir::OpResult affine_grid_grad(const pir::Value& input,
                               const pir::Value& output_grad,
                               const std::vector<int64_t>& output_shape,
                               bool align_corners = true);

pir::OpResult affine_grid_grad(const pir::Value& input,
                               const pir::Value& output_grad,
                               pir::Value output_shape,
                               bool align_corners = true);

pir::OpResult affine_grid_grad(const pir::Value& input,
                               const pir::Value& output_grad,
                               std::vector<pir::Value> output_shape,
                               bool align_corners = true);

pir::OpResult angle_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult argsort_grad(const pir::Value& indices,
                           const pir::Value& x,
                           const pir::Value& out_grad,
                           int axis,
                           bool descending);

pir::OpResult as_strided_grad(const pir::Value& input,
                              const pir::Value& out_grad,
                              const std::vector<int64_t>& dims = {},
                              const std::vector<int64_t>& stride = {},
                              int64_t offset = 0);

pir::OpResult asin_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult asin_grad_(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult asinh_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult asinh_grad_(const pir::Value& x, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> atan2_grad(const pir::Value& x,
                                                    const pir::Value& y,
                                                    const pir::Value& out_grad);

pir::OpResult atan_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult atan_grad_(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult atanh_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult atanh_grad_(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult bce_loss_grad(const pir::Value& input,
                            const pir::Value& label,
                            const pir::Value& out_grad);

pir::OpResult bce_loss_grad_(const pir::Value& input,
                             const pir::Value& label,
                             const pir::Value& out_grad);

pir::OpResult bicubic_interp_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const pir::Value& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
bilinear_grad(const pir::Value& x,
              const pir::Value& y,
              const pir::Value& weight,
              const pir::Value& out_grad);

pir::OpResult bilinear_interp_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const pir::Value& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode);

std::tuple<pir::OpResult, pir::OpResult> bmm_grad(const pir::Value& x,
                                                  const pir::Value& y,
                                                  const pir::Value& out_grad);

std::vector<pir::OpResult> broadcast_tensors_grad(
    const std::vector<pir::Value>& input,
    const std::vector<pir::Value>& out_grad);

pir::OpResult ceil_grad(const pir::Value& out_grad);

pir::OpResult ceil_grad_(const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> celu_double_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float alpha);

std::tuple<pir::OpResult, pir::OpResult> celu_double_grad_(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float alpha);

pir::OpResult celu_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        float alpha);

pir::OpResult celu_grad_(const pir::Value& x,
                         const pir::Value& out_grad,
                         float alpha);

pir::OpResult cholesky_grad(const pir::Value& out,
                            const pir::Value& out_grad,
                            bool upper);

std::tuple<pir::OpResult, pir::OpResult> cholesky_solve_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out,
    const pir::Value& out_grad,
    bool upper);

pir::OpResult clip_double_grad(const pir::Value& x,
                               const pir::Value& grad_x_grad,
                               float min = 0.,
                               float max = 0.);

pir::OpResult clip_double_grad(const pir::Value& x,
                               const pir::Value& grad_x_grad,
                               pir::Value min,
                               pir::Value max);

pir::OpResult clip_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        float min = 0.,
                        float max = 0.);

pir::OpResult clip_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        pir::Value min,
                        pir::Value max);

pir::OpResult clip_grad_(const pir::Value& x,
                         const pir::Value& out_grad,
                         float min = 0.,
                         float max = 0.);

pir::OpResult clip_grad_(const pir::Value& x,
                         const pir::Value& out_grad,
                         pir::Value min,
                         pir::Value max);

std::tuple<pir::OpResult, pir::OpResult> complex_grad(
    const pir::Value& real, const pir::Value& imag, const pir::Value& out_grad);

std::vector<pir::OpResult> concat_grad(const std::vector<pir::Value>& x,
                                       const pir::Value& out_grad,
                                       int axis = 0);

std::vector<pir::OpResult> concat_grad(const std::vector<pir::Value>& x,
                                       const pir::Value& out_grad,
                                       pir::Value axis);

std::tuple<pir::OpResult, pir::OpResult> conv2d_grad(
    const pir::Value& input,
    const pir::Value& filter,
    const pir::Value& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> conv2d_grad_grad(
    const pir::Value& input,
    const pir::Value& filter,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_input_grad,
    const paddle::optional<pir::Value>& grad_filter_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> conv3d_double_grad(
    const pir::Value& input,
    const pir::Value& filter,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_input_grad,
    const paddle::optional<pir::Value>& grad_filter_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult> conv3d_grad(
    const pir::Value& input,
    const pir::Value& filter,
    const pir::Value& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult> conv3d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::vector<int>& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult> cos_double_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out,
    const pir::Value& grad_x_grad);

std::tuple<pir::OpResult, pir::OpResult> cos_double_grad_(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out,
    const pir::Value& grad_x_grad);

pir::OpResult cos_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult cos_grad_(const pir::Value& x, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> cos_triple_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out_forward,
    const paddle::optional<pir::Value>& grad_x_grad_forward,
    const pir::Value& grad_x_grad,
    const paddle::optional<pir::Value>& grad_out_grad_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> cos_triple_grad_(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out_forward,
    const paddle::optional<pir::Value>& grad_x_grad_forward,
    const pir::Value& grad_x_grad,
    const paddle::optional<pir::Value>& grad_out_grad_grad);

pir::OpResult cosh_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult cosh_grad_(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult crop_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& offsets);

pir::OpResult crop_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        pir::Value offsets);

pir::OpResult crop_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        std::vector<pir::Value> offsets);

pir::OpResult cross_entropy_with_softmax_grad(const pir::Value& label,
                                              const pir::Value& softmax,
                                              const pir::Value& loss_grad,
                                              bool soft_label,
                                              bool use_softmax,
                                              bool numeric_stable_mode,
                                              int ignore_index,
                                              int axis);

pir::OpResult cross_entropy_with_softmax_grad_(const pir::Value& label,
                                               const pir::Value& softmax,
                                               const pir::Value& loss_grad,
                                               bool soft_label,
                                               bool use_softmax,
                                               bool numeric_stable_mode,
                                               int ignore_index,
                                               int axis);

std::tuple<pir::OpResult, pir::OpResult> cross_grad(const pir::Value& x,
                                                    const pir::Value& y,
                                                    const pir::Value& out_grad,
                                                    int axis);

pir::OpResult cummax_grad(const pir::Value& x,
                          const pir::Value& indices,
                          const pir::Value& out_grad,
                          int axis,
                          phi::DataType dtype);

pir::OpResult cummin_grad(const pir::Value& x,
                          const pir::Value& indices,
                          const pir::Value& out_grad,
                          int axis,
                          phi::DataType dtype);

pir::OpResult cumprod_grad(const pir::Value& x,
                           const pir::Value& out,
                           const pir::Value& out_grad,
                           int dim);

pir::OpResult cumsum_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          int axis,
                          bool flatten,
                          bool exclusive,
                          bool reverse);

pir::OpResult cumsum_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          pir::Value axis,
                          bool flatten,
                          bool exclusive,
                          bool reverse);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
depthwise_conv2d_double_grad(
    const pir::Value& input,
    const pir::Value& filter,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_input_grad,
    const paddle::optional<pir::Value>& grad_filter_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult> depthwise_conv2d_grad(
    const pir::Value& input,
    const pir::Value& filter,
    const pir::Value& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format);

pir::OpResult det_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad);

pir::OpResult diag_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        int offset);

pir::OpResult diagonal_grad(const pir::Value& x,
                            const pir::Value& out_grad,
                            int offset = 0,
                            int axis1 = 0,
                            int axis2 = 1);

pir::OpResult digamma_grad(const pir::Value& x, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> dist_grad(const pir::Value& x,
                                                   const pir::Value& y,
                                                   const pir::Value& out,
                                                   const pir::Value& out_grad,
                                                   float p);

std::tuple<pir::OpResult, pir::OpResult> dot_grad(const pir::Value& x,
                                                  const pir::Value& y,
                                                  const pir::Value& out_grad);

pir::OpResult eig_grad(const pir::Value& out_w,
                       const pir::Value& out_v,
                       const pir::Value& out_w_grad,
                       const pir::Value& out_v_grad);

pir::OpResult eigh_grad(const pir::Value& out_w,
                        const pir::Value& out_v,
                        const pir::Value& out_w_grad,
                        const pir::Value& out_v_grad);

pir::OpResult eigvalsh_grad(const pir::Value& eigenvectors,
                            const pir::Value& eigenvalues_grad,
                            const std::string& uplo,
                            bool is_test);

std::tuple<pir::OpResult, pir::OpResult> elu_double_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float alpha);

std::tuple<pir::OpResult, pir::OpResult> elu_double_grad_(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float alpha);

pir::OpResult elu_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       float alpha);

pir::OpResult elu_grad_(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad,
                        float alpha);

pir::OpResult erf_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult erfinv_grad(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult exp_grad(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult exp_grad_(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult expand_as_grad(const pir::Value& x,
                             const pir::Value& out_grad,
                             const std::vector<int>& target_shape);

pir::OpResult expand_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          const std::vector<int64_t>& shape);

pir::OpResult expand_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          pir::Value shape);

pir::OpResult expand_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          std::vector<pir::Value> shape);

pir::OpResult expm1_grad(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult expm1_grad_(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult fft_c2c_grad(const pir::Value& out_grad,
                           const std::vector<int64_t>& axes,
                           const std::string& normalization,
                           bool forward);

pir::OpResult fft_c2r_grad(const pir::Value& out_grad,
                           const std::vector<int64_t>& axes,
                           const std::string& normalization,
                           bool forward,
                           int64_t last_dim_size);

pir::OpResult fft_r2c_grad(const pir::Value& x,
                           const pir::Value& out_grad,
                           const std::vector<int64_t>& axes,
                           const std::string& normalization,
                           bool forward,
                           bool onesided);

pir::OpResult fill_diagonal_grad(const pir::Value& out_grad,
                                 float value,
                                 int offset,
                                 bool wrap);

pir::OpResult fill_diagonal_tensor_grad(const pir::Value& out_grad,
                                        int64_t offset,
                                        int dim1,
                                        int dim2);

pir::OpResult fill_diagonal_tensor_grad_(const pir::Value& out_grad,
                                         int64_t offset,
                                         int dim1,
                                         int dim2);

pir::OpResult fill_grad(const pir::Value& out_grad, float value);

pir::OpResult fill_grad(const pir::Value& out_grad, pir::Value value);

pir::OpResult fill_grad_(const pir::Value& out_grad, float value);

pir::OpResult fill_grad_(const pir::Value& out_grad, pir::Value value);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> flash_attn_grad(
    const pir::Value& q,
    const pir::Value& k,
    const pir::Value& v,
    const pir::Value& out,
    const pir::Value& softmax_lse,
    const pir::Value& seed_offset,
    const paddle::optional<pir::Value>& attn_mask,
    const pir::Value& out_grad,
    float dropout = 0.0,
    bool causal = false);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
flash_attn_unpadded_grad(const pir::Value& q,
                         const pir::Value& k,
                         const pir::Value& v,
                         const pir::Value& cu_seqlens_q,
                         const pir::Value& cu_seqlens_k,
                         const pir::Value& out,
                         const pir::Value& softmax_lse,
                         const pir::Value& seed_offset,
                         const paddle::optional<pir::Value>& attn_mask,
                         const pir::Value& out_grad,
                         int64_t max_seqlen_q,
                         int64_t max_seqlen_k,
                         float scale,
                         float dropout = 0.0,
                         bool causal = false);

pir::OpResult flatten_grad(const pir::Value& xshape,
                           const pir::Value& out_grad);

pir::OpResult flatten_grad_(const pir::Value& xshape,
                            const pir::Value& out_grad);

pir::OpResult floor_grad(const pir::Value& out_grad);

pir::OpResult floor_grad_(const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> fmax_grad(const pir::Value& x,
                                                   const pir::Value& y,
                                                   const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> fmin_grad(const pir::Value& x,
                                                   const pir::Value& y,
                                                   const pir::Value& out_grad);

pir::OpResult fold_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        const std::vector<int>& output_sizes,
                        const std::vector<int>& kernel_sizes,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::vector<int>& dilations);

pir::OpResult frame_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         int frame_length,
                         int hop_length,
                         int axis);

pir::OpResult gammaln_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult gather_grad(const pir::Value& x,
                          const pir::Value& index,
                          const pir::Value& out_grad,
                          int axis = 0);

pir::OpResult gather_grad(const pir::Value& x,
                          const pir::Value& index,
                          const pir::Value& out_grad,
                          pir::Value axis);

pir::OpResult gather_nd_grad(const pir::Value& x,
                             const pir::Value& index,
                             const pir::Value& out_grad);

pir::OpResult gaussian_inplace_grad(const pir::Value& out_grad,
                                    float mean = 0,
                                    float std = 1.0,
                                    int seed = 0);

pir::OpResult gaussian_inplace_grad_(const pir::Value& out_grad,
                                     float mean = 0,
                                     float std = 1.0,
                                     int seed = 0);

pir::OpResult gelu_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        bool approximate);

std::tuple<pir::OpResult, pir::OpResult> grid_sample_grad(
    const pir::Value& x,
    const pir::Value& grid,
    const pir::Value& out_grad,
    const std::string& mode,
    const std::string& padding_mode,
    bool align_corners);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> group_norm_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const paddle::optional<pir::Value>& bias,
    const pir::Value& y,
    const pir::Value& mean,
    const pir::Value& variance,
    const pir::Value& y_grad,
    float epsilon,
    int groups,
    const std::string& data_layout);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> group_norm_grad_(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const paddle::optional<pir::Value>& bias,
    const pir::Value& y,
    const pir::Value& mean,
    const pir::Value& variance,
    const pir::Value& y_grad,
    float epsilon,
    int groups,
    const std::string& data_layout);

pir::OpResult gumbel_softmax_grad(const pir::Value& out,
                                  const pir::Value& out_grad,
                                  int axis);

pir::OpResult hardshrink_grad(const pir::Value& x,
                              const pir::Value& out_grad,
                              float threshold);

pir::OpResult hardshrink_grad_(const pir::Value& x,
                               const pir::Value& out_grad,
                               float threshold);

pir::OpResult hardsigmoid_grad(const pir::Value& out,
                               const pir::Value& out_grad,
                               float slope,
                               float offset);

pir::OpResult hardsigmoid_grad_(const pir::Value& out,
                                const pir::Value& out_grad,
                                float slope,
                                float offset);

pir::OpResult hardtanh_grad(const pir::Value& x,
                            const pir::Value& out_grad,
                            float t_min,
                            float t_max);

pir::OpResult hardtanh_grad_(const pir::Value& x,
                             const pir::Value& out_grad,
                             float t_min,
                             float t_max);

std::tuple<pir::OpResult, pir::OpResult> heaviside_grad(
    const pir::Value& x, const pir::Value& y, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> huber_loss_grad(
    const pir::Value& residual, const pir::Value& out_grad, float delta);

pir::OpResult i0_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult i0e_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad);

pir::OpResult i1_grad(const pir::Value& x,
                      const pir::Value& out,
                      const pir::Value& out_grad);

pir::OpResult i1e_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad);

pir::OpResult identity_loss_grad(const pir::Value& x,
                                 const pir::Value& out_grad,
                                 int reduction);

pir::OpResult identity_loss_grad_(const pir::Value& x,
                                  const pir::Value& out_grad,
                                  int reduction);

pir::OpResult imag_grad(const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> index_add_grad(
    const pir::Value& index,
    const pir::Value& add_value,
    const pir::Value& out_grad,
    int axis);

std::tuple<pir::OpResult, pir::OpResult> index_add_grad_(
    const pir::Value& index,
    const pir::Value& add_value,
    const pir::Value& out_grad,
    int axis);

std::tuple<pir::OpResult, pir::OpResult> index_put_grad(
    const pir::Value& x,
    const std::vector<pir::Value>& indices,
    const pir::Value& value,
    const pir::Value& out_grad,
    bool accumulate = false);

pir::OpResult index_sample_grad(const pir::Value& x,
                                const pir::Value& index,
                                const pir::Value& out_grad);

pir::OpResult index_select_grad(const pir::Value& x,
                                const pir::Value& index,
                                const pir::Value& out_grad,
                                int axis);

pir::OpResult index_select_strided_grad(const pir::Value& x,
                                        const pir::Value& out_grad,
                                        int64_t index,
                                        int axis);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
instance_norm_double_grad(const pir::Value& x,
                          const paddle::optional<pir::Value>& fwd_scale,
                          const pir::Value& saved_mean,
                          const pir::Value& saved_variance,
                          const pir::Value& grad_y,
                          const paddle::optional<pir::Value>& grad_x_grad,
                          const paddle::optional<pir::Value>& grad_scale_grad,
                          const paddle::optional<pir::Value>& grad_bias_grad,
                          float epsilon);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> instance_norm_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const pir::Value& saved_mean,
    const pir::Value& saved_variance,
    const pir::Value& y_grad,
    float epsilon = 1e-5);

pir::OpResult inverse_grad(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult kldiv_loss_grad(const pir::Value& x,
                              const pir::Value& label,
                              const pir::Value& out_grad,
                              const std::string& reduction);

std::tuple<pir::OpResult, pir::OpResult> kron_grad(const pir::Value& x,
                                                   const pir::Value& y,
                                                   const pir::Value& out_grad);

pir::OpResult kthvalue_grad(const pir::Value& x,
                            const pir::Value& indices,
                            const pir::Value& out_grad,
                            int k,
                            int axis,
                            bool keepdim);

pir::OpResult label_smooth_grad(const pir::Value& out_grad, float epsilon);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> layer_norm_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const paddle::optional<pir::Value>& bias,
    const pir::Value& mean,
    const pir::Value& variance,
    const pir::Value& out_grad,
    float epsilon = 1e-5,
    int begin_norm_axis = 1);

pir::OpResult leaky_relu_double_grad(const pir::Value& x,
                                     const pir::Value& grad_x_grad,
                                     float negative_slope);

pir::OpResult leaky_relu_double_grad_(const pir::Value& x,
                                      const pir::Value& grad_x_grad,
                                      float negative_slope);

pir::OpResult leaky_relu_grad(const pir::Value& x,
                              const pir::Value& out_grad,
                              float negative_slope);

pir::OpResult leaky_relu_grad_(const pir::Value& x,
                               const pir::Value& out_grad,
                               float negative_slope);

std::tuple<pir::OpResult, pir::OpResult> lerp_grad(const pir::Value& x,
                                                   const pir::Value& y,
                                                   const pir::Value& weight,
                                                   const pir::Value& out,
                                                   const pir::Value& out_grad);

pir::OpResult lgamma_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult linear_interp_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const pir::Value& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode);

pir::OpResult log10_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult log10_grad_(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult log1p_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult log1p_grad_(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult log2_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult log2_grad_(const pir::Value& x, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> log_double_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad);

std::tuple<pir::OpResult, pir::OpResult> log_double_grad_(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad);

pir::OpResult log_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult log_grad_(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult log_loss_grad(const pir::Value& input,
                            const pir::Value& label,
                            const pir::Value& out_grad,
                            float epsilon);

pir::OpResult log_softmax_grad(const pir::Value& out,
                               const pir::Value& out_grad,
                               int axis);

pir::OpResult logcumsumexp_grad(const pir::Value& x,
                                const pir::Value& out,
                                const pir::Value& out_grad,
                                int axis,
                                bool flatten,
                                bool exclusive,
                                bool reverse);

pir::OpResult logit_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         float eps);

pir::OpResult logsigmoid_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult logsigmoid_grad_(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult lu_grad(const pir::Value& x,
                      const pir::Value& out,
                      const pir::Value& pivots,
                      const pir::Value& out_grad,
                      bool pivot);

pir::OpResult lu_grad_(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& pivots,
                       const pir::Value& out_grad,
                       bool pivot);

pir::OpResult lu_unpack_grad(const pir::Value& x,
                             const pir::Value& y,
                             const pir::Value& l,
                             const pir::Value& u,
                             const pir::Value& pmat,
                             const pir::Value& l_grad,
                             const pir::Value& u_grad,
                             bool unpack_ludata,
                             bool unpack_pivots);

pir::OpResult margin_cross_entropy_grad(const pir::Value& logits,
                                        const pir::Value& label,
                                        const pir::Value& softmax,
                                        const pir::Value& loss_grad,
                                        bool return_softmax,
                                        int ring_id,
                                        int rank,
                                        int nranks,
                                        float margin1,
                                        float margin2,
                                        float margin3,
                                        float scale);

pir::OpResult margin_cross_entropy_grad_(const pir::Value& logits,
                                         const pir::Value& label,
                                         const pir::Value& softmax,
                                         const pir::Value& loss_grad,
                                         bool return_softmax,
                                         int ring_id,
                                         int rank,
                                         int nranks,
                                         float margin1,
                                         float margin2,
                                         float margin3,
                                         float scale);

pir::OpResult masked_select_grad(const pir::Value& x,
                                 const pir::Value& mask,
                                 const pir::Value& out_grad);

pir::OpResult matrix_power_grad(const pir::Value& x,
                                const pir::Value& out,
                                const pir::Value& out_grad,
                                int n);

pir::OpResult max_pool2d_with_index_grad(const pir::Value& x,
                                         const pir::Value& mask,
                                         const pir::Value& out_grad,
                                         const std::vector<int>& kernel_size,
                                         const std::vector<int>& strides,
                                         const std::vector<int>& paddings,
                                         bool global_pooling,
                                         bool adaptive);

pir::OpResult max_pool3d_with_index_grad(const pir::Value& x,
                                         const pir::Value& mask,
                                         const pir::Value& out_grad,
                                         const std::vector<int>& kernel_size,
                                         const std::vector<int>& strides,
                                         const std::vector<int>& paddings,
                                         bool global_pooling,
                                         bool adaptive);

pir::OpResult maxout_grad(const pir::Value& x,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          int groups,
                          int axis);

pir::OpResult mean_all_grad(const pir::Value& x, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
memory_efficient_attention_grad(
    const pir::Value& query,
    const pir::Value& key,
    const pir::Value& value,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& cu_seqlens_q,
    const paddle::optional<pir::Value>& cu_seqlens_k,
    const pir::Value& output,
    const pir::Value& logsumexp,
    const pir::Value& seed_and_offset,
    const pir::Value& output_grad,
    float max_seqlen_q,
    float max_seqlen_k,
    bool causal,
    double dropout_p,
    float scale);

std::vector<pir::OpResult> meshgrid_grad(
    const std::vector<pir::Value>& inputs,
    const std::vector<pir::Value>& outputs_grad);

pir::OpResult mode_grad(const pir::Value& x,
                        const pir::Value& indices,
                        const pir::Value& out_grad,
                        int axis,
                        bool keepdim);

std::vector<pir::OpResult> multi_dot_grad(const std::vector<pir::Value>& x,
                                          const pir::Value& out_grad);

std::vector<pir::OpResult> multiplex_grad(const std::vector<pir::Value>& inputs,
                                          const pir::Value& index,
                                          const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> mv_grad(const pir::Value& x,
                                                 const pir::Value& vec,
                                                 const pir::Value& out_grad);

pir::OpResult nanmedian_grad(const pir::Value& x,
                             const pir::Value& medians,
                             const pir::Value& out_grad,
                             const std::vector<int64_t>& axis,
                             bool keepdim);

pir::OpResult nearest_interp_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const pir::Value& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode);

pir::OpResult nll_loss_grad(const pir::Value& input,
                            const pir::Value& label,
                            const paddle::optional<pir::Value>& weight,
                            const pir::Value& total_weight,
                            const pir::Value& out_grad,
                            int64_t ignore_index,
                            const std::string& reduction);

pir::OpResult overlap_add_grad(const pir::Value& x,
                               const pir::Value& out_grad,
                               int hop_length,
                               int axis);

pir::OpResult p_norm_grad(const pir::Value& x,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          float porder,
                          int axis,
                          float epsilon,
                          bool keepdim,
                          bool asvector);

pir::OpResult pad3d_double_grad(const pir::Value& grad_x_grad,
                                const std::vector<int64_t>& paddings,
                                const std::string& mode,
                                float pad_value,
                                const std::string& data_format);

pir::OpResult pad3d_double_grad(const pir::Value& grad_x_grad,
                                pir::Value paddings,
                                const std::string& mode,
                                float pad_value,
                                const std::string& data_format);

pir::OpResult pad3d_double_grad(const pir::Value& grad_x_grad,
                                std::vector<pir::Value> paddings,
                                const std::string& mode,
                                float pad_value,
                                const std::string& data_format);

pir::OpResult pad3d_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         const std::vector<int64_t>& paddings,
                         const std::string& mode,
                         float pad_value,
                         const std::string& data_format);

pir::OpResult pad3d_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         pir::Value paddings,
                         const std::string& mode,
                         float pad_value,
                         const std::string& data_format);

pir::OpResult pad3d_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         std::vector<pir::Value> paddings,
                         const std::string& mode,
                         float pad_value,
                         const std::string& data_format);

pir::OpResult pixel_shuffle_grad(const pir::Value& out_grad,
                                 int upscale_factor,
                                 const std::string& data_format);

pir::OpResult pixel_unshuffle_grad(const pir::Value& out_grad,
                                   int downscale_factor,
                                   const std::string& data_format);

pir::OpResult poisson_grad(const pir::Value& out_grad);

pir::OpResult polygamma_grad(const pir::Value& x,
                             const pir::Value& out_grad,
                             int n);

std::tuple<pir::OpResult, pir::OpResult> pow_double_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float y);

std::tuple<pir::OpResult, pir::OpResult> pow_double_grad_(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float y);

pir::OpResult pow_grad(const pir::Value& x,
                       const pir::Value& out_grad,
                       float y = -1);

pir::OpResult pow_grad_(const pir::Value& x,
                        const pir::Value& out_grad,
                        float y = -1);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> pow_triple_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_grad_x,
    const pir::Value& grad_x_grad,
    const paddle::optional<pir::Value>& grad_grad_out_grad,
    float y);

std::tuple<pir::OpResult, pir::OpResult> prelu_grad(
    const pir::Value& x,
    const pir::Value& alpha,
    const pir::Value& out_grad,
    const std::string& data_format,
    const std::string& mode);

pir::OpResult psroi_pool_grad(const pir::Value& x,
                              const pir::Value& boxes,
                              const paddle::optional<pir::Value>& boxes_num,
                              const pir::Value& out_grad,
                              int pooled_height,
                              int pooled_width,
                              int output_channels,
                              float spatial_scale);

std::tuple<pir::OpResult, pir::OpResult> put_along_axis_grad(
    const pir::Value& arr,
    const pir::Value& indices,
    const pir::Value& values,
    const pir::Value& out,
    const pir::Value& out_grad,
    int axis,
    const std::string& reduce,
    bool include_self);

pir::OpResult qr_grad(const pir::Value& x,
                      const pir::Value& q,
                      const pir::Value& r,
                      const pir::Value& q_grad,
                      const pir::Value& r_grad,
                      const std::string& mode);

pir::OpResult real_grad(const pir::Value& out_grad);

pir::OpResult reciprocal_grad(const pir::Value& out,
                              const pir::Value& out_grad);

pir::OpResult reciprocal_grad_(const pir::Value& out,
                               const pir::Value& out_grad);

pir::OpResult relu6_grad(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult relu6_grad_(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult relu_double_grad(const pir::Value& out,
                               const pir::Value& grad_x_grad);

pir::OpResult relu_double_grad_(const pir::Value& out,
                                const pir::Value& grad_x_grad);

pir::OpResult relu_grad(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult relu_grad_(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult renorm_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          float p,
                          int axis,
                          float max_norm);

pir::OpResult roi_align_grad(const pir::Value& x,
                             const pir::Value& boxes,
                             const paddle::optional<pir::Value>& boxes_num,
                             const pir::Value& out_grad,
                             int pooled_height,
                             int pooled_width,
                             float spatial_scale,
                             int sampling_ratio,
                             bool aligned);

pir::OpResult roi_pool_grad(const pir::Value& x,
                            const pir::Value& boxes,
                            const paddle::optional<pir::Value>& boxes_num,
                            const pir::Value& arg_max,
                            const pir::Value& out_grad,
                            int pooled_height,
                            int pooled_width,
                            float spatial_scale);

pir::OpResult roll_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& shifts,
                        const std::vector<int64_t>& axis);

pir::OpResult roll_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        pir::Value shifts,
                        const std::vector<int64_t>& axis);

pir::OpResult roll_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        std::vector<pir::Value> shifts,
                        const std::vector<int64_t>& axis);

pir::OpResult round_grad(const pir::Value& out_grad);

pir::OpResult round_grad_(const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> rsqrt_double_grad(
    const pir::Value& out,
    const pir::Value& grad_x,
    const pir::Value& grad_x_grad);

std::tuple<pir::OpResult, pir::OpResult> rsqrt_double_grad_(
    const pir::Value& out,
    const pir::Value& grad_x,
    const pir::Value& grad_x_grad);

pir::OpResult rsqrt_grad(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult rsqrt_grad_(const pir::Value& out, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> scatter_grad(
    const pir::Value& index,
    const pir::Value& updates,
    const pir::Value& out_grad,
    bool overwrite);

std::tuple<pir::OpResult, pir::OpResult> scatter_nd_add_grad(
    const pir::Value& index,
    const pir::Value& updates,
    const pir::Value& out_grad);

pir::OpResult segment_pool_grad(const pir::Value& x,
                                const pir::Value& segment_ids,
                                const pir::Value& out,
                                const paddle::optional<pir::Value>& summed_ids,
                                const pir::Value& out_grad,
                                const std::string& pooltype);

pir::OpResult selu_grad(const pir::Value& out,
                        const pir::Value& out_grad,
                        float scale,
                        float alpha);

pir::OpResult send_u_recv_grad(const pir::Value& x,
                               const pir::Value& src_index,
                               const pir::Value& dst_index,
                               const paddle::optional<pir::Value>& out,
                               const paddle::optional<pir::Value>& dst_count,
                               const pir::Value& out_grad,
                               const std::string& reduce_op = "SUM");

std::tuple<pir::OpResult, pir::OpResult> send_ue_recv_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& src_index,
    const pir::Value& dst_index,
    const paddle::optional<pir::Value>& out,
    const paddle::optional<pir::Value>& dst_count,
    const pir::Value& out_grad,
    const std::string& message_op,
    const std::string& reduce_op);

std::tuple<pir::OpResult, pir::OpResult> send_uv_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& src_index,
    const pir::Value& dst_index,
    const pir::Value& out_grad,
    const std::string& message_op = "ADD");

pir::OpResult sigmoid_cross_entropy_with_logits_grad(
    const pir::Value& x,
    const pir::Value& label,
    const paddle::optional<pir::Value>& pos_weight,
    const pir::Value& out_grad,
    bool normalize,
    int ignore_index);

pir::OpResult sigmoid_cross_entropy_with_logits_grad_(
    const pir::Value& x,
    const pir::Value& label,
    const paddle::optional<pir::Value>& pos_weight,
    const pir::Value& out_grad,
    bool normalize,
    int ignore_index);

std::tuple<pir::OpResult, pir::OpResult> sigmoid_double_grad(
    const pir::Value& out,
    const pir::Value& fwd_grad_out,
    const pir::Value& grad_x_grad);

std::tuple<pir::OpResult, pir::OpResult> sigmoid_double_grad_(
    const pir::Value& out,
    const pir::Value& fwd_grad_out,
    const pir::Value& grad_x_grad);

pir::OpResult sigmoid_grad(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult sigmoid_grad_(const pir::Value& out, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sigmoid_triple_grad(
    const pir::Value& out,
    const pir::Value& fwd_grad_out,
    const pir::Value& grad_grad_x,
    const pir::Value& grad_out_grad,
    const paddle::optional<pir::Value>& grad_grad_out_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sigmoid_triple_grad_(
    const pir::Value& out,
    const pir::Value& fwd_grad_out,
    const pir::Value& grad_grad_x,
    const pir::Value& grad_out_grad,
    const paddle::optional<pir::Value>& grad_grad_out_grad);

pir::OpResult silu_grad(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad);

pir::OpResult silu_grad_(const pir::Value& x,
                         const pir::Value& out,
                         const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> sin_double_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out,
    const pir::Value& grad_x_grad);

std::tuple<pir::OpResult, pir::OpResult> sin_double_grad_(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out,
    const pir::Value& grad_x_grad);

pir::OpResult sin_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult sin_grad_(const pir::Value& x, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sin_triple_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out_forward,
    const paddle::optional<pir::Value>& grad_x_grad_forward,
    const pir::Value& grad_x_grad,
    const paddle::optional<pir::Value>& grad_out_grad_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sin_triple_grad_(
    const pir::Value& x,
    const paddle::optional<pir::Value>& grad_out_forward,
    const paddle::optional<pir::Value>& grad_x_grad_forward,
    const pir::Value& grad_x_grad,
    const paddle::optional<pir::Value>& grad_out_grad_grad);

pir::OpResult sinh_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult sinh_grad_(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult slogdet_grad(const pir::Value& x,
                           const pir::Value& out,
                           const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> softplus_double_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float beta,
    float threshold);

std::tuple<pir::OpResult, pir::OpResult> softplus_double_grad_(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad,
    float beta,
    float threshold);

pir::OpResult softplus_grad(const pir::Value& x,
                            const pir::Value& out_grad,
                            float beta,
                            float threshold);

pir::OpResult softplus_grad_(const pir::Value& x,
                             const pir::Value& out_grad,
                             float beta,
                             float threshold);

pir::OpResult softshrink_grad(const pir::Value& x,
                              const pir::Value& out_grad,
                              float threshold);

pir::OpResult softshrink_grad_(const pir::Value& x,
                               const pir::Value& out_grad,
                               float threshold);

pir::OpResult softsign_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult softsign_grad_(const pir::Value& x, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> solve_grad(const pir::Value& x,
                                                    const pir::Value& y,
                                                    const pir::Value& out,
                                                    const pir::Value& out_grad);

pir::OpResult spectral_norm_grad(const pir::Value& weight,
                                 const pir::Value& u,
                                 const pir::Value& v,
                                 const pir::Value& out_grad,
                                 int dim,
                                 int power_iters,
                                 float eps);

std::tuple<pir::OpResult, pir::OpResult> sqrt_double_grad(
    const pir::Value& out,
    const pir::Value& grad_x,
    const pir::Value& grad_x_grad);

std::tuple<pir::OpResult, pir::OpResult> sqrt_double_grad_(
    const pir::Value& out,
    const pir::Value& grad_x,
    const pir::Value& grad_x_grad);

pir::OpResult sqrt_grad(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult sqrt_grad_(const pir::Value& out, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> square_double_grad(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad);

std::tuple<pir::OpResult, pir::OpResult> square_double_grad_(
    const pir::Value& x,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad);

pir::OpResult square_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult square_grad_(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult squared_l2_norm_grad(const pir::Value& x,
                                   const pir::Value& out_grad);

pir::OpResult squeeze_grad(const pir::Value& xshape,
                           const pir::Value& out_grad,
                           const std::vector<int64_t>& axis);

pir::OpResult squeeze_grad(const pir::Value& xshape,
                           const pir::Value& out_grad,
                           pir::Value axis);

pir::OpResult squeeze_grad(const pir::Value& xshape,
                           const pir::Value& out_grad,
                           std::vector<pir::Value> axis);

pir::OpResult squeeze_grad_(const pir::Value& xshape,
                            const pir::Value& out_grad,
                            const std::vector<int64_t>& axis);

pir::OpResult squeeze_grad_(const pir::Value& xshape,
                            const pir::Value& out_grad,
                            pir::Value axis);

pir::OpResult squeeze_grad_(const pir::Value& xshape,
                            const pir::Value& out_grad,
                            std::vector<pir::Value> axis);

std::vector<pir::OpResult> stack_grad(const std::vector<pir::Value>& x,
                                      const pir::Value& out_grad,
                                      int axis);

pir::OpResult stanh_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         float scale_a,
                         float scale_b);

pir::OpResult svd_grad(const pir::Value& x,
                       const pir::Value& u,
                       const pir::Value& vh,
                       const pir::Value& s,
                       const paddle::optional<pir::Value>& u_grad,
                       const paddle::optional<pir::Value>& vh_grad,
                       const paddle::optional<pir::Value>& s_grad,
                       bool full_matrices);

pir::OpResult take_along_axis_grad(const pir::Value& arr,
                                   const pir::Value& indices,
                                   const pir::Value& out_grad,
                                   int axis);

pir::OpResult tan_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult tan_grad_(const pir::Value& x, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> tanh_double_grad(
    const pir::Value& out,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad);

std::tuple<pir::OpResult, pir::OpResult> tanh_double_grad_(
    const pir::Value& out,
    const pir::Value& grad_out,
    const pir::Value& grad_x_grad);

pir::OpResult tanh_grad(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult tanh_grad_(const pir::Value& out, const pir::Value& out_grad);

pir::OpResult tanh_shrink_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult tanh_shrink_grad_(const pir::Value& x,
                                const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> tanh_triple_grad(
    const pir::Value& out,
    const pir::Value& grad_out_forward,
    const pir::Value& grad_x_grad_forward,
    const paddle::optional<pir::Value>& grad_out_new_grad,
    const paddle::optional<pir::Value>& grad_out_grad_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> tanh_triple_grad_(
    const pir::Value& out,
    const pir::Value& grad_out_forward,
    const pir::Value& grad_x_grad_forward,
    const paddle::optional<pir::Value>& grad_out_new_grad,
    const paddle::optional<pir::Value>& grad_out_grad_grad);

pir::OpResult temporal_shift_grad(const pir::Value& out_grad,
                                  int seg_num,
                                  float shift_ratio,
                                  const std::string& data_format);

pir::OpResult tensor_unfold_grad(const pir::Value& input,
                                 const pir::Value& out_grad,
                                 int64_t axis,
                                 int64_t size,
                                 int64_t step);

pir::OpResult thresholded_relu_grad(const pir::Value& x,
                                    const pir::Value& out_grad,
                                    float threshold);

pir::OpResult thresholded_relu_grad_(const pir::Value& x,
                                     const pir::Value& out_grad,
                                     float threshold);

pir::OpResult topk_grad(const pir::Value& x,
                        const pir::Value& indices,
                        const pir::Value& out_grad,
                        int k,
                        int axis,
                        bool largest,
                        bool sorted);

pir::OpResult topk_grad(const pir::Value& x,
                        const pir::Value& indices,
                        const pir::Value& out_grad,
                        pir::Value k,
                        int axis,
                        bool largest,
                        bool sorted);

pir::OpResult trace_grad(const pir::Value& x,
                         const pir::Value& out_grad,
                         int offset,
                         int axis1,
                         int axis2);

std::tuple<pir::OpResult, pir::OpResult> triangular_solve_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out,
    const pir::Value& out_grad,
    bool upper,
    bool transpose,
    bool unitriangular);

pir::OpResult trilinear_interp_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& out_size,
    const paddle::optional<std::vector<pir::Value>>& size_tensor,
    const paddle::optional<pir::Value>& scale_tensor,
    const pir::Value& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode);

pir::OpResult trunc_grad(const pir::Value& out_grad);

pir::OpResult unfold_grad(const pir::Value& x,
                          const pir::Value& out_grad,
                          const std::vector<int>& kernel_sizes,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          const std::vector<int>& dilations);

pir::OpResult uniform_inplace_grad(const pir::Value& out_grad,
                                   float min = -1.0,
                                   float max = 1.0,
                                   int seed = 0,
                                   int diag_num = 0,
                                   int diag_step = 0,
                                   float diag_val = 1.0);

pir::OpResult uniform_inplace_grad_(const pir::Value& out_grad,
                                    float min = -1.0,
                                    float max = 1.0,
                                    int seed = 0,
                                    int diag_num = 0,
                                    int diag_step = 0,
                                    float diag_val = 1.0);

pir::OpResult unsqueeze_grad(const pir::Value& xshape,
                             const pir::Value& out_grad,
                             const std::vector<int64_t>& axis);

pir::OpResult unsqueeze_grad(const pir::Value& xshape,
                             const pir::Value& out_grad,
                             pir::Value axis);

pir::OpResult unsqueeze_grad(const pir::Value& xshape,
                             const pir::Value& out_grad,
                             std::vector<pir::Value> axis);

pir::OpResult unsqueeze_grad_(const pir::Value& xshape,
                              const pir::Value& out_grad,
                              const std::vector<int64_t>& axis);

pir::OpResult unsqueeze_grad_(const pir::Value& xshape,
                              const pir::Value& out_grad,
                              pir::Value axis);

pir::OpResult unsqueeze_grad_(const pir::Value& xshape,
                              const pir::Value& out_grad,
                              std::vector<pir::Value> axis);

pir::OpResult unstack_grad(const std::vector<pir::Value>& out_grad, int axis);

pir::OpResult view_dtype_grad(const pir::Value& input,
                              const pir::Value& out_grad,
                              phi::DataType dtype);

pir::OpResult view_shape_grad(const pir::Value& input,
                              const pir::Value& out_grad,
                              const std::vector<int64_t>& dims = {});

pir::OpResult warpctc_grad(const pir::Value& logits,
                           const paddle::optional<pir::Value>& logits_length,
                           const pir::Value& warpctcgrad,
                           const pir::Value& loss_grad,
                           int blank,
                           bool norm_by_times);

pir::OpResult warprnnt_grad(const pir::Value& input,
                            const pir::Value& input_lengths,
                            const pir::Value& warprnntgrad,
                            const pir::Value& loss_grad,
                            int blank = 0,
                            float fastemit_lambda = 0.0);

pir::OpResult weight_only_linear_grad(const pir::Value& x,
                                      const pir::Value& weight,
                                      const paddle::optional<pir::Value>& bias,
                                      const pir::Value& weight_scale,
                                      const pir::Value& out_grad,
                                      const std::string& weight_dtype,
                                      int arch,
                                      int group_size);

std::tuple<pir::OpResult, pir::OpResult> where_grad(const pir::Value& condition,
                                                    const pir::Value& x,
                                                    const pir::Value& y,
                                                    const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
yolo_loss_grad(const pir::Value& x,
               const pir::Value& gt_box,
               const pir::Value& gt_label,
               const paddle::optional<pir::Value>& gt_score,
               const pir::Value& objectness_mask,
               const pir::Value& gt_match_mask,
               const pir::Value& loss_grad,
               const std::vector<int>& anchors,
               const std::vector<int>& anchor_mask,
               int class_num,
               float ignore_thresh,
               int downsample_ratio,
               bool use_label_smooth,
               float scale_x_y);

pir::OpResult unpool3d_grad(const pir::Value& x,
                            const pir::Value& indices,
                            const pir::Value& out,
                            const pir::Value& out_grad,
                            const std::vector<int>& ksize,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            const std::vector<int>& output_size,
                            const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult> add_act_xpu(
    const pir::Value& x,
    const paddle::optional<pir::Value>& x_max,
    const pir::Value& y,
    const paddle::optional<pir::Value>& y_max,
    int act_type);

pir::OpResult add_layernorm_xpu(const pir::Value& x,
                                const pir::Value& y,
                                const pir::Value& scale,
                                const pir::Value& bias,
                                int begin_norm_axis,
                                float epsilon);

pir::OpResult addcmul_xpu(const pir::Value& x,
                          const pir::Value& y,
                          const pir::Value& w);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
block_multihead_attention_(
    const pir::Value& qkv,
    const pir::Value& key_cache,
    const pir::Value& value_cache,
    const pir::Value& seq_lens_encoder,
    const pir::Value& seq_lens_decoder,
    const pir::Value& seq_lens_this_time,
    const pir::Value& padding_offsets,
    const pir::Value& cum_offsets,
    const pir::Value& cu_seqlens_q,
    const pir::Value& cu_seqlens_k,
    const pir::Value& block_tables,
    const paddle::optional<pir::Value>& pre_key_cache,
    const paddle::optional<pir::Value>& pre_value_cache,
    const paddle::optional<pir::Value>& rope_emb,
    const paddle::optional<pir::Value>& mask,
    const paddle::optional<pir::Value>& tgt_mask,
    const paddle::optional<pir::Value>& cache_k_quant_scales,
    const paddle::optional<pir::Value>& cache_v_quant_scales,
    const paddle::optional<pir::Value>& cache_k_dequant_scales,
    const paddle::optional<pir::Value>& cache_v_dequant_scales,
    const paddle::optional<pir::Value>& qkv_out_scale,
    const paddle::optional<pir::Value>& qkv_bias,
    const paddle::optional<pir::Value>& out_shift,
    const paddle::optional<pir::Value>& out_smooth,
    int max_seq_len,
    int block_size,
    bool use_neox_style,
    bool dynamic_cachekv_quant = false,
    int quant_round_type = 1,
    float quant_max_bound = 127.0,
    float quant_min_bound = -127.0,
    float out_scale = -1,
    const std::string& compute_dtype = "default");

pir::OpResult bn_act_xpu(const pir::Value& x,
                         const pir::Value& mean,
                         const pir::Value& variance,
                         const pir::Value& scale,
                         const pir::Value& bias,
                         float momentum,
                         float epsilon,
                         const std::string& data_layout,
                         int act_type);

std::tuple<pir::OpResult, pir::OpResult> conv1d_xpu(
    const pir::Value& x,
    const paddle::optional<pir::Value>& x_max,
    const pir::Value& filter,
    const pir::Value& filter_max,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& branch,
    const paddle::optional<pir::Value>& branch_max,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    int dilations,
    int strides,
    int groups,
    int act_type,
    float act_param);

std::tuple<pir::OpResult, pir::OpResult> conv2d_transpose_xpu(
    const pir::Value& x,
    const paddle::optional<pir::Value>& x_max,
    const pir::Value& filter,
    const pir::Value& filter_max,
    const paddle::optional<pir::Value>& bias,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::vector<int64_t>& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format,
    bool has_bias,
    bool with_act,
    const std::string& act_type);

std::tuple<pir::OpResult, pir::OpResult> conv2d_xpu(
    const pir::Value& x,
    const paddle::optional<pir::Value>& x_max,
    const pir::Value& filter,
    const pir::Value& filter_max,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& branch,
    const paddle::optional<pir::Value>& branch_max,
    const paddle::optional<pir::Value>& scale_max,
    const paddle::optional<pir::Value>& out_max_in,
    const std::vector<int>& paddings,
    const std::vector<int>& dilations,
    const std::vector<int>& strides,
    const std::string& padding_algorithm,
    int groups,
    int act_type,
    float act_param,
    phi::DataType out_dtype);

pir::OpResult dequantize_xpu(const pir::Value& x,
                             phi::DataType out_dtype,
                             float scale = 1.0f);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
embedding_with_eltwise_add_xpu(const std::vector<pir::Value>& ids,
                               const std::vector<pir::Value>& tables,
                               const paddle::optional<pir::Value>& mask,
                               int64_t padding_idx);

pir::OpResult fast_layernorm_xpu(const pir::Value& x,
                                 const pir::Value& scale,
                                 const pir::Value& bias,
                                 int begin_norm_axis,
                                 float epsilon);

pir::OpResult fast_where_xpu(const pir::Value& condition,
                             const pir::Value& x,
                             const pir::Value& y);

pir::OpResult fc(const pir::Value& input,
                 const pir::Value& w,
                 const paddle::optional<pir::Value>& bias,
                 int in_num_col_dims = 1,
                 const std::string& activation_type = "",
                 bool padding_weights = false);

std::tuple<pir::OpResult, pir::OpResult> fc_xpu(
    const pir::Value& x,
    const paddle::optional<pir::Value>& x_max,
    const pir::Value& w,
    const pir::Value& w_max,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& scale_max,
    const paddle::optional<pir::Value>& out_max_in,
    int in_num_col_dims,
    bool transpose_x,
    float alpha,
    float beta,
    int act_type,
    float act_alpha,
    phi::DataType out_dtype);

pir::OpResult fused_bias_act(const pir::Value& x,
                             const paddle::optional<pir::Value>& bias,
                             const paddle::optional<pir::Value>& dequant_scales,
                             const paddle::optional<pir::Value>& shift,
                             const paddle::optional<pir::Value>& smooth,
                             const std::string& act_method = "gelu",
                             const std::string& compute_dtype = "default",
                             float quant_scale = -1,
                             int quant_round_type = 1,
                             float quant_max_bound = 127.0,
                             float quant_min_bound = -127.0);

pir::OpResult fused_bias_dropout_residual_layer_norm(
    const pir::Value& x,
    const pir::Value& residual,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& ln_scale,
    const paddle::optional<pir::Value>& ln_bias,
    float dropout_rate = 0.5f,
    bool is_test = false,
    bool dropout_fix_seed = true,
    int dropout_seed = true,
    const std::string& dropout_implementation = "downgrade_in_infer",
    float ln_epsilon = 1e-5);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
fused_bias_residual_layernorm(const pir::Value& x,
                              const paddle::optional<pir::Value>& bias,
                              const paddle::optional<pir::Value>& residual,
                              const paddle::optional<pir::Value>& norm_weight,
                              const paddle::optional<pir::Value>& norm_bias,
                              float epsilon,
                              float residual_alpha,
                              int begin_norm_axis,
                              float quant_scale,
                              int quant_round_type,
                              float quant_max_bound,
                              float quant_min_bound);

std::tuple<pir::OpResult, std::vector<pir::OpResult>> fused_conv2d_add_act(
    const pir::Value& input,
    const pir::Value& filter,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& residual_data,
    const std::vector<int>& strides = {1, 1},
    const std::vector<int>& paddings = {0, 0},
    const std::string& padding_algorithm = "EXPLICIT",
    const std::vector<int>& dilations = {1, 1},
    int groups = 1,
    const std::string& data_format = "NCHW",
    const std::string& activation = "relu",
    const std::vector<int>& split_channels = {},
    bool exhaustive_search = false,
    int workspace_size_MB = 32,
    float fuse_alpha = 0.0f);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_dconv_drelu_dbn(const pir::Value& grad_output,
                      const pir::Value& weight,
                      const paddle::optional<pir::Value>& grad_output_add,
                      const paddle::optional<pir::Value>& residual_input,
                      const paddle::optional<pir::Value>& bn1_eqscale,
                      const paddle::optional<pir::Value>& bn1_eqbias,
                      const paddle::optional<pir::Value>& conv_input,
                      const pir::Value& bn1_mean,
                      const pir::Value& bn1_inv_std,
                      const pir::Value& bn1_gamma,
                      const pir::Value& bn1_beta,
                      const pir::Value& bn1_input,
                      const paddle::optional<pir::Value>& bn2_mean,
                      const paddle::optional<pir::Value>& bn2_inv_std,
                      const paddle::optional<pir::Value>& bn2_gamma,
                      const paddle::optional<pir::Value>& bn2_beta,
                      const paddle::optional<pir::Value>& bn2_input,
                      const std::vector<int>& paddings,
                      const std::vector<int>& dilations,
                      const std::vector<int>& strides,
                      const std::string& padding_algorithm,
                      int groups,
                      const std::string& data_format,
                      bool fuse_shortcut,
                      bool fuse_dual,
                      bool fuse_add,
                      bool exhaustive_search);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
fused_dot_product_attention(const pir::Value& q,
                            const pir::Value& k,
                            const pir::Value& v,
                            const pir::Value& mask,
                            float scaling_factor,
                            float dropout_probability,
                            bool is_training = false,
                            bool is_causal_masking = false);

std::tuple<pir::OpResult, pir::OpResult> fused_dropout_add(
    const pir::Value& x,
    const pir::Value& y,
    const paddle::optional<pir::Value>& seed_tensor,
    float p,
    bool is_test,
    const std::string& mode,
    int seed = 0,
    bool fix_seed = false);

pir::OpResult fused_embedding_eltwise_layernorm(
    const std::vector<pir::Value>& ids,
    const std::vector<pir::Value>& embs,
    const pir::Value& bias,
    const pir::Value& scale,
    float epsilon = 0.00001f);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
fused_fc_elementwise_layernorm(const pir::Value& x,
                               const pir::Value& w,
                               const pir::Value& y,
                               const paddle::optional<pir::Value>& bias0,
                               const paddle::optional<pir::Value>& scale,
                               const paddle::optional<pir::Value>& bias1,
                               int x_num_col_dims = 1,
                               const std::string& activation_type = "",
                               float epsilon = 0.00001f,
                               int begin_norm_axis = 1);

std::tuple<pir::OpResult, pir::OpResult> fused_linear_param_grad_add(
    const pir::Value& x,
    const pir::Value& dout,
    const paddle::optional<pir::Value>& dweight,
    const paddle::optional<pir::Value>& dbias,
    bool multi_precision = true,
    bool has_bias = true);

std::tuple<pir::OpResult, std::vector<pir::OpResult>>
fused_multi_transformer_int8_xpu(
    const pir::Value& x,
    const std::vector<pir::Value>& ln_scale,
    const std::vector<pir::Value>& ln_bias,
    const std::vector<pir::Value>& qkv_in_max,
    const std::vector<pir::Value>& qkvw,
    const std::vector<pir::Value>& qkv_bias,
    const std::vector<pir::Value>& qkv_scales,
    const std::vector<pir::Value>& out_linear_in_max,
    const std::vector<pir::Value>& out_linear_w,
    const std::vector<pir::Value>& out_linear_bias,
    const std::vector<pir::Value>& out_linear_scales,
    const std::vector<pir::Value>& ffn_ln_scale,
    const std::vector<pir::Value>& ffn_ln_bias,
    const std::vector<pir::Value>& ffn1_in_max,
    const std::vector<pir::Value>& ffn1_weight,
    const std::vector<pir::Value>& ffn1_bias,
    const std::vector<pir::Value>& ffn1_scales,
    const std::vector<pir::Value>& ffn2_in_max,
    const std::vector<pir::Value>& ffn2_weight,
    const std::vector<pir::Value>& ffn2_bias,
    const std::vector<pir::Value>& ffn2_scales,
    const paddle::optional<std::vector<pir::Value>>& cache_kv,
    const paddle::optional<std::vector<pir::Value>>& pre_caches,
    const paddle::optional<pir::Value>& rotary_pos_emb,
    const paddle::optional<pir::Value>& time_step,
    const paddle::optional<pir::Value>& seq_lengths,
    const paddle::optional<pir::Value>& src_mask,
    const paddle::optional<pir::Value>& gather_index,
    const pir::Value& max_buffer,
    bool pre_layer_norm,
    int rotary_emb_dims,
    float epsilon,
    float dropout_rate,
    bool is_test,
    const std::string& dropout_implementation,
    const std::string& act_method,
    bool trans_qkvw,
    int ring_id,
    int gather_axis);

std::tuple<pir::OpResult, std::vector<pir::OpResult>>
fused_multi_transformer_xpu(
    const pir::Value& x,
    const std::vector<pir::Value>& ln_scale,
    const std::vector<pir::Value>& ln_bias,
    const std::vector<pir::Value>& qkvw,
    const std::vector<pir::Value>& qkvw_max,
    const std::vector<pir::Value>& qkv_bias,
    const std::vector<pir::Value>& out_linear_w,
    const std::vector<pir::Value>& out_linear_wmax,
    const std::vector<pir::Value>& out_linear_bias,
    const std::vector<pir::Value>& ffn_ln_scale,
    const std::vector<pir::Value>& ffn_ln_bias,
    const std::vector<pir::Value>& ffn1_weight,
    const std::vector<pir::Value>& ffn1_weight_max,
    const std::vector<pir::Value>& ffn1_bias,
    const std::vector<pir::Value>& ffn2_weight,
    const std::vector<pir::Value>& ffn2_weight_max,
    const std::vector<pir::Value>& ffn2_bias,
    const paddle::optional<std::vector<pir::Value>>& cache_kv,
    const paddle::optional<std::vector<pir::Value>>& pre_caches,
    const paddle::optional<pir::Value>& rotary_pos_emb,
    const paddle::optional<pir::Value>& time_step,
    const paddle::optional<pir::Value>& seq_lengths,
    const paddle::optional<pir::Value>& src_mask,
    const paddle::optional<pir::Value>& gather_index,
    const pir::Value& max_buffer,
    bool pre_layer_norm,
    int rotary_emb_dims,
    float epsilon,
    float dropout_rate,
    bool is_test,
    const std::string& dropout_implementation,
    const std::string& act_method,
    bool trans_qkvw,
    int ring_id,
    int gather_axis);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
fused_rotary_position_embedding(
    const pir::Value& q,
    const paddle::optional<pir::Value>& k,
    const paddle::optional<pir::Value>& v,
    const paddle::optional<pir::Value>& sin,
    const paddle::optional<pir::Value>& cos,
    const paddle::optional<pir::Value>& position_ids,
    bool use_neox_rotary_style = true);

pir::OpResult fused_scale_bias_add_relu(
    const pir::Value& x1,
    const pir::Value& scale1,
    const pir::Value& bias1,
    const pir::Value& x2,
    const paddle::optional<pir::Value>& scale2,
    const paddle::optional<pir::Value>& bias2,
    bool fuse_dual,
    bool exhaustive_search);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_scale_bias_relu_conv_bn(const pir::Value& x,
                              const pir::Value& w,
                              const paddle::optional<pir::Value>& scale,
                              const paddle::optional<pir::Value>& bias,
                              const pir::Value& bn_scale,
                              const pir::Value& bn_bias,
                              const pir::Value& input_running_mean,
                              const pir::Value& input_running_var,
                              const std::vector<int>& paddings,
                              const std::vector<int>& dilations,
                              const std::vector<int>& strides,
                              const std::string& padding_algorithm,
                              int groups,
                              const std::string& data_format,
                              float momentum,
                              float epsilon,
                              bool fuse_prologue,
                              bool exhaustive_search,
                              int64_t accumulation_count = 0);

pir::OpResult fusion_gru(const pir::Value& x,
                         const paddle::optional<pir::Value>& h0,
                         const pir::Value& weight_x,
                         const pir::Value& weight_h,
                         const paddle::optional<pir::Value>& bias,
                         const std::string& activation = "tanh",
                         const std::string& gate_activation = "sigmoid",
                         bool is_reverse = false,
                         bool use_seq = true,
                         bool origin_mode = false,
                         bool use_mkldnn = false,
                         const std::string& mkldnn_data_type = "float32",
                         float scale_data = 1.0f,
                         float shift_data = 0.0f,
                         const std::vector<float>& scale_weights = {1.0f},
                         bool force_fp32_output = false);

pir::OpResult fusion_repeated_fc_relu(const pir::Value& x,
                                      const std::vector<pir::Value>& w,
                                      const std::vector<pir::Value>& bias);

pir::OpResult fusion_seqconv_eltadd_relu(const pir::Value& x,
                                         const pir::Value& filter,
                                         const pir::Value& bias,
                                         int context_length,
                                         int context_start = 0,
                                         int context_stride = 1);

pir::OpResult fusion_seqexpand_concat_fc(
    const std::vector<pir::Value>& x,
    const pir::Value& fc_weight,
    const paddle::optional<pir::Value>& fc_bias,
    const std::string& fc_activation = "identity");

pir::OpResult fusion_squared_mat_sub(const pir::Value& x,
                                     const pir::Value& y,
                                     float scalar = 1.0f);

pir::OpResult fusion_transpose_flatten_concat(
    const std::vector<pir::Value>& x,
    const std::vector<int>& trans_axis,
    int flatten_axis,
    int concat_axis);

pir::OpResult generate_sequence_xpu(const pir::Value& x, phi::DataType dtype);

pir::OpResult layer_norm_act_xpu(const pir::Value& x,
                                 const pir::Value& scale,
                                 const pir::Value& bias,
                                 int begin_norm_axis,
                                 float epsilon,
                                 int act_type,
                                 float act_param);

pir::OpResult max_pool2d_v2(const pir::Value& x,
                            const std::vector<int>& kernel_size,
                            const std::vector<int>& strides = {1, 1},
                            const std::vector<int>& paddings = {0, 0},
                            const std::string& data_format = "NCHW",
                            bool global_pooling = false,
                            bool adaptive = false);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> multi_encoder_xpu(
    const pir::Value& x,
    const std::vector<pir::Value>& fc_weight,
    const std::vector<pir::Value>& fc_weight_max,
    const std::vector<pir::Value>& fc_bias,
    const std::vector<pir::Value>& ln_scale,
    const std::vector<pir::Value>& ln_bias,
    const paddle::optional<pir::Value>& mask,
    const paddle::optional<pir::Value>& seq_lod,
    const paddle::optional<pir::Value>& max_seq_len,
    int layer_num,
    bool norm_before,
    int hidden_dim,
    int head_num,
    int size_per_head,
    int ffn_hidden_dim_scale,
    int act_type,
    int relative_type,
    int slice_idx);

pir::OpResult multihead_matmul(const pir::Value& input,
                               const pir::Value& w,
                               const pir::Value& bias,
                               const paddle::optional<pir::Value>& bias_qk,
                               bool transpose_q = false,
                               bool transpose_k = true,
                               bool transpose_v = false,
                               float alpha = 1.0f,
                               int head_number = 1);

std::tuple<pir::OpResult, pir::OpResult> qkv_attention_xpu(
    const pir::Value& q,
    const pir::Value& k,
    const pir::Value& v,
    const paddle::optional<pir::Value>& q_max,
    const paddle::optional<pir::Value>& k_max,
    const paddle::optional<pir::Value>& v_max,
    float alpha,
    int head_num,
    int head_dim,
    bool qkv_fc_fusion,
    phi::DataType out_dtype);

pir::OpResult quantize_xpu(const pir::Value& x,
                           phi::DataType out_dtype,
                           float scale = 1.0f);

pir::OpResult self_dp_attention(const pir::Value& x,
                                float alpha = 1.0f,
                                int head_number = 1);

pir::OpResult sine_pos_xpu(const pir::Value& x, const pir::Value& y);

pir::OpResult skip_layernorm(const pir::Value& x,
                             const pir::Value& y,
                             const pir::Value& scale,
                             const pir::Value& bias,
                             float epsilon,
                             int begin_norm_axis);

pir::OpResult squeeze_excitation_block(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& filter_max,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& branch,
    const std::vector<int>& act_type,
    const std::vector<float>& act_param,
    const std::vector<int>& filter_dims);

pir::OpResult variable_length_memory_efficient_attention(
    const pir::Value& query,
    const pir::Value& key,
    const pir::Value& value,
    const pir::Value& seq_lens,
    const pir::Value& kv_seq_lens,
    const paddle::optional<pir::Value>& mask,
    float scale,
    bool causal,
    int pre_cache_length);

std::tuple<pir::OpResult, pir::OpResult> yolo_box_xpu(
    const pir::Value& x,
    const paddle::optional<pir::Value>& x_max,
    const pir::Value& grid,
    const pir::Value& stride,
    const pir::Value& anchor_grid,
    float offset);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_bias_dropout_residual_layer_norm_grad(
    const pir::Value& y_grad,
    const pir::Value& x,
    const pir::Value& residual,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& ln_scale,
    const paddle::optional<pir::Value>& ln_bias,
    const pir::Value& ln_mean,
    const pir::Value& ln_variance,
    const pir::Value& bias_dropout_residual_out,
    const pir::Value& dropout_mask_out,
    float dropout_rate = 0.5f,
    bool is_test = false,
    bool dropout_fix_seed = true,
    int dropout_seed = true,
    const std::string& dropout_implementation = "downgrade_in_infer",
    float ln_epsilon = 1e-5);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
fused_dot_product_attention_grad(const pir::Value& q,
                                 const pir::Value& k,
                                 const pir::Value& v,
                                 const pir::Value& out,
                                 const pir::Value& softmax_out,
                                 const pir::Value& rng_state,
                                 const pir::Value& mask,
                                 const pir::Value& out_grad,
                                 float scaling_factor,
                                 float dropout_probability,
                                 bool is_causal_masking = false);

std::tuple<pir::OpResult, pir::OpResult> fused_dropout_add_grad(
    const pir::Value& seed_offset,
    const pir::Value& out_grad,
    float p,
    bool is_test,
    const std::string& mode,
    bool fix_seed);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
fused_rotary_position_embedding_grad(
    const paddle::optional<pir::Value>& sin,
    const paddle::optional<pir::Value>& cos,
    const paddle::optional<pir::Value>& position_ids,
    const pir::Value& out_q_grad,
    const paddle::optional<pir::Value>& out_k_grad,
    const paddle::optional<pir::Value>& out_v_grad,
    bool use_neox_rotary_style);

pir::OpResult max_pool2d_v2_grad(const pir::Value& x,
                                 const pir::Value& out,
                                 const pir::Value& saved_idx,
                                 const pir::Value& out_grad,
                                 const std::vector<int>& kernel_size,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 const std::string& data_format,
                                 bool global_pooling,
                                 bool adaptive);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           paddle::optional<pir::OpResult>>
adadelta_(const pir::Value& param,
          const pir::Value& grad,
          const pir::Value& avg_squared_grad,
          const pir::Value& avg_squared_update,
          const pir::Value& learning_rate,
          const paddle::optional<pir::Value>& master_param,
          float rho,
          float epsilon,
          bool multi_precision);

pir::OpResult add(const pir::Value& x, const pir::Value& y);

pir::OpResult add_(const pir::Value& x, const pir::Value& y);

pir::OpResult add_n(const std::vector<pir::Value>& inputs);

pir::OpResult add_n_(const std::vector<pir::Value>& inputs);

pir::OpResult add_n_with_kernel(const std::vector<pir::Value>& inputs);

pir::OpResult all(const pir::Value& x,
                  const std::vector<int64_t>& axis = {},
                  bool keepdim = false);

pir::OpResult amax(const pir::Value& x,
                   const std::vector<int64_t>& axis = {},
                   bool keepdim = false);

pir::OpResult amin(const pir::Value& x,
                   const std::vector<int64_t>& axis = {},
                   bool keepdim = false);

pir::OpResult any(const pir::Value& x,
                  const std::vector<int64_t>& axis = {},
                  bool keepdim = false);

pir::OpResult assign(const pir::Value& x);

pir::OpResult assign_(const pir::Value& x);

pir::OpResult assign_out_(const pir::Value& x, const pir::Value& output);

pir::OpResult assign_value(const std::vector<int>& shape,
                           phi::DataType dtype,
                           std::vector<phi::Scalar> values,
                           const Place& place = {});

pir::OpResult assign_value_(const pir::Value& output,
                            const std::vector<int>& shape,
                            phi::DataType dtype,
                            std::vector<phi::Scalar> values,
                            const Place& place = {});

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
batch_norm(const pir::Value& x,
           const pir::Value& mean,
           const pir::Value& variance,
           const paddle::optional<pir::Value>& scale,
           const paddle::optional<pir::Value>& bias,
           bool is_test,
           float momentum,
           float epsilon,
           const std::string& data_layout,
           bool use_global_stats,
           bool trainable_statistics);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
batch_norm_(const pir::Value& x,
            const pir::Value& mean,
            const pir::Value& variance,
            const paddle::optional<pir::Value>& scale,
            const paddle::optional<pir::Value>& bias,
            bool is_test,
            float momentum,
            float epsilon,
            const std::string& data_layout,
            bool use_global_stats,
            bool trainable_statistics);

pir::OpResult c_allgather(const pir::Value& x,
                          int ring_id,
                          int nranks,
                          bool use_calc_stream);

pir::OpResult c_allreduce_max(const pir::Value& x,
                              int ring_id,
                              bool use_calc_stream,
                              bool use_model_parallel);

pir::OpResult c_allreduce_max_(const pir::Value& x,
                               int ring_id,
                               bool use_calc_stream,
                               bool use_model_parallel);

pir::OpResult c_allreduce_sum(const pir::Value& x,
                              int ring_id,
                              bool use_calc_stream,
                              bool use_model_parallel);

pir::OpResult c_allreduce_sum_(const pir::Value& x,
                               int ring_id,
                               bool use_calc_stream,
                               bool use_model_parallel);

pir::OpResult c_broadcast(const pir::Value& x,
                          int ring_id = 0,
                          int root = 0,
                          bool use_calc_stream = false);

pir::OpResult c_broadcast_(const pir::Value& x,
                           int ring_id = 0,
                           int root = 0,
                           bool use_calc_stream = false);

pir::OpResult c_concat(const pir::Value& x,
                       int rank,
                       int nranks,
                       int ring_id,
                       bool use_calc_stream,
                       bool use_model_parallel);

pir::OpResult c_embedding(const pir::Value& weight,
                          const pir::Value& x,
                          int64_t start_index = 0,
                          int64_t vocab_size = -1);

pir::OpResult c_identity(const pir::Value& x,
                         int ring_id,
                         bool use_calc_stream,
                         bool use_model_parallel);

pir::OpResult c_identity_(const pir::Value& x,
                          int ring_id,
                          bool use_calc_stream,
                          bool use_model_parallel);

pir::OpResult c_reduce_min(const pir::Value& x,
                           int ring_id,
                           int root_id,
                           bool use_calc_stream);

pir::OpResult c_reduce_min_(const pir::Value& x,
                            int ring_id,
                            int root_id,
                            bool use_calc_stream);

pir::OpResult c_reduce_sum(const pir::Value& x,
                           int ring_id,
                           int root_id,
                           bool use_calc_stream);

pir::OpResult c_reduce_sum_(const pir::Value& x,
                            int ring_id,
                            int root_id,
                            bool use_calc_stream);

pir::OpResult c_reducescatter(const pir::Value& x,
                              int ring_id = 0,
                              int nranks = 1,
                              bool use_calc_stream = false);

pir::OpResult c_sync_calc_stream(const pir::Value& x);

pir::OpResult c_sync_calc_stream_(const pir::Value& x);

pir::OpResult c_sync_comm_stream(const pir::Value& x, int ring_id);

pir::OpResult c_sync_comm_stream_(const pir::Value& x, int ring_id);

pir::OpResult cast(const pir::Value& x, phi::DataType dtype);

pir::OpResult cast_(const pir::Value& x, phi::DataType dtype);

pir::OpResult channel_shuffle(const pir::Value& x,
                              int groups,
                              const std::string& data_format = "NCHW");

pir::OpResult conv2d_transpose(
    const pir::Value& x,
    const pir::Value& filter,
    const std::vector<int>& strides = {1, 1},
    const std::vector<int>& paddings = {0, 0},
    const std::vector<int>& output_padding = {},
    const std::vector<int64_t>& output_size = {},
    const std::string& padding_algorithm = "EXPLICIT",
    int groups = 1,
    const std::vector<int>& dilations = {1, 1},
    const std::string& data_format = "NCHW");

pir::OpResult conv2d_transpose(
    const pir::Value& x,
    const pir::Value& filter,
    pir::Value output_size,
    const std::vector<int>& strides = {1, 1},
    const std::vector<int>& paddings = {0, 0},
    const std::vector<int>& output_padding = {},
    const std::string& padding_algorithm = "EXPLICIT",
    int groups = 1,
    const std::vector<int>& dilations = {1, 1},
    const std::string& data_format = "NCHW");

pir::OpResult conv2d_transpose(
    const pir::Value& x,
    const pir::Value& filter,
    std::vector<pir::Value> output_size,
    const std::vector<int>& strides = {1, 1},
    const std::vector<int>& paddings = {0, 0},
    const std::vector<int>& output_padding = {},
    const std::string& padding_algorithm = "EXPLICIT",
    int groups = 1,
    const std::vector<int>& dilations = {1, 1},
    const std::string& data_format = "NCHW");

std::tuple<pir::OpResult, pir::OpResult> decayed_adagrad(
    const pir::Value& param,
    const pir::Value& grad,
    const pir::Value& moment,
    const pir::Value& learning_rate,
    float decay = 0.95f,
    float epsilon = 1.0e-6f);

pir::OpResult decode_jpeg(const pir::Value& x,
                          const std::string& mode,
                          const Place& place);

pir::OpResult deformable_conv(const pir::Value& x,
                              const pir::Value& offset,
                              const pir::Value& filter,
                              const paddle::optional<pir::Value>& mask,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              const std::vector<int>& dilations,
                              int deformable_groups,
                              int groups,
                              int im2col_step);

pir::OpResult depthwise_conv2d_transpose(
    const pir::Value& x,
    const pir::Value& filter,
    const std::vector<int>& strides = {1, 1},
    const std::vector<int>& paddings = {0, 0},
    const std::vector<int>& output_padding = {},
    const std::vector<int64_t>& output_size = {},
    const std::string& padding_algorithm = "EXPLICIT",
    int groups = 1,
    const std::vector<int>& dilations = {1, 1},
    const std::string& data_format = "NCHW");

pir::OpResult depthwise_conv2d_transpose(
    const pir::Value& x,
    const pir::Value& filter,
    pir::Value output_size,
    const std::vector<int>& strides = {1, 1},
    const std::vector<int>& paddings = {0, 0},
    const std::vector<int>& output_padding = {},
    const std::string& padding_algorithm = "EXPLICIT",
    int groups = 1,
    const std::vector<int>& dilations = {1, 1},
    const std::string& data_format = "NCHW");

pir::OpResult depthwise_conv2d_transpose(
    const pir::Value& x,
    const pir::Value& filter,
    std::vector<pir::Value> output_size,
    const std::vector<int>& strides = {1, 1},
    const std::vector<int>& paddings = {0, 0},
    const std::vector<int>& output_padding = {},
    const std::string& padding_algorithm = "EXPLICIT",
    int groups = 1,
    const std::vector<int>& dilations = {1, 1},
    const std::string& data_format = "NCHW");

pir::OpResult disable_check_model_nan_inf(const pir::Value& x, int flag = 0);

std::
    tuple<std::vector<pir::OpResult>, std::vector<pir::OpResult>, pir::OpResult>
    distribute_fpn_proposals(const pir::Value& fpn_rois,
                             const paddle::optional<pir::Value>& rois_num,
                             int min_level,
                             int max_level,
                             int refer_level,
                             int refer_scale,
                             bool pixel_offset);

pir::OpResult divide(const pir::Value& x, const pir::Value& y);

pir::OpResult divide_(const pir::Value& x, const pir::Value& y);

pir::OpResult dropout(const pir::Value& x,
                      const paddle::optional<pir::Value>& seed_tensor,
                      float p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed);

std::
    tuple<pir::OpResult, std::vector<pir::OpResult>, std::vector<pir::OpResult>>
    einsum(const std::vector<pir::Value>& x, const std::string& equation);

pir::OpResult elementwise_pow(const pir::Value& x, const pir::Value& y);

pir::OpResult embedding(const pir::Value& x,
                        const pir::Value& weight,
                        int64_t padding_idx = -1,
                        bool sparse = false);

pir::OpResult empty(const std::vector<int64_t>& shape,
                    phi::DataType dtype = phi::DataType::FLOAT32,
                    const Place& place = phi::CPUPlace());

pir::OpResult empty(pir::Value shape,
                    phi::DataType dtype = phi::DataType::FLOAT32,
                    const Place& place = phi::CPUPlace());

pir::OpResult empty(std::vector<pir::Value> shape,
                    phi::DataType dtype = phi::DataType::FLOAT32,
                    const Place& place = phi::CPUPlace());

pir::OpResult empty_like(const pir::Value& x,
                         phi::DataType dtype = phi::DataType::UNDEFINED,
                         const Place& place = {});

pir::OpResult enable_check_model_nan_inf(const pir::Value& x, int flag = 1);

pir::OpResult equal(const pir::Value& x, const pir::Value& y);

pir::OpResult equal_(const pir::Value& x, const pir::Value& y);

pir::OpResult exponential_(const pir::Value& x, float lam);

pir::OpResult eye(float num_rows,
                  float num_columns,
                  phi::DataType dtype = phi::DataType::FLOAT32,
                  const Place& place = {});

pir::OpResult eye(pir::Value num_rows,
                  pir::Value num_columns,
                  phi::DataType dtype = phi::DataType::FLOAT32,
                  const Place& place = {});

pir::OpResult fetch(const pir::Value& x, const std::string& name, int col);

pir::OpResult floor_divide(const pir::Value& x, const pir::Value& y);

pir::OpResult floor_divide_(const pir::Value& x, const pir::Value& y);

pir::OpResult frobenius_norm(const pir::Value& x,
                             const std::vector<int64_t>& axis,
                             bool keep_dim,
                             bool reduce_all);

pir::OpResult frobenius_norm(const pir::Value& x,
                             pir::Value axis,
                             bool keep_dim,
                             bool reduce_all);

pir::OpResult frobenius_norm(const pir::Value& x,
                             std::vector<pir::Value> axis,
                             bool keep_dim,
                             bool reduce_all);

pir::OpResult full(const std::vector<int64_t>& shape,
                   float value,
                   phi::DataType dtype = phi::DataType::FLOAT32,
                   const Place& place = phi::CPUPlace());

pir::OpResult full_(const pir::Value& output,
                    const std::vector<int64_t>& shape,
                    float value,
                    phi::DataType dtype = phi::DataType::FLOAT32,
                    const Place& place = phi::CPUPlace());

pir::OpResult full_batch_size_like(const pir::Value& input,
                                   const std::vector<int>& shape,
                                   phi::DataType dtype,
                                   float value,
                                   int input_dim_idx,
                                   int output_dim_idx,
                                   const Place& place = phi::CPUPlace());

pir::OpResult full_like(const pir::Value& x,
                        float value,
                        phi::DataType dtype = phi::DataType::UNDEFINED,
                        const Place& place = {});

pir::OpResult full_like(const pir::Value& x,
                        pir::Value value,
                        phi::DataType dtype = phi::DataType::UNDEFINED,
                        const Place& place = {});

pir::OpResult full_with_tensor(const pir::Value& shape,
                               const pir::Value& value,
                               phi::DataType dtype = phi::DataType::FLOAT32);

std::tuple<std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           paddle::optional<std::vector<pir::OpResult>>>
fused_adam_(const std::vector<pir::Value>& params,
            const std::vector<pir::Value>& grads,
            const pir::Value& learning_rate,
            const std::vector<pir::Value>& moments1,
            const std::vector<pir::Value>& moments2,
            const std::vector<pir::Value>& beta1_pows,
            const std::vector<pir::Value>& beta2_pows,
            const paddle::optional<std::vector<pir::Value>>& master_params,
            const paddle::optional<pir::Value>& skip_update,
            float beta1,
            float beta2,
            float epsilon,
            int chunk_size,
            float weight_decay,
            bool use_adamw,
            bool multi_precision,
            bool use_global_beta_pow);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_batch_norm_act(const pir::Value& x,
                     const pir::Value& scale,
                     const pir::Value& bias,
                     const pir::Value& mean,
                     const pir::Value& variance,
                     float momentum,
                     float epsilon,
                     const std::string& act_type);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_batch_norm_act_(const pir::Value& x,
                      const pir::Value& scale,
                      const pir::Value& bias,
                      const pir::Value& mean,
                      const pir::Value& variance,
                      float momentum,
                      float epsilon,
                      const std::string& act_type);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_bn_add_activation(const pir::Value& x,
                        const pir::Value& z,
                        const pir::Value& scale,
                        const pir::Value& bias,
                        const pir::Value& mean,
                        const pir::Value& variance,
                        float momentum,
                        float epsilon,
                        const std::string& act_type);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_bn_add_activation_(const pir::Value& x,
                         const pir::Value& z,
                         const pir::Value& scale,
                         const pir::Value& bias,
                         const pir::Value& mean,
                         const pir::Value& variance,
                         float momentum,
                         float epsilon,
                         const std::string& act_type);

pir::OpResult fused_softmax_mask_upper_triangle(const pir::Value& X);

pir::OpResult gaussian(const std::vector<int64_t>& shape,
                       float mean,
                       float std,
                       int seed,
                       phi::DataType dtype,
                       const Place& place = {});

pir::OpResult gaussian(pir::Value shape,
                       float mean,
                       float std,
                       int seed,
                       phi::DataType dtype,
                       const Place& place = {});

pir::OpResult gaussian(std::vector<pir::Value> shape,
                       float mean,
                       float std,
                       int seed,
                       phi::DataType dtype,
                       const Place& place = {});

pir::OpResult get_tensor_from_selected_rows(const pir::Value& x);

pir::OpResult greater_equal(const pir::Value& x, const pir::Value& y);

pir::OpResult greater_equal_(const pir::Value& x, const pir::Value& y);

pir::OpResult greater_than(const pir::Value& x, const pir::Value& y);

pir::OpResult greater_than_(const pir::Value& x, const pir::Value& y);

pir::OpResult hardswish(const pir::Value& x);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> hsigmoid_loss(
    const pir::Value& x,
    const pir::Value& label,
    const pir::Value& w,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& path,
    const paddle::optional<pir::Value>& code,
    int num_classes,
    bool is_sparse);

pir::OpResult increment(const pir::Value& x, float value = 1.0);

pir::OpResult increment_(const pir::Value& x, float value = 1.0);

pir::OpResult less_equal(const pir::Value& x, const pir::Value& y);

pir::OpResult less_equal_(const pir::Value& x, const pir::Value& y);

pir::OpResult less_than(const pir::Value& x, const pir::Value& y);

pir::OpResult less_than_(const pir::Value& x, const pir::Value& y);

pir::OpResult linspace(const pir::Value& start,
                       const pir::Value& stop,
                       const pir::Value& number,
                       phi::DataType dtype,
                       const Place& place);

pir::OpResult logspace(const pir::Value& start,
                       const pir::Value& stop,
                       const pir::Value& num,
                       const pir::Value& base,
                       phi::DataType dtype,
                       const Place& place = {});

pir::OpResult logsumexp(const pir::Value& x,
                        const std::vector<int64_t>& axis,
                        bool keepdim,
                        bool reduce_all);

pir::OpResult matmul(const pir::Value& x,
                     const pir::Value& y,
                     bool transpose_x = false,
                     bool transpose_y = false);

pir::OpResult matrix_rank(const pir::Value& x,
                          float tol,
                          bool use_default_tol = true,
                          bool hermitian = false);

pir::OpResult matrix_rank_tol(const pir::Value& x,
                              const pir::Value& atol_tensor,
                              bool use_default_tol = true,
                              bool hermitian = false);

pir::OpResult max(const pir::Value& x,
                  const std::vector<int64_t>& axis = {},
                  bool keepdim = false);

pir::OpResult max(const pir::Value& x, pir::Value axis, bool keepdim = false);

pir::OpResult max(const pir::Value& x,
                  std::vector<pir::Value> axis,
                  bool keepdim = false);

pir::OpResult maximum(const pir::Value& x, const pir::Value& y);

pir::OpResult mean(const pir::Value& x,
                   const std::vector<int64_t>& axis = {},
                   bool keepdim = false);

pir::OpResult memcpy(const pir::Value& x, int dst_place_type);

pir::OpResult memcpy_d2h(const pir::Value& x, int dst_place_type);

pir::OpResult memcpy_h2d(const pir::Value& x, int dst_place_type);

pir::OpResult min(const pir::Value& x,
                  const std::vector<int64_t>& axis = {},
                  bool keepdim = false);

pir::OpResult min(const pir::Value& x, pir::Value axis, bool keepdim = false);

pir::OpResult min(const pir::Value& x,
                  std::vector<pir::Value> axis,
                  bool keepdim = false);

pir::OpResult minimum(const pir::Value& x, const pir::Value& y);

pir::OpResult mish(const pir::Value& x, float lambda);

pir::OpResult multiply(const pir::Value& x, const pir::Value& y);

pir::OpResult multiply_(const pir::Value& x, const pir::Value& y);

std::tuple<pir::OpResult, pir::OpResult> norm(const pir::Value& x,
                                              int axis,
                                              float epsilon,
                                              bool is_test);

pir::OpResult not_equal(const pir::Value& x, const pir::Value& y);

pir::OpResult not_equal_(const pir::Value& x, const pir::Value& y);

pir::OpResult one_hot(const pir::Value& x, int num_classes);

pir::OpResult one_hot(const pir::Value& x, pir::Value num_classes);

pir::OpResult pad(const pir::Value& x,
                  const std::vector<int>& paddings,
                  float pad_value);

pir::OpResult pad(const pir::Value& x,
                  pir::Value pad_value,
                  const std::vector<int>& paddings);

pir::OpResult pool2d(const pir::Value& x,
                     const std::vector<int64_t>& kernel_size,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     bool ceil_mode,
                     bool exclusive,
                     const std::string& data_format,
                     const std::string& pooling_type,
                     bool global_pooling,
                     bool adaptive,
                     const std::string& padding_algorithm);

pir::OpResult pool2d(const pir::Value& x,
                     pir::Value kernel_size,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     bool ceil_mode,
                     bool exclusive,
                     const std::string& data_format,
                     const std::string& pooling_type,
                     bool global_pooling,
                     bool adaptive,
                     const std::string& padding_algorithm);

pir::OpResult pool2d(const pir::Value& x,
                     std::vector<pir::Value> kernel_size,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     bool ceil_mode,
                     bool exclusive,
                     const std::string& data_format,
                     const std::string& pooling_type,
                     bool global_pooling,
                     bool adaptive,
                     const std::string& padding_algorithm);

pir::OpResult pool3d(const pir::Value& x,
                     const std::vector<int>& kernel_size,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     bool ceil_mode,
                     bool exclusive,
                     const std::string& data_format,
                     const std::string& pooling_type,
                     bool global_pooling,
                     bool adaptive,
                     const std::string& padding_algorithm);

pir::OpResult print(const pir::Value& in,
                    int first_n,
                    const std::string& message,
                    int summarize,
                    bool print_tensor_name = true,
                    bool print_tensor_type = true,
                    bool print_tensor_shape = true,
                    bool print_tensor_layout = true,
                    bool print_tensor_lod = true,
                    const std::string& print_phase = "BOTH",
                    bool is_forward = true);

pir::OpResult prod(const pir::Value& x,
                   const std::vector<int64_t>& dims,
                   bool keep_dim,
                   bool reduce_all);

pir::OpResult prod(const pir::Value& x,
                   pir::Value dims,
                   bool keep_dim,
                   bool reduce_all);

pir::OpResult prod(const pir::Value& x,
                   std::vector<pir::Value> dims,
                   bool keep_dim,
                   bool reduce_all);

pir::OpResult randint(int low,
                      int high,
                      const std::vector<int64_t>& shape,
                      phi::DataType dtype = phi::DataType::INT64,
                      const Place& place = {});

pir::OpResult randint(pir::Value shape,
                      int low,
                      int high,
                      phi::DataType dtype = phi::DataType::INT64,
                      const Place& place = {});

pir::OpResult randint(std::vector<pir::Value> shape,
                      int low,
                      int high,
                      phi::DataType dtype = phi::DataType::INT64,
                      const Place& place = {});

pir::OpResult randperm(int n, phi::DataType dtype, const Place& place = {});

pir::OpResult read_file(const std::string& filename = "",
                        phi::DataType dtype = phi::DataType::UINT8,
                        const Place& place = phi::CPUPlace());

pir::OpResult recv_v2(const std::vector<int>& out_shape = {},
                      phi::DataType dtype = phi::DataType::FLOAT32,
                      int peer = 0,
                      int ring_id = 0,
                      bool use_calc_stream = false,
                      bool dynamic_shape = false);

pir::OpResult remainder(const pir::Value& x, const pir::Value& y);

pir::OpResult remainder_(const pir::Value& x, const pir::Value& y);

pir::OpResult repeat_interleave(const pir::Value& x, int repeats, int axis);

pir::OpResult repeat_interleave_with_tensor_index(const pir::Value& x,
                                                  const pir::Value& repeats,
                                                  int axis);

pir::OpResult reshape(const pir::Value& x, const std::vector<int64_t>& shape);

pir::OpResult reshape(const pir::Value& x, pir::Value shape);

pir::OpResult reshape(const pir::Value& x, std::vector<pir::Value> shape);

pir::OpResult reshape_(const pir::Value& x, const std::vector<int64_t>& shape);

pir::OpResult reshape_(const pir::Value& x, pir::Value shape);

pir::OpResult reshape_(const pir::Value& x, std::vector<pir::Value> shape);

std::tuple<pir::OpResult, pir::OpResult, std::vector<pir::OpResult>> rnn(
    const pir::Value& x,
    const std::vector<pir::Value>& pre_state,
    const std::vector<pir::Value>& weight_list,
    const paddle::optional<pir::Value>& sequence_length,
    const pir::Value& dropout_state_in,
    float dropout_prob = 0.0,
    bool is_bidirec = false,
    int input_size = 10,
    int hidden_size = 100,
    int num_layers = 1,
    const std::string& mode = "RNN_TANH",
    int seed = 0,
    bool is_test = false);

std::tuple<pir::OpResult, pir::OpResult, std::vector<pir::OpResult>> rnn_(
    const pir::Value& x,
    const std::vector<pir::Value>& pre_state,
    const std::vector<pir::Value>& weight_list,
    const paddle::optional<pir::Value>& sequence_length,
    const pir::Value& dropout_state_in,
    float dropout_prob = 0.0,
    bool is_bidirec = false,
    int input_size = 10,
    int hidden_size = 100,
    int num_layers = 1,
    const std::string& mode = "RNN_TANH",
    int seed = 0,
    bool is_test = false);

pir::OpResult row_conv(const pir::Value& x, const pir::Value& filter);

pir::OpResult rrelu(const pir::Value& x,
                    float lower,
                    float upper,
                    bool is_test);

pir::OpResult seed(int seed,
                   bool deterministic,
                   const std::string& rng_name,
                   bool force_cpu);

void send_v2(const pir::Value& x,
             int ring_id = 0,
             int peer = 0,
             bool use_calc_stream = false,
             bool dynamic_shape = false);

pir::OpResult set_value(const pir::Value& x,
                        const std::vector<int64_t>& starts,
                        const std::vector<int64_t>& ends,
                        const std::vector<int64_t>& steps,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& decrease_axes,
                        const std::vector<int64_t>& none_axes,
                        const std::vector<int64_t>& shape,
                        std::vector<phi::Scalar> values);

pir::OpResult set_value(const pir::Value& x,
                        pir::Value starts,
                        pir::Value ends,
                        pir::Value steps,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& decrease_axes,
                        const std::vector<int64_t>& none_axes,
                        const std::vector<int64_t>& shape,
                        std::vector<phi::Scalar> values);

pir::OpResult set_value(const pir::Value& x,
                        std::vector<pir::Value> starts,
                        std::vector<pir::Value> ends,
                        std::vector<pir::Value> steps,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& decrease_axes,
                        const std::vector<int64_t>& none_axes,
                        const std::vector<int64_t>& shape,
                        std::vector<phi::Scalar> values);

pir::OpResult set_value_(const pir::Value& x,
                         const std::vector<int64_t>& starts,
                         const std::vector<int64_t>& ends,
                         const std::vector<int64_t>& steps,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& decrease_axes,
                         const std::vector<int64_t>& none_axes,
                         const std::vector<int64_t>& shape,
                         std::vector<phi::Scalar> values);

pir::OpResult set_value_(const pir::Value& x,
                         pir::Value starts,
                         pir::Value ends,
                         pir::Value steps,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& decrease_axes,
                         const std::vector<int64_t>& none_axes,
                         const std::vector<int64_t>& shape,
                         std::vector<phi::Scalar> values);

pir::OpResult set_value_(const pir::Value& x,
                         std::vector<pir::Value> starts,
                         std::vector<pir::Value> ends,
                         std::vector<pir::Value> steps,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& decrease_axes,
                         const std::vector<int64_t>& none_axes,
                         const std::vector<int64_t>& shape,
                         std::vector<phi::Scalar> values);

pir::OpResult set_value_with_tensor(const pir::Value& x,
                                    const pir::Value& values,
                                    const std::vector<int64_t>& starts,
                                    const std::vector<int64_t>& ends,
                                    const std::vector<int64_t>& steps,
                                    const std::vector<int64_t>& axes,
                                    const std::vector<int64_t>& decrease_axes,
                                    const std::vector<int64_t>& none_axes);

pir::OpResult set_value_with_tensor(const pir::Value& x,
                                    const pir::Value& values,
                                    pir::Value starts,
                                    pir::Value ends,
                                    pir::Value steps,
                                    const std::vector<int64_t>& axes,
                                    const std::vector<int64_t>& decrease_axes,
                                    const std::vector<int64_t>& none_axes);

pir::OpResult set_value_with_tensor(const pir::Value& x,
                                    const pir::Value& values,
                                    std::vector<pir::Value> starts,
                                    std::vector<pir::Value> ends,
                                    std::vector<pir::Value> steps,
                                    const std::vector<int64_t>& axes,
                                    const std::vector<int64_t>& decrease_axes,
                                    const std::vector<int64_t>& none_axes);

pir::OpResult set_value_with_tensor_(const pir::Value& x,
                                     const pir::Value& values,
                                     const std::vector<int64_t>& starts,
                                     const std::vector<int64_t>& ends,
                                     const std::vector<int64_t>& steps,
                                     const std::vector<int64_t>& axes,
                                     const std::vector<int64_t>& decrease_axes,
                                     const std::vector<int64_t>& none_axes);

pir::OpResult set_value_with_tensor_(const pir::Value& x,
                                     const pir::Value& values,
                                     pir::Value starts,
                                     pir::Value ends,
                                     pir::Value steps,
                                     const std::vector<int64_t>& axes,
                                     const std::vector<int64_t>& decrease_axes,
                                     const std::vector<int64_t>& none_axes);

pir::OpResult set_value_with_tensor_(const pir::Value& x,
                                     const pir::Value& values,
                                     std::vector<pir::Value> starts,
                                     std::vector<pir::Value> ends,
                                     std::vector<pir::Value> steps,
                                     const std::vector<int64_t>& axes,
                                     const std::vector<int64_t>& decrease_axes,
                                     const std::vector<int64_t>& none_axes);

pir::OpResult shadow_feed(const pir::Value& x);

pir::OpResult share_data(const pir::Value& x);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> shuffle_batch(
    const pir::Value& x, const pir::Value& seed, int startup_seed = 0);

pir::OpResult slice(const pir::Value& input,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis);

pir::OpResult slice(const pir::Value& input,
                    pir::Value starts,
                    pir::Value ends,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis);

pir::OpResult slice(const pir::Value& input,
                    std::vector<pir::Value> starts,
                    std::vector<pir::Value> ends,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis);

pir::OpResult soft_relu(const pir::Value& x, float threshold = 20.0f);

pir::OpResult softmax(const pir::Value& x, int axis);

pir::OpResult softmax_(const pir::Value& x, int axis);

std::vector<pir::OpResult> split(const pir::Value& x,
                                 const std::vector<int64_t>& sections,
                                 int axis);

std::vector<pir::OpResult> split(const pir::Value& x,
                                 pir::Value sections,
                                 pir::Value axis);

std::vector<pir::OpResult> split(const pir::Value& x,
                                 std::vector<pir::Value> sections,
                                 pir::Value axis);

std::vector<pir::OpResult> split_with_num(const pir::Value& x,
                                          int num,
                                          int axis);

std::vector<pir::OpResult> split_with_num(const pir::Value& x,
                                          pir::Value axis,
                                          int num);

pir::OpResult strided_slice(const pir::Value& x,
                            const std::vector<int>& axes,
                            const std::vector<int64_t>& starts,
                            const std::vector<int64_t>& ends,
                            const std::vector<int64_t>& strides);

pir::OpResult strided_slice(const pir::Value& x,
                            pir::Value starts,
                            pir::Value ends,
                            pir::Value strides,
                            const std::vector<int>& axes);

pir::OpResult strided_slice(const pir::Value& x,
                            std::vector<pir::Value> starts,
                            std::vector<pir::Value> ends,
                            std::vector<pir::Value> strides,
                            const std::vector<int>& axes);

pir::OpResult subtract(const pir::Value& x, const pir::Value& y);

pir::OpResult subtract_(const pir::Value& x, const pir::Value& y);

pir::OpResult sum(const pir::Value& x,
                  const std::vector<int64_t>& axis = {},
                  phi::DataType dtype = phi::DataType::UNDEFINED,
                  bool keepdim = false);

pir::OpResult sum(const pir::Value& x,
                  pir::Value axis,
                  phi::DataType dtype = phi::DataType::UNDEFINED,
                  bool keepdim = false);

pir::OpResult sum(const pir::Value& x,
                  std::vector<pir::Value> axis,
                  phi::DataType dtype = phi::DataType::UNDEFINED,
                  bool keepdim = false);

pir::OpResult swish(const pir::Value& x);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
sync_batch_norm_(const pir::Value& x,
                 const pir::Value& mean,
                 const pir::Value& variance,
                 const pir::Value& scale,
                 const pir::Value& bias,
                 bool is_test,
                 float momentum,
                 float epsilon,
                 const std::string& data_layout,
                 bool use_global_stats,
                 bool trainable_statistics);

pir::OpResult tile(const pir::Value& x,
                   const std::vector<int64_t>& repeat_times = {});

pir::OpResult tile(const pir::Value& x, pir::Value repeat_times);

pir::OpResult tile(const pir::Value& x, std::vector<pir::Value> repeat_times);

pir::OpResult trans_layout(const pir::Value& x, const std::vector<int>& perm);

pir::OpResult transpose(const pir::Value& x, const std::vector<int>& perm);

pir::OpResult transpose_(const pir::Value& x, const std::vector<int>& perm);

pir::OpResult tril(const pir::Value& x, int diagonal);

pir::OpResult tril_(const pir::Value& x, int diagonal);

pir::OpResult tril_indices(int rows,
                           int cols,
                           int offset,
                           phi::DataType dtype,
                           const Place& place = {});

pir::OpResult triu(const pir::Value& x, int diagonal);

pir::OpResult triu_(const pir::Value& x, int diagonal);

pir::OpResult triu_indices(
    int row, int col, int offset, phi::DataType dtype, const Place& place = {});

pir::OpResult truncated_gaussian_random(
    const std::vector<int>& shape,
    float mean,
    float std,
    int seed,
    phi::DataType dtype = phi::DataType::FLOAT32,
    const Place& place = {});

pir::OpResult uniform(const std::vector<int64_t>& shape,
                      phi::DataType dtype,
                      float min,
                      float max,
                      int seed,
                      const Place& place = {});

pir::OpResult uniform(pir::Value shape,
                      pir::Value min,
                      pir::Value max,
                      phi::DataType dtype,
                      int seed,
                      const Place& place = {});

pir::OpResult uniform(std::vector<pir::Value> shape,
                      pir::Value min,
                      pir::Value max,
                      phi::DataType dtype,
                      int seed,
                      const Place& place = {});

pir::OpResult uniform_random_batch_size_like(
    const pir::Value& input,
    const std::vector<int>& shape,
    int input_dim_idx = 0,
    int output_dim_idx = 0,
    float min = -1.0f,
    float max = 1.0f,
    int seed = 0,
    int diag_num = 0,
    int diag_step = 0,
    float diag_val = 1.0f,
    phi::DataType dtype = phi::DataType::FLOAT32);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult> unique(
    const pir::Value& x,
    bool return_index = false,
    bool return_inverse = false,
    bool return_counts = false,
    const std::vector<int>& axis = {},
    phi::DataType dtype = phi::DataType::INT64,
    bool is_sorted = false);

pir::OpResult unpool(const pir::Value& x,
                     const pir::Value& indices,
                     const std::vector<int>& ksize,
                     const std::vector<int>& strides,
                     const std::vector<int>& padding,
                     const std::vector<int64_t>& output_size,
                     const std::string& data_format);

pir::OpResult unpool(const pir::Value& x,
                     const pir::Value& indices,
                     pir::Value output_size,
                     const std::vector<int>& ksize,
                     const std::vector<int>& strides,
                     const std::vector<int>& padding,
                     const std::string& data_format);

pir::OpResult unpool(const pir::Value& x,
                     const pir::Value& indices,
                     std::vector<pir::Value> output_size,
                     const std::vector<int>& ksize,
                     const std::vector<int>& strides,
                     const std::vector<int>& padding,
                     const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult> c_softmax_with_cross_entropy(
    const pir::Value& logits,
    const pir::Value& label,
    int64_t ignore_index = -100,
    int ring_id = 0,
    int rank = 0,
    int nranks = 0);

pir::OpResult dpsgd(const pir::Value& param,
                    const pir::Value& grad,
                    const pir::Value& learning_rate,
                    float clip = 10.0f,
                    float batch_size = 16.0f,
                    float sigma = 1.0f,
                    int seed = 0);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> ftrl(
    const pir::Value& param,
    const pir::Value& squared_accumulator,
    const pir::Value& linear_accumulator,
    const pir::Value& grad,
    const pir::Value& learning_rate,
    float l1 = 0.0f,
    float l2 = 0.0f,
    float lr_power = -0.5f);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_attention(const pir::Value& x,
                const paddle::optional<pir::Value>& ln_scale,
                const paddle::optional<pir::Value>& ln_bias,
                const pir::Value& qkv_weight,
                const paddle::optional<pir::Value>& qkv_bias,
                const paddle::optional<pir::Value>& cache_kv,
                const paddle::optional<pir::Value>& src_mask,
                const pir::Value& out_linear_weight,
                const paddle::optional<pir::Value>& out_linear_bias,
                const paddle::optional<pir::Value>& ln_scale_2,
                const paddle::optional<pir::Value>& ln_bias_2,
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
                int ring_id);

pir::OpResult fused_elemwise_add_activation(
    const pir::Value& x,
    const pir::Value& y,
    const std::vector<std::string>& functor_list,
    float scale = 0.0,
    int axis = -1,
    bool save_intermediate_out = false);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_feedforward(const pir::Value& x,
                  const paddle::optional<pir::Value>& dropout1_seed,
                  const paddle::optional<pir::Value>& dropout2_seed,
                  const pir::Value& linear1_weight,
                  const paddle::optional<pir::Value>& linear1_bias,
                  const pir::Value& linear2_weight,
                  const paddle::optional<pir::Value>& linear2_bias,
                  const paddle::optional<pir::Value>& ln1_scale,
                  const paddle::optional<pir::Value>& ln1_bias,
                  const paddle::optional<pir::Value>& ln2_scale,
                  const paddle::optional<pir::Value>& ln2_bias,
                  bool pre_layer_norm,
                  float ln1_epsilon,
                  float ln2_epsilon,
                  const std::string& act_method,
                  float dropout1_prob,
                  float dropout2_prob,
                  const std::string& dropout1_implementation,
                  const std::string& dropout2_implementation,
                  bool is_test,
                  bool dropout1_fix_seed,
                  bool dropout2_fix_seed,
                  int dropout1_seed_val,
                  int dropout2_seed_val,
                  bool add_residual,
                  int ring_id);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> nce(
    const pir::Value& input,
    const pir::Value& label,
    const pir::Value& weight,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& sample_weight,
    const paddle::optional<pir::Value>& custom_dist_probs,
    const paddle::optional<pir::Value>& custom_dist_alias,
    const paddle::optional<pir::Value>& custom_dist_alias_probs,
    int num_total_classes,
    const std::vector<int>& custom_neg_classes = {},
    int num_neg_samples = 10,
    int sampler = 0,
    int seed = 0,
    bool is_sparse = false,
    bool remote_prefetch = false,
    bool is_test = false);

pir::OpResult number_count(const pir::Value& numbers, int upper_range);

pir::OpResult onednn_to_paddle_layout(const pir::Value& x, int dst_layout);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sparse_momentum(
    const pir::Value& param,
    const pir::Value& grad,
    const pir::Value& velocity,
    const pir::Value& index,
    const pir::Value& learning_rate,
    const paddle::optional<pir::Value>& master_param,
    float mu,
    float axis = 0,
    bool use_nesterov = false,
    const std::string& regularization_method = "",
    float regularization_coeff = 0.0f,
    bool multi_precision = false,
    float rescale_grad = 1.0f);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sparse_momentum(
    const pir::Value& param,
    const pir::Value& grad,
    const pir::Value& velocity,
    const pir::Value& index,
    const pir::Value& learning_rate,
    const paddle::optional<pir::Value>& master_param,
    pir::Value axis,
    float mu,
    bool use_nesterov = false,
    const std::string& regularization_method = "",
    float regularization_coeff = 0.0f,
    bool multi_precision = false,
    float rescale_grad = 1.0f);

std::tuple<pir::OpResult, pir::OpResult> match_matrix_tensor(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& w,
    int dim_t = 1);

std::tuple<std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           paddle::optional<std::vector<pir::OpResult>>>
lars_momentum(const std::vector<pir::Value>& param,
              const std::vector<pir::Value>& grad,
              const std::vector<pir::Value>& velocity,
              const std::vector<pir::Value>& learning_rate,
              const paddle::optional<std::vector<pir::Value>>& master_param,
              float mu,
              float lars_coeff = 0.001f,
              const std::vector<float>& lars_weight_decay = {0.0005f},
              float epsilon = 0.0f,
              bool multi_precision = false,
              float rescale_grad = 1.0f);

std::tuple<std::vector<pir::OpResult>,
           std::vector<pir::OpResult>,
           paddle::optional<std::vector<pir::OpResult>>>
lars_momentum_(const std::vector<pir::Value>& param,
               const std::vector<pir::Value>& grad,
               const std::vector<pir::Value>& velocity,
               const std::vector<pir::Value>& learning_rate,
               const paddle::optional<std::vector<pir::Value>>& master_param,
               float mu,
               float lars_coeff = 0.001f,
               const std::vector<float>& lars_weight_decay = {0.0005f},
               float epsilon = 0.0f,
               bool multi_precision = false,
               float rescale_grad = 1.0f);

pir::OpResult add_double_grad(const pir::Value& y,
                              const pir::Value& grad_out,
                              const paddle::optional<pir::Value>& grad_x_grad,
                              const paddle::optional<pir::Value>& grad_y_grad,
                              int axis = -1);

pir::OpResult add_double_grad_(const pir::Value& y,
                               const pir::Value& grad_out,
                               const paddle::optional<pir::Value>& grad_x_grad,
                               const paddle::optional<pir::Value>& grad_y_grad,
                               int axis = -1);

std::tuple<pir::OpResult, pir::OpResult> add_grad(const pir::Value& x,
                                                  const pir::Value& y,
                                                  const pir::Value& out_grad,
                                                  int axis = -1);

std::tuple<pir::OpResult, pir::OpResult> add_grad_(const pir::Value& x,
                                                   const pir::Value& y,
                                                   const pir::Value& out_grad,
                                                   int axis = -1);

std::tuple<pir::OpResult, pir::OpResult> add_triple_grad(
    const pir::Value& grad_grad_x,
    const pir::Value& grad_grad_y,
    const pir::Value& grad_grad_out_grad,
    int axis = -1);

std::tuple<pir::OpResult, pir::OpResult> add_triple_grad_(
    const pir::Value& grad_grad_x,
    const pir::Value& grad_grad_y,
    const pir::Value& grad_grad_out_grad,
    int axis = -1);

pir::OpResult amax_grad(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& axis = {},
                        bool keepdim = false,
                        bool reduce_all = false);

pir::OpResult amin_grad(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& axis = {},
                        bool keepdim = false,
                        bool reduce_all = false);

pir::OpResult assign_out__grad(const pir::Value& out_grad);

pir::OpResult assign_out__grad_(const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> batch_norm_double_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const paddle::optional<pir::Value>& out_mean,
    const paddle::optional<pir::Value>& out_variance,
    const pir::Value& saved_mean,
    const pir::Value& saved_variance,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_scale_grad,
    const paddle::optional<pir::Value>& grad_bias_grad,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> batch_norm_double_grad_(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const paddle::optional<pir::Value>& out_mean,
    const paddle::optional<pir::Value>& out_variance,
    const pir::Value& saved_mean,
    const pir::Value& saved_variance,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_scale_grad,
    const paddle::optional<pir::Value>& grad_bias_grad,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> batch_norm_grad(
    const pir::Value& x,
    const paddle::optional<pir::Value>& scale,
    const paddle::optional<pir::Value>& bias,
    const paddle::optional<pir::Value>& mean_out,
    const paddle::optional<pir::Value>& variance_out,
    const pir::Value& saved_mean,
    const pir::Value& saved_variance,
    const paddle::optional<pir::Value>& reserve_space,
    const pir::Value& out_grad,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics);

pir::OpResult c_embedding_grad(const pir::Value& weight,
                               const pir::Value& x,
                               const pir::Value& out_grad,
                               int64_t start_index = 0);

pir::OpResult c_softmax_with_cross_entropy_grad(const pir::Value& softmax,
                                                const pir::Value& label,
                                                const pir::Value& loss_grad,
                                                int64_t ignore_index = -100,
                                                int ring_id = 0,
                                                int rank = 0,
                                                int nranks = 0);

pir::OpResult channel_shuffle_grad(const pir::Value& out_grad,
                                   int groups,
                                   const std::string& data_format = "NCHW");

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
conv2d_transpose_double_grad(const pir::Value& x,
                             const pir::Value& filter,
                             const pir::Value& grad_out,
                             const pir::Value& grad_x_grad,
                             const pir::Value& grad_filter_grad,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::vector<int>& output_padding,
                             const std::vector<int64_t>& output_size,
                             const std::string& padding_algorithm,
                             int groups,
                             const std::vector<int>& dilations,
                             const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
conv2d_transpose_double_grad(const pir::Value& x,
                             const pir::Value& filter,
                             const pir::Value& grad_out,
                             const pir::Value& grad_x_grad,
                             const pir::Value& grad_filter_grad,
                             pir::Value output_size,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::vector<int>& output_padding,
                             const std::string& padding_algorithm,
                             int groups,
                             const std::vector<int>& dilations,
                             const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
conv2d_transpose_double_grad(const pir::Value& x,
                             const pir::Value& filter,
                             const pir::Value& grad_out,
                             const pir::Value& grad_x_grad,
                             const pir::Value& grad_filter_grad,
                             std::vector<pir::Value> output_size,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::vector<int>& output_padding,
                             const std::string& padding_algorithm,
                             int groups,
                             const std::vector<int>& dilations,
                             const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult> conv2d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::vector<int64_t>& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult> conv2d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    pir::Value output_size,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult> conv2d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    std::vector<pir::Value> output_size,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
deformable_conv_grad(const pir::Value& x,
                     const pir::Value& offset,
                     const pir::Value& filter,
                     const paddle::optional<pir::Value>& mask,
                     const pir::Value& out_grad,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     int deformable_groups,
                     int groups,
                     int im2col_step);

std::tuple<pir::OpResult, pir::OpResult> depthwise_conv2d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::vector<int64_t>& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult> depthwise_conv2d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    pir::Value output_size,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult> depthwise_conv2d_transpose_grad(
    const pir::Value& x,
    const pir::Value& filter,
    const pir::Value& out_grad,
    std::vector<pir::Value> output_size,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> divide_double_grad(
    const pir::Value& y,
    const pir::Value& out,
    const pir::Value& grad_x,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    int axis = -1);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> divide_double_grad_(
    const pir::Value& y,
    const pir::Value& out,
    const pir::Value& grad_x,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    int axis = -1);

std::tuple<pir::OpResult, pir::OpResult> divide_grad(const pir::Value& x,
                                                     const pir::Value& y,
                                                     const pir::Value& out,
                                                     const pir::Value& out_grad,
                                                     int axis = -1);

pir::OpResult dropout_grad(const pir::Value& mask,
                           const pir::Value& out_grad,
                           float p,
                           bool is_test,
                           const std::string& mode);

std::vector<pir::OpResult> einsum_grad(
    const std::vector<pir::Value>& x_shape,
    const std::vector<pir::Value>& inner_cache,
    const pir::Value& out_grad,
    const std::string& equation);

std::tuple<pir::OpResult, pir::OpResult> elementwise_pow_grad(
    const pir::Value& x, const pir::Value& y, const pir::Value& out_grad);

pir::OpResult frobenius_norm_grad(const pir::Value& x,
                                  const pir::Value& out,
                                  const pir::Value& out_grad,
                                  const std::vector<int64_t>& axis,
                                  bool keep_dim,
                                  bool reduce_all);

pir::OpResult frobenius_norm_grad(const pir::Value& x,
                                  const pir::Value& out,
                                  const pir::Value& out_grad,
                                  pir::Value axis,
                                  bool keep_dim,
                                  bool reduce_all);

pir::OpResult frobenius_norm_grad(const pir::Value& x,
                                  const pir::Value& out,
                                  const pir::Value& out_grad,
                                  std::vector<pir::Value> axis,
                                  bool keep_dim,
                                  bool reduce_all);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_attention_grad(
    const pir::Value& out_grad,
    const pir::Value& x,
    const pir::Value& qkv_weight,
    const paddle::optional<pir::Value>& qkv_bias,
    const paddle::optional<pir::Value>& qkv_bias_out,
    const paddle::optional<pir::Value>& src_mask,
    const paddle::optional<pir::Value>& src_mask_out,
    const pir::Value& out_linear_weight,
    const paddle::optional<pir::Value>& out_linear_bias,
    const paddle::optional<pir::Value>& ln_scale,
    const paddle::optional<pir::Value>& ln_bias,
    const paddle::optional<pir::Value>& ln_scale_2,
    const paddle::optional<pir::Value>& ln_bias_2,
    const paddle::optional<pir::Value>& ln_out,
    const paddle::optional<pir::Value>& ln_mean,
    const paddle::optional<pir::Value>& ln_var,
    const paddle::optional<pir::Value>& ln_mean_2,
    const paddle::optional<pir::Value>& ln_var_2,
    const paddle::optional<pir::Value>& bias_dropout_residual_out,
    const pir::Value& qkv_out,
    const pir::Value& transpose_out_2,
    const pir::Value& qk_out,
    const pir::Value& qktv_out,
    const pir::Value& softmax_out,
    const pir::Value& attn_dropout_mask_out,
    const pir::Value& attn_dropout_out,
    const pir::Value& fmha_out,
    const pir::Value& out_linear_out,
    const pir::Value& dropout_mask_out,
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
    int ring_id);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
fused_batch_norm_act_grad(const pir::Value& x,
                          const pir::Value& scale,
                          const pir::Value& bias,
                          const pir::Value& out,
                          const pir::Value& saved_mean,
                          const pir::Value& saved_variance,
                          const paddle::optional<pir::Value>& reserve_space,
                          const pir::Value& out_grad,
                          float momentum,
                          float epsilon,
                          const std::string& act_type);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult, pir::OpResult>
fused_bn_add_activation_grad(const pir::Value& x,
                             const pir::Value& scale,
                             const pir::Value& bias,
                             const pir::Value& out,
                             const pir::Value& saved_mean,
                             const pir::Value& saved_variance,
                             const paddle::optional<pir::Value>& reserve_space,
                             const pir::Value& out_grad,
                             float momentum,
                             float epsilon,
                             const std::string& act_type);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
fused_feedforward_grad(const pir::Value& out_grad,
                       const pir::Value& x,
                       const pir::Value& linear1_weight,
                       const paddle::optional<pir::Value>& linear1_bias,
                       const pir::Value& linear2_weight,
                       const pir::Value& dropout1_mask,
                       const pir::Value& dropout2_mask,
                       const pir::Value& linear1_out,
                       const pir::Value& dropout1_out,
                       const paddle::optional<pir::Value>& dropout2_out,
                       const paddle::optional<pir::Value>& ln1_scale,
                       const paddle::optional<pir::Value>& ln1_bias,
                       const paddle::optional<pir::Value>& ln1_out,
                       const paddle::optional<pir::Value>& ln1_mean,
                       const paddle::optional<pir::Value>& ln1_variance,
                       const paddle::optional<pir::Value>& ln2_scale,
                       const paddle::optional<pir::Value>& ln2_bias,
                       const paddle::optional<pir::Value>& ln2_mean,
                       const paddle::optional<pir::Value>& ln2_variance,
                       const paddle::optional<pir::Value>& linear2_bias,
                       bool pre_layer_norm,
                       float ln1_epsilon,
                       float ln2_epsilon,
                       const std::string& act_method,
                       float dropout1_prob,
                       float dropout2_prob,
                       const std::string& dropout1_implementation,
                       const std::string& dropout2_implementation,
                       bool is_test,
                       bool dropout1_fix_seed,
                       bool dropout2_fix_seed,
                       int dropout1_seed_val,
                       int dropout2_seed_val,
                       bool add_residual,
                       int ring_id);

pir::OpResult fused_softmax_mask_upper_triangle_grad(
    const pir::Value& Out, const pir::Value& Out_grad);

pir::OpResult hardswish_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult hardswish_grad_(const pir::Value& x, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> hsigmoid_loss_grad(
    const pir::Value& x,
    const pir::Value& w,
    const pir::Value& label,
    const paddle::optional<pir::Value>& path,
    const paddle::optional<pir::Value>& code,
    const paddle::optional<pir::Value>& bias,
    const pir::Value& pre_out,
    const pir::Value& out_grad,
    int num_classes,
    bool is_sparse);

pir::OpResult logsumexp_grad(const pir::Value& x,
                             const pir::Value& out,
                             const pir::Value& out_grad,
                             const std::vector<int64_t>& axis,
                             bool keepdim,
                             bool reduce_all);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> matmul_double_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    bool transpose_x = false,
    bool transpose_y = false);

std::tuple<pir::OpResult, pir::OpResult> matmul_grad(const pir::Value& x,
                                                     const pir::Value& y,
                                                     const pir::Value& out_grad,
                                                     bool transpose_x = false,
                                                     bool transpose_y = false);

pir::OpResult max_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       const std::vector<int64_t>& axis = {},
                       bool keepdim = false,
                       bool reduce_all = false);

pir::OpResult max_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       pir::Value axis,
                       bool keepdim = false,
                       bool reduce_all = false);

pir::OpResult max_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       std::vector<pir::Value> axis,
                       bool keepdim = false,
                       bool reduce_all = false);

std::tuple<pir::OpResult, pir::OpResult> maximum_grad(
    const pir::Value& x, const pir::Value& y, const pir::Value& out_grad);

pir::OpResult mean_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& axis = {},
                        bool keepdim = false,
                        bool reduce_all = false);

pir::OpResult min_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       const std::vector<int64_t>& axis = {},
                       bool keepdim = false,
                       bool reduce_all = false);

pir::OpResult min_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       pir::Value axis,
                       bool keepdim = false,
                       bool reduce_all = false);

pir::OpResult min_grad(const pir::Value& x,
                       const pir::Value& out,
                       const pir::Value& out_grad,
                       std::vector<pir::Value> axis,
                       bool keepdim = false,
                       bool reduce_all = false);

std::tuple<pir::OpResult, pir::OpResult> minimum_grad(
    const pir::Value& x, const pir::Value& y, const pir::Value& out_grad);

pir::OpResult mish_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        float lambda);

pir::OpResult mish_grad_(const pir::Value& x,
                         const pir::Value& out_grad,
                         float lambda);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> multiply_double_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    int axis = -1);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> multiply_double_grad_(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    int axis = -1);

std::tuple<pir::OpResult, pir::OpResult> multiply_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out_grad,
    int axis = -1);

std::tuple<pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult,
           pir::OpResult>
multiply_triple_grad(const pir::Value& x,
                     const pir::Value& y,
                     const pir::Value& fwd_grad_out,
                     const paddle::optional<pir::Value>& fwd_grad_grad_x,
                     const paddle::optional<pir::Value>& fwd_grad_grad_y,
                     const paddle::optional<pir::Value>& grad_x_grad,
                     const paddle::optional<pir::Value>& grad_y_grad,
                     const paddle::optional<pir::Value>& grad_grad_out_grad,
                     int axis = -1);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> nce_grad(
    const pir::Value& input,
    const pir::Value& label,
    const paddle::optional<pir::Value>& bias,
    const pir::Value& weight,
    const pir::Value& sample_logits,
    const pir::Value& sample_labels,
    const paddle::optional<pir::Value>& sample_weight,
    const paddle::optional<pir::Value>& custom_dist_probs,
    const paddle::optional<pir::Value>& custom_dist_alias,
    const paddle::optional<pir::Value>& custom_dist_alias_probs,
    const pir::Value& cost_grad,
    int num_total_classes,
    const std::vector<int>& custom_neg_classes = {},
    int num_neg_samples = 10,
    int sampler = 0,
    int seed = 0,
    bool is_sparse = false,
    bool remote_prefetch = false,
    bool is_test = false);

pir::OpResult norm_grad(const pir::Value& x,
                        const pir::Value& norm,
                        const pir::Value& out_grad,
                        int axis,
                        float epsilon,
                        bool is_test);

pir::OpResult pad_double_grad(const pir::Value& grad_x_grad,
                              const std::vector<int>& paddings,
                              float pad_value);

pir::OpResult pad_double_grad(const pir::Value& grad_x_grad,
                              pir::Value pad_value,
                              const std::vector<int>& paddings);

pir::OpResult pad_grad(const pir::Value& x,
                       const pir::Value& out_grad,
                       const std::vector<int>& paddings,
                       float pad_value);

pir::OpResult pad_grad(const pir::Value& x,
                       const pir::Value& out_grad,
                       pir::Value pad_value,
                       const std::vector<int>& paddings);

pir::OpResult pool2d_double_grad(const pir::Value& x,
                                 const pir::Value& grad_x_grad,
                                 const std::vector<int64_t>& kernel_size,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 bool ceil_mode,
                                 bool exclusive,
                                 const std::string& data_format,
                                 const std::string& pooling_type,
                                 bool global_pooling,
                                 bool adaptive,
                                 const std::string& padding_algorithm);

pir::OpResult pool2d_double_grad(const pir::Value& x,
                                 const pir::Value& grad_x_grad,
                                 pir::Value kernel_size,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 bool ceil_mode,
                                 bool exclusive,
                                 const std::string& data_format,
                                 const std::string& pooling_type,
                                 bool global_pooling,
                                 bool adaptive,
                                 const std::string& padding_algorithm);

pir::OpResult pool2d_double_grad(const pir::Value& x,
                                 const pir::Value& grad_x_grad,
                                 std::vector<pir::Value> kernel_size,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 bool ceil_mode,
                                 bool exclusive,
                                 const std::string& data_format,
                                 const std::string& pooling_type,
                                 bool global_pooling,
                                 bool adaptive,
                                 const std::string& padding_algorithm);

pir::OpResult pool2d_grad(const pir::Value& x,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          const std::vector<int64_t>& kernel_size,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          bool ceil_mode,
                          bool exclusive,
                          const std::string& data_format,
                          const std::string& pooling_type,
                          bool global_pooling,
                          bool adaptive,
                          const std::string& padding_algorithm);

pir::OpResult pool2d_grad(const pir::Value& x,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          pir::Value kernel_size,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          bool ceil_mode,
                          bool exclusive,
                          const std::string& data_format,
                          const std::string& pooling_type,
                          bool global_pooling,
                          bool adaptive,
                          const std::string& padding_algorithm);

pir::OpResult pool2d_grad(const pir::Value& x,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          std::vector<pir::Value> kernel_size,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          bool ceil_mode,
                          bool exclusive,
                          const std::string& data_format,
                          const std::string& pooling_type,
                          bool global_pooling,
                          bool adaptive,
                          const std::string& padding_algorithm);

pir::OpResult pool3d_grad(const pir::Value& x,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          const std::vector<int>& kernel_size,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          bool ceil_mode,
                          bool exclusive,
                          const std::string& data_format,
                          const std::string& pooling_type,
                          bool global_pooling,
                          bool adaptive,
                          const std::string& padding_algorithm);

pir::OpResult prod_grad(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& dims,
                        bool keep_dim,
                        bool reduce_all);

pir::OpResult prod_grad(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad,
                        pir::Value dims,
                        bool keep_dim,
                        bool reduce_all);

pir::OpResult prod_grad(const pir::Value& x,
                        const pir::Value& out,
                        const pir::Value& out_grad,
                        std::vector<pir::Value> dims,
                        bool keep_dim,
                        bool reduce_all);

pir::OpResult repeat_interleave_grad(const pir::Value& x,
                                     const pir::Value& out_grad,
                                     int repeats,
                                     int axis);

pir::OpResult repeat_interleave_with_tensor_index_grad(
    const pir::Value& x,
    const pir::Value& repeats,
    const pir::Value& out_grad,
    int axis);

pir::OpResult reshape_double_grad(const pir::Value& grad_out,
                                  const pir::Value& grad_x_grad);

pir::OpResult reshape_double_grad_(const pir::Value& grad_out,
                                   const pir::Value& grad_x_grad);

pir::OpResult reshape_grad(const pir::Value& xshape,
                           const pir::Value& out_grad);

pir::OpResult reshape_grad_(const pir::Value& xshape,
                            const pir::Value& out_grad);

std::
    tuple<pir::OpResult, std::vector<pir::OpResult>, std::vector<pir::OpResult>>
    rnn_grad(const pir::Value& x,
             const std::vector<pir::Value>& pre_state,
             const std::vector<pir::Value>& weight_list,
             const paddle::optional<pir::Value>& sequence_length,
             const pir::Value& out,
             const pir::Value& dropout_state_out,
             const pir::Value& reserve,
             const pir::Value& out_grad,
             const std::vector<pir::Value>& state_grad,
             float dropout_prob,
             bool is_bidirec,
             int input_size,
             int hidden_size,
             int num_layers,
             const std::string& mode,
             int seed,
             bool is_test);

std::tuple<pir::OpResult, pir::OpResult> row_conv_grad(
    const pir::Value& x, const pir::Value& filter, const pir::Value& out_grad);

pir::OpResult rrelu_grad(const pir::Value& x,
                         const pir::Value& noise,
                         const pir::Value& out_grad);

pir::OpResult set_value_grad(const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult> set_value_with_tensor_grad(
    const pir::Value& values,
    const pir::Value& out_grad,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& ends,
    const std::vector<int64_t>& steps,
    const std::vector<int64_t>& axes,
    const std::vector<int64_t>& decrease_axes,
    const std::vector<int64_t>& none_axes);

std::tuple<pir::OpResult, pir::OpResult> set_value_with_tensor_grad(
    const pir::Value& values,
    const pir::Value& out_grad,
    pir::Value starts,
    pir::Value ends,
    pir::Value steps,
    const std::vector<int64_t>& axes,
    const std::vector<int64_t>& decrease_axes,
    const std::vector<int64_t>& none_axes);

std::tuple<pir::OpResult, pir::OpResult> set_value_with_tensor_grad(
    const pir::Value& values,
    const pir::Value& out_grad,
    std::vector<pir::Value> starts,
    std::vector<pir::Value> ends,
    std::vector<pir::Value> steps,
    const std::vector<int64_t>& axes,
    const std::vector<int64_t>& decrease_axes,
    const std::vector<int64_t>& none_axes);

pir::OpResult slice_grad(const pir::Value& input,
                         const pir::Value& out_grad,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& starts,
                         const std::vector<int64_t>& ends,
                         const std::vector<int64_t>& infer_flags,
                         const std::vector<int64_t>& decrease_axis);

pir::OpResult slice_grad(const pir::Value& input,
                         const pir::Value& out_grad,
                         pir::Value starts,
                         pir::Value ends,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& infer_flags,
                         const std::vector<int64_t>& decrease_axis);

pir::OpResult slice_grad(const pir::Value& input,
                         const pir::Value& out_grad,
                         std::vector<pir::Value> starts,
                         std::vector<pir::Value> ends,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& infer_flags,
                         const std::vector<int64_t>& decrease_axis);

pir::OpResult soft_relu_grad(const pir::Value& out,
                             const pir::Value& out_grad,
                             float threshold);

pir::OpResult softmax_grad(const pir::Value& out,
                           const pir::Value& out_grad,
                           int axis);

pir::OpResult split_grad(const std::vector<pir::Value>& out_grad,
                         int axis = -1);

pir::OpResult split_grad(const std::vector<pir::Value>& out_grad,
                         pir::Value axis);

pir::OpResult strided_slice_grad(const pir::Value& x,
                                 const pir::Value& out_grad,
                                 const std::vector<int>& axes,
                                 const std::vector<int64_t>& starts,
                                 const std::vector<int64_t>& ends,
                                 const std::vector<int64_t>& strides);

pir::OpResult strided_slice_grad(const pir::Value& x,
                                 const pir::Value& out_grad,
                                 pir::Value starts,
                                 pir::Value ends,
                                 pir::Value strides,
                                 const std::vector<int>& axes);

pir::OpResult strided_slice_grad(const pir::Value& x,
                                 const pir::Value& out_grad,
                                 std::vector<pir::Value> starts,
                                 std::vector<pir::Value> ends,
                                 std::vector<pir::Value> strides,
                                 const std::vector<int>& axes);

pir::OpResult subtract_double_grad(
    const pir::Value& y,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    int axis = -1);

pir::OpResult subtract_double_grad_(
    const pir::Value& y,
    const pir::Value& grad_out,
    const paddle::optional<pir::Value>& grad_x_grad,
    const paddle::optional<pir::Value>& grad_y_grad,
    int axis = -1);

std::tuple<pir::OpResult, pir::OpResult> subtract_grad(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out_grad,
    int axis = -1);

std::tuple<pir::OpResult, pir::OpResult> subtract_grad_(
    const pir::Value& x,
    const pir::Value& y,
    const pir::Value& out_grad,
    int axis = -1);

pir::OpResult sum_grad(const pir::Value& x,
                       const pir::Value& out_grad,
                       const std::vector<int64_t>& axis,
                       bool keepdim,
                       bool reduce_all = false);

pir::OpResult sum_grad(const pir::Value& x,
                       const pir::Value& out_grad,
                       pir::Value axis,
                       bool keepdim,
                       bool reduce_all = false);

pir::OpResult sum_grad(const pir::Value& x,
                       const pir::Value& out_grad,
                       std::vector<pir::Value> axis,
                       bool keepdim,
                       bool reduce_all = false);

pir::OpResult swish_grad(const pir::Value& x, const pir::Value& out_grad);

pir::OpResult swish_grad_(const pir::Value& x, const pir::Value& out_grad);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult> sync_batch_norm_grad(
    const pir::Value& x,
    const pir::Value& scale,
    const pir::Value& bias,
    const pir::Value& saved_mean,
    const pir::Value& saved_variance,
    const paddle::optional<pir::Value>& reserve_space,
    const pir::Value& out_grad,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics);

pir::OpResult tile_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        const std::vector<int64_t>& repeat_times);

pir::OpResult tile_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        pir::Value repeat_times);

pir::OpResult tile_grad(const pir::Value& x,
                        const pir::Value& out_grad,
                        std::vector<pir::Value> repeat_times);

pir::OpResult trans_layout_grad(const pir::Value& x,
                                const pir::Value& out_grad,
                                const std::vector<int>& perm);

pir::OpResult transpose_grad(const pir::Value& out_grad,
                             const std::vector<int>& perm);

pir::OpResult tril_grad(const pir::Value& out_grad, int diagonal);

pir::OpResult triu_grad(const pir::Value& out_grad, int diagonal);

pir::OpResult disable_check_model_nan_inf_grad(const pir::Value& out_grad,
                                               int unsetflag = 1);

pir::OpResult enable_check_model_nan_inf_grad(const pir::Value& out_grad,
                                              int unsetflag = 0);

std::tuple<pir::OpResult, pir::OpResult> fused_elemwise_add_activation_grad(
    const paddle::optional<pir::Value>& x,
    const pir::Value& y,
    const pir::Value& out,
    const paddle::optional<pir::Value>& intermediate_out,
    const pir::Value& out_grad,
    const std::vector<std::string>& functor_list,
    float scale = 0.0,
    int axis = -1,
    bool save_intermediate_out = false);

pir::OpResult shuffle_batch_grad(const pir::Value& shuffle_idx,
                                 const pir::Value& out_grad,
                                 int startup_seed = 0);

pir::OpResult unpool_grad(const pir::Value& x,
                          const pir::Value& indices,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          const std::vector<int>& ksize,
                          const std::vector<int>& strides,
                          const std::vector<int>& padding,
                          const std::vector<int64_t>& output_size,
                          const std::string& data_format);

pir::OpResult unpool_grad(const pir::Value& x,
                          const pir::Value& indices,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          pir::Value output_size,
                          const std::vector<int>& ksize,
                          const std::vector<int>& strides,
                          const std::vector<int>& padding,
                          const std::string& data_format);

pir::OpResult unpool_grad(const pir::Value& x,
                          const pir::Value& indices,
                          const pir::Value& out,
                          const pir::Value& out_grad,
                          std::vector<pir::Value> output_size,
                          const std::vector<int>& ksize,
                          const std::vector<int>& strides,
                          const std::vector<int>& padding,
                          const std::string& data_format);

std::tuple<pir::OpResult, pir::OpResult, pir::OpResult>
match_matrix_tensor_grad(const pir::Value& x,
                         const pir::Value& y,
                         const pir::Value& w,
                         const pir::Value& tmp,
                         const pir::Value& out_grad,
                         int dim_t = 1);

pir::OpResult arange(float start,
                     float end,
                     float step,
                     phi::DataType dtype = phi::DataType::FLOAT64,
                     const Place& place = phi::CPUPlace());

pir::OpResult arange(pir::Value start,
                     pir::Value end,
                     pir::Value step,
                     phi::DataType dtype = phi::DataType::FLOAT64,
                     const Place& place = phi::CPUPlace());

pir::OpResult sequence_mask(const pir::Value& x, int max_len, int out_dtype);

pir::OpResult sequence_mask(const pir::Value& x,
                            pir::Value max_len,
                            int out_dtype);

}  // namespace dialect

}  // namespace paddle
