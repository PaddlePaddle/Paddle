#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {


PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor> flash_attn_unpadded_intermediate(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, const paddle::optional<Tensor>& fixed_seed_offset, const paddle::optional<Tensor>& attn_mask, int64_t max_seqlen_q, int64_t max_seqlen_k, float scale, float dropout = 0.0, bool causal = false, bool return_softmax = false, bool is_test = false, const std::string& rng_name = "");

PADDLE_API std::tuple<Tensor, Tensor> flatten_intermediate(const Tensor& x, int start_axis = 1, int stop_axis = 1);

PADDLE_API std::tuple<Tensor&, Tensor> flatten_intermediate_(Tensor& x, int start_axis = 1, int stop_axis = 1);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> group_norm_intermediate(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon = 1e-5, int groups = -1, const std::string& data_format = "NCHW");

PADDLE_API std::tuple<Tensor, Tensor> huber_loss_intermediate(const Tensor& input, const Tensor& label, float delta);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> instance_norm_intermediate(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon = 1e-5);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> layer_norm_intermediate(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon = 1e-5, int begin_norm_axis = 1);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> rms_norm_intermediate(const Tensor& x, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& residual, const Tensor& norm_weight, const paddle::optional<Tensor>& norm_bias, float epsilon, int begin_norm_axis, float quant_scale, int quant_round_type, float quant_max_bound, float quant_min_bound);

PADDLE_API std::tuple<Tensor, Tensor> roi_pool_intermediate(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height = 1, int pooled_width = 1, float spatial_scale = 1.0);

PADDLE_API std::tuple<Tensor, Tensor> segment_pool_intermediate(const Tensor& x, const Tensor& segment_ids, const std::string& pooltype = "SUM");

PADDLE_API std::tuple<Tensor, Tensor> send_u_recv_intermediate(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const std::string& reduce_op = "SUM", const IntArray& out_size = {0});

PADDLE_API std::tuple<Tensor, Tensor> send_ue_recv_intermediate(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const std::string& message_op = "ADD", const std::string& reduce_op = "SUM", const IntArray& out_size = {0});

PADDLE_API std::tuple<Tensor, Tensor> squeeze_intermediate(const Tensor& x, const IntArray& axis = {});

PADDLE_API std::tuple<Tensor&, Tensor> squeeze_intermediate_(Tensor& x, const IntArray& axis = {});

PADDLE_API std::tuple<Tensor, Tensor> unsqueeze_intermediate(const Tensor& x, const IntArray& axis = {});

PADDLE_API std::tuple<Tensor&, Tensor> unsqueeze_intermediate_(Tensor& x, const IntArray& axis = {});

PADDLE_API std::tuple<Tensor, Tensor> warpctc_intermediate(const Tensor& logits, const Tensor& label, const paddle::optional<Tensor>& logits_length, const paddle::optional<Tensor>& labels_length, int blank = 0, bool norm_by_times = false);

PADDLE_API std::tuple<Tensor, Tensor> warprnnt_intermediate(const Tensor& input, const Tensor& label, const Tensor& input_lengths, const Tensor& label_lengths, int blank = 0, float fastemit_lambda = 0.0);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> yolo_loss_intermediate(const Tensor& x, const Tensor& gt_box, const Tensor& gt_label, const paddle::optional<Tensor>& gt_score, const std::vector<int>& anchors = {}, const std::vector<int>& anchor_mask = {}, int class_num = 1, float ignore_thresh = 0.7, int downsample_ratio = 32, bool use_label_smooth = true, float scale_x_y = 1.0);

PADDLE_API std::tuple<Tensor, Tensor> dropout_intermediate(const Tensor& x, const paddle::optional<Tensor>& seed_tensor, const Scalar& p, bool is_test, const std::string& mode, int seed, bool fix_seed);

PADDLE_API std::tuple<Tensor, Tensor> reshape_intermediate(const Tensor& x, const IntArray& shape);

PADDLE_API std::tuple<Tensor&, Tensor> reshape_intermediate_(Tensor& x, const IntArray& shape);

PADDLE_API std::tuple<Tensor, Tensor, std::vector<Tensor>, Tensor> rnn_intermediate(const Tensor& x, const std::vector<Tensor>& pre_state, const std::vector<Tensor>& weight_list, const paddle::optional<Tensor>& sequence_length, const Tensor& dropout_state_in, float dropout_prob = 0.0, bool is_bidirec = false, int input_size = 10, int hidden_size = 100, int num_layers = 1, const std::string& mode = "RNN_TANH", int seed = 0, bool is_test = false);

PADDLE_API std::tuple<Tensor, Tensor> rrelu_intermediate(const Tensor& x, float lower, float upper, bool is_test);

namespace sparse {

// out, rulebook, counter

PADDLE_API std::tuple<Tensor, Tensor, Tensor> conv3d_intermediate(const Tensor& x, const Tensor& kernel, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides, int groups, bool subm, const std::string& key = "");


// out, softmax

PADDLE_API std::tuple<Tensor, Tensor> fused_attention_intermediate(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& sparse_mask, const paddle::optional<Tensor>& key_padding_mask, const paddle::optional<Tensor>& attn_mask);


// out, rulebook, counter

PADDLE_API std::tuple<Tensor, Tensor, Tensor> maxpool_intermediate(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides);


}  // namespace sparse


}  // namespace experimental
}  // namespace paddle
