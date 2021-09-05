/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/beam_search.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace framework {
class LoDTensor;
class Tensor;
}  // namespace framework
namespace platform {
class NPUDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class BeamSearchFunctor<platform::NPUDeviceContext, T> {
 public:
  template <typename U>
  void PrintTensor(const framework::Tensor& src,
                   const platform::NPUDeviceContext& ctx) {
    std::vector<U> vec(src.numel());
    TensorToVector(src, ctx, &vec);
    for (int i = 0; i < static_cast<int>(vec.size()); ++i) {
      VLOG(3) << "vec[" << i << "] : " << vec[i];
    }
  }

  void operator()(const platform::NPUDeviceContext& ctx,
                  const framework::LoDTensor* pre_ids,
                  const framework::LoDTensor* pre_scores,
                  const framework::LoDTensor* ids,
                  const framework::LoDTensor* scores,
                  framework::LoDTensor* selected_ids,
                  framework::LoDTensor* selected_scores,
                  framework::Tensor* parent_idx, size_t level, size_t beam_size,
                  int end_id, bool is_accumulated) {
    auto abs_lod = framework::ToAbsOffset(scores->lod());
    auto& high_level = abs_lod[level];

    // construct seq_id to calculate parent_idx
    int64_t num_seqs = 0;
    std::vector<int> vector_seq_ids;
    for (int64_t seq_id = 0;
         seq_id < static_cast<int64_t>(scores->NumElements(level)); ++seq_id) {
      int64_t seq_offset_start = abs_lod[level][seq_id];
      int64_t seq_offset_end = abs_lod[level][seq_id + 1];
      int64_t seq_offset_interval = seq_offset_end - seq_offset_start;
      if (seq_offset_interval > 0) {
        num_seqs++;
      }
      for (int64_t offset = seq_offset_start; offset < seq_offset_end;
           ++offset) {
        vector_seq_ids.push_back(static_cast<int>(seq_offset_start));
      }
    }

    VLOG(3) << "yoki: scores";
    PrintTensor<float>(*scores, ctx);

    VLOG(3) << "yoki: num_seqs: " << num_seqs;
    for (int i = 0; i < static_cast<int>(vector_seq_ids.size()); ++i) {
      VLOG(3) << "vec[" << i << "] : " << vector_seq_ids[i];
    }

    // size of the first beam is 1, others are equal to beam_size
    int64_t real_beam_size = static_cast<int64_t>(scores->dims()[0] / num_seqs);
    // K
    int64_t seq_width = 1;
    for (int i = 1; i < scores->dims().size(); i++) {
      seq_width *= scores->dims()[i];
    }

    auto place = ctx.GetPlace();
    auto stream = ctx.stream();

    int64_t total_length = num_seqs * beam_size;
    int64_t batch_size = static_cast<int64_t>(scores->dims()[0]);

    // Step1: Define Tensors and Preprocess the situation that pre_id == end_id

    // cast ids and pre_ids from int to float32
    Tensor ids_int32(framework::proto::VarType::INT32);
    if (ids->type() != framework::proto::VarType::INT32) {
      ids_int32.Resize(ids->dims());
      ids_int32.mutable_data<int>(ctx.GetPlace());
      auto dst_dtype_ids_int32 = ConvertToNpuDtype(ids_int32.type());
      const auto& runner_ids_int32 =
          NpuOpRunner("Cast", {*ids}, {ids_int32},
                      {{"dst_type", static_cast<int>(dst_dtype_ids_int32)}});
      runner_ids_int32.Run(stream);
    } else {
      ids_int32.ShareDataWith(*ids);
    }

    Tensor pre_ids_int32(framework::proto::VarType::INT32);
    if (pre_ids->type() != framework::proto::VarType::INT32) {
      pre_ids_int32.Resize(pre_ids->dims());
      pre_ids_int32.mutable_data<int>(ctx.GetPlace());
      auto dst_dtype_pre_ids_int32 = ConvertToNpuDtype(pre_ids_int32.type());
      const auto& runner_pre_ids_int32 = NpuOpRunner(
          "Cast", {*pre_ids}, {pre_ids_int32},
          {{"dst_type", static_cast<int>(dst_dtype_pre_ids_int32)}});
      runner_pre_ids_int32.Run(stream);
    } else {
      pre_ids_int32.ShareDataWith(*pre_ids);
    }

    Tensor expand_pre_ids(pre_ids_int32.type());
    expand_pre_ids.Resize(framework::make_ddim({batch_size, seq_width}));
    expand_pre_ids.mutable_data<int>(place);
    const auto& runner_tile_pre_ids =
        NpuOpRunner("TileWithAxis", {pre_ids_int32}, {expand_pre_ids},
                    {{"axis", 1}, {"tiles", seq_width}});
    runner_tile_pre_ids.Run(stream);
    expand_pre_ids.Resize(ids_int32.dims());

    Tensor expand_pre_scores(pre_scores->type());
    expand_pre_scores.Resize(framework::make_ddim({batch_size, seq_width}));
    expand_pre_scores.mutable_data<float>(place);
    const auto& runner_tile_pre_scores =
        NpuOpRunner("TileWithAxis", {*pre_scores}, {expand_pre_scores},
                    {{"axis", 1}, {"tiles", seq_width}});
    runner_tile_pre_scores.Run(stream);
    expand_pre_scores.Resize(scores->dims());

    // End_id Tensors
    Tensor end_id_tmp_tensor(framework::proto::VarType::INT32);
    end_id_tmp_tensor.mutable_data<int>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<int>(&end_id_tmp_tensor, end_id);

    Tensor end_id_tensors(ids_int32.type());
    end_id_tensors.mutable_data<int>(ids_int32.dims(), place);
    const auto& runner_fill_end_id =
        NpuOpRunner("FillD", {end_id_tmp_tensor}, {end_id_tensors},
                    {{"dims", framework::vectorize(ids_int32.dims())}});
    runner_fill_end_id.Run(stream);

    // whether expand_pre_ids == end_ids?
    Tensor equal_end_ids(framework::proto::VarType::BOOL);
    equal_end_ids.mutable_data<bool>(ids_int32.dims(), place);
    const auto& runner_equal_end_ids = NpuOpRunner(
        "Equal", {expand_pre_ids, end_id_tensors}, {equal_end_ids}, {});
    runner_equal_end_ids.Run(stream);

    // construct a Tensor with dimension ids->dims():
    // [[False, True, True, True, ...],
    //  [False, True, True, True, ...],
    //  ...]
    Tensor false_tmp_tensor(framework::proto::VarType::INT32);
    false_tmp_tensor.mutable_data<int>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<int>(&false_tmp_tensor, static_cast<int>(false));

    Tensor first_pos_false_tensors(framework::proto::VarType::INT32);
    first_pos_false_tensors.Resize(framework::make_ddim({batch_size, 1}));
    first_pos_false_tensors.mutable_data<int>(place);
    std::vector<int64_t> fill_dims = {batch_size, 1};
    framework::NPUAttributeMap fill_attr = {{"dims", fill_dims}};
    const auto& runner_fill_false_tensors = NpuOpRunner(
        "FillD", {false_tmp_tensor}, {first_pos_false_tensors}, fill_attr);
    runner_fill_false_tensors.Run(stream);

    Tensor pos_tensors(framework::proto::VarType::INT32);
    if (seq_width > 1) {
      pos_tensors.Resize(framework::make_ddim({batch_size, seq_width}));
      pos_tensors.mutable_data<int>(place);

      Tensor true_tmp_tensor(framework::proto::VarType::INT32);
      true_tmp_tensor.mutable_data<int>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<int>(&true_tmp_tensor, static_cast<int>(true));

      Tensor second_pos_true_tensors(framework::proto::VarType::INT32);
      second_pos_true_tensors.Resize(
          framework::make_ddim({batch_size, seq_width - 1}));
      second_pos_true_tensors.mutable_data<int>(place);
      std::vector<int64_t> fill_dims2 = {batch_size, seq_width - 1};
      framework::NPUAttributeMap fill_attr2 = {{"dims", fill_dims2}};
      const auto& runner_fill_true_tensors = NpuOpRunner(
          "FillD", {true_tmp_tensor}, {second_pos_true_tensors}, fill_attr2);
      runner_fill_true_tensors.Run(stream);

      std::vector<framework::Tensor> concat_inputs = {first_pos_false_tensors,
                                                      second_pos_true_tensors};
      std::vector<std::string> concat_names = {"x0", "x1"};
      NpuOpRunner runner_concat_false_true{"ConcatD",
                                           {concat_inputs},
                                           {pos_tensors},
                                           {{"concat_dim", 1}, {"N", 2}}};
      runner_concat_false_true.AddInputNames(concat_names);
      runner_concat_false_true.Run(stream);
      pos_tensors.Resize(ids_int32.dims());
    } else {
      pos_tensors.ShareDataWith(first_pos_false_tensors);
    }

    Tensor cast_pos_tensors_bool(framework::proto::VarType::BOOL);
    cast_pos_tensors_bool.Resize(pos_tensors.dims());
    cast_pos_tensors_bool.mutable_data<bool>(ctx.GetPlace());
    auto dst_dtype = ConvertToNpuDtype(cast_pos_tensors_bool.type());
    const auto& runner_cast_pos_tensors =
        NpuOpRunner("Cast", {pos_tensors}, {cast_pos_tensors_bool},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_pos_tensors.Run(stream);

    // if pre_ids == end_ids, save only one score, and others become -inf
    // construct pre_ids == end_ids and save only one score
    Tensor save_one_end_score(framework::proto::VarType::BOOL);
    save_one_end_score.mutable_data<bool>(ids_int32.dims(), place);
    const auto& runner_logical_and =
        NpuOpRunner("LogicalAnd", {equal_end_ids, cast_pos_tensors_bool},
                    {save_one_end_score}, {});
    runner_logical_and.Run(stream);

    // if save_one_end_score is True, set score to -inf
    // define -Inf Tensors
    Tensor ninf_tmp_tensor(scores->type());
    ninf_tmp_tensor.mutable_data<float>({1}, ctx.GetPlace());
    float ninf_value =
        static_cast<float>(-std::numeric_limits<float>::infinity());
    FillNpuTensorWithConstant<float>(&ninf_tmp_tensor, ninf_value);

    Tensor ninf_tensors(scores->type());
    ninf_tensors.mutable_data<float>(scores->dims(), place);
    const auto& runner_fill_ninf =
        NpuOpRunner("FillD", {ninf_tmp_tensor}, {ninf_tensors},
                    {{"dims", framework::vectorize(scores->dims())}});
    runner_fill_ninf.Run(stream);

    // Step2: calculate topk scores

    // get scores used in topk op
    Tensor tmp_scores(scores->type());
    tmp_scores.mutable_data<float>(scores->dims(), place);
    if (!is_accumulated) {
      // if pre_id == end_id, cal_scores = pre_score, and id = end_id
      // else, cal_score = pre_score + log(score)

      // calculate log(scores)
      Tensor log_scores(scores->type());
      log_scores.mutable_data<float>(scores->dims(), place);

      Tensor one(scores->type());
      one.mutable_data<float>(scores->dims(), place);
      const auto& runner_one = NpuOpRunner("OnesLike", {*scores}, {one}, {});
      runner_one.Run(stream);

      Tensor sub(scores->type());
      sub.mutable_data<float>(scores->dims(), place);
      const auto& runner_sub = NpuOpRunner("Sub", {*scores, one}, {sub}, {});
      runner_sub.Run(stream);

      const auto& runner_log_scores =
          NpuOpRunner("Log1p", {sub}, {log_scores}, {});
      runner_log_scores.Run(stream);

      // tmp_scores = pre_score + log(scores)
      const auto& runner_add_scores =
          NpuOpRunner("Add", {log_scores, *pre_scores}, {tmp_scores}, {});
      runner_add_scores.Run(stream);

      // if pre_ids == end_ids, use pre_score rather than score
      const auto& runner_select_equal_end_score =
          NpuOpRunner("Select", {equal_end_ids, expand_pre_scores, tmp_scores},
                      {tmp_scores}, {});
      runner_select_equal_end_score.Run(stream);
    } else {
      // if pre_ids == end_ids, use pre_score rather than score
      const auto& runner_select_equal_end_score2 =
          NpuOpRunner("Select", {equal_end_ids, expand_pre_scores, *scores},
                      {tmp_scores}, {});
      runner_select_equal_end_score2.Run(stream);
    }

    // if pre_ids == end_ids, save only one score, and others become -inf
    Tensor cal_scores(scores->type());
    cal_scores.mutable_data<float>(scores->dims(), place);
    const auto& runner_select_inf_score =
        NpuOpRunner("Select", {save_one_end_score, ninf_tensors, tmp_scores},
                    {cal_scores}, {});
    runner_select_inf_score.Run(stream);

    // resize scores from [num_seqs * beam_size, K] to [num_seqs, beam_size * K]
    // real_beam_size = 1 or beam_size
    cal_scores.Resize(
        framework::make_ddim({num_seqs, real_beam_size * seq_width}));

    Tensor topk_scores(scores->type());
    topk_scores.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));
    topk_scores.mutable_data<float>(ctx.GetPlace());

    Tensor tmp_indices(framework::proto::VarType::INT32);
    tmp_indices.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));
    tmp_indices.mutable_data<int>(ctx.GetPlace());

    VLOG(3) << "yoki: cal_scores";
    PrintTensor<float>(cal_scores, ctx);

    // run topk op
    NpuOpRunner runner_topk;
    runner_topk.SetType("TopKV2")
        .AddInput(cal_scores)
        .AddInput(std::vector<int>{static_cast<int>(beam_size)})
        .AddOutput(topk_scores)
        .AddOutput(tmp_indices)
        .AddAttr("sorted", true)
        .AddAttr("dim", -1)
        .AddAttr("largest", true);
    runner_topk.Run(stream);

    VLOG(3) << "yoki: topk_scores";
    PrintTensor<float>(topk_scores, ctx);

    // cast tmp_indices from int to float32 for Sort op
    Tensor cast_tmp_indices(framework::proto::VarType::FP32);
    cast_tmp_indices.Resize(tmp_indices.dims());
    cast_tmp_indices.mutable_data<float>(ctx.GetPlace());
    auto dst_dtype_tmp_indices_fp32 =
        ConvertToNpuDtype(cast_tmp_indices.type());
    const auto& runner_cast_tmp_indices = NpuOpRunner(
        "Cast", {tmp_indices}, {cast_tmp_indices},
        {{"dst_type", static_cast<int>(dst_dtype_tmp_indices_fp32)}});
    runner_cast_tmp_indices.Run(stream);

    // sort tmp_indices
    Tensor sorted_tmp_indices(framework::proto::VarType::FP32);
    sorted_tmp_indices.Resize(tmp_indices.dims());
    sorted_tmp_indices.mutable_data<float>(ctx.GetPlace());
    Tensor sorted_score_indices(framework::proto::VarType::INT32);
    sorted_score_indices.Resize(tmp_indices.dims());
    sorted_score_indices.mutable_data<int>(ctx.GetPlace());
    const auto& runner_sort_tmp_indices = NpuOpRunner(
        "Sort", {cast_tmp_indices}, {sorted_tmp_indices, sorted_score_indices},
        {{"axis", 1}, {"descending", false}});
    runner_sort_tmp_indices.Run(stream);

    // cast sorted_tmp_indices from float32 to int
    Tensor cast_sort_tmp_indices(framework::proto::VarType::INT32);
    cast_sort_tmp_indices.Resize(sorted_tmp_indices.dims());
    cast_sort_tmp_indices.mutable_data<int>(ctx.GetPlace());
    auto dst_dtype_tmp_indices_int32 =
        ConvertToNpuDtype(cast_sort_tmp_indices.type());
    const auto& runner_cast_sort_tmp_indices = NpuOpRunner(
        "Cast", {sorted_tmp_indices}, {cast_sort_tmp_indices},
        {{"dst_type", static_cast<int>(dst_dtype_tmp_indices_int32)}});
    runner_cast_sort_tmp_indices.Run(stream);

    // Step 3: infer selected ids from tmp_indices and ids

    // if pre_ids == end_ids, use pre_ids rather than ids
    Tensor cal_ids(ids_int32.type());
    cal_ids.mutable_data<int>(ids_int32.dims(), place);
    const auto& runner_select_equal_end_id = NpuOpRunner(
        "Select", {equal_end_ids, expand_pre_ids, ids_int32}, {cal_ids}, {});
    runner_select_equal_end_id.Run(stream);

    VLOG(3) << "yoki: cal_ids";
    PrintTensor<int>(cal_ids, ctx);

    // resize ids from [num_seqs * real_beam_size, K] to [num_seqs,
    // real_beam_size * K]
    // real_beam_size = 1 or beam_size
    cal_ids.Resize(
        framework::make_ddim({num_seqs, real_beam_size * seq_width}));

    // construct batch_ids like [[0, 0, 0], [1, 1, 1], ..., [bs-1, bs-1, bs-1]]
    // construct arange(num_seqs*beam_size).reshape((num_seqs, beam_size)) //
    // beam_size
    Tensor batch_ids(framework::proto::VarType::INT32);
    batch_ids.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 1}));
    batch_ids.mutable_data<int>(place);

    std::vector<int> vector_batch_ids;
    for (int i = 0; i < num_seqs * static_cast<int>(beam_size); ++i) {
      vector_batch_ids.push_back(static_cast<int>(i / beam_size));
    }
    framework::TensorFromVector(vector_batch_ids, ctx, &batch_ids);
    batch_ids.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 1}));

    // sort topk_scores to get selected_scores
    // get indices of gather_nd op for calculating selected_scores
    Tensor gather_nd_score_indices(framework::proto::VarType::INT32);
    gather_nd_score_indices.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 2}));
    gather_nd_score_indices.mutable_data<int>(place);

    sorted_score_indices.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 1}));
    std::vector<framework::Tensor> concat_inputs2 = {batch_ids,
                                                     sorted_score_indices};
    std::vector<std::string> concat_names = {"x0", "x1"};
    NpuOpRunner runner_concat_score_indices{"ConcatD",
                                            {concat_inputs2},
                                            {gather_nd_score_indices},
                                            {{"concat_dim", 2}, {"N", 2}}};
    runner_concat_score_indices.AddInputNames(concat_names);
    runner_concat_score_indices.Run(stream);

    VLOG(3) << "yoki: gather_nd_score_indices";
    PrintTensor<int>(gather_nd_score_indices, ctx);

    // use gather_nd to get selected_scores
    Tensor tmp_selected_scores(framework::proto::VarType::FP32);
    tmp_selected_scores.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));
    tmp_selected_scores.mutable_data<float>(ctx.GetPlace());
    const auto& runner_gather_nd_scores =
        NpuOpRunner("GatherNd", {topk_scores, gather_nd_score_indices},
                    {tmp_selected_scores}, {});
    runner_gather_nd_scores.Run(stream);

    // get indices of gather_nd op
    cast_sort_tmp_indices.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 1}));
    Tensor gather_nd_id_indices(framework::proto::VarType::INT32);
    gather_nd_id_indices.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 2}));
    gather_nd_id_indices.mutable_data<int>(place);

    std::vector<framework::Tensor> concat_inputs3 = {batch_ids,
                                                     cast_sort_tmp_indices};
    NpuOpRunner runner_concat_id_indices{"ConcatD",
                                         {concat_inputs3},
                                         {gather_nd_id_indices},
                                         {{"concat_dim", 2}, {"N", 2}}};
    runner_concat_id_indices.AddInputNames(concat_names);
    runner_concat_id_indices.Run(stream);

    // use gather_nd to get selected_ids
    Tensor topk_ids(framework::proto::VarType::INT32);
    topk_ids.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));
    topk_ids.mutable_data<int>(ctx.GetPlace());

    const auto& runner_gather_nd_ids = NpuOpRunner(
        "GatherNd", {cal_ids, gather_nd_id_indices}, {topk_ids}, {});
    runner_gather_nd_ids.Run(stream);

    VLOG(3) << "yoki: gather_nd_id_indices";
    PrintTensor<int>(gather_nd_id_indices, ctx);

    VLOG(3) << "yoki: topk_ids";
    PrintTensor<int>(topk_ids, ctx);

    // Step 4: set lod of output Tensor
    // define Tensor with value `seq_width`
    Tensor seq_width_tensor(framework::proto::VarType::INT32);
    seq_width_tensor.mutable_data<int>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<int>(&seq_width_tensor,
                                   static_cast<int>(seq_width));

    // beam_ids = tmp_indices // seq_width
    Tensor beam_ids(framework::proto::VarType::INT32);
    beam_ids.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));
    beam_ids.mutable_data<int>(ctx.GetPlace());
    cast_sort_tmp_indices.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));

    const auto& runner_div = NpuOpRunner(
        "Div", {cast_sort_tmp_indices, seq_width_tensor}, {beam_ids}, {});
    runner_div.Run(stream);

    VLOG(3) << "yoki: beam_ids";
    PrintTensor<int>(beam_ids, ctx);

    Tensor seq_ids(framework::proto::VarType::INT32);
    seq_ids.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));
    seq_ids.mutable_data<int>(place);
    framework::TensorFromVector(vector_seq_ids, ctx, &seq_ids);
    seq_ids.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));

    VLOG(3) << "yoki: seq_ids";
    PrintTensor<int>(seq_ids, ctx);

    // calculate parent_idx
    Tensor tmp_parent_idx(framework::proto::VarType::INT32);
    tmp_parent_idx.Resize(framework::make_ddim({total_length}));
    tmp_parent_idx.mutable_data<int>(place);
    const auto& runner_add_beam_id =
        NpuOpRunner("Add", {beam_ids, seq_ids}, {tmp_parent_idx}, {});
    runner_add_beam_id.Run(stream);

    VLOG(3) << "yoki: tmp_parent_idx";
    PrintTensor<int>(tmp_parent_idx, ctx);

    // Step 5: PruneEndBeams
    // End_id Tensors
    Tensor end_id_tensors_after_topk(topk_ids.type());
    end_id_tensors_after_topk.mutable_data<int>(topk_ids.dims(), place);
    const auto& runner_fill_end_id_after_topk =
        NpuOpRunner("FillD", {end_id_tmp_tensor}, {end_id_tensors_after_topk},
                    {{"dims", framework::vectorize(topk_ids.dims())}});
    runner_fill_end_id_after_topk.Run(stream);

    // whether topk_ids == end_ids?
    Tensor topk_ids_equal_end_ids(framework::proto::VarType::BOOL);
    topk_ids_equal_end_ids.mutable_data<bool>(topk_ids.dims(), place);
    const auto& runner_topk_ids_equal_end_ids =
        NpuOpRunner("Equal", {topk_ids, end_id_tensors_after_topk},
                    {topk_ids_equal_end_ids}, {});
    runner_topk_ids_equal_end_ids.Run(stream);

    // cast topk_ids_equal_end_ids from bool to float32
    Tensor cast_topk_ids_equal_end_ids(framework::proto::VarType::FP32);
    cast_topk_ids_equal_end_ids.Resize(topk_ids_equal_end_ids.dims());
    cast_topk_ids_equal_end_ids.mutable_data<float>(ctx.GetPlace());
    auto dst_dtype_float32 =
        ConvertToNpuDtype(cast_topk_ids_equal_end_ids.type());
    const auto& runner_cast_topk_ids_equal_end_ids = NpuOpRunner(
        "Cast", {topk_ids_equal_end_ids}, {cast_topk_ids_equal_end_ids},
        {{"dst_type", static_cast<int>(dst_dtype_float32)}});
    runner_cast_topk_ids_equal_end_ids.Run(stream);

    // reduce_max(topk_ids_equal_end_ids) => finish flag
    Tensor reduce_max_equal(cast_topk_ids_equal_end_ids.type());
    reduce_max_equal.Resize(framework::make_ddim({num_seqs}));
    reduce_max_equal.mutable_data<float>(place);
    const auto& runner_reduce_max_equal = NpuOpRunner(
        "ReduceMaxD", {cast_topk_ids_equal_end_ids}, {reduce_max_equal},
        {{"axes", std::vector<int>{1}}, {"keep_dims", false}});
    runner_reduce_max_equal.Run(stream);

    // get 1 - reduce_max_equal => not finish flag
    Tensor one(reduce_max_equal.type());
    one.mutable_data<float>(reduce_max_equal.dims(), place);
    const auto& runner_one =
        NpuOpRunner("OnesLike", {reduce_max_equal}, {one}, {});
    runner_one.Run(stream);

    Tensor one_sub_reduce_max_equal(reduce_max_equal.type());
    one_sub_reduce_max_equal.mutable_data<float>(reduce_max_equal.dims(),
                                                 place);
    const auto& runner_one_sub_reduce_max_equal = NpuOpRunner(
        "Sub", {one, reduce_max_equal}, {one_sub_reduce_max_equal}, {});
    runner_one_sub_reduce_max_equal.Run(stream);

    // reduce_sum one_sub_reduce_max_equal to get new num_seqs
    Tensor new_num_seqs_tensor(one_sub_reduce_max_equal.type());
    new_num_seqs_tensor.Resize(framework::make_ddim({1}));
    new_num_seqs_tensor.mutable_data<float>(place);
    const auto& runner_new_num_seqs_tensor = NpuOpRunner(
        "ReduceSumD", {one_sub_reduce_max_equal}, {new_num_seqs_tensor},
        {{"axes", std::vector<int>{0}}, {"keep_dims", true}});
    runner_new_num_seqs_tensor.Run(stream);

    std::vector<float> vector_new_num_seqs;
    framework::TensorToVector(new_num_seqs_tensor, ctx, &vector_new_num_seqs);
    int64_t new_num_seqs = static_cast<int64_t>(vector_new_num_seqs[0]);

    // expand flag_not_finish to {num_seqs, beam_size}
    one_sub_reduce_max_equal.Resize(framework::make_ddim({num_seqs, 1}));
    Tensor expand_flag_not_finish(one_sub_reduce_max_equal.type());
    expand_flag_not_finish.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));
    expand_flag_not_finish.mutable_data<float>(place);
    const auto& runner_tile_expand_flag_not_finish = NpuOpRunner(
        "TileWithAxis", {one_sub_reduce_max_equal}, {expand_flag_not_finish},
        {{"axis", 1}, {"tiles", static_cast<int64_t>(beam_size)}});
    runner_tile_expand_flag_not_finish.Run(stream);

    // cast expand_flag_not_finish from float32 to bool
    Tensor cast_expand_flag_not_finish(framework::proto::VarType::BOOL);
    cast_expand_flag_not_finish.Resize(expand_flag_not_finish.dims());
    cast_expand_flag_not_finish.mutable_data<bool>(ctx.GetPlace());
    auto dst_dtype_bool = ConvertToNpuDtype(cast_expand_flag_not_finish.type());
    const auto& runner_cast_expand_flag_not_finish = NpuOpRunner(
        "Cast", {expand_flag_not_finish}, {cast_expand_flag_not_finish},
        {{"dst_type", static_cast<int>(dst_dtype_bool)}});
    runner_cast_expand_flag_not_finish.Run(stream);

    VLOG(3) << "yoki: cast_expand_flag_not_finish";
    PrintTensor<bool>(cast_expand_flag_not_finish, ctx);

    cast_expand_flag_not_finish.Resize(
        framework::make_ddim({num_seqs * static_cast<int64_t>(beam_size)}));

    // mask topk_id
    // cast topk_id from int to float32
    Tensor cast_topk_ids(framework::proto::VarType::FP32);
    cast_topk_ids.Resize(topk_ids.dims());
    cast_topk_ids.mutable_data<float>(ctx.GetPlace());
    const auto& runner_cast_topk_ids =
        NpuOpRunner("Cast", {topk_ids}, {cast_topk_ids},
                    {{"dst_type", static_cast<int>(dst_dtype_float32)}});
    runner_cast_topk_ids.Run(stream);

    Tensor mask_topk_ids(cast_topk_ids.type());
    mask_topk_ids.Resize(
        framework::make_ddim({new_num_seqs * static_cast<int64_t>(beam_size)}));
    mask_topk_ids.mutable_data<float>(ctx.GetPlace());

    VLOG(3) << "yoki: cast_topk_ids";
    PrintTensor<float>(cast_topk_ids, ctx);
    VLOG(3) << "yoki: cast_expand_flag_not_finish";
    PrintTensor<bool>(cast_expand_flag_not_finish, ctx);

    cast_topk_ids.Resize(
        framework::make_ddim({num_seqs * static_cast<int64_t>(beam_size)}));
    const auto& runner_mask_topk_ids = NpuOpRunner(
        "MaskedSelect", {cast_topk_ids, cast_expand_flag_not_finish},
        {mask_topk_ids}, {});
    runner_mask_topk_ids.Run(stream);
    mask_topk_ids.Resize(framework::make_ddim(
        {new_num_seqs * static_cast<int64_t>(beam_size), 1}));

    VLOG(3) << "yoki: cast_topk_ids";
    PrintTensor<float>(cast_topk_ids, ctx);
    VLOG(3) << "yoki: mask_topk_ids";
    PrintTensor<float>(mask_topk_ids, ctx);

    // mask tmp_selected_scores
    // // cast tmp_selected_scores from int to float32
    // Tensor cast_selected_scores(framework::proto::VarType::FP32);
    // cast_selected_scores.Resize(tmp_selected_scores.dims());
    // cast_selected_scores.mutable_data<float>(ctx.GetPlace());
    // const auto& runner_cast_selected_scores_fp32 = NpuOpRunner(
    //     "Cast", {tmp_selected_scores}, {cast_selected_scores},
    //     {{"dst_type", static_cast<int>(dst_dtype_float32)}});
    // runner_cast_selected_scores_fp32.Run(stream);

    selected_scores->mutable_data<float>(
        framework::make_ddim(
            {new_num_seqs * static_cast<int64_t>(beam_size), 1}),
        place);
    Tensor mask_selected_scores(tmp_selected_scores.type());
    mask_selected_scores.ShareDataWith(*selected_scores);
    mask_selected_scores.Resize(
        framework::make_ddim({new_num_seqs * static_cast<int64_t>(beam_size)}));
    // mask_selected_scores.mutable_data<float>(ctx.GetPlace());

    tmp_selected_scores.Resize(
        framework::make_ddim({num_seqs * static_cast<int64_t>(beam_size)}));
    const auto& runner_mask_selected_scores = NpuOpRunner(
        "MaskedSelect", {tmp_selected_scores, cast_expand_flag_not_finish},
        {mask_selected_scores}, {});
    runner_mask_selected_scores.Run(stream);
    // mask_selected_scores.Resize(framework::make_ddim({new_num_seqs *
    // static_cast<int64_t>(beam_size), 1}));

    VLOG(3) << "yoki: mask_selected_scores";
    PrintTensor<float>(mask_selected_scores, ctx);

    // mask parent_idx
    // cast parent_idx from int to float32
    Tensor cast_parent_idx(framework::proto::VarType::FP32);
    cast_parent_idx.Resize(tmp_parent_idx.dims());
    cast_parent_idx.mutable_data<float>(ctx.GetPlace());
    const auto& runner_cast_parent_idx_fp32 =
        NpuOpRunner("Cast", {tmp_parent_idx}, {cast_parent_idx},
                    {{"dst_type", static_cast<int>(dst_dtype_float32)}});
    runner_cast_parent_idx_fp32.Run(stream);

    Tensor mask_parent_idx(cast_parent_idx.type());
    mask_parent_idx.Resize(
        framework::make_ddim({new_num_seqs * static_cast<int64_t>(beam_size)}));
    mask_parent_idx.mutable_data<float>(ctx.GetPlace());

    cast_parent_idx.Resize(
        framework::make_ddim({num_seqs * static_cast<int64_t>(beam_size)}));
    const auto& runner_mask_parent_idx = NpuOpRunner(
        "MaskedSelect", {cast_parent_idx, cast_expand_flag_not_finish},
        {mask_parent_idx}, {});
    runner_mask_parent_idx.Run(stream);

    VLOG(3) << "yoki: cast_parent_idx";
    PrintTensor<float>(cast_parent_idx, ctx);
    VLOG(3) << "yoki: mask_parent_idx";
    PrintTensor<float>(mask_parent_idx, ctx);

    // cast topk_id from float to int64_t to get selected_ids
    selected_ids->mutable_data<int64_t>(mask_topk_ids.dims(), place);
    auto dst_dtype_int64 = ConvertToNpuDtype(selected_ids->type());
    const auto& runner_cast_selected_ids =
        NpuOpRunner("Cast", {mask_topk_ids}, {*selected_ids},
                    {{"dst_type", static_cast<int>(dst_dtype_int64)}});
    runner_cast_selected_ids.Run(stream);

    VLOG(3) << "yoki: selected_ids";
    PrintTensor<int64_t>(*selected_ids, ctx);

    // // cast mask_selected_scores from float to int64_t to get selected_scores
    // selected_scores->mutable_data<float>(mask_selected_scores.dims(), place);
    // const auto& runner_cast_selected_scores = NpuOpRunner(
    //     "Cast", {mask_selected_scores}, {*selected_scores},
    //     {{"dst_type", static_cast<int>(dst_dtype_int64)}});
    // runner_cast_selected_scores.Run(stream);

    // VLOG(3) << "yoki: selected_scores";
    // PrintTensor<int64_t>(*selected_scores, ctx);

    // cast mask_parent_idx from float to int64_t to get parent_idx
    parent_idx->mutable_data<int64_t>(mask_parent_idx.dims(), place);
    const auto& runner_cast_parent_idx =
        NpuOpRunner("Cast", {mask_parent_idx}, {*parent_idx},
                    {{"dst_type", static_cast<int>(dst_dtype_int64)}});
    runner_cast_parent_idx.Run(stream);

    VLOG(3) << "yoki: parent_idx";
    PrintTensor<int64_t>(*parent_idx, ctx);

    VLOG(3) << "yoki: parent_idx";
    PrintTensor<int64_t>(*parent_idx, ctx);

    std::vector<int> vector_parent_idx;
    framework::TensorToVector(tmp_parent_idx, ctx, &vector_parent_idx);

    VLOG(3) << "yoki: vector_parent_idx";
    for (int i = 0; i < static_cast<int>(vector_parent_idx.size()); ++i) {
      VLOG(3) << "vec[" << i << "] : " << vector_parent_idx[i];
    }

    // set low level, len(low_level) = high_level[-1]
    std::vector<int> low_level;
    std::vector<int> num_parent_ids(num_seqs * beam_size,
                                    static_cast<int64_t>(0));
    size_t low_level_size = high_level[high_level.size() - 1];
    size_t sum_parent_id = 0;

    // calculate number of every parent_id
    for (size_t i = 0; i < num_seqs * beam_size; ++i) {
      num_parent_ids[vector_parent_idx[i]]++;
    }

    // update low_level
    low_level.push_back(0);
    for (size_t i = 0; i < low_level_size; ++i) {
      sum_parent_id += num_parent_ids[i];
      low_level.push_back(sum_parent_id);
    }

    // fill lod
    framework::LoD lod(2);
    lod[0].assign(high_level.begin(), high_level.end());
    lod[1].assign(low_level.begin(), low_level.end());
    if (!framework::CheckLoD(lod)) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "lod %s is not right in"
          " beam_search, please check your code.",
          framework::LoDToString(lod)));
    }
    selected_ids->set_lod(lod);
    selected_scores->set_lod(lod);
  }
};

template class BeamSearchFunctor<platform::NPUDeviceContext, int>;
template class BeamSearchFunctor<platform::NPUDeviceContext, int64_t>;
template class BeamSearchFunctor<platform::NPUDeviceContext, float>;
template class BeamSearchFunctor<platform::NPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
