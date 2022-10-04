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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#include "paddle/phi/common/data_type.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {}  // namespace framework
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
  void operator()(const platform::NPUDeviceContext& ctx,
                  const framework::LoDTensor* pre_ids,
                  const framework::LoDTensor* pre_scores,
                  const framework::LoDTensor* ids,
                  const framework::LoDTensor* scores,
                  framework::LoDTensor* selected_ids,
                  framework::LoDTensor* selected_scores,
                  phi::DenseTensor* parent_idx,
                  size_t level,
                  size_t beam_size,
                  int end_id,
                  bool is_accumulated) {
    auto abs_lod = framework::ToAbsOffset(scores->lod());
    auto& high_level = abs_lod[level];

    int64_t num_seqs = scores->NumElements(level);
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
    selected_ids->mutable_data<int64_t>(phi::make_ddim({total_length, 1}),
                                        place);
    selected_scores->mutable_data<float>(phi::make_ddim({total_length, 1}),
                                         place);
    parent_idx->mutable_data<int64_t>(phi::make_ddim({total_length}), place);

    // Step1: Define Tensors and Preprocess the situation that pre_id == end_id

    // cast ids and pre_ids from int to float32
    Tensor ids_int32(experimental::DataType::INT32);
    if (framework::TransToProtoVarType(ids->dtype()) !=
        framework::proto::VarType::INT32) {
      ids_int32.Resize(ids->dims());
      ids_int32.mutable_data<int>(ctx.GetPlace());
      auto dst_dtype_ids_int32 =
          ConvertToNpuDtype(framework::TransToProtoVarType(ids_int32.dtype()));
      const auto& runner_ids_int32 =
          NpuOpRunner("Cast",
                      {*ids},
                      {ids_int32},
                      {{"dst_type", static_cast<int>(dst_dtype_ids_int32)}});
      runner_ids_int32.Run(stream);
    } else {
      ids_int32.ShareDataWith(*ids);
    }

    Tensor pre_ids_int32(experimental::DataType::INT32);
    if (framework::TransToProtoVarType(pre_ids->dtype()) !=
        framework::proto::VarType::INT32) {
      pre_ids_int32.Resize(pre_ids->dims());
      pre_ids_int32.mutable_data<int>(ctx.GetPlace());
      auto dst_dtype_pre_ids_int32 = ConvertToNpuDtype(
          framework::TransToProtoVarType(pre_ids_int32.dtype()));
      const auto& runner_pre_ids_int32 = NpuOpRunner(
          "Cast",
          {*pre_ids},
          {pre_ids_int32},
          {{"dst_type", static_cast<int>(dst_dtype_pre_ids_int32)}});
      runner_pre_ids_int32.Run(stream);
    } else {
      pre_ids_int32.ShareDataWith(*pre_ids);
    }

    Tensor expand_pre_ids(pre_ids_int32.dtype());
    expand_pre_ids.Resize(phi::make_ddim({batch_size, seq_width}));
    expand_pre_ids.mutable_data<int>(place);
    const auto& runner_tile_pre_ids =
        NpuOpRunner("TileWithAxis",
                    {pre_ids_int32},
                    {expand_pre_ids},
                    {{"axis", 1}, {"tiles", seq_width}});
    runner_tile_pre_ids.Run(stream);
    expand_pre_ids.Resize(ids_int32.dims());

    Tensor expand_pre_scores(pre_scores->dtype());
    expand_pre_scores.Resize(phi::make_ddim({batch_size, seq_width}));
    expand_pre_scores.mutable_data<float>(place);
    const auto& runner_tile_pre_scores =
        NpuOpRunner("TileWithAxis",
                    {*pre_scores},
                    {expand_pre_scores},
                    {{"axis", 1}, {"tiles", seq_width}});
    runner_tile_pre_scores.Run(stream);
    expand_pre_scores.Resize(scores->dims());

    // End_id Tensors
    Tensor end_id_tmp_tensor(experimental::DataType::INT32);
    end_id_tmp_tensor.mutable_data<int>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<int>(&end_id_tmp_tensor, end_id);

    Tensor end_id_tensors(ids_int32.dtype());
    end_id_tensors.mutable_data<int>(ids_int32.dims(), place);
    const auto& runner_fill_end_id =
        NpuOpRunner("FillD",
                    {end_id_tmp_tensor},
                    {end_id_tensors},
                    {{"dims", phi::vectorize(ids_int32.dims())}});
    runner_fill_end_id.Run(stream);

    // whether expand_pre_ids == end_ids?
    Tensor equal_end_ids(experimental::DataType::BOOL);
    equal_end_ids.mutable_data<bool>(ids_int32.dims(), place);
    const auto& runner_equal_end_ids = NpuOpRunner(
        "Equal", {expand_pre_ids, end_id_tensors}, {equal_end_ids}, {});
    runner_equal_end_ids.Run(stream);

    // construct a Tensor with dimension ids->dims():
    // [[False, True, True, True, ...],
    //  [False, True, True, True, ...],
    //  ...]
    Tensor false_tmp_tensor(experimental::DataType::INT32);
    false_tmp_tensor.mutable_data<int>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<int>(&false_tmp_tensor, static_cast<int>(false));

    Tensor first_pos_false_tensors(experimental::DataType::INT32);
    first_pos_false_tensors.Resize(phi::make_ddim({batch_size, 1}));
    first_pos_false_tensors.mutable_data<int>(place);
    std::vector<int64_t> fill_dims = {batch_size, 1};
    framework::NPUAttributeMap fill_attr = {{"dims", fill_dims}};
    const auto& runner_fill_false_tensors = NpuOpRunner(
        "FillD", {false_tmp_tensor}, {first_pos_false_tensors}, fill_attr);
    runner_fill_false_tensors.Run(stream);

    Tensor pos_tensors(experimental::DataType::INT32);
    if (seq_width > 1) {
      pos_tensors.Resize(phi::make_ddim({batch_size, seq_width}));
      pos_tensors.mutable_data<int>(place);

      Tensor true_tmp_tensor(experimental::DataType::INT32);
      true_tmp_tensor.mutable_data<int>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<int>(&true_tmp_tensor, static_cast<int>(true));

      Tensor second_pos_true_tensors(experimental::DataType::INT32);
      second_pos_true_tensors.Resize(
          phi::make_ddim({batch_size, seq_width - 1}));
      second_pos_true_tensors.mutable_data<int>(place);
      std::vector<int64_t> fill_dims2 = {batch_size, seq_width - 1};
      framework::NPUAttributeMap fill_attr2 = {{"dims", fill_dims2}};
      const auto& runner_fill_true_tensors = NpuOpRunner(
          "FillD", {true_tmp_tensor}, {second_pos_true_tensors}, fill_attr2);
      runner_fill_true_tensors.Run(stream);

      std::vector<phi::DenseTensor> concat_inputs = {first_pos_false_tensors,
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

    Tensor cast_pos_tensors_bool(experimental::DataType::BOOL);
    cast_pos_tensors_bool.Resize(pos_tensors.dims());
    cast_pos_tensors_bool.mutable_data<bool>(ctx.GetPlace());
    auto dst_dtype = ConvertToNpuDtype(
        framework::TransToProtoVarType(cast_pos_tensors_bool.type()));
    const auto& runner_cast_pos_tensors =
        NpuOpRunner("Cast",
                    {pos_tensors},
                    {cast_pos_tensors_bool},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_pos_tensors.Run(stream);

    // if pre_ids == end_ids, save only one score, and others become -inf
    // construct pre_ids == end_ids and save only one score
    Tensor save_one_end_score(experimental::DataType::BOOL);
    save_one_end_score.mutable_data<bool>(ids_int32.dims(), place);
    const auto& runner_logical_and =
        NpuOpRunner("LogicalAnd",
                    {equal_end_ids, cast_pos_tensors_bool},
                    {save_one_end_score},
                    {});
    runner_logical_and.Run(stream);

    // if save_one_end_score is True, set score to -inf
    // define -Inf Tensors
    Tensor ninf_tmp_tensor(scores->dtype());
    ninf_tmp_tensor.mutable_data<float>({1}, ctx.GetPlace());
    float ninf_value =
        static_cast<float>(-std::numeric_limits<float>::infinity());
    FillNpuTensorWithConstant<float>(&ninf_tmp_tensor, ninf_value);

    Tensor ninf_tensors(scores->dtype());
    ninf_tensors.mutable_data<float>(scores->dims(), place);
    const auto& runner_fill_ninf =
        NpuOpRunner("FillD",
                    {ninf_tmp_tensor},
                    {ninf_tensors},
                    {{"dims", phi::vectorize(scores->dims())}});
    runner_fill_ninf.Run(stream);

    // Step2: calculate topk scores

    // get scores used in topk op
    Tensor tmp_scores(scores->dtype());
    tmp_scores.mutable_data<float>(scores->dims(), place);
    if (!is_accumulated) {
      // if pre_id == end_id, cal_scores = pre_score, and id = end_id
      // else, cal_score = pre_score + log(score)

      // calculate log(scores)
      Tensor log_scores(scores->dtype());
      log_scores.mutable_data<float>(scores->dims(), place);

      Tensor one(scores->dtype());
      one.mutable_data<float>(scores->dims(), place);
      const auto& runner_one = NpuOpRunner("OnesLike", {*scores}, {one}, {});
      runner_one.Run(stream);

      Tensor sub(scores->dtype());
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
          NpuOpRunner("Select",
                      {equal_end_ids, expand_pre_scores, tmp_scores},
                      {tmp_scores},
                      {});
      runner_select_equal_end_score.Run(stream);
    } else {
      // if pre_ids == end_ids, use pre_score rather than score
      const auto& runner_select_equal_end_score2 =
          NpuOpRunner("Select",
                      {equal_end_ids, expand_pre_scores, *scores},
                      {tmp_scores},
                      {});
      runner_select_equal_end_score2.Run(stream);
    }

    // if pre_ids == end_ids, save only one score, and others become -inf
    Tensor cal_scores(scores->dtype());
    cal_scores.mutable_data<float>(scores->dims(), place);
    const auto& runner_select_inf_score =
        NpuOpRunner("Select",
                    {save_one_end_score, ninf_tensors, tmp_scores},
                    {cal_scores},
                    {});
    runner_select_inf_score.Run(stream);

    // resize scores from [num_seqs * beam_size, K] to [num_seqs, beam_size * K]
    // real_beam_size = 1 or beam_size
    cal_scores.Resize(phi::make_ddim({num_seqs, real_beam_size * seq_width}));

    Tensor topk_scores(scores->dtype());
    topk_scores.Resize(
        phi::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));
    topk_scores.mutable_data<float>(ctx.GetPlace());

    Tensor tmp_indices(experimental::DataType::INT32);
    tmp_indices.Resize(
        phi::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));
    tmp_indices.mutable_data<int>(ctx.GetPlace());

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

    // cast tmp_indices from int to float32 for Sort op
    Tensor cast_tmp_indices(experimental::DataType::FLOAT32);
    cast_tmp_indices.Resize(tmp_indices.dims());
    cast_tmp_indices.mutable_data<float>(ctx.GetPlace());
    auto dst_dtype_tmp_indices_fp32 = ConvertToNpuDtype(
        framework::TransToProtoVarType(cast_tmp_indices.type()));
    const auto& runner_cast_tmp_indices = NpuOpRunner(
        "Cast",
        {tmp_indices},
        {cast_tmp_indices},
        {{"dst_type", static_cast<int>(dst_dtype_tmp_indices_fp32)}});
    runner_cast_tmp_indices.Run(stream);

    // sort tmp_indices
    Tensor sorted_tmp_indices(experimental::DataType::FLOAT32);
    sorted_tmp_indices.Resize(tmp_indices.dims());
    sorted_tmp_indices.mutable_data<float>(ctx.GetPlace());
    Tensor sorted_score_indices(experimental::DataType::INT32);
    sorted_score_indices.Resize(tmp_indices.dims());
    sorted_score_indices.mutable_data<int>(ctx.GetPlace());
    const auto& runner_sort_tmp_indices =
        NpuOpRunner("Sort",
                    {cast_tmp_indices},
                    {sorted_tmp_indices, sorted_score_indices},
                    {{"axis", 1}, {"descending", false}});
    runner_sort_tmp_indices.Run(stream);

    // cast sorted_tmp_indices from float32 to int
    Tensor cast_sort_tmp_indices(experimental::DataType::INT32);
    cast_sort_tmp_indices.Resize(sorted_tmp_indices.dims());
    cast_sort_tmp_indices.mutable_data<int>(ctx.GetPlace());
    auto dst_dtype_tmp_indices_int32 = ConvertToNpuDtype(
        framework::TransToProtoVarType(cast_sort_tmp_indices.type()));
    const auto& runner_cast_sort_tmp_indices = NpuOpRunner(
        "Cast",
        {sorted_tmp_indices},
        {cast_sort_tmp_indices},
        {{"dst_type", static_cast<int>(dst_dtype_tmp_indices_int32)}});
    runner_cast_sort_tmp_indices.Run(stream);

    // Step 3: infer selected ids from tmp_indices and ids

    // if pre_ids == end_ids, use pre_ids rather than ids
    Tensor cal_ids(ids_int32.dtype());
    cal_ids.mutable_data<int>(ids_int32.dims(), place);
    const auto& runner_select_equal_end_id = NpuOpRunner(
        "Select", {equal_end_ids, expand_pre_ids, ids_int32}, {cal_ids}, {});
    runner_select_equal_end_id.Run(stream);

    // resize ids from [num_seqs * real_beam_size, K] to [num_seqs,
    // real_beam_size * K]
    // real_beam_size = 1 or beam_size
    cal_ids.Resize(phi::make_ddim({num_seqs, real_beam_size * seq_width}));

    // construct batch_ids like [[0, 0, 0], [1, 1, 1], ..., [bs-1, bs-1, bs-1]]
    // construct arange(num_seqs*beam_size).reshape((num_seqs, beam_size)) //
    // beam_size
    Tensor batch_ids(experimental::DataType::INT32);
    batch_ids.Resize(
        phi::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 1}));
    batch_ids.mutable_data<int>(place);

    std::vector<int> vector_batch_ids;
    for (int i = 0; i < num_seqs * static_cast<int>(beam_size); ++i) {
      vector_batch_ids.push_back(static_cast<int>(i / beam_size));
    }
    framework::TensorFromVector(vector_batch_ids, ctx, &batch_ids);
    batch_ids.Resize(
        phi::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 1}));

    // sort topk_scores to get selected_scores
    // get indices of gather_nd op for calculating selected_scores
    Tensor gather_nd_score_indices(experimental::DataType::INT32);
    gather_nd_score_indices.Resize(
        phi::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 2}));
    gather_nd_score_indices.mutable_data<int>(place);

    sorted_score_indices.Resize(
        phi::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 1}));
    std::vector<phi::DenseTensor> concat_inputs2 = {batch_ids,
                                                    sorted_score_indices};
    std::vector<std::string> concat_names = {"x0", "x1"};
    NpuOpRunner runner_concat_score_indices{"ConcatD",
                                            {concat_inputs2},
                                            {gather_nd_score_indices},
                                            {{"concat_dim", 2}, {"N", 2}}};
    runner_concat_score_indices.AddInputNames(concat_names);
    runner_concat_score_indices.Run(stream);

    // use gather_nd to get selected_scores
    const auto& runner_gather_nd_scores =
        NpuOpRunner("GatherNd",
                    {topk_scores, gather_nd_score_indices},
                    {*selected_scores},
                    {});
    runner_gather_nd_scores.Run(stream);

    // get indices of gather_nd op
    cast_sort_tmp_indices.Resize(
        phi::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 1}));
    Tensor gather_nd_id_indices(experimental::DataType::INT32);
    gather_nd_id_indices.Resize(
        phi::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 2}));
    gather_nd_id_indices.mutable_data<int>(place);

    std::vector<phi::DenseTensor> concat_inputs3 = {batch_ids,
                                                    cast_sort_tmp_indices};
    NpuOpRunner runner_concat_id_indices{"ConcatD",
                                         {concat_inputs3},
                                         {gather_nd_id_indices},
                                         {{"concat_dim", 2}, {"N", 2}}};
    runner_concat_id_indices.AddInputNames(concat_names);
    runner_concat_id_indices.Run(stream);

    // use gather_nd to get selected_ids
    Tensor topk_ids(experimental::DataType::INT32);
    topk_ids.Resize(
        phi::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));
    topk_ids.mutable_data<int>(ctx.GetPlace());

    const auto& runner_gather_nd_ids = NpuOpRunner(
        "GatherNd", {cal_ids, gather_nd_id_indices}, {topk_ids}, {});
    runner_gather_nd_ids.Run(stream);

    // cast topk_ids from int to int64 to get selected_ids
    auto dst_dtype_selected_ids =
        ConvertToNpuDtype(framework::TransToProtoVarType(selected_ids->type()));
    const auto& runner_cast_selected_ids =
        NpuOpRunner("Cast",
                    {topk_ids},
                    {*selected_ids},
                    {{"dst_type", static_cast<int>(dst_dtype_selected_ids)}});
    runner_cast_selected_ids.Run(stream);

    // TODO(pangyoki): PruneEndBeams

    // Step 4: set lod of output Tensor
    // define Tensor with value `seq_width`
    Tensor seq_width_tensor(experimental::DataType::INT32);
    seq_width_tensor.mutable_data<int>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<int>(&seq_width_tensor,
                                   static_cast<int>(seq_width));

    // beam_ids = tmp_indices // seq_width
    Tensor beam_ids(experimental::DataType::INT32);
    beam_ids.Resize(
        phi::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));
    beam_ids.mutable_data<int>(ctx.GetPlace());
    cast_sort_tmp_indices.Resize(
        phi::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));

    const auto& runner_div = NpuOpRunner(
        "Div", {cast_sort_tmp_indices, seq_width_tensor}, {beam_ids}, {});
    runner_div.Run(stream);

    // get parent_idx by adding batch_ids to beam_ids
    // construct scale_batch_ids like [[0, 0, 0], [bw, bw, bw], ..., [bs-1*bw,
    // bs-1*bw, bs-1*bw]]
    batch_ids.Resize(
        phi::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));

    // cast batch_ids from int to float32
    Tensor cast_batch_ids(experimental::DataType::FLOAT32);
    cast_batch_ids.Resize(batch_ids.dims());
    cast_batch_ids.mutable_data<float>(ctx.GetPlace());
    auto dst_dtype1 = ConvertToNpuDtype(
        framework::TransToProtoVarType(cast_batch_ids.type()));
    const auto& runner_cast_batch_ids =
        NpuOpRunner("Cast",
                    {batch_ids},
                    {cast_batch_ids},
                    {{"dst_type", static_cast<int>(dst_dtype1)}});
    runner_cast_batch_ids.Run(stream);

    // scale batch_ids with beam_size
    Tensor scale_batch_ids(experimental::DataType::FLOAT32);
    scale_batch_ids.Resize(batch_ids.dims());
    scale_batch_ids.mutable_data<float>(place);
    const auto& runner_power =
        NpuOpRunner("Power",
                    {cast_batch_ids},
                    {scale_batch_ids},
                    {{"power", static_cast<float>(1.0)},
                     {"scale", static_cast<float>(beam_size)},
                     {"shift", static_cast<float>(0.0)}});
    runner_power.Run(stream);

    // cast cast_scale_batch_ids from float32 to int
    Tensor cast_scale_batch_ids(experimental::DataType::INT32);
    cast_scale_batch_ids.Resize(scale_batch_ids.dims());
    cast_scale_batch_ids.mutable_data<int>(ctx.GetPlace());
    auto dst_dtype2 = ConvertToNpuDtype(
        framework::TransToProtoVarType(cast_scale_batch_ids.type()));
    const auto& runner_cast_scale_batch_ids =
        NpuOpRunner("Cast",
                    {scale_batch_ids},
                    {cast_scale_batch_ids},
                    {{"dst_type", static_cast<int>(dst_dtype2)}});
    runner_cast_scale_batch_ids.Run(stream);

    // calculate parent_idx
    Tensor tmp_parent_idx(experimental::DataType::INT32);
    tmp_parent_idx.Resize(parent_idx->dims());
    tmp_parent_idx.mutable_data<int>(place);
    const auto& runner_add_beam_id = NpuOpRunner(
        "Add", {beam_ids, cast_scale_batch_ids}, {tmp_parent_idx}, {});
    runner_add_beam_id.Run(stream);

    // cast tmp_parent_idx from int to int64 to get parent_idx
    auto dst_dtype_parent_idx =
        ConvertToNpuDtype(framework::TransToProtoVarType(parent_idx->type()));
    const auto& runner_cast_parent_idx =
        NpuOpRunner("Cast",
                    {tmp_parent_idx},
                    {*parent_idx},
                    {{"dst_type", static_cast<int>(dst_dtype_parent_idx)}});
    runner_cast_parent_idx.Run(stream);

    std::vector<int> vector_parent_idx;
    framework::TensorToVector(tmp_parent_idx, ctx, &vector_parent_idx);

    // set low level, len(low_level) = high_level[-1]
    std::vector<int> low_level;
    std::vector<int> num_parent_ids(num_seqs * beam_size,
                                    static_cast<int64_t>(0));
    size_t low_level_size = high_level[num_seqs];
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
