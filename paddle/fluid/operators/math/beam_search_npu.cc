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

    int64_t num_seqs = scores->NumElements(level);
    // size of the first beam is 1, others are equal to beam_size
    int64_t real_beam_size = static_cast<int64_t>(scores->dims()[0] / num_seqs);
    // K(=beam_size)
    int64_t seq_width = 1;
    for (int i = 1; i < scores->dims().size(); i++) {
      seq_width *= scores->dims()[i];
    }

    auto place = ctx.GetPlace();
    auto stream = ctx.stream();

    int64_t total_length = num_seqs * beam_size;
    int64_t batch_size = static_cast<int64_t>(scores->dims()[0]);
    selected_ids->mutable_data<int64_t>(framework::make_ddim({total_length, 1}),
                                        place);
    selected_scores->mutable_data<T>(framework::make_ddim({total_length, 1}),
                                     place);
    parent_idx->mutable_data<int64_t>(framework::make_ddim({total_length}),
                                      place);

    // Step1: Define Tensors and Preprocess the situation that pre_id == end_id

    // expand pre_ids to shape of ids
    Tensor expand_pre_ids(pre_ids->type());
    expand_pre_ids.Resize(framework::make_ddim({batch_size, seq_width}));
    expand_pre_ids.mutable_data<int64_t>(place);
    const auto& runner_tile_pre_ids =
        NpuOpRunner("TileWithAxis", {*pre_ids}, {expand_pre_ids},
                    {{"axis", 1}, {"tiles", seq_width}});
    runner_tile_pre_ids.Run(stream);
    expand_pre_ids.Resize(ids->dims());

    // End_id Tensors
    Tensor end_id_tmp_tensor(framework::proto::VarType::INT64);
    end_id_tmp_tensor.mutable_data<int64_t>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<int64_t>(&end_id_tmp_tensor, end_id);

    Tensor end_id_tensors(ids->type());
    end_id_tensors.mutable_data<int64_t>(ids->dims(), place);
    const auto& runner_fill_end_id =
        NpuOpRunner("FillD", {end_id_tmp_tensor}, {end_id_tensors},
                    {{"dims", framework::vectorize(ids->dims())}});
    runner_fill_end_id.Run(stream);

    // whether expand_pre_ids == end_ids?
    Tensor equal_end_ids(framework::proto::VarType::BOOL);
    equal_end_ids.mutable_data<bool>(ids->dims(), place);
    const auto& runner_equal_end_ids = NpuOpRunner(
        "Equal", {expand_pre_ids, end_id_tensors}, {equal_end_ids}, {});
    runner_equal_end_ids.Run(stream);

    // construct a Tensor with dimension ids->dims():
    // [[False, True, True, True, ...],
    //  [False, True, True, True, ...],
    //  ...]
    Tensor false_tmp_tensor(framework::proto::VarType::BOOL);
    false_tmp_tensor.mutable_data<bool>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<bool>(&false_tmp_tensor, false);

    Tensor first_pos_false_tensors(framework::proto::VarType::BOOL);
    first_pos_false_tensors.Resize(framework::make_ddim({batch_size, 1}));
    first_pos_false_tensors.mutable_data<bool>(place);
    std::vector<int64_t> fill_dims = {batch_size, 1};
    framework::NPUAttributeMap fill_attr = {{"dims", fill_dims}};
    const auto& runner_fill_false_tensors = NpuOpRunner(
        "FillD", {false_tmp_tensor}, {first_pos_false_tensors}, fill_attr);
    runner_fill_false_tensors.Run(stream);

    Tensor true_tmp_tensor(framework::proto::VarType::BOOL);
    true_tmp_tensor.mutable_data<bool>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<bool>(&true_tmp_tensor, true);

    Tensor second_pos_true_tensors(framework::proto::VarType::BOOL);
    second_pos_true_tensors.Resize(
        framework::make_ddim({batch_size, seq_width - 1}));
    second_pos_true_tensors.mutable_data<bool>(place);
    std::vector<int64_t> fill_dims2 = {batch_size, seq_width - 1};
    framework::NPUAttributeMap fill_attr2 = {{"dims", fill_dims2}};
    const auto& runner_fill_true_tensors = NpuOpRunner(
        "FillD", {true_tmp_tensor}, {second_pos_true_tensors}, fill_attr2);
    runner_fill_true_tensors.Run(stream);

    Tensor pos_tensors(framework::proto::VarType::BOOL);
    pos_tensors.Resize(framework::make_ddim({batch_size, seq_width}));
    pos_tensors.mutable_data<bool>(place);
    const auto& runner_concat_false_true = NpuOpRunner(
        "ConcatD", {first_pos_false_tensors, second_pos_true_tensors},
        {pos_tensors}, {{"concat_dim", 1}, {"N", 2}});
    runner_concat_false_true.Run(stream);
    pos_tensors.Resize(ids->dims());

    // if pre_ids == end_ids, save only one score, and others become -inf
    // construct pre_ids == end_ids and save only one score
    Tensor save_one_end_score(framework::proto::VarType::BOOL);
    save_one_end_score.mutable_data<bool>(ids->dims(), place);
    const auto& runner_logical_and = NpuOpRunner(
        "LogicalAnd", {equal_end_ids, pos_tensors}, {save_one_end_score}, {});
    runner_logical_and.Run(stream);

    // if save_one_end_score is True, set score to -inf
    // define -Inf Tensors
    Tensor ninf_tmp_tensor(scores->type());
    ninf_tmp_tensor.mutable_data<float>({1}, ctx.GetPlace());
    float ninf_value =
        static_cast<float>(-std::numeric_limits<float>::infinity());
    FillNpuTensorWithConstant<float>(&ninf_tmp_tensor, ninf_value);

    Tensor ninf_tensors(scores->type());
    ninf_tensors.mutable_data<T>(scores->dims(), place);
    const auto& runner_fill_ninf =
        NpuOpRunner("FillD", {ninf_tmp_tensor}, {ninf_tensors},
                    {{"dims", framework::vectorize(scores->dims())}});
    runner_fill_ninf.Run(stream);

    // Step2: calculate topk scores

    // get scores used in topk op
    Tensor tmp_scores(scores->type());
    tmp_scores.mutable_data<T>(scores->dims(), place);
    if (!is_accumulated) {
      // if pre_id == end_id, cal_scores = pre_score, and id = end_id
      // else, cal_score = pre_score + log(score)

      // calculate log(scores)
      Tensor log_scores(scores->type());
      log_scores.mutable_data<T>(scores->dims(), place);

      Tensor one(scores->type());
      one.mutable_data<T>(scores->dims(), place);
      const auto& runner_one = NpuOpRunner("OnesLike", {*scores}, {one}, {});
      runner_one.Run(stream);

      Tensor sub(scores->type());
      sub.mutable_data<T>(scores->dims(), place);
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
      const auto& runner_select_equal_end_score = NpuOpRunner(
          "Select", {equal_end_ids, *pre_scores, tmp_scores}, {tmp_scores}, {});
      runner_select_equal_end_score.Run(stream);
    } else {
      // if pre_ids == end_ids, use pre_score rather than score
      const auto& runner_select_equal_end_score2 = NpuOpRunner(
          "Select", {equal_end_ids, *pre_scores, *scores}, {tmp_scores}, {});
      runner_select_equal_end_score2.Run(stream);
    }

    // if pre_ids == end_ids, save only one score, and others become -inf
    Tensor cal_scores(scores->type());
    cal_scores.mutable_data<T>(scores->dims(), place);
    const auto& runner_select_inf_score =
        NpuOpRunner("Select", {save_one_end_score, ninf_tensors, tmp_scores},
                    {cal_scores}, {});
    runner_select_inf_score.Run(stream);

    // resize scores from [num_seqs * beam_size, K] to [num_seqs, beam_size * K]
    // real_beam_size = 1 or beam_size
    cal_scores.Resize(
        framework::make_ddim({num_seqs, real_beam_size * seq_width}));

    // select topk scores
    // prepare assit
    auto dim = cal_scores.dims().size();
    framework::Tensor assist_seq_tensor;
    assist_seq_tensor.Resize({2 * dim});
    assist_seq_tensor.mutable_data<T>(ctx.GetPlace());
    gen_assist_seq(&assist_seq_tensor, dim, ctx);

    framework::NPUAttributeMap attr_input = {{"sorted", "true"},
                                             {"k", static_cast<int>(beam_size)},
                                             {"dim", -1},
                                             {"largest", true}};

    Tensor topk_scores(scores->type());
    topk_scores.ShareDataWith(*selected_scores);
    topk_scores.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));

    Tensor tmp_indices(framework::proto::VarType::INT64);
    tmp_indices.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));
    tmp_indices.mutable_data<int64_t>(ctx.GetPlace());

    // run topk op
    const auto& runner_topk =
        NpuOpRunner("TopKD", {cal_scores, assist_seq_tensor},
                    {topk_scores, tmp_indices}, attr_input);
    runner_topk.Run(stream);

    // Step 3: infer selected ids from tmp_indices and ids

    // if pre_ids == end_ids, use pre_ids rather than ids
    Tensor cal_ids(ids->type());
    cal_ids.mutable_data<int64_t>(ids->dims(), place);
    const auto& runner_select_equal_end_id =
        NpuOpRunner("Select", {equal_end_ids, *pre_ids, *ids}, {cal_ids}, {});
    runner_select_equal_end_id.Run(stream);

    // resize ids from [num_seqs * real_beam_size, K] to [num_seqs,
    // real_beam_size * K]
    // real_beam_size = 1 or beam_size
    cal_ids.Resize(
        framework::make_ddim({num_seqs, real_beam_size * seq_width}));

    // construct arange(num_seqs*beam_size).reshape((num_seqs, beam_size)) //
    // beam_size
    Tensor batch_ids(framework::proto::VarType::INT64);
    batch_ids.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 1}));
    batch_ids.mutable_data<int64_t>(place);

    std::vector<int64_t> vector_batch_ids;
    for (int64_t i = 0; i < num_seqs * static_cast<int64_t>(beam_size); ++i) {
      vector_batch_ids.push_back(static_cast<int64_t>(i / beam_size));
    }
    framework::TensorFromVector(vector_batch_ids, ctx, &batch_ids);

    // get indices of gather_nd op
    tmp_indices.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 1}));
    Tensor gather_nd_indices(framework::proto::VarType::INT64);
    gather_nd_indices.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size), 2}));
    gather_nd_indices.mutable_data<int64_t>(place);
    const auto& runner_concat_indices =
        NpuOpRunner("ConcatD", {batch_ids, tmp_indices}, {gather_nd_indices},
                    {{"concat_dim", 1}, {"N", 2}});
    runner_concat_indices.Run(stream);

    // use gather_nd to get selected_ids
    Tensor topk_ids(selected_ids->type());
    topk_ids.ShareDataWith(*selected_ids);
    topk_ids.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));

    const auto& runner_gather_nd =
        NpuOpRunner("GatherNd", {cal_ids, gather_nd_indices}, {topk_ids}, {});
    runner_gather_nd.Run(stream);

    // Step 4: set lod of output Tensor
    // define Tensor with value `beam_size`
    Tensor beam_size_tensor(framework::proto::VarType::INT64);
    beam_size_tensor.mutable_data<int64_t>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<int64_t>(&beam_size_tensor, beam_size);

    // beam_ids = tmp_indices % beam_size
    Tensor beam_ids(framework::proto::VarType::INT64);
    beam_ids.ShareDataWith(*parent_idx);
    beam_ids.Resize(
        framework::make_ddim({num_seqs, static_cast<int64_t>(beam_size)}));

    const auto& runner_mod =
        NpuOpRunner("Mod", {tmp_indices, beam_size_tensor}, {beam_ids}, {});
    runner_mod.Run(stream);

    std::vector<int64_t> vector_beam_ids;
    framework::TensorToVector(beam_ids, ctx, &vector_beam_ids);

    // set low level, len(low_level) = high_level[-1]
    std::vector<int64_t> low_level;
    std::vector<int64_t> num_beam_ids(num_seqs * beam_size,
                                      static_cast<int64_t>(0));
    size_t low_level_size = high_level[num_seqs - 1];
    size_t sum_beam_id = 0;

    // calculate number of every beam_id
    for (size_t i = 0; i < num_seqs * beam_size; ++i) {
      num_beam_ids[vector_beam_ids[i]]++;
    }

    // update low_level
    low_level.push_back(0);
    for (size_t i = 0; i < low_level_size; ++i) {
      sum_beam_id += num_beam_ids[i];
      low_level.push_back(sum_beam_id);
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

 protected:
  // used in topk NPU OP
  void gen_assist_seq(framework::Tensor* assit_tensor, int64_t dim,
                      const platform::NPUDeviceContext& ctx) {
    const int64_t dimx2 = dim;
    std::vector<paddle::platform::float16> assit;
    assit.resize(2 * dimx2);
    for (int64_t i = 0; i < dimx2; i++) {
      // for i in range [0, dim]
      assit[i] = static_cast<paddle::platform::float16>(i);

      // for i in range [dim, dimx2]
      int64_t idx =
          static_cast<int64_t>(static_cast<paddle::platform::float16>(i));
      int64_t gap = i - idx;
      assit[i + dim] = static_cast<paddle::platform::float16>(gap);
    }
    framework::TensorFromVector(assit, ctx, assit_tensor);
  }
};

template class BeamSearchFunctor<platform::NPUDeviceContext, int>;
template class BeamSearchFunctor<platform::NPUDeviceContext, int64_t>;
template class BeamSearchFunctor<platform::NPUDeviceContext, float>;
template class BeamSearchFunctor<platform::NPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
