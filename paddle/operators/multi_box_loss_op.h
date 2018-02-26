/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/cross_entropy.h"
#include "paddle/operators/math/detection_util.h"
#include "paddle/operators/math/softmax.h"

namespace paddle {
namespace operators {

template <typename T>
T MultiBoxLossSmoothL1(const framework::ExecutionContext& ctx,
                       const framework::Tensor& output,
                       const framework::Tensor& label, int match_num,
                       T dest_scale) {
  auto sample_num = output.dims()[0];
  auto dim = output.numel() / sample_num;

  PADDLE_ENFORCE_EQ(label.dims()[0], sample_num);
  PADDLE_ENFORCE_EQ(label.numel(), output.numel());

  const T* out_data = output.data<T>();
  const T* label_data = label.data<T>();

  T cost = 0.0;
  for (int i = 0; i < sample_num; ++i, out_data += dim, label_data += dim) {
    T cost_i = 0.0;
    for (int j = 0; j < dim; ++j) {
      T abs = std::fabs(out_data[j] - label_data[j]);
      cost_i *= dest_scale;
      if (abs < 1.0)
        cost_i += 0.5 * abs * abs;
      else
        cost_i += abs - 0.5;
    }
    cost += cost_i;
  }
  return cost / match_num;
}

template <typename DeviceContext, typename T>
class MultiBoxLossOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins_loc = ctx.MultiInput<framework::Tensor>("Loc");
    auto ins_conf = ctx.MultiInput<framework::Tensor>("Conf");
    auto* in_priorbox = ctx.Input<framework::Tensor>("PriorBox");
    auto* in_label = ctx.Input<framework::LoDTensor>("Label");

    auto* out_loss = ctx.Output<framework::Tensor>("Loss");
    auto* out_inter_couter = ctx.Output<framework::Tensor>("InterCounter");
    auto* out_all_match_indices =
        ctx.Output<framework::Tensor>("AllMatchIndices");

    auto* out_all_neg_indices = ctx.Output<framework::Tensor>("AllNegIndices");

    auto* out_loc_gt = ctx.Output<framework::Tensor>("LocGTData");
    auto* out_conf_gt = ctx.Output<framework::Tensor>("ConfGTData");
    auto* out_loc_diff = ctx.Output<framework::Tensor>("LocDiff");
    auto* out_conf_prob = ctx.Output<framework::Tensor>("ConfProb");

    int class_num = ctx.template Attr<int>("class_num");
    float overlap_threshold = ctx.template Attr<float>("overlap_threshold");
    float neg_pos_ratio = ctx.template Attr<float>("neg_pos_ratio");
    float neg_overlap = ctx.template Attr<float>("neg_overlap");
    int background_label_id = ctx.template Attr<int>("background_label_id");

    int input_num = ins_loc.size();
    int batch_size = ins_loc[0]->dims()[0];
    int prior_num = in_priorbox->numel() / 8;

    framework::Tensor loc_buffer_cpu;
    framework::Tensor conf_buffer_cpu;

    int loc_size_sum = 0;
    int conf_size_sum = 0;
    for (int i = 0; i < input_num; ++i) {
      loc_size_sum += ins_loc[i]->numel();
      conf_size_sum += ins_conf[i]->numel();
    }

    auto loc_buffer_dim = framework::make_ddim({1, loc_size_sum});
    loc_buffer_cpu.mutable_data<T>(loc_buffer_dim, platform::CPUPlace());

    auto conf_buffer_dim = framework::make_ddim({1, conf_size_sum});
    conf_buffer_cpu.mutable_data<T>(conf_buffer_dim, platform::CPUPlace());

    math::SetConstant<DeviceContext, T> set_constant;
    set_constant(ctx.template device_context<DeviceContext>(), &loc_buffer_cpu,
                 0);
    set_constant(ctx.template device_context<DeviceContext>(), &conf_buffer_cpu,
                 0);

    int loc_offset = 0;
    int conf_offset = 0;

    platform::CPUPlace cpu_place;
    platform::CPUDeviceContext cpu_ctx(cpu_place);

    for (int i = 0; i < input_num; ++i) {
      auto in_loc = ins_loc[i];
      auto in_conf = ins_conf[i];

      if (platform::is_gpu_place(ctx.GetPlace())) {
        framework::Tensor loc;
        framework::Tensor conf;
        loc.mutable_data<T>(in_loc->dims(), platform::CPUPlace());
        conf.mutable_data<T>(in_conf->dims(), platform::CPUPlace());
        framework::CopyFrom(*in_loc, platform::CPUPlace(), ctx.device_context(),
                            &loc);
        framework::CopyFrom(*in_conf, platform::CPUPlace(),
                            ctx.device_context(), &conf);

        loc_offset +=
            math::TransposeFromNCHWToNHWC<platform::CPUDeviceContext, T>(
                cpu_ctx, loc, loc_buffer_cpu, loc_size_sum, loc_offset);
        conf_offset +=
            math::TransposeFromNCHWToNHWC<platform::CPUDeviceContext, T>(
                cpu_ctx, conf, conf_buffer_cpu, conf_size_sum, conf_offset);
      } else {
        loc_offset +=
            math::TransposeFromNCHWToNHWC<platform::CPUDeviceContext, T>(
                cpu_ctx, *in_loc, loc_buffer_cpu, loc_size_sum, loc_offset);
        conf_offset +=
            math::TransposeFromNCHWToNHWC<platform::CPUDeviceContext, T>(
                cpu_ctx, *in_conf, conf_buffer_cpu, conf_size_sum, conf_offset);
      }
    }

    auto loc_buffer_cpu_data = loc_buffer_cpu.data<T>();
    printf("\nloc_buffer_cpu_data\n");
    for (int i = 0; i < loc_buffer_cpu.numel(); ++i) {
      printf("%f[%d], ", loc_buffer_cpu_data[i], (int)i);
    }
    printf("\nconf_buffer_cpu_data\n");
    auto conf_buffer_cpu_data = conf_buffer_cpu.data<T>();
    for (int i = 0; i < conf_buffer_cpu.numel(); ++i) {
      printf("%f[%d], ", conf_buffer_cpu_data[i], (int)i);
    }

    std::vector<std::vector<T>> all_max_conf_score;
    GetMaxConfidenceScores(conf_buffer_cpu, batch_size, prior_num, class_num,
                           background_label_id, all_max_conf_score);

    out_all_match_indices->mutable_data<int>({batch_size, prior_num},
                                             platform::CPUPlace());

    out_all_neg_indices->mutable_data<int>({batch_size, prior_num},
                                           platform::CPUPlace());

    printf("\nall_max_conf_score\n");
    for (auto& s : all_max_conf_score) {
      for (auto i : s) {
        printf("%f, ", i);
      }
      printf("\n");
    }

    auto all_match_indices =
        framework::EigenMatrix<int>::From(*out_all_match_indices);
    all_match_indices.setConstant(-1);

    auto all_neg_indices =
        framework::EigenMatrix<int>::From(*out_all_neg_indices);
    all_neg_indices.setConstant(-1);

    int total_match = 0;
    int total_neg = 0;
    framework::Tensor priorbox_cpu;
    framework::LoDTensor label_cpu;
    if (platform::is_gpu_place(ctx.GetPlace())) {
      priorbox_cpu.mutable_data<T>(in_priorbox->dims(), platform::CPUPlace());
      label_cpu.mutable_data<T>(in_label->dims(), platform::CPUPlace());
      framework::CopyFrom(*in_priorbox, platform::CPUPlace(),
                          ctx.device_context(), &priorbox_cpu);
      framework::CopyFrom(*in_label, platform::CPUPlace(), ctx.device_context(),
                          &label_cpu);
      label_cpu.set_lod(in_label->lod());

      GenerateMatchIndices(priorbox_cpu, prior_num, label_cpu,
                           all_max_conf_score, batch_size, overlap_threshold,
                           neg_overlap, neg_pos_ratio, all_match_indices,
                           all_neg_indices, total_match, total_neg);
    } else {
      printf("\npriorValue %d\n", (int)prior_num);
      auto priorbox_cpu_data = in_priorbox->data<T>();
      for (int n = 0; n < prior_num; ++n) {
        for (int i = 0; i < 8; ++i) {
          printf("%f, ", priorbox_cpu_data[n * 8 + i]);
        }
        printf("\n");
      }

      auto label_lod = in_label->lod();
      printf("\nseqNum %d\n", (int)label_lod[0].size());
      for (size_t i = 0; i < label_lod[0].size(); ++i) {
        printf("%d, ", (int)label_lod[0][i]);
      }
      printf("\nseqNum %d\n", (int)label_lod[0].size());
      auto label_print_data = in_label->data<T>();
      for (int n = 0; n < batch_size; ++n) {
        for (size_t i = label_lod[0][n]; i < label_lod[0][n + 1]; i++) {
          printf("%f, ", label_print_data[i * 6]);
          printf("%f, ", label_print_data[i * 6 + 1]);
          printf("%f, ", label_print_data[i * 6 + 2]);
          printf("%f, ", label_print_data[i * 6 + 3]);
          printf("%f, ", label_print_data[i * 6 + 4]);
          printf("%f, ", label_print_data[i * 6 + 5]);
          printf("\n");
        }
        printf("\n");
      }

      GenerateMatchIndices(*in_priorbox, prior_num, *in_label,
                           all_max_conf_score, batch_size, overlap_threshold,
                           neg_overlap, neg_pos_ratio, all_match_indices,
                           all_neg_indices, total_match, total_neg);
    }

    T loc_loss = 0.0;
    T conf_loss = 0.0;
    int num_conf = total_match + total_neg;
    if (platform::is_gpu_place(ctx.GetPlace())) {
      if (total_match >= 1) {
        loc_loss =
            CalcLocationLoss(ctx, priorbox_cpu, loc_buffer_cpu, label_cpu,
                             total_match, batch_size, prior_num,
                             all_match_indices, *out_loc_gt, *out_loc_diff);
      }

      if (num_conf >= 1) {
        conf_loss = CalcConfidenceLoss(
            ctx, priorbox_cpu, conf_buffer_cpu, label_cpu, total_match,
            total_neg, batch_size, prior_num, background_label_id, class_num,
            all_match_indices, all_neg_indices, *out_conf_gt, *out_conf_prob);
      }
    } else {
      if (total_match >= 1) {
        loc_loss =
            CalcLocationLoss(ctx, *in_priorbox, loc_buffer_cpu, *in_label,
                             total_match, batch_size, prior_num,
                             all_match_indices, *out_loc_gt, *out_loc_diff);
      }

      if (num_conf >= 1) {
        conf_loss = CalcConfidenceLoss(
            ctx, *in_priorbox, conf_buffer_cpu, *in_label, total_match,
            total_neg, batch_size, prior_num, background_label_id, class_num,
            all_match_indices, all_neg_indices, *out_conf_gt, *out_conf_prob);
      }
    }

    T loss = loc_loss + conf_loss;

    printf("loc_loss %f\n", loc_loss);
    printf("conf_loss %f\n", conf_loss);

    out_loss->mutable_data<T>(ctx.GetPlace());
    if (platform::is_gpu_place(ctx.GetPlace())) {
      framework::Tensor loss_cpu;
      T* loss_data =
          loss_cpu.mutable_data<T>(out_loss->dims(), platform::CPUPlace());
      loss_data[0] = loss;
      framework::CopyFrom(loss_cpu, platform::CPUPlace(), ctx.device_context(),
                          out_loss);
    } else {
      T* loss_data = out_loss->mutable_data<T>(ctx.GetPlace());
      loss_data[0] = loss;
    }

    out_inter_couter->mutable_data<int>(platform::CPUPlace());
    auto inter_couter = framework::EigenVector<int>::Flatten(*out_inter_couter);

    inter_couter(0) = total_match;
    inter_couter(1) = total_neg;
    inter_couter(2) = num_conf;
  }  // namespace operators

 private:
  void GetMaxConfidenceScores(
      const framework::Tensor& conf, int batch_size, int prior_num,
      int class_num, int background_label_id,
      std::vector<std::vector<T>>& all_max_conf_score) const {
    all_max_conf_score.clear();
    const T* conf_data = conf.data<T>();
    for (int i = 0; i < batch_size; ++i) {
      std::vector<T> max_conf_score;
      for (int j = 0; j < prior_num; ++j) {
        int offset = j * class_num;
        T max_val = -std::numeric_limits<T>::max();
        T max_pos_val = -std::numeric_limits<T>::max();
        T max_score = 0.0;
        for (int c = 0; c < class_num; ++c) {
          max_val = std::max<T>(conf_data[offset + c], max_val);
          if (c != background_label_id)
            max_pos_val = std::max<T>(conf_data[offset + c], max_pos_val);
        }
        T sum = 0.0;
        for (int c = 0; c < class_num; ++c)
          sum += std::exp(conf_data[offset + c] - max_val);
        max_score = std::exp(max_pos_val - max_val) / sum;
        max_conf_score.push_back(max_score);
      }
      conf_data += prior_num * class_num;
      all_max_conf_score.push_back(max_conf_score);
    }
    return;
  }

  void GenerateMatchIndices(
      const framework::Tensor& priorbox, int num_prior_bboxes,
      const framework::LoDTensor& label,
      const std::vector<std::vector<T>>& max_conf_score, int batch_size,
      float overlap_threshold, float neg_overlap_threshold, int neg_pos_ratio,
      framework::EigenMatrix<int>::Type& all_match_indices,
      framework::EigenMatrix<int>::Type& all_neg_indices, int& total_match,
      int& total_neg) const {
    std::vector<math::BBox<T>> prior_bboxes;
    GetBBoxFromPriorData(priorbox.data<T>(), num_prior_bboxes, prior_bboxes);

    auto label_lod = label.lod();
    auto label_index = label_lod[0];
    auto label_data_num = static_cast<int>(label_lod[0].size());

    total_match = 0;
    total_neg = 0;
    for (int n = 0; n < batch_size; ++n) {
      std::vector<int> match_indices;
      std::vector<int> neg_indices;
      std::vector<float> match_overlaps;
      match_indices.resize(num_prior_bboxes, -1);
      match_overlaps.resize(num_prior_bboxes, 0.0);
      size_t num_gt_bboxes = 0;
      if (n < label_data_num)
        num_gt_bboxes = label_index[n + 1] - label_index[n];
      if (num_gt_bboxes == 0) {
        continue;
      }
      std::vector<math::BBox<T>> gt_bboxes;
      GetBBoxFromLabelData(label.data<T>() + label_index[n] * 6, num_gt_bboxes,
                           gt_bboxes);

      MatchBBox(prior_bboxes, gt_bboxes, overlap_threshold, match_indices,
                match_overlaps);

      size_t num_pos = 0;
      size_t neg_num = 0;
      for (size_t i = 0; i < match_indices.size(); ++i)
        if (match_indices[i] != -1) ++num_pos;
      total_match += num_pos;
      std::vector<std::pair<float, size_t>> scores_indices;
      for (size_t i = 0; i < match_indices.size(); ++i)
        if (match_indices[i] == -1 &&
            match_overlaps[i] < neg_overlap_threshold) {
          scores_indices.push_back(std::make_pair(max_conf_score[n][i], i));
          ++neg_num;
        }
      neg_num = std::min(static_cast<size_t>(num_pos * neg_pos_ratio), neg_num);
      std::sort(scores_indices.begin(), scores_indices.end(),
                math::SortScorePairDescend<size_t>);
      for (size_t i = 0; i < neg_num; ++i)
        neg_indices.push_back(scores_indices[i].second);
      total_neg += neg_num;
      for (size_t i = 0; i < match_indices.size(); ++i) {
        all_match_indices(n, i) = match_indices[i];
      }

      for (size_t i = 0; i < neg_indices.size(); ++i) {
        all_neg_indices(n, i) = neg_indices[i];
      }
    }
    return;
  }

  T CalcLocationLoss(const framework::ExecutionContext& ctx,
                     const framework::Tensor& priorbox,
                     const framework::Tensor& loc_buffer,
                     const framework::LoDTensor& label, int match_num,
                     int batch_size, int prior_num,
                     framework::EigenMatrix<int>::Type& all_match_indices,
                     framework::Tensor& loc_gt,
                     framework::Tensor& loc_diff) const {
    T loc_loss = 0.0;
    auto label_lod = label.lod();
    auto label_index = label_lod[0];

    size_t count = 0;
    auto loc_dim = framework::make_ddim({match_num * 4, 1});
    T* loc_diff_data = loc_diff.mutable_data<T>(loc_dim, platform::CPUPlace());
    T* loc_gt_data = loc_gt.mutable_data<T>(loc_dim, platform::CPUPlace());

    int loc_gt_offset = 0;
    const T* loc_buffer_data = loc_buffer.data<T>();
    for (int n = 0; n < batch_size; ++n) {
      for (int i = 0; i < prior_num; ++i) {
        if (all_match_indices(n, i) == -1) continue;  // match none
        size_t loc_offset = n * (loc_buffer.numel() / batch_size) + i * 4;
        std::copy(loc_buffer_data + loc_offset,
                  loc_buffer_data + loc_offset + 4, loc_diff_data + count);
        count += 4;
        const int gt_idx = all_match_indices(n, i);
        size_t prior_offset = i * 8;
        std::vector<math::BBox<T>> prior_bboxes;
        GetBBoxFromPriorData(priorbox.data<T>() + prior_offset, 1,
                             prior_bboxes);
        std::vector<std::vector<T>> prior_bbox_var;
        math::GetBBoxVarFromPriorData<T>(priorbox.data<T>() + prior_offset, 1,
                                         prior_bbox_var);
        size_t label_offset = (label_index[n] + gt_idx) * 6;
        std::vector<math::BBox<T>> gt_bboxes;
        GetBBoxFromLabelData(label.data<T>() + label_offset, 1, gt_bboxes);
        std::vector<T> gt_encode;
        EncodeBBoxWithVar(prior_bboxes[0], prior_bbox_var[0], gt_bboxes[0],
                          gt_encode);
        std::copy(gt_encode.begin(), gt_encode.end(),
                  loc_gt_data + loc_gt_offset);
        loc_gt_offset += gt_encode.size();
      }
    }
    loc_loss = MultiBoxLossSmoothL1<T>(ctx, loc_diff, loc_gt, match_num, 0.0);
    return loc_loss;
  }

  T CalcConfidenceLoss(const framework::ExecutionContext& ctx,
                       const framework::Tensor& priorbox,
                       const framework::Tensor& conf_buffer,
                       const framework::LoDTensor& label, int match_num,
                       int neg_num, int batch_size, int prior_num,
                       int background_label_id, int class_num,
                       framework::EigenMatrix<int>::Type& all_match_indices,
                       framework::EigenMatrix<int>::Type& all_neg_indices,
                       framework::Tensor& conf_gt,
                       framework::Tensor& conf_prob) const {
    T conf_loss = 0;
    auto label_lod = label.lod();
    auto label_index = label_lod[0];

    size_t count = 0;
    // std::vector<T> conf_pred_data;
    T* conf_prob_data = conf_prob.mutable_data<T>(
        {match_num + neg_num, class_num}, platform::CPUPlace());
    int64_t* conf_gt_data = conf_gt.mutable_data<int64_t>(
        {match_num + neg_num, 1}, platform::CPUPlace());
    const T* conf_buffer_data = conf_buffer.data<T>();

    math::SetConstant<DeviceContext, T> set_constant_t;
    math::SetConstant<DeviceContext, int64_t> set_constant_i;
    set_constant_t(ctx.template device_context<DeviceContext>(), &conf_prob, 0);
    set_constant_i(ctx.template device_context<DeviceContext>(), &conf_gt, 0);

    for (int n = 0; n < batch_size; ++n) {
      for (int i = 0; i < prior_num; ++i) {
        if (all_match_indices(n, i) == -1) continue;
        size_t label_offset = (label_index[n] + all_match_indices(n, i)) * 6;
        const int gt_label = (label.data<T>() + label_offset)[0];
        printf("\n count:%d [%d %d]\n", (int)count, (int)n, (int)i);
        printf("\n label_offset:%d [%d %d]\n", (int)label_offset, (int)n,
               (int)i);
        printf("\n label_index:%d [%d %d]\n", (int)label_index[n], (int)n,
               (int)i);
        printf("\n gt_label:%d [%d %d]\n", (int)gt_label, (int)n, (int)i);

        conf_gt_data[count] = gt_label;
        size_t conf_offset = n * prior_num * class_num + i * class_num;
        printf("\n conf_offset:%d [%d %d]\n", (int)conf_offset, (int)n, (int)i);
        printf("\n class_num:%d [%d %d]\n", (int)class_num, (int)n, (int)i);
        std::copy(conf_buffer_data + conf_offset,
                  conf_buffer_data + conf_offset + class_num,
                  conf_prob_data + count * class_num);
        // conf_pred_data.reserve(conf_pred_data.size() + class_num);
        // conf_pred_data.insert(conf_pred_data.end(),
        //                       conf_buffer_data + conf_offset,
        //                       conf_buffer_data + conf_offset + class_num);
        ++count;
      }
      // Negative mining samples
      for (int i = 0; i < prior_num; ++i) {
        if (all_neg_indices(n, i) == -1) continue;
        conf_gt_data[count] = background_label_id;
        size_t conf_offset =
            n * prior_num * class_num + all_neg_indices(n, i) * class_num;
        printf("\n count:%d [%d %d]\n", (int)count, (int)n, (int)i);
        printf("\n Neg conf_offset:%d [%d %d]\n", (int)conf_offset, (int)n,
               (int)i);
        printf("\n all_neg_indices:%d [%d %d]\n", (int)all_neg_indices(n, i),
               (int)n, (int)i);
        std::copy(conf_buffer_data + conf_offset,
                  conf_buffer_data + conf_offset + class_num,
                  conf_prob_data + count * class_num);
        // conf_pred_data.reserve(conf_pred_data.size() + class_num);
        // conf_pred_data.insert(conf_pred_data.end(),
        //                       conf_buffer_data + conf_offset,
        //                       conf_buffer_data + conf_offset + class_num);
        ++count;
      }
    }

    platform::CPUPlace cpu_place;
    platform::CPUDeviceContext cpu_ctx(cpu_place);

    for (int i = 0; i < conf_prob.numel(); ++i) {
      printf("1 conf_prob[%d]=%f\n", i, conf_prob.data<T>()[i]);
    }

    math::SoftmaxFunctor<platform::CPUDeviceContext, T>()(cpu_ctx, &conf_prob,
                                                          &conf_prob);

    for (int i = 0; i < conf_prob.numel(); ++i) {
      printf("2 conf_prob[%d]=%f\n", i, conf_prob.data<T>()[i]);
    }
    framework::Tensor conf_loss_out;
    auto conf_loss_data = conf_loss_out.mutable_data<T>(
        {match_num + neg_num, 1}, platform::CPUPlace());

    // framework::Tensor tensor_temp;
    // auto tensor_temp_data =
    // tensor_temp.mutable_data<int64_t>(conf_gt.dims(), platform::CPUPlace());
    for (int i = 0; i < conf_gt.numel(); ++i) {
      printf("conf_gt[%d]=%d\n", i, (int)conf_gt.data<int64_t>()[i]);
    }
    // math::CrossEntropyFunctor<platform::CPUDeviceContext, T>()(
    // cpu_ctx, &conf_loss_out, &conf_prob, &tensor_temp, false);
    math::CrossEntropyFunctor<platform::CPUDeviceContext, T>()(
        cpu_ctx, &conf_loss_out, &conf_prob, &conf_gt, false);

    conf_loss = 0.0;
    for (int i = 0; i < conf_loss_out.numel(); ++i) {
      printf("conf_loss_data[%d]=%f\n", i, conf_loss_data[i]);
      conf_loss += conf_loss_data[i];
    }
    conf_loss = conf_loss / match_num;
    return conf_loss;
  }
};  // namespace operators

template <typename T>
void MultiBoxLossSmoothL1BP(const framework::ExecutionContext& ctx,
                            const framework::Tensor& output,
                            const framework::Tensor& label,
                            framework::Tensor& grad, int match_num,
                            T dest_scale) {
  auto sample_num = output.dims()[0];
  auto dim = output.numel() / sample_num;

  const T* out_data = output.data<T>();
  const T* label_data = label.data<T>();
  T* grad_data = grad.mutable_data<T>(platform::CPUPlace());

  for (int i = 0; i < sample_num;
       ++i, out_data += dim, grad_data += dim, label_data += dim) {
    for (int j = 0; j < dim; ++j) {
      T val = out_data[j] - label_data[j];
      grad_data[j] *= dest_scale;
      if (std::fabs(val) < 1) {
        grad_data[j] += val;
      } else {
        grad_data[j] += (T(0) < val) - (val < T(0));
      }
    }
  }
}

template <typename DeviceContext, typename T>
class MultiBoxLossGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto d_ins_loc =
        ctx.MultiOutput<framework::Tensor>(framework::GradVarName("Loc"));
    auto d_ins_conf =
        ctx.MultiOutput<framework::Tensor>(framework::GradVarName("Conf"));

    auto ins_loc = ctx.MultiInput<framework::Tensor>("Loc");
    auto ins_conf = ctx.MultiInput<framework::Tensor>("Conf");
    auto* in_priorbox = ctx.Input<framework::Tensor>("PriorBox");

    auto* in_loc_gt = ctx.Input<framework::Tensor>("LocGTData");
    auto* in_conf_gt = ctx.Input<framework::Tensor>("ConfGTData");
    auto* in_loc_diff = ctx.Input<framework::Tensor>("LocDiff");
    auto* in_conf_prob = ctx.Input<framework::Tensor>("ConfProb");

    auto* in_inter_couter = ctx.Input<framework::Tensor>("InterCounter");

    auto inter_couter = framework::EigenVector<int>::Flatten(*in_inter_couter);

    int class_num = ctx.template Attr<int>("class_num");

    auto* in_all_match_indices =
        ctx.Input<framework::Tensor>("AllMatchIndices");

    auto* in_all_neg_indices = ctx.Input<framework::Tensor>("AllNegIndices");

    int batch_size = ins_loc[0]->dims()[0];
    int match_num = inter_couter(0);
    // int neg_num = inter_couter(1);
    int conf_num = inter_couter(2);
    int prior_num = in_priorbox->numel() / 8;
    int input_num = ins_loc.size();

    framework::Tensor loc_buffer;
    framework::Tensor conf_buffer;

    int loc_size_sum = 0;
    int conf_size_sum = 0;
    for (int i = 0; i < input_num; ++i) {
      loc_size_sum += ins_loc[i]->numel();
      conf_size_sum += ins_conf[i]->numel();
    }

    auto loc_buffer_dim = framework::make_ddim({1, loc_size_sum});
    loc_buffer.mutable_data<T>(loc_buffer_dim, platform::CPUPlace());

    auto conf_buffer_dim = framework::make_ddim({1, conf_size_sum});
    conf_buffer.mutable_data<T>(conf_buffer_dim, platform::CPUPlace());

    auto all_match_indices =
        framework::EigenMatrix<const int>::From(*in_all_match_indices);

    auto all_neg_indices =
        framework::EigenMatrix<const int>::From(*in_all_neg_indices);

    printf("\n in_loc_diff %lld\n", in_loc_diff->numel());
    for (int i = 0; i < in_loc_diff->numel(); ++i) {
      printf("%f[%d], ", in_loc_diff->data<T>()[i], (int)i);
    }
    printf("\n in_loc_gt %lld\n", in_loc_gt->numel());
    for (int i = 0; i < in_loc_gt->numel(); ++i) {
      printf("%f[%d], ", in_loc_gt->data<T>()[i], (int)i);
    }
    if (match_num > 1) {
      CalcLocationLossBP(ctx, *in_loc_diff, *in_loc_gt, loc_buffer, match_num,
                         batch_size, prior_num, all_match_indices);
    }
    printf("\n loc_buffer %lld\n", loc_buffer.numel());
    for (int i = 0; i < loc_buffer.numel(); ++i) {
      printf("%f[%d], ", loc_buffer.data<T>()[i], (int)i);
    }

    printf("\n in_conf_gt %lld\n", in_conf_gt->numel());
    for (int i = 0; i < in_conf_gt->numel(); ++i) {
      printf("%lld[%d], ", in_conf_gt->data<int64_t>()[i], (int)i);
    }
    printf("\n in_conf_prob %lld\n", in_conf_prob->numel());
    for (int i = 0; i < in_conf_prob->numel(); ++i) {
      printf("%f[%d], ", in_conf_prob->data<T>()[i], (int)i);
    }

    if (conf_num > 1) {
      CalcConfidenceLossBP(ctx, *in_conf_gt, *in_conf_prob, conf_buffer,
                           match_num, conf_num, batch_size, prior_num,
                           class_num, all_match_indices, all_neg_indices);
    }
    printf("\n conf_buffer %lld\n", conf_buffer.numel());
    for (int i = 0; i < conf_buffer.numel(); ++i) {
      printf("%f[%d], ", conf_buffer.data<T>()[i], (int)i);
    }

    int loc_offset = 0;
    int conf_offset = 0;

    // printf("!! 123 \n");
    // auto& place = *ctx.template
    // device_context<DeviceContext>().eigen_device();
    platform::CPUPlace cpu_place;
    platform::CPUDeviceContext cpu_ctx(cpu_place);

    for (int i = 0; i < input_num; ++i) {
      auto d_loc = d_ins_loc[i];
      auto d_conf = d_ins_conf[i];

      d_loc->mutable_data<T>(ins_loc[i]->dims(), platform::CPUPlace());
      d_conf->mutable_data<T>(ins_conf[i]->dims(), platform::CPUPlace());

      math::SetConstant<DeviceContext, T> set_constant;
      set_constant(ctx.template device_context<DeviceContext>(), d_loc, 0);
      set_constant(ctx.template device_context<DeviceContext>(), d_conf, 0);

      printf("\n d_loc %d: %lld\n", i, d_loc->numel());
      for (int i = 0; i < d_loc->numel(); ++i) {
        printf("%f[%d], ", d_loc->data<T>()[i], (int)i);
      }

      printf("\n d_conf %d: %lld\n", i, d_conf->numel());
      for (int i = 0; i < d_conf->numel(); ++i) {
        printf("%f[%d], ", d_conf->data<T>()[i], (int)i);
      }

      // framework::Tensor loc_g_buffer;
      // framework::Tensor conf_g_buffer;
      // loc_g_buffer.mutable_data<T>(d_loc->dims(), platform::CPUPlace());
      // conf_g_buffer.mutable_data<T>(d_conf->dims(), platform::CPUPlace());

      loc_offset +=
          math::TransposeFromNHWCToNCHW<platform::CPUDeviceContext, T>(
              cpu_ctx, loc_buffer, loc_size_sum, loc_offset, *d_loc);
      conf_offset +=
          math::TransposeFromNHWCToNCHW<platform::CPUDeviceContext, T>(
              cpu_ctx, conf_buffer, conf_size_sum, conf_offset, *d_conf);

      // auto d_loc_e = framework::EigenTensor<T, 4>::From(*d_loc);
      // auto loc_g_buffer_e = framework::EigenTensor<T, 4>::From(loc_g_buffer);
      // d_loc_e.device(place) = loc_g_buffer_e;

      // auto d_conf_e = framework::EigenTensor<T, 4>::From(*d_conf);
      // auto conf_g_buffer_e = framework::EigenTensor<T,
      // 4>::From(conf_g_buffer); d_conf_e.device(place) = conf_g_buffer_e;

      printf("\n out d_loc %d: %lld\n", i, d_loc->numel());
      for (int i = 0; i < d_loc->numel(); ++i) {
        printf("%f[%d], ", d_loc->data<T>()[i], (int)i);
      }

      printf("\n out d_conf %d: %lld\n", i, d_conf->numel());
      for (int i = 0; i < d_conf->numel(); ++i) {
        printf("%f[%d], ", d_conf->data<T>()[i], (int)i);
      }
    }
  }

 private:
  void CalcLocationLossBP(
      const framework::ExecutionContext& ctx, const framework::Tensor& loc_diff,
      const framework::Tensor& loc_gt, framework::Tensor& loc_buffer,
      int match_num, int batch_size, int prior_num,
      framework::EigenMatrix<const int>::Type& all_match_indices) const {
    framework::Tensor loc_diff_buffer;
    loc_diff_buffer.mutable_data<T>(loc_diff.dims(), platform::CPUPlace());
    MultiBoxLossSmoothL1BP<T>(ctx, loc_diff, loc_gt, loc_diff_buffer, match_num,
                              0.0);
    // scale gradient
    auto loc_diff_data = loc_diff_buffer.data<T>();
    printf("\n loc_diff_buffer[%d]\n", (int)loc_diff_buffer.numel());
    for (int i = 0; i < loc_diff_buffer.numel(); ++i) {
      printf("%f[%d], ", loc_diff_data[i], (int)i);
    }
    for (int i = 0; i < match_num * 4; ++i)
      loc_diff_data[i] *= (1. / match_num);
    // Copy gradient back
    size_t count = 0;
    for (int n = 0; n < batch_size; ++n) {
      for (int i = 0; i < prior_num; ++i) {
        if (all_match_indices(n, i) == -1) continue;
        T* loc_buffer_data = loc_buffer.data<T>() + n * prior_num * 4 + i * 4;
        std::copy(loc_diff_data + count * 4, loc_diff_data + (count + 1) * 4,
                  loc_buffer_data);
        ++count;
      }
    }
  }

  void CalcConfidenceLossBP(
      const framework::ExecutionContext& ctx, const framework::Tensor& conf_gt,
      const framework::Tensor& conf_prob, framework::Tensor& conf_buffer,
      int match_num, int conf_num, int batch_size, int prior_num, int class_num,
      framework::EigenMatrix<const int>::Type& all_match_indices,
      framework::EigenMatrix<const int>::Type& all_neg_indices) const {
    framework::Tensor conf_prob_temp;
    conf_prob_temp.mutable_data<T>(conf_prob.dims(), platform::CPUPlace());
    framework::CopyFrom(conf_prob, platform::CPUPlace(), ctx.device_context(),
                        &conf_prob_temp);
    auto conf_prob_data = conf_prob_temp.data<T>();
    auto conf_gt_data = conf_gt.data<int64_t>();
    for (int i = 0; i < conf_num; ++i)
      // conf_prob_data[i * class_num + static_cast<int>(conf_gt_data[i])] -= 1;
      conf_prob_data[i * class_num + conf_gt_data[i]] -= 1;

    for (int i = 0; i < conf_num * class_num; ++i)
      conf_prob_data[i] *= (1. / match_num);
    size_t count = 0;
    for (int n = 0; n < batch_size; ++n) {
      for (int i = 0; i < prior_num; ++i) {
        if (all_match_indices(n, i) == -1) continue;
        T* conf_diff_data =
            conf_buffer.data<T>() + n * prior_num * class_num + i * class_num;
        std::copy(conf_prob_data + count * class_num,
                  conf_prob_data + (count + 1) * class_num, conf_diff_data);
        ++count;
      }
      for (int i = 0; i < prior_num; ++i) {
        if (all_neg_indices(n, i) == -1) continue;
        int idx = all_neg_indices(n, i);
        T* conf_diff_data =
            conf_buffer.data<T>() + n * prior_num * class_num + idx * class_num;
        std::copy(conf_prob_data + count * class_num,
                  conf_prob_data + (count + 1) * class_num, conf_diff_data);
        ++count;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
