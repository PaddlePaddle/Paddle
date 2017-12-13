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
#include "paddle/operators/math/detection_util.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
inline void GetAccumulation(std::vector<std::pair<T, int>> in_pairs,
                            std::vector<int>* accu_vec) {
  std::stable_sort(in_pairs.begin(), in_pairs.end(),
                   math::SortScorePairDescend<int>);
  accu_vec->clear();
  size_t sum = 0;
  for (size_t i = 0; i < in_pairs.size(); ++i) {
    // auto score = in_pairs[i].first;
    auto count = in_pairs[i].second;
    sum += count;
    accu_vec->push_back(sum);
  }
}

template <typename Place, typename T>
class DetectionMAPOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input_label = ctx.Input<framework::LoDTensor>("Label");
    auto* input_detect = ctx.Input<framework::Tensor>("Detect");
    auto* map_out = ctx.Output<framework::Tensor>("MAP");

    float overlap_threshold = ctx.Attr<float>("overlap_threshold");
    float evaluate_difficult = ctx.Attr<bool>("evaluate_difficult");
    std::string ap_type = ctx.Attr<std::string>("ap_type");

    auto label_lod = input_label->lod();
    PADDLE_ENFORCE_EQ(label_lod.size(), 1UL,
                      "Only support one level sequence now.");
    auto batch_size = label_lod[0].size() - 1;

    std::vector<std::map<int, std::vector<operators::math::BBox<T>>>> gt_bboxes;

    std::vector<
        std::map<int, std::vector<std::pair<T, operators::math::BBox<T>>>>>
        detect_bboxes;

    if (platform::is_gpu_place(ctx.GetPlace())) {
      framework::LoDTensor input_label_cpu;
      framework::Tensor input_detect_cpu;
      input_label_cpu.set_lod(input_label->lod());
      input_label_cpu.Resize(input_label->dims());
      input_detect_cpu.Resize(input_detect->dims());
      input_label_cpu.mutable_data<T>(platform::CPUPlace());
      input_detect_cpu.mutable_data<T>(platform::CPUPlace());
      framework::CopyFrom(*input_label, platform::CPUPlace(),
                          ctx.device_context(), &input_label_cpu);
      framework::CopyFrom(*input_detect, platform::CPUPlace(),
                          ctx.device_context(), &input_detect_cpu);
      GetBBoxes(input_label_cpu, input_detect_cpu, gt_bboxes, detect_bboxes);
    } else {
      GetBBoxes(*input_label, *input_detect, gt_bboxes, detect_bboxes);
    }

    std::map<int, int> label_pos_count;
    std::map<int, std::vector<std::pair<T, int>>> true_pos;
    std::map<int, std::vector<std::pair<T, int>>> false_pos;

    CalcTrueAndFalsePositive(batch_size, evaluate_difficult, overlap_threshold,
                             gt_bboxes, detect_bboxes, label_pos_count,
                             true_pos, false_pos);

    T map = CalcMAP(ap_type, label_pos_count, true_pos, false_pos);

    T* map_data = nullptr;
    framework::Tensor map_cpu;
    map_out->mutable_data<T>(ctx.GetPlace());
    if (platform::is_gpu_place(ctx.GetPlace())) {
      map_data = map_cpu.mutable_data<T>(map_out->dims(), platform::CPUPlace());
      map_data[0] = map;
      framework::CopyFrom(map_cpu, platform::CPUPlace(), ctx.device_context(),
                          map_out);
    } else {
      map_data = map_out->mutable_data<T>(ctx.GetPlace());
      map_data[0] = map;
    }
  }

 protected:
  void GetBBoxes(
      const framework::LoDTensor& input_label,
      const framework::Tensor& input_detect,
      std::vector<std::map<int, std::vector<operators::math::BBox<T>>>>&
          gt_bboxes,
      std::vector<
          std::map<int, std::vector<std::pair<T, operators::math::BBox<T>>>>>&
          detect_bboxes) const {
    const T* label_data = input_label.data<T>();
    const T* detect_data = input_detect.data<T>();

    auto label_lod = input_label.lod();
    auto batch_size = label_lod[0].size() - 1;
    auto label_index = label_lod[0];

    for (size_t n = 0; n < batch_size; ++n) {
      std::map<int, std::vector<operators::math::BBox<T>>> bboxes;
      for (int i = label_index[n]; i < label_index[n + 1]; ++i) {
        std::vector<operators::math::BBox<T>> bbox;
        math::GetBBoxFromLabelData<T>(label_data + i * 6, 1, bbox);
        int label = static_cast<int>(label_data[i * 6]);
        bboxes[label].push_back(bbox[0]);
      }
      gt_bboxes.push_back(bboxes);
    }

    size_t n = 0;
    size_t detect_box_count = input_detect.dims()[0];
    for (size_t img_id = 0; img_id < batch_size; ++img_id) {
      std::map<int, std::vector<std::pair<T, operators::math::BBox<T>>>> bboxes;
      size_t cur_img_id = static_cast<size_t>((detect_data + n * 7)[0]);
      while (cur_img_id == img_id && n < detect_box_count) {
        std::vector<T> label;
        std::vector<T> score;
        std::vector<operators::math::BBox<T>> bbox;
        math::GetBBoxFromDetectData<T>(detect_data + n * 7, 1, label, score,
                                       bbox);
        bboxes[label[0]].push_back(std::make_pair(score[0], bbox[0]));
        ++n;
        cur_img_id = static_cast<size_t>((detect_data + n * 7)[0]);
      }
      detect_bboxes.push_back(bboxes);
    }
  }

  void CalcTrueAndFalsePositive(
      size_t batch_size, bool evaluate_difficult, float overlap_threshold,
      const std::vector<std::map<int, std::vector<operators::math::BBox<T>>>>&
          gt_bboxes,
      const std::vector<
          std::map<int, std::vector<std::pair<T, operators::math::BBox<T>>>>>&
          detect_bboxes,
      std::map<int, int>& label_pos_count,
      std::map<int, std::vector<std::pair<T, int>>>& true_pos,
      std::map<int, std::vector<std::pair<T, int>>>& false_pos) const {
    for (size_t n = 0; n < batch_size; ++n) {
      auto image_gt_bboxes = gt_bboxes[n];
      for (auto it = image_gt_bboxes.begin(); it != image_gt_bboxes.end();
           ++it) {
        size_t count = 0;
        auto labeled_bboxes = it->second;
        if (evaluate_difficult) {
          count = labeled_bboxes.size();
        } else {
          for (size_t i = 0; i < labeled_bboxes.size(); ++i)
            if (!(labeled_bboxes[i].is_difficult)) ++count;
        }
        if (count == 0) {
          continue;
        }
        int label = it->first;
        if (label_pos_count.find(label) == label_pos_count.end()) {
          label_pos_count[label] = count;
        } else {
          label_pos_count[label] += count;
        }
      }
    }

    for (size_t n = 0; n < detect_bboxes.size(); ++n) {
      auto image_gt_bboxes = gt_bboxes[n];
      auto detections = detect_bboxes[n];

      if (image_gt_bboxes.size() == 0) {
        for (auto it = detections.begin(); it != detections.end(); ++it) {
          auto pred_bboxes = it->second;
          int label = it->first;
          for (size_t i = 0; i < pred_bboxes.size(); ++i) {
            auto score = pred_bboxes[i].first;
            true_pos[label].push_back(std::make_pair(score, 0));
            false_pos[label].push_back(std::make_pair(score, 1));
          }
        }
        continue;
      }

      for (auto it = detections.begin(); it != detections.end(); ++it) {
        int label = it->first;
        auto pred_bboxes = it->second;
        if (image_gt_bboxes.find(label) == image_gt_bboxes.end()) {
          for (size_t i = 0; i < pred_bboxes.size(); ++i) {
            auto score = pred_bboxes[i].first;
            true_pos[label].push_back(std::make_pair(score, 0));
            false_pos[label].push_back(std::make_pair(score, 1));
          }
          continue;
        }

        auto matched_bboxes = image_gt_bboxes.find(label)->second;
        std::vector<bool> visited(matched_bboxes.size(), false);
        // Sort detections in descend order based on scores
        std::sort(pred_bboxes.begin(), pred_bboxes.end(),
                  math::SortScorePairDescend<operators::math::BBox<T>>);
        for (size_t i = 0; i < pred_bboxes.size(); ++i) {
          float max_overlap = -1.0;
          size_t max_idx = 0;
          auto score = pred_bboxes[i].first;
          for (size_t j = 0; j < matched_bboxes.size(); ++j) {
            float overlap =
                JaccardOverlap(pred_bboxes[i].second, matched_bboxes[j]);
            if (overlap > max_overlap) {
              max_overlap = overlap;
              max_idx = j;
            }
          }
          if (max_overlap > overlap_threshold) {
            bool match_evaluate_difficult =
                evaluate_difficult ||
                (!evaluate_difficult && !matched_bboxes[max_idx].is_difficult);
            if (match_evaluate_difficult) {
              if (!visited[max_idx]) {
                true_pos[label].push_back(std::make_pair(score, 1));
                false_pos[label].push_back(std::make_pair(score, 0));
                visited[max_idx] = true;
              } else {
                true_pos[label].push_back(std::make_pair(score, 0));
                false_pos[label].push_back(std::make_pair(score, 1));
              }
            }
          } else {
            true_pos[label].push_back(std::make_pair(score, 0));
            false_pos[label].push_back(std::make_pair(score, 1));
          }
        }
      }
    }
  }

  T CalcMAP(
      std::string ap_type, const std::map<int, int>& label_pos_count,
      const std::map<int, std::vector<std::pair<T, int>>>& true_pos,
      const std::map<int, std::vector<std::pair<T, int>>>& false_pos) const {
    T mAP = 0.0;
    int count = 0;
    for (auto it = label_pos_count.begin(); it != label_pos_count.end(); ++it) {
      int label = it->first;
      int label_num_pos = it->second;
      if (label_num_pos == 0 || true_pos.find(label) == true_pos.end())
        continue;
      auto label_true_pos = true_pos.find(label)->second;
      auto label_false_pos = false_pos.find(label)->second;
      // Compute average precision.
      std::vector<int> tp_sum;
      GetAccumulation<T>(label_true_pos, &tp_sum);
      std::vector<int> fp_sum;
      GetAccumulation<T>(label_false_pos, &fp_sum);
      std::vector<float> precision, recall;
      size_t num = tp_sum.size();
      // Compute Precision.
      for (size_t i = 0; i < num; ++i) {
        // CHECK_LE(tpCumSum[i], labelNumPos);
        precision.push_back(static_cast<float>(tp_sum[i]) /
                            static_cast<float>(tp_sum[i] + fp_sum[i]));
        recall.push_back(static_cast<float>(tp_sum[i]) / label_num_pos);
      }
      // VOC2007 style
      if (ap_type == "11point") {
        std::vector<float> max_precisions(11, 0.0);
        int start_idx = num - 1;
        for (int j = 10; j >= 0; --j)
          for (int i = start_idx; i >= 0; --i) {
            if (recall[i] < j / 10.) {
              start_idx = i;
              if (j > 0) max_precisions[j - 1] = max_precisions[j];
              break;
            } else {
              if (max_precisions[j] < precision[i])
                max_precisions[j] = precision[i];
            }
          }
        for (int j = 10; j >= 0; --j) mAP += max_precisions[j] / 11;
        ++count;
      } else if (ap_type == "Integral") {
        // Nature integral
        float average_precisions = 0.;
        float prev_recall = 0.;
        for (size_t i = 0; i < num; ++i) {
          if (fabs(recall[i] - prev_recall) > 1e-6)
            average_precisions += precision[i] * fabs(recall[i] - prev_recall);
          prev_recall = recall[i];
        }
        mAP += average_precisions;
        ++count;
      } else {
        LOG(FATAL) << "Unkown ap version: " << ap_type;
      }
    }
    if (count != 0) mAP /= count;
    return mAP * 100;
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle
