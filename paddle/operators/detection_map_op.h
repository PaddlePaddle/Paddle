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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

enum APType { kNone = 0, kIntegral, k11point };

APType GetAPType(std::string str) {
  if (str == "integral") {
    return APType::kIntegral;
  } else if (str == "11point") {
    return APType::k11point;
  } else {
    return APType::kNone;
  }
}

template <typename T>
inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                                 const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

template <typename T>
inline void GetAccumulation(std::vector<std::pair<T, int>> in_pairs,
                            std::vector<int>* accu_vec) {
  std::stable_sort(in_pairs.begin(), in_pairs.end(), SortScorePairDescend<int>);
  accu_vec->clear();
  size_t sum = 0;
  for (size_t i = 0; i < in_pairs.size(); ++i) {
    auto count = in_pairs[i].second;
    sum += count;
    accu_vec->push_back(sum);
  }
}

template <typename Place, typename T>
class DetectionMAPOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_detect = ctx.Input<framework::LoDTensor>("Detection");
    auto* in_label = ctx.Input<framework::LoDTensor>("Label");
    auto* out_map = ctx.Output<framework::Tensor>("MAP");

    float overlap_threshold = ctx.Attr<float>("overlap_threshold");
    float evaluate_difficult = ctx.Attr<bool>("evaluate_difficult");
    auto ap_type = GetAPType(ctx.Attr<std::string>("ap_type"));

    auto label_lod = in_label->lod();
    auto detect_lod = in_detect->lod();
    PADDLE_ENFORCE_EQ(label_lod.size(), 1UL,
                      "Only support one level sequence now.");
    PADDLE_ENFORCE_EQ(label_lod[0].size(), detect_lod[0].size(),
                      "The batch_size of input(Label) and input(Detection) "
                      "must be the same.");

    std::vector<std::map<int, std::vector<Box>>> gt_boxes;
    std::vector<std::map<int, std::vector<std::pair<T, Box>>>> detect_boxes;

    GetBoxes(*in_label, *in_detect, gt_boxes, detect_boxes);

    std::map<int, int> label_pos_count;
    std::map<int, std::vector<std::pair<T, int>>> true_pos;
    std::map<int, std::vector<std::pair<T, int>>> false_pos;

    CalcTrueAndFalsePositive(gt_boxes, detect_boxes, evaluate_difficult,
                             overlap_threshold, label_pos_count, true_pos,
                             false_pos);

    T map = CalcMAP(ap_type, label_pos_count, true_pos, false_pos);

    T* map_data = out_map->mutable_data<T>(ctx.GetPlace());
    map_data[0] = map;
  }

 protected:
  struct Box {
    Box(T xmin, T ymin, T xmax, T ymax)
        : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax), is_difficult(false) {}

    T xmin, ymin, xmax, ymax;
    bool is_difficult;
  };

  inline T JaccardOverlap(const Box& box1, const Box& box2) const {
    if (box2.xmin > box1.xmax || box2.xmax < box1.xmin ||
        box2.ymin > box1.ymax || box2.ymax < box1.ymin) {
      return 0.0;
    } else {
      T inter_xmin = std::max(box1.xmin, box2.xmin);
      T inter_ymin = std::max(box1.ymin, box2.ymin);
      T inter_xmax = std::min(box1.xmax, box2.xmax);
      T inter_ymax = std::min(box1.ymax, box2.ymax);

      T inter_width = inter_xmax - inter_xmin;
      T inter_height = inter_ymax - inter_ymin;
      T inter_area = inter_width * inter_height;

      T bbox_area1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin);
      T bbox_area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin);

      return inter_area / (bbox_area1 + bbox_area2 - inter_area);
    }
  }

  void GetBoxes(const framework::LoDTensor& input_label,
                const framework::LoDTensor& input_detect,
                std::vector<std::map<int, std::vector<Box>>>& gt_boxes,
                std::vector<std::map<int, std::vector<std::pair<T, Box>>>>&
                    detect_boxes) const {
    auto labels = framework::EigenTensor<T, 2>::From(input_label);
    auto detect = framework::EigenTensor<T, 2>::From(input_detect);

    auto label_lod = input_label.lod();
    auto detect_lod = input_detect.lod();

    int batch_size = label_lod[0].size() - 1;
    auto label_index = label_lod[0];

    for (int n = 0; n < batch_size; ++n) {
      std::map<int, std::vector<Box>> boxes;
      for (int i = label_index[n]; i < label_index[n + 1]; ++i) {
        Box box(labels(i, 2), labels(i, 3), labels(i, 4), labels(i, 5));
        int label = labels(i, 0);
        auto is_difficult = labels(i, 1);
        if (std::abs(is_difficult - 0.0) < 1e-6)
          box.is_difficult = false;
        else
          box.is_difficult = true;
        boxes[label].push_back(box);
      }
      gt_boxes.push_back(boxes);
    }

    auto detect_index = detect_lod[0];
    for (int n = 0; n < batch_size; ++n) {
      std::map<int, std::vector<std::pair<T, Box>>> boxes;
      for (int i = detect_index[n]; i < detect_index[n + 1]; ++i) {
        Box box(detect(i, 2), detect(i, 3), detect(i, 4), detect(i, 5));
        int label = detect(i, 0);
        auto score = detect(i, 1);
        boxes[label].push_back(std::make_pair(score, box));
      }
      detect_boxes.push_back(boxes);
    }
  }

  void CalcTrueAndFalsePositive(
      const std::vector<std::map<int, std::vector<Box>>>& gt_boxes,
      const std::vector<std::map<int, std::vector<std::pair<T, Box>>>>&
          detect_boxes,
      bool evaluate_difficult, float overlap_threshold,
      std::map<int, int>& label_pos_count,
      std::map<int, std::vector<std::pair<T, int>>>& true_pos,
      std::map<int, std::vector<std::pair<T, int>>>& false_pos) const {
    int batch_size = gt_boxes.size();
    for (int n = 0; n < batch_size; ++n) {
      auto image_gt_boxes = gt_boxes[n];
      for (auto it = image_gt_boxes.begin(); it != image_gt_boxes.end(); ++it) {
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

    for (size_t n = 0; n < detect_boxes.size(); ++n) {
      auto image_gt_boxes = gt_boxes[n];
      auto detections = detect_boxes[n];

      if (image_gt_boxes.size() == 0) {
        for (auto it = detections.begin(); it != detections.end(); ++it) {
          auto pred_boxes = it->second;
          int label = it->first;
          for (size_t i = 0; i < pred_boxes.size(); ++i) {
            auto score = pred_boxes[i].first;
            true_pos[label].push_back(std::make_pair(score, 0));
            false_pos[label].push_back(std::make_pair(score, 1));
          }
        }
        continue;
      }

      for (auto it = detections.begin(); it != detections.end(); ++it) {
        int label = it->first;
        auto pred_boxes = it->second;
        if (image_gt_boxes.find(label) == image_gt_boxes.end()) {
          for (size_t i = 0; i < pred_boxes.size(); ++i) {
            auto score = pred_boxes[i].first;
            true_pos[label].push_back(std::make_pair(score, 0));
            false_pos[label].push_back(std::make_pair(score, 1));
          }
          continue;
        }

        auto matched_bboxes = image_gt_boxes.find(label)->second;
        std::vector<bool> visited(matched_bboxes.size(), false);
        // Sort detections in descend order based on scores
        std::sort(pred_boxes.begin(), pred_boxes.end(),
                  SortScorePairDescend<Box>);
        for (size_t i = 0; i < pred_boxes.size(); ++i) {
          T max_overlap = -1.0;
          size_t max_idx = 0;
          auto score = pred_boxes[i].first;
          for (size_t j = 0; j < matched_bboxes.size(); ++j) {
            T overlap = JaccardOverlap(pred_boxes[i].second, matched_bboxes[j]);
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
      APType ap_type, const std::map<int, int>& label_pos_count,
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
      std::vector<T> precision, recall;
      size_t num = tp_sum.size();
      // Compute Precision.
      for (size_t i = 0; i < num; ++i) {
        // CHECK_LE(tpCumSum[i], labelNumPos);
        precision.push_back(static_cast<T>(tp_sum[i]) /
                            static_cast<T>(tp_sum[i] + fp_sum[i]));
        recall.push_back(static_cast<T>(tp_sum[i]) / label_num_pos);
      }
      // VOC2007 style
      if (ap_type == APType::k11point) {
        std::vector<T> max_precisions(11, 0.0);
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
      } else if (ap_type == APType::kIntegral) {
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
