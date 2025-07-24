// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {
namespace funcs {

template <typename T>
HOSTDEVICE inline T sigmoid(T x) {
  return 1.0 / (1.0 + std::exp(-x));
}

template <typename T>
HOSTDEVICE inline void GetYoloBox(T* box,
                                  const T* x,
                                  const int* anchors,
                                  int64_t i,
                                  int64_t j,
                                  int64_t an_idx,
                                  int64_t grid_size_h,
                                  int64_t grid_size_w,
                                  int64_t input_size_h,
                                  int64_t input_size_w,
                                  int64_t index,
                                  int64_t stride,
                                  int64_t img_height,
                                  int64_t img_width,
                                  float scale,
                                  float bias) {
  box[0] = (i + sigmoid<T>(x[index]) * scale + bias) * img_width / grid_size_w;
  box[1] = (j + sigmoid<T>(x[index + stride]) * scale + bias) * img_height /
           grid_size_h;
  box[2] = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] * img_width /
           input_size_w;
  box[3] = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] *
           img_height / input_size_h;
}

HOSTDEVICE inline int64_t GetEntryIndex(int64_t batch,
                                        int64_t an_idx,
                                        int64_t hw_idx,
                                        int an_num,
                                        int64_t an_stride,
                                        int64_t stride,
                                        int64_t entry,
                                        bool iou_aware) {
  if (iou_aware) {
    return (batch * an_num + an_idx) * an_stride +
           (batch * an_num + an_num + entry) * stride + hw_idx;
  } else {
    return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
  }
}

HOSTDEVICE inline int GetIoUIndex(int64_t batch,
                                  int64_t an_idx,
                                  int64_t hw_idx,
                                  int an_num,
                                  int64_t an_stride,
                                  int64_t stride) {
  return batch * an_num * an_stride + (batch * an_num + an_idx) * stride +
         hw_idx;
}

template <typename T>
HOSTDEVICE inline void CalcDetectionBox(T* boxes,
                                        T* box,
                                        const int64_t box_idx,
                                        const int64_t img_height,
                                        const int64_t img_width,
                                        bool clip_bbox) {
  boxes[box_idx] = box[0] - box[2] / 2;
  boxes[box_idx + 1] = box[1] - box[3] / 2;
  boxes[box_idx + 2] = box[0] + box[2] / 2;
  boxes[box_idx + 3] = box[1] + box[3] / 2;

  if (clip_bbox) {
    boxes[box_idx] = boxes[box_idx] > 0 ? boxes[box_idx] : static_cast<T>(0);
    boxes[box_idx + 1] =
        boxes[box_idx + 1] > 0 ? boxes[box_idx + 1] : static_cast<T>(0);
    boxes[box_idx + 2] = boxes[box_idx + 2] < img_width - 1
                             ? boxes[box_idx + 2]
                             : static_cast<T>(img_width - 1);
    boxes[box_idx + 3] = boxes[box_idx + 3] < img_height - 1
                             ? boxes[box_idx + 3]
                             : static_cast<T>(img_height - 1);
  }
}

template <typename T>
HOSTDEVICE inline void CalcLabelScore(T* scores,
                                      const T* input,
                                      const int64_t label_idx,
                                      const int64_t score_idx,
                                      const int class_num,
                                      const T conf,
                                      const int64_t stride) {
  for (int i = 0; i < class_num; i++) {
    scores[score_idx + i] = conf * sigmoid<T>(input[label_idx + i * stride]);
  }
}

}  // namespace funcs
}  // namespace phi
