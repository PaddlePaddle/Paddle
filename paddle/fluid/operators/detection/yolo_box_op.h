/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
HOSTDEVICE inline T sigmoid(T x) {
  return 1.0 / (1.0 + std::exp(-x));
}

template <typename T>
HOSTDEVICE inline void GetYoloBox(T* box, const T* x, const int* anchors, int i,
                                  int j, int an_idx, int grid_size,
                                  int input_size, int index, int stride,
                                  int img_height, int img_width, float scale,
                                  float bias) {
  box[0] = (i + sigmoid<T>(x[index]) * scale + bias) * img_width / grid_size;
  box[1] = (j + sigmoid<T>(x[index + stride]) * scale + bias) * img_height /
           grid_size;
  box[2] = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] * img_width /
           input_size;
  box[3] = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] *
           img_height / input_size;
}

HOSTDEVICE inline int GetEntryIndex(int batch, int an_idx, int hw_idx,
                                    int an_num, int an_stride, int stride,
                                    int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

template <typename T>
HOSTDEVICE inline void CalcDetectionBox(T* boxes, T* box, const int box_idx,
                                        const int img_height,
                                        const int img_width, bool clip_bbox) {
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
HOSTDEVICE inline void CalcLabelScore(T* scores, const T* input,
                                      const int label_idx, const int score_idx,
                                      const int class_num, const T conf,
                                      const int stride) {
  for (int i = 0; i < class_num; i++) {
    scores[score_idx + i] = conf * sigmoid<T>(input[label_idx + i * stride]);
  }
}

template <typename T>
class YoloBoxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* imgsize = ctx.Input<Tensor>("ImgSize");
    auto* boxes = ctx.Output<Tensor>("Boxes");
    auto* scores = ctx.Output<Tensor>("Scores");
    auto anchors = ctx.Attr<std::vector<int>>("anchors");
    int class_num = ctx.Attr<int>("class_num");
    float conf_thresh = ctx.Attr<float>("conf_thresh");
    int downsample_ratio = ctx.Attr<int>("downsample_ratio");
    bool clip_bbox = ctx.Attr<bool>("clip_bbox");
    float scale = ctx.Attr<float>("scale_x_y");
    float bias = -0.5 * (scale - 1.);

    const int n = input->dims()[0];
    const int h = input->dims()[2];
    const int w = input->dims()[3];
    const int box_num = boxes->dims()[1];
    const int an_num = anchors.size() / 2;
    int input_size = downsample_ratio * h;

    const int stride = h * w;
    const int an_stride = (class_num + 5) * stride;

    Tensor anchors_;
    auto anchors_data =
        anchors_.mutable_data<int>({an_num * 2}, ctx.GetPlace());
    std::copy(anchors.begin(), anchors.end(), anchors_data);

    const T* input_data = input->data<T>();
    const int* imgsize_data = imgsize->data<int>();
    T* boxes_data = boxes->mutable_data<T>({n, box_num, 4}, ctx.GetPlace());
    memset(boxes_data, 0, boxes->numel() * sizeof(T));
    T* scores_data =
        scores->mutable_data<T>({n, box_num, class_num}, ctx.GetPlace());
    memset(scores_data, 0, scores->numel() * sizeof(T));

    T box[4];
    for (int i = 0; i < n; i++) {
      int img_height = imgsize_data[2 * i];
      int img_width = imgsize_data[2 * i + 1];

      for (int j = 0; j < an_num; j++) {
        for (int k = 0; k < h; k++) {
          for (int l = 0; l < w; l++) {
            int obj_idx =
                GetEntryIndex(i, j, k * w + l, an_num, an_stride, stride, 4);
            T conf = sigmoid<T>(input_data[obj_idx]);
            if (conf < conf_thresh) {
              continue;
            }

            int box_idx =
                GetEntryIndex(i, j, k * w + l, an_num, an_stride, stride, 0);
            GetYoloBox<T>(box, input_data, anchors_data, l, k, j, h, input_size,
                          box_idx, stride, img_height, img_width, scale, bias);
            box_idx = (i * box_num + j * stride + k * w + l) * 4;
            CalcDetectionBox<T>(boxes_data, box, box_idx, img_height, img_width,
                                clip_bbox);

            int label_idx =
                GetEntryIndex(i, j, k * w + l, an_num, an_stride, stride, 5);
            int score_idx = (i * box_num + j * stride + k * w + l) * class_num;
            CalcLabelScore<T>(scores_data, input_data, label_idx, score_idx,
                              class_num, conf, stride);
          }
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
