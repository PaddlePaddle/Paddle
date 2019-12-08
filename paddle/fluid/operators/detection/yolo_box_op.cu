/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/detection/yolo_box_op.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void KeYoloBoxFw(const T* input, const int* imgsize, T* boxes,
                            T* scores, const float conf_thresh,
                            const int* anchors, const int n, const int h,
                            const int w, const int an_num, const int class_num,
                            const int box_num, int input_size, bool clip_bbox) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  T box[4];
  for (; tid < n * box_num; tid += stride) {
    int grid_num = h * w;
    int i = tid / box_num;
    int j = (tid % box_num) / grid_num;
    int k = (tid % grid_num) / w;
    int l = tid % w;

    int an_stride = (5 + class_num) * grid_num;
    int img_height = imgsize[2 * i];
    int img_width = imgsize[2 * i + 1];

    int obj_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, grid_num, 4);
    T conf = sigmoid<T>(input[obj_idx]);
    if (conf < conf_thresh) {
      continue;
    }

    int box_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, grid_num, 0);
    GetYoloBox<T>(box, input, anchors, l, k, j, h, input_size, box_idx,
                  grid_num, img_height, img_width);
    box_idx = (i * box_num + j * grid_num + k * w + l) * 4;
    CalcDetectionBox<T>(boxes, box, box_idx, img_height, img_width, clip_bbox);

    int label_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, grid_num, 5);
    int score_idx = (i * box_num + j * grid_num + k * w + l) * class_num;
    CalcLabelScore<T>(scores, input, label_idx, score_idx, class_num, conf,
                      grid_num);
  }
}

template <typename T>
class YoloBoxOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* img_size = ctx.Input<Tensor>("ImgSize");
    auto* boxes = ctx.Output<Tensor>("Boxes");
    auto* scores = ctx.Output<Tensor>("Scores");

    auto anchors = ctx.Attr<std::vector<int>>("anchors");
    int class_num = ctx.Attr<int>("class_num");
    float conf_thresh = ctx.Attr<float>("conf_thresh");
    int downsample_ratio = ctx.Attr<int>("downsample_ratio");
    bool clip_bbox = ctx.Attr<bool>("clip_bbox");

    const int n = input->dims()[0];
    const int h = input->dims()[2];
    const int w = input->dims()[3];
    const int box_num = boxes->dims()[1];
    const int an_num = anchors.size() / 2;
    int input_size = downsample_ratio * h;

    auto& dev_ctx = ctx.cuda_device_context();
    int bytes = sizeof(int) * anchors.size();
    auto anchors_ptr = memory::Alloc(dev_ctx, sizeof(int) * anchors.size());
    int* anchors_data = reinterpret_cast<int*>(anchors_ptr->ptr());
    const auto gplace = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    const auto cplace = platform::CPUPlace();
    memory::Copy(gplace, anchors_data, cplace, anchors.data(), bytes,
                 dev_ctx.stream());

    const T* input_data = input->data<T>();
    const int* imgsize_data = img_size->data<int>();
    T* boxes_data = boxes->mutable_data<T>({n, box_num, 4}, ctx.GetPlace());
    T* scores_data =
        scores->mutable_data<T>({n, box_num, class_num}, ctx.GetPlace());
    math::SetConstant<platform::CUDADeviceContext, T> set_zero;
    set_zero(dev_ctx, boxes, static_cast<T>(0));
    set_zero(dev_ctx, scores, static_cast<T>(0));

    int grid_dim = (n * box_num + 512 - 1) / 512;
    grid_dim = grid_dim > 8 ? 8 : grid_dim;

    KeYoloBoxFw<T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
        input_data, imgsize_data, boxes_data, scores_data, conf_thresh,
        anchors_data, n, h, w, an_num, class_num, box_num, input_size,
        clip_bbox);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(yolo_box, ops::YoloBoxOpCUDAKernel<float>,
                        ops::YoloBoxOpCUDAKernel<double>);
