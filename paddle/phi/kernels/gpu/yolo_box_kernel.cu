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

#include "paddle/phi/kernels/yolo_box_kernel.h"

#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/yolo_box_util.h"

namespace phi {

template <typename T>
__global__ void KeYoloBoxFw(const T* input,
                            const int* imgsize,
                            T* boxes,
                            T* scores,
                            const float conf_thresh,
                            const int* anchors,
                            const int n,
                            const int h,
                            const int w,
                            const int an_num,
                            const int class_num,
                            const int box_num,
                            int input_size_h,
                            int input_size_w,
                            bool clip_bbox,
                            const float scale,
                            const float bias,
                            bool iou_aware,
                            const float iou_aware_factor) {
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

    int obj_idx = funcs::GetEntryIndex(
        i, j, k * w + l, an_num, an_stride, grid_num, 4, iou_aware);
    T conf = funcs::sigmoid<T>(input[obj_idx]);
    if (iou_aware) {
      int iou_idx =
          funcs::GetIoUIndex(i, j, k * w + l, an_num, an_stride, grid_num);
      T iou = funcs::sigmoid<T>(input[iou_idx]);
      conf = pow(conf, static_cast<T>(1. - iou_aware_factor)) *
             pow(iou, static_cast<T>(iou_aware_factor));
    }
    if (conf < conf_thresh) {
      continue;
    }

    int box_idx = funcs::GetEntryIndex(
        i, j, k * w + l, an_num, an_stride, grid_num, 0, iou_aware);
    funcs::GetYoloBox<T>(box,
                         input,
                         anchors,
                         l,
                         k,
                         j,
                         h,
                         w,
                         input_size_h,
                         input_size_w,
                         box_idx,
                         grid_num,
                         img_height,
                         img_width,
                         scale,
                         bias);
    box_idx = (i * box_num + j * grid_num + k * w + l) * 4;
    funcs::CalcDetectionBox<T>(
        boxes, box, box_idx, img_height, img_width, clip_bbox);

    int label_idx = funcs::GetEntryIndex(
        i, j, k * w + l, an_num, an_stride, grid_num, 5, iou_aware);
    int score_idx = (i * box_num + j * grid_num + k * w + l) * class_num;
    funcs::CalcLabelScore<T>(
        scores, input, label_idx, score_idx, class_num, conf, grid_num);
  }
}

template <typename T, typename Context>
void YoloBoxKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& img_size,
                   const std::vector<int>& anchors,
                   int class_num,
                   float conf_thresh,
                   int downsample_ratio,
                   bool clip_bbox,
                   float scale_x_y,
                   bool iou_aware,
                   float iou_aware_factor,
                   DenseTensor* boxes,
                   DenseTensor* scores) {
  auto* input = &x;
  float scale = scale_x_y;
  float bias = -0.5 * (scale - 1.);

  const int n = input->dims()[0];
  const int h = input->dims()[2];
  const int w = input->dims()[3];
  const int box_num = boxes->dims()[1];
  const int an_num = anchors.size() / 2;
  int input_size_h = downsample_ratio * h;
  int input_size_w = downsample_ratio * w;

  int bytes = sizeof(int) * anchors.size();
  DenseTensor tmp_anchors;
  tmp_anchors.Resize(phi::make_dim(anchors.size()));
  int* anchors_data = dev_ctx.template Alloc<int>(&tmp_anchors);
  const auto gplace = dev_ctx.GetPlace();
  const auto cplace = phi::CPUPlace();
  paddle::memory::Copy(
      gplace, anchors_data, cplace, anchors.data(), bytes, dev_ctx.stream());

  const T* input_data = input->data<T>();
  const int* imgsize_data = img_size.data<int>();
  boxes->Resize({n, box_num, 4});
  T* boxes_data = dev_ctx.template Alloc<T>(boxes);
  scores->Resize({n, box_num, class_num});
  T* scores_data = dev_ctx.template Alloc<T>(scores);
  phi::funcs::SetConstant<phi::GPUContext, T> set_zero;
  set_zero(dev_ctx, boxes, static_cast<T>(0));
  set_zero(dev_ctx, scores, static_cast<T>(0));
  backends::gpu::GpuLaunchConfig config =
      backends::gpu::GetGpuLaunchConfig1D(dev_ctx, n * box_num);

  dim3 thread_num = config.thread_per_block;
#ifdef WITH_NV_JETSON
  if (config.compute_capability == 53 || config.compute_capability == 62) {
    thread_num = 512;
  }
#endif

  KeYoloBoxFw<T><<<config.block_per_grid, thread_num, 0, dev_ctx.stream()>>>(
      input_data,
      imgsize_data,
      boxes_data,
      scores_data,
      conf_thresh,
      anchors_data,
      n,
      h,
      w,
      an_num,
      class_num,
      box_num,
      input_size_h,
      input_size_w,
      clip_bbox,
      scale,
      bias,
      iou_aware,
      iou_aware_factor);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    yolo_box, GPU, ALL_LAYOUT, phi::YoloBoxKernel, float, double) {}
