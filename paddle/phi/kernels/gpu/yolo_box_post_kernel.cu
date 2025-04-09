// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/yolo_box_util.h"
namespace phi {
struct Box {
  float x, y, w, h;
};

struct Detection {
  Box bbox;
  int classes;
  float* prob;
  float* mask;
  float objectness;
  int sort_class;
  int max_prob_class_index;
};

struct TensorInfo {
  int bbox_count_host;  // record bbox numbers
  int bbox_count_max_alloc{50};
  float* bboxes_dev_ptr;
  float* bboxes_host_ptr;
  int* bbox_count_device_ptr;  // Box counter in gpu memory, used by atomicAdd
};

static int NMSComparator(const void* pa, const void* pb) {
  const Detection a = *reinterpret_cast<const Detection*>(pa);
  const Detection b = *reinterpret_cast<const Detection*>(pb);
  if (a.max_prob_class_index > b.max_prob_class_index)
    return 1;
  else if (a.max_prob_class_index < b.max_prob_class_index)
    return -1;

  float diff = 0;
  if (b.sort_class >= 0) {
    diff = a.prob[b.sort_class] - b.prob[b.sort_class];
  } else {
    diff = a.objectness - b.objectness;
  }

  if (diff < 0)
    return 1;
  else if (diff > 0)
    return -1;
  return 0;
}

static float Overlap(float x1, float w1, float x2, float w2) {
  float l1 = x1 - w1 / 2;
  float l2 = x2 - w2 / 2;
  float left = l1 > l2 ? l1 : l2;
  float r1 = x1 + w1 / 2;
  float r2 = x2 + w2 / 2;
  float right = r1 < r2 ? r1 : r2;
  return right - left;
}

static float BoxIntersection(Box a, Box b) {
  float w = Overlap(a.x, a.w, b.x, b.w);
  float h = Overlap(a.y, a.h, b.y, b.h);
  if (w < 0 || h < 0) return 0;
  float area = w * h;
  return area;
}

static float BoxUnion(Box a, Box b) {
  float i = BoxIntersection(a, b);
  float u = a.w * a.h + b.w * b.h - i;
  return u;
}

static float BoxIOU(Box a, Box b) {
  return BoxIntersection(a, b) / BoxUnion(a, b);
}

static void PostNMS(std::vector<Detection>* det_bboxes,
                    float thresh,
                    int classes) {
  int total = det_bboxes->size();
  if (total <= 0) {
    return;
  }

  Detection* dets = det_bboxes->data();
  int i, j, k;
  k = total - 1;
  for (i = 0; i <= k; ++i) {
    if (dets[i].objectness == 0) {
      Detection swap = dets[i];
      dets[i] = dets[k];
      dets[k] = swap;
      --k;
      --i;
    }
  }
  total = k + 1;

  qsort(dets, total, sizeof(Detection), NMSComparator);

  for (i = 0; i < total; ++i) {
    if (dets[i].objectness == 0) continue;
    Box a = dets[i].bbox;
    for (j = i + 1; j < total; ++j) {
      if (dets[j].objectness == 0) continue;
      if (dets[j].max_prob_class_index != dets[i].max_prob_class_index) break;
      Box b = dets[j].bbox;
      if (BoxIOU(a, b) > thresh) {
        dets[j].objectness = 0;
        for (k = 0; k < classes; ++k) {
          dets[j].prob[k] = 0;
        }
      }
    }
  }
}

__global__ void YoloBoxNum(const float* input,
                           int* bbox_count,
                           const int grid_size,
                           const int class_num,
                           const int anchors_num,
                           float prob_thresh) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int y_id = blockIdx.y * blockDim.y + threadIdx.y;
  int z_id = blockIdx.z * blockDim.z + threadIdx.z;
  if ((x_id >= grid_size) || (y_id >= grid_size) || (z_id >= anchors_num)) {
    return;
  }

  const int grids_num = grid_size * grid_size;
  const int bbindex = y_id * grid_size + x_id;
  float objectness = input[bbindex + grids_num * (z_id * (5 + class_num) + 4)];
  if (objectness < prob_thresh) {
    return;
  }

  atomicAdd(bbox_count, 1);
}

__global__ void YoloTensorParseKernel(const float* input,
                                      const float* im_shape_data,
                                      const float* im_scale_data,
                                      float* output,
                                      int* bbox_index,
                                      const int grid_size,
                                      const int class_num,
                                      const int anchors_num,
                                      const int netw,
                                      const int neth,
                                      int* biases,
                                      float prob_thresh) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int y_id = blockIdx.y * blockDim.y + threadIdx.y;
  int z_id = blockIdx.z * blockDim.z + threadIdx.z;
  if ((x_id >= grid_size) || (y_id >= grid_size) || (z_id >= anchors_num)) {
    return;
  }

  const float pic_h = im_shape_data[0] / im_scale_data[0];
  const float pic_w = im_shape_data[1] / im_scale_data[1];
  const int grids_num = grid_size * grid_size;
  const int bbindex = y_id * grid_size + x_id;
  float objectness = input[bbindex + grids_num * (z_id * (5 + class_num) + 4)];
  if (objectness < prob_thresh) {
    return;
  }

  int cur_bbox_index = atomicAdd(bbox_index, 1);
  int tensor_index = cur_bbox_index * (5 + class_num);

  // x
  float x = input[bbindex + grids_num * (z_id * (5 + class_num) + 0)];
  x = (x + static_cast<float>(x_id)) * static_cast<float>(pic_w) /
      static_cast<float>(grid_size);
  // y
  float y = input[bbindex + grids_num * (z_id * (5 + class_num) + 1)];
  y = (y + static_cast<float>(y_id)) * static_cast<float>(pic_h) /
      static_cast<float>(grid_size);
  // w
  float w = input[bbindex + grids_num * (z_id * (5 + class_num) + 2)];
  w = w * biases[2 * z_id] * pic_w / netw;
  // h
  float h = input[bbindex + grids_num * (z_id * (5 + class_num) + 3)];
  h = h * biases[2 * z_id + 1] * pic_h / neth;

  output[tensor_index] = objectness;
  output[tensor_index + 1] = x - w / 2;
  output[tensor_index + 2] = y - h / 2;
  output[tensor_index + 3] = x + w / 2;
  output[tensor_index + 4] = y + h / 2;
  output[tensor_index + 1] =
      output[tensor_index + 1] > 0 ? output[tensor_index + 1] : 0.f;
  output[tensor_index + 2] =
      output[tensor_index + 2] > 0 ? output[tensor_index + 2] : 0.f;
  output[tensor_index + 3] = output[tensor_index + 3] < pic_w - 1
                                 ? output[tensor_index + 3]
                                 : pic_w - 1;
  output[tensor_index + 4] = output[tensor_index + 4] < pic_h - 1
                                 ? output[tensor_index + 4]
                                 : pic_h - 1;

  // Probabilities of classes
  for (int i = 0; i < class_num; ++i) {
    float prob =
        input[bbindex + grids_num * (z_id * (5 + class_num) + (5 + i))] *
        objectness;
    output[tensor_index + 5 + i] = prob;
  }
}

static void YoloTensorParseCuda(
    const float* input_data,  // [in] YOLO_BOX_HEAD layer output
    const float* image_shape_data,
    const float* image_scale_data,
    float** bboxes_tensor_ptr,  // [out] Bounding boxes output tensor
    int* bbox_count_max_alloc,  // [in/out] maximum bounding Box number
                                // allocated in dev
    int* bbox_count_host,  // [in/out] bounding boxes number recorded in host
    int* bbox_count_device_ptr,  // [in/out] bounding boxes number calculated
                                 // in
                                 // device side
    int* bbox_index_device_ptr,  // [in] bounding Box index for kernel threads
                                 // shared access
    int grid_size,
    int class_num,
    int anchors_num,
    int netw,
    int neth,
    int* biases_device,
    float prob_thresh) {
  dim3 threads_per_block(16, 16, 4);
  dim3 number_of_blocks((grid_size / threads_per_block.x) + 1,
                        (grid_size / threads_per_block.y) + 1,
                        (anchors_num / threads_per_block.z) + 1);

  // Estimate how many boxes will be chosen
  int bbox_count = 0;
#ifdef PADDLE_WITH_HIP
  hipMemcpy(
      bbox_count_device_ptr, &bbox_count, sizeof(int), hipMemcpyHostToDevice);
#else
  cudaMemcpy(
      bbox_count_device_ptr, &bbox_count, sizeof(int), cudaMemcpyHostToDevice);
#endif
  YoloBoxNum<<<number_of_blocks, threads_per_block, 0>>>(input_data,
                                                         bbox_count_device_ptr,
                                                         grid_size,
                                                         class_num,
                                                         anchors_num,
                                                         prob_thresh);
#ifdef PADDLE_WITH_HIP
  hipMemcpy(
      &bbox_count, bbox_count_device_ptr, sizeof(int), hipMemcpyDeviceToHost);
#else
  cudaMemcpy(
      &bbox_count, bbox_count_device_ptr, sizeof(int), cudaMemcpyDeviceToHost);
#endif

  // Record actual bbox number
  *bbox_count_host = bbox_count;

  // Obtain previous allocated bbox tensor in device side
  float* bbox_tensor = *bboxes_tensor_ptr;
  // Update previous maximum bbox number
  if (bbox_count > *bbox_count_max_alloc) {
#ifdef PADDLE_WITH_HIP
    hipFree(bbox_tensor);
    hipMalloc(&bbox_tensor, bbox_count * (5 + class_num) * sizeof(float));
#else
    cudaFree(bbox_tensor);
    cudaMalloc(&bbox_tensor, bbox_count * (5 + class_num) * sizeof(float));
#endif
    *bbox_count_max_alloc = bbox_count;
    *bboxes_tensor_ptr = bbox_tensor;
  }

  // Now generate bboxes
  int bbox_index = 0;
#ifdef PADDLE_WITH_HIP
  hipMemcpy(
      bbox_index_device_ptr, &bbox_index, sizeof(int), hipMemcpyHostToDevice);
#else
  cudaMemcpy(
      bbox_index_device_ptr, &bbox_index, sizeof(int), cudaMemcpyHostToDevice);
#endif
  YoloTensorParseKernel<<<number_of_blocks, threads_per_block, 0>>>(
      input_data,
      image_shape_data,
      image_scale_data,
      bbox_tensor,
      bbox_index_device_ptr,
      grid_size,
      class_num,
      anchors_num,
      netw,
      neth,
      biases_device,
      prob_thresh);
}

template <typename T, typename Context>
void YoloBoxPostKernel(const Context& dev_ctx,
                       const DenseTensor& boxes0,
                       const DenseTensor& boxes1,
                       const DenseTensor& boxes2,
                       const DenseTensor& image_shape,
                       const DenseTensor& image_scale,
                       const std::vector<int>& anchors0,
                       const std::vector<int>& anchors1,
                       const std::vector<int>& anchors2,
                       int class_num,
                       float conf_thresh,
                       int downsample_ratio0,
                       int downsample_ratio1,
                       int downsample_ratio2,
                       bool clip_bbox UNUSED,
                       float scale_x_y UNUSED,
                       float nms_threshold,
                       DenseTensor* out,
                       DenseTensor* nms_rois_num) {
  // prepare inputs
  std::vector<const float*> boxes_input(3);
  std::vector<std::vector<int32_t>> boxes_input_dims(3);
  const DenseTensor* boxes_tensor;
  for (int i = 0; i < 3; i++) {
    if (i == 0) {
      boxes_tensor = &boxes0;
    } else if (i == 1) {
      boxes_tensor = &boxes1;
    } else if (i == 2) {
      boxes_tensor = &boxes2;
    }
    boxes_input[i] = boxes_tensor->data<float>();
    auto dims = boxes_tensor->dims();
    for (int j = 0; j < dims.size(); j++) {
      boxes_input_dims[i].push_back(dims[j]);
    }
  }
  const float* image_shape_data = image_shape.data<float>();
  const float* image_scale_data = image_scale.data<float>();

  // prepare outputs
  auto* boxes_scores_tensor = out;
  auto* boxes_num_tensor = nms_rois_num;

  // prepare anchors
  std::vector<int32_t> anchors;
  anchors.insert(anchors.end(), anchors0.begin(), anchors0.end());
  anchors.insert(anchors.end(), anchors1.begin(), anchors1.end());
  anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
  int* device_anchors;
#ifdef PADDLE_WITH_HIP
  hipMalloc(reinterpret_cast<void**>(&device_anchors),
            anchors.size() * sizeof(int));
  hipMemcpy(device_anchors,
            anchors.data(),
            anchors.size() * sizeof(int),
            hipMemcpyHostToDevice);
#else
  cudaMalloc(reinterpret_cast<void**>(&device_anchors),
             anchors.size() * sizeof(int));
  cudaMemcpy(device_anchors,
             anchors.data(),
             anchors.size() * sizeof(int),
             cudaMemcpyHostToDevice);
#endif
  int* device_anchors_ptr[3];
  device_anchors_ptr[0] = device_anchors;
  device_anchors_ptr[1] = device_anchors_ptr[0] + anchors0.size();
  device_anchors_ptr[2] = device_anchors_ptr[1] + anchors1.size();
  std::vector<int> anchors_num{static_cast<int>(anchors0.size()) / 2,
                               static_cast<int>(anchors1.size()) / 2,
                               static_cast<int>(anchors2.size()) / 2};

  // prepare other attrs
  std::vector<int> downsample_ratio{
      downsample_ratio0, downsample_ratio1, downsample_ratio2};
  // clip_bbox and scale_x_y is not used now!

  int batch = image_shape.dims()[0];
  TensorInfo* ts_info = new TensorInfo[batch * boxes_input.size()];
  for (int i = 0; i < batch * static_cast<int>(boxes_input.size()); i++) {
#ifdef PADDLE_WITH_HIP
    hipMalloc(
        reinterpret_cast<void**>(&ts_info[i].bboxes_dev_ptr),
        ts_info[i].bbox_count_max_alloc * (5 + class_num) * sizeof(float));
#else
    cudaMalloc(
        reinterpret_cast<void**>(&ts_info[i].bboxes_dev_ptr),
        ts_info[i].bbox_count_max_alloc * (5 + class_num) * sizeof(float));
#endif
    ts_info[i].bboxes_host_ptr = reinterpret_cast<float*>(malloc(
        ts_info[i].bbox_count_max_alloc * (5 + class_num) * sizeof(float)));
#ifdef PADDLE_WITH_HIP
    hipMalloc(reinterpret_cast<void**>(&ts_info[i].bbox_count_device_ptr),
              sizeof(int));
#else
    cudaMalloc(reinterpret_cast<void**>(&ts_info[i].bbox_count_device_ptr),
               sizeof(int));
#endif
  }

  // Box index counter in gpu memory
  // *bbox_index_device_ptr used by atomicAdd
  int* bbox_index_device_ptr;
#ifdef PADDLE_WITH_HIP
  hipMalloc(reinterpret_cast<void**>(&bbox_index_device_ptr), sizeof(int));
#else
  cudaMalloc(reinterpret_cast<void**>(&bbox_index_device_ptr), sizeof(int));
#endif

  int total_bbox = 0;
  for (int batch_id = 0; batch_id < batch; batch_id++) {
    for (int input_id = 0; input_id < boxes_input.size(); input_id++) {
      int c = boxes_input_dims[input_id][1];
      int h = boxes_input_dims[input_id][2];
      int w = boxes_input_dims[input_id][3];
      int ts_id = batch_id * boxes_input.size() + input_id;
      int bbox_count_max_alloc = ts_info[ts_id].bbox_count_max_alloc;

      YoloTensorParseCuda(
          boxes_input[input_id] + batch_id * c * h * w,
          image_shape_data + batch_id * 2,
          image_scale_data + batch_id * 2,
          &(ts_info[ts_id].bboxes_dev_ptr),   // output in gpu,must use 2-level
                                              // pointer, because we may
                                              // re-malloc
          &bbox_count_max_alloc,              // bbox_count_alloc_ptr boxes we
                                              // pre-allocate
          &(ts_info[ts_id].bbox_count_host),  // record bbox numbers
          ts_info[ts_id].bbox_count_device_ptr,  // for atomicAdd
          bbox_index_device_ptr,                 // for atomicAdd
          h,
          class_num,
          anchors_num[input_id],
          downsample_ratio[input_id] * h,
          downsample_ratio[input_id] * w,
          device_anchors_ptr[input_id],
          conf_thresh);

      // batch info update
      if (bbox_count_max_alloc > ts_info[ts_id].bbox_count_max_alloc) {
        ts_info[ts_id].bbox_count_max_alloc = bbox_count_max_alloc;
        ts_info[ts_id].bboxes_host_ptr = reinterpret_cast<float*>(
            realloc(ts_info[ts_id].bboxes_host_ptr,
                    bbox_count_max_alloc * (5 + class_num) * sizeof(float)));
      }
// we need copy bbox_count_host boxes to cpu memory
#ifdef PADDLE_WITH_HIP
      hipMemcpyAsync(
          ts_info[ts_id].bboxes_host_ptr,
          ts_info[ts_id].bboxes_dev_ptr,
          ts_info[ts_id].bbox_count_host * (5 + class_num) * sizeof(float),
          hipMemcpyDeviceToHost);
#else
      cudaMemcpyAsync(
          ts_info[ts_id].bboxes_host_ptr,
          ts_info[ts_id].bboxes_dev_ptr,
          ts_info[ts_id].bbox_count_host * (5 + class_num) * sizeof(float),
          cudaMemcpyDeviceToHost);
#endif
      total_bbox += ts_info[ts_id].bbox_count_host;
    }
  }

  boxes_scores_tensor->Resize({total_bbox > 0 ? total_bbox : 1, 6});
  float* boxes_scores_data =
      dev_ctx.template HostAlloc<float>(boxes_scores_tensor);
  memset(boxes_scores_data, 0, sizeof(float) * 6);
  boxes_num_tensor->Resize({batch});
  int* boxes_num_data = dev_ctx.template HostAlloc<int>(boxes_num_tensor);
  int boxes_scores_id = 0;

  // NMS
  for (int batch_id = 0; batch_id < batch; batch_id++) {
    std::vector<Detection> bbox_det_vec;
    for (int input_id = 0; input_id < boxes_input.size(); input_id++) {
      int ts_id = batch_id * boxes_input.size() + input_id;
      int bbox_count = ts_info[ts_id].bbox_count_host;
      if (bbox_count <= 0) {
        continue;
      }

      float* bbox_host_ptr = ts_info[ts_id].bboxes_host_ptr;
      for (int bbox_index = 0; bbox_index < bbox_count; ++bbox_index) {
        Detection bbox_det;
        memset(&bbox_det, 0, sizeof(Detection));
        bbox_det.objectness = bbox_host_ptr[bbox_index * (5 + class_num) + 0];
        bbox_det.bbox.x = bbox_host_ptr[bbox_index * (5 + class_num) + 1];
        bbox_det.bbox.y = bbox_host_ptr[bbox_index * (5 + class_num) + 2];
        bbox_det.bbox.w =
            bbox_host_ptr[bbox_index * (5 + class_num) + 3] - bbox_det.bbox.x;
        bbox_det.bbox.h =
            bbox_host_ptr[bbox_index * (5 + class_num) + 4] - bbox_det.bbox.y;
        bbox_det.classes = class_num;
        bbox_det.prob =
            reinterpret_cast<float*>(malloc(class_num * sizeof(float)));
        int max_prob_class_id = -1;
        float max_class_prob = 0.0;
        for (int class_id = 0; class_id < class_num; class_id++) {
          float prob =
              bbox_host_ptr[bbox_index * (5 + class_num) + 5 + class_id];
          bbox_det.prob[class_id] = prob;
          if (prob > max_class_prob) {
            max_class_prob = prob;
            max_prob_class_id = class_id;
          }
        }
        bbox_det.max_prob_class_index = max_prob_class_id;
        bbox_det.sort_class = max_prob_class_id;
        bbox_det_vec.push_back(bbox_det);
      }
    }
    PostNMS(&bbox_det_vec, nms_threshold, class_num);
    for (int i = 0; i < bbox_det_vec.size(); i++) {
      boxes_scores_data[boxes_scores_id++] =
          bbox_det_vec[i].max_prob_class_index;
      boxes_scores_data[boxes_scores_id++] = bbox_det_vec[i].objectness;
      boxes_scores_data[boxes_scores_id++] = bbox_det_vec[i].bbox.x;
      boxes_scores_data[boxes_scores_id++] = bbox_det_vec[i].bbox.y;
      boxes_scores_data[boxes_scores_id++] =
          bbox_det_vec[i].bbox.w + bbox_det_vec[i].bbox.x;
      boxes_scores_data[boxes_scores_id++] =
          bbox_det_vec[i].bbox.h + bbox_det_vec[i].bbox.y;
      free(bbox_det_vec[i].prob);
    }
    boxes_num_data[batch_id] = bbox_det_vec.size();
  }

#ifdef PADDLE_WITH_HIP
  hipFree(bbox_index_device_ptr);
#else
  cudaFree(bbox_index_device_ptr);
#endif
  for (int i = 0; i < batch * boxes_input.size(); i++) {
#ifdef PADDLE_WITH_HIP
    hipFree(ts_info[i].bboxes_dev_ptr);
    hipFree(ts_info[i].bbox_count_device_ptr);
#else
    cudaFree(ts_info[i].bboxes_dev_ptr);
    cudaFree(ts_info[i].bbox_count_device_ptr);
#endif
    free(ts_info[i].bboxes_host_ptr);
  }
  delete[] ts_info;
}
}  // namespace phi

PD_REGISTER_KERNEL(
    yolo_box_post, GPU, ALL_LAYOUT, phi::YoloBoxPostKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
}
