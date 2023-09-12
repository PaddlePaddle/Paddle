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

#include "paddle/fluid/inference/tensorrt/plugin/yolo_box_head_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

inline __device__ float SigmoidGPU(const float& x) {
  return 1.0f / (1.0f + __expf(-x));
}

__global__ void YoloBoxHeadKernel(const float* input,
                                  float* output,
                                  const int grid_size_x,
                                  const int grid_size_y,
                                  const int class_num,
                                  const int anchors_num) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int y_id = blockIdx.y * blockDim.y + threadIdx.y;
  int z_id = blockIdx.z * blockDim.z + threadIdx.z;
  if ((x_id >= grid_size_x) || (y_id >= grid_size_y) || (z_id >= anchors_num)) {
    return;
  }
  const int grids_num = grid_size_x * grid_size_y;
  const int bbindex = y_id * grid_size_x + x_id;

  // objectness
  output[bbindex + grids_num * (z_id * (5 + class_num) + 4)] =
      SigmoidGPU(input[bbindex + grids_num * (z_id * (5 + class_num) + 4)]);
  // x
  output[bbindex + grids_num * (z_id * (5 + class_num) + 0)] =
      SigmoidGPU(input[bbindex + grids_num * (z_id * (5 + class_num) + 0)]);
  // y
  output[bbindex + grids_num * (z_id * (5 + class_num) + 1)] =
      SigmoidGPU(input[bbindex + grids_num * (z_id * (5 + class_num) + 1)]);
  // w
  output[bbindex + grids_num * (z_id * (5 + class_num) + 2)] =
      __expf(input[bbindex + grids_num * (z_id * (5 + class_num) + 2)]);
  // h
  output[bbindex + grids_num * (z_id * (5 + class_num) + 3)] =
      __expf(input[bbindex + grids_num * (z_id * (5 + class_num) + 3)]);
  // Probabilities of classes
  for (int i = 0; i < class_num; ++i) {
    output[bbindex + grids_num * (z_id * (5 + class_num) + (5 + i))] =
        SigmoidGPU(
            input[bbindex + grids_num * (z_id * (5 + class_num) + (5 + i))]);
  }
}

int YoloBoxHeadPlugin::enqueue(int batch_size,
                               const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
                               void** outputs,
#else
                               void* const* outputs,
#endif
                               void* workspace,
                               cudaStream_t stream) TRT_NOEXCEPT {
  const int h = input_dims_[0].d[1];
  const int w = input_dims_[0].d[2];
  const int grid_size_x = w;
  const int grid_size_y = h;
  const int anchors_num = anchors_.size() / 2;
  const float* input_data = static_cast<const float*>(inputs[0]);
  float* output_data = static_cast<float*>(outputs[0]);
  const int volume = input_dims_[0].d[0] * h * w;
  dim3 block(16, 16, 4);
  dim3 grid((grid_size_x / block.x) + 1,
            (grid_size_y / block.y) + 1,
            (anchors_num / block.z) + 1);
  for (int n = 0; n < batch_size; n++) {
    YoloBoxHeadKernel<<<grid, block, 0, stream>>>(input_data + n * volume,
                                                  output_data + n * volume,
                                                  grid_size_x,
                                                  grid_size_y,
                                                  class_num_,
                                                  anchors_num);
  }
  return 0;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
