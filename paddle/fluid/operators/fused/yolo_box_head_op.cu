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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

template <typename T>
inline __device__ T SigmoidGPU(const T& x) {
  return 1.0f / (1.0f + __expf(-x));
}

template <typename T>
__global__ void YoloBoxHeadCudaKernel(const T* input,
                                      T* output,
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

template <typename T>
class YoloBoxHeadKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using Tensor = phi::DenseTensor;
    auto* x = context.Input<phi::DenseTensor>("X");
    auto* out = context.Output<phi::DenseTensor>("Out");
    auto anchors = context.Attr<std::vector<int>>("anchors");
    auto class_num = context.Attr<int>("class_num");
    auto& device_ctx = context.template device_context<phi::GPUContext>();
    auto x_dims = x->dims();
    const int batch_size = x_dims[0];
    const int h = x_dims[2];
    const int w = x_dims[3];
    const int grid_size_x = w;
    const int grid_size_y = h;
    const int anchors_num = anchors.size() / 2;
    const T* input_data = x->data<T>();
    T* output_data = device_ctx.Alloc<T>(out, out->numel() * sizeof(T));
    auto stream = device_ctx.stream();
    const int volume = x_dims[1] * h * w;
    dim3 block(16, 16, 4);
    dim3 grid((grid_size_x / block.x) + 1,
              (grid_size_y / block.y) + 1,
              (anchors_num / block.z) + 1);
    for (int n = 0; n < batch_size; n++) {
      YoloBoxHeadCudaKernel<<<grid, block, 0, stream>>>(
          input_data + n * volume,
          output_data + n * volume,
          grid_size_x,
          grid_size_y,
          class_num,
          anchors_num);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(yolo_box_head, ops::YoloBoxHeadKernel<float>);
