/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/detection/box_coder_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void EncodeCenterSizeKernel(
    const T* prior_box_data, const T* prior_box_var_data,
    const T* target_box_data, const int row, const int col, const int len,
    const bool normalized, const T prior_box_var_size, const float* variance,
    const int var_size, T* output) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < row * col) {
    const int row_idx = idx / col;
    const int col_idx = idx % col;
    T prior_box_width = prior_box_data[col_idx * len + 2] -
                        prior_box_data[col_idx * len] + (normalized == false);
    T prior_box_height = prior_box_data[col_idx * len + 3] -
                         prior_box_data[col_idx * len + 1] +
                         (normalized == false);
    T prior_box_center_x = prior_box_data[col_idx * len] + prior_box_width / 2;
    T prior_box_center_y =
        prior_box_data[col_idx * len + 1] + prior_box_height / 2;

    T target_box_center_x =
        (target_box_data[row_idx * len + 2] + target_box_data[row_idx * len]) /
        2;
    T target_box_center_y = (target_box_data[row_idx * len + 3] +
                             target_box_data[row_idx * len + 1]) /
                            2;
    T target_box_width = target_box_data[row_idx * len + 2] -
                         target_box_data[row_idx * len] + (normalized == false);
    T target_box_height = target_box_data[row_idx * len + 3] -
                          target_box_data[row_idx * len + 1] +
                          (normalized == false);

    output[idx * len] =
        (target_box_center_x - prior_box_center_x) / prior_box_width;
    output[idx * len + 1] =
        (target_box_center_y - prior_box_center_y) / prior_box_height;
    output[idx * len + 2] = log(fabs(target_box_width / prior_box_width));
    output[idx * len + 3] = log(fabs(target_box_height / prior_box_height));
    if (prior_box_var_data) {
      int prior_var_offset = col_idx * len;
      output[idx * len] /= prior_box_var_data[prior_var_offset];
      output[idx * len + 1] /= prior_box_var_data[prior_var_offset + 1];
      output[idx * len + 2] /= prior_box_var_data[prior_var_offset + 2];
      output[idx * len + 3] /= prior_box_var_data[prior_var_offset + 3];
    } else if (var_size == 4) {
      for (int k = 0; k < 4; ++k) {
        output[idx * len + k] /= static_cast<T>(variance[k]);
      }
    }
  }
}

template <typename T>
__global__ void DecodeCenterSizeKernel(
    const T* prior_box_data, const T* prior_box_var_data,
    const T* target_box_data, const int row, const int col, const int len,
    const bool normalized, const T prior_box_var_size, const float* variance,
    const int var_size, const int axis, T* output) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int prior_box_offset = 0;
  if (idx < row * col) {
    const int col_idx = idx % col;
    const int row_idx = idx / col;
    prior_box_offset = axis == 0 ? col_idx * len : row_idx * len;
    T prior_box_width = prior_box_data[prior_box_offset + 2] -
                        prior_box_data[prior_box_offset] +
                        (normalized == false);
    T prior_box_height = prior_box_data[prior_box_offset + 3] -
                         prior_box_data[prior_box_offset + 1] +
                         (normalized == false);
    T prior_box_center_x =
        prior_box_data[prior_box_offset] + prior_box_width / 2;
    T prior_box_center_y =
        prior_box_data[prior_box_offset + 1] + prior_box_height / 2;
    T target_box_width, target_box_height;
    T target_box_center_x, target_box_center_y;
    T box_var_x = T(1), box_var_y = T(1);
    T box_var_w = T(1), box_var_h = T(1);
    if (prior_box_var_data) {
      int prior_var_offset = axis == 0 ? col_idx * len : row_idx * len;
      box_var_x = prior_box_var_data[prior_var_offset];
      box_var_y = prior_box_var_data[prior_var_offset + 1];
      box_var_w = prior_box_var_data[prior_var_offset + 2];
      box_var_h = prior_box_var_data[prior_var_offset + 3];
    } else if (var_size == 4) {
      box_var_x = static_cast<T>(variance[0]);
      box_var_y = static_cast<T>(variance[1]);
      box_var_w = static_cast<T>(variance[2]);
      box_var_h = static_cast<T>(variance[3]);
    }
    target_box_width =
        exp(box_var_w * target_box_data[idx * len + 2]) * prior_box_width;
    target_box_height =
        exp(box_var_h * target_box_data[idx * len + 3]) * prior_box_height;
    target_box_center_x =
        box_var_x * target_box_data[idx * len] * prior_box_width +
        prior_box_center_x;
    target_box_center_y =
        box_var_y * target_box_data[idx * len + 1] * prior_box_height +
        prior_box_center_y;

    output[idx * len] = target_box_center_x - target_box_width / 2;
    output[idx * len + 1] = target_box_center_y - target_box_height / 2;
    output[idx * len + 2] =
        target_box_center_x + target_box_width / 2 - (normalized == false);
    output[idx * len + 3] =
        target_box_center_y + target_box_height / 2 - (normalized == false);
  }
}

template <typename DeviceContext, typename T>
class BoxCoderCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto* prior_box = context.Input<framework::Tensor>("PriorBox");
    auto* prior_box_var = context.Input<framework::Tensor>("PriorBoxVar");
    auto* target_box = context.Input<framework::LoDTensor>("TargetBox");
    auto* output_box = context.Output<framework::Tensor>("OutputBox");
    std::vector<float> variance = context.Attr<std::vector<float>>("variance");
    const T* prior_box_data = prior_box->data<T>();
    const T* target_box_data = target_box->data<T>();
    const T* prior_box_var_data = nullptr;
    auto prior_box_var_size = 0;
    if (prior_box_var) {
      PADDLE_ENFORCE(variance.empty(),
                     "Input 'PriorBoxVar' and attribute 'variance' should not"
                     "be used at the same time.");
      prior_box_var_data = prior_box_var->data<T>();
      prior_box_var_size = prior_box_var->dims().size();
    }
    if (!(variance.empty())) {
      PADDLE_ENFORCE(static_cast<int>(variance.size()) == 4,
                     "Size of attribute 'variance' should be 4");
    }

    if (target_box->lod().size()) {
      PADDLE_ENFORCE_EQ(target_box->lod().size(), 1,
                        "Only support 1 level of LoD.");
    }
    const int var_size = static_cast<int>(variance.size());

    auto code_type = GetBoxCodeType(context.Attr<std::string>("code_type"));
    bool normalized = context.Attr<bool>("box_normalized");
    int axis = context.Attr<int>("axis");

    auto row = target_box->dims()[0];
    auto col = prior_box->dims()[0];
    if (code_type == BoxCodeType::kDecodeCenterSize) {
      col = target_box->dims()[1];
    }
    auto len = prior_box->dims()[1];
    int block = 512;
    int grid = (row * col + block - 1) / block;
    auto& device_ctx = context.cuda_device_context();

    auto& allocator =
        platform::DeviceTemporaryAllocator::Instance().Get(device_ctx);
    int bytes = var_size * sizeof(float);
    auto dev_var = allocator.Allocate(bytes);
    float* dev_var_data = reinterpret_cast<float*>(dev_var->ptr());
    auto cplace = platform::CPUPlace();
    const auto gplace = boost::get<platform::CUDAPlace>(context.GetPlace());
    memory::Copy(gplace, dev_var_data, cplace, &variance[0], bytes,
                 device_ctx.stream());

    output_box->mutable_data<T>({row, col, len}, context.GetPlace());
    T* output = output_box->data<T>();

    if (code_type == BoxCodeType::kEncodeCenterSize) {
      EncodeCenterSizeKernel<T><<<grid, block, 0, device_ctx.stream()>>>(
          prior_box_data, prior_box_var_data, target_box_data, row, col, len,
          normalized, prior_box_var_size, dev_var_data, var_size, output);
    } else if (code_type == BoxCodeType::kDecodeCenterSize) {
      DecodeCenterSizeKernel<T><<<grid, block, 0, device_ctx.stream()>>>(
          prior_box_data, prior_box_var_data, target_box_data, row, col, len,
          normalized, prior_box_var_size, dev_var_data, var_size, axis, output);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    box_coder,
    ops::BoxCoderCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::BoxCoderCUDAKernel<paddle::platform::CUDADeviceContext, double>);
