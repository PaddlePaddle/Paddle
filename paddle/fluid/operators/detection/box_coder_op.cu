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

#include "paddle/fluid/operators/detection/box_coder_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void EncodeCenterSizeKernel(const T* prior_box_data,
                                       const T* prior_box_var_data,
                                       const T* target_box_data, const int row,
                                       const int col, const int len,
                                       const bool normalized, T* output) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < row * col) {
    const int row_idx = idx / col;
    const int col_idx = idx % col;
    T prior_box_width = prior_box_data[col_idx * len + 2] -
                        prior_box_data[col_idx * len] + (normalized == false);
    T prior_box_height = prior_box_data[col_idx * len + 3] -
                         prior_box_data[col_idx * len + 1] +
                         (normalized == false);
    T prior_box_center_x =
        (prior_box_data[col_idx * len + 2] + prior_box_data[col_idx * len]) / 2;
    T prior_box_center_y = (prior_box_data[col_idx * len + 3] +
                            prior_box_data[col_idx * len + 1]) /
                           2;

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
      output[idx * len] /= prior_box_var_data[col_idx * len];
      output[idx * len + 1] /= prior_box_var_data[col_idx * len + 1];
      output[idx * len + 2] /= prior_box_var_data[col_idx * len + 2];
      output[idx * len + 3] /= prior_box_var_data[col_idx * len + 3];
    }
  }
}

template <typename T>
__global__ void DecodeCenterSizeKernel(const T* prior_box_data,
                                       const T* prior_box_var_data,
                                       const T* target_box_data, const int row,
                                       const int col, const int len,
                                       const bool normalized, T* output) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < row * col) {
    const int col_idx = idx % col;
    T prior_box_width = prior_box_data[col_idx * len + 2] -
                        prior_box_data[col_idx * len] + (normalized == false);
    T prior_box_height = prior_box_data[col_idx * len + 3] -
                         prior_box_data[col_idx * len + 1] +
                         (normalized == false);
    T prior_box_center_x =
        (prior_box_data[col_idx * len + 2] + prior_box_data[col_idx * len]) / 2;
    T prior_box_center_y = (prior_box_data[col_idx * len + 3] +
                            prior_box_data[col_idx * len + 1]) /
                           2;
    T target_box_width, target_box_height;
    T target_box_center_x, target_box_center_y;
    if (prior_box_var_data) {
      target_box_width = exp(prior_box_var_data[col_idx * len + 2] *
                             target_box_data[idx * len + 2]) *
                         prior_box_width;
      target_box_height = exp(prior_box_var_data[col_idx * len + 3] *
                              target_box_data[idx * len + 3]) *
                          prior_box_height;
      target_box_center_x = prior_box_var_data[col_idx * len] *
                                target_box_data[idx * len] * prior_box_width +
                            prior_box_center_x;
      target_box_center_y = prior_box_var_data[col_idx * len + 1] *
                                target_box_data[idx * len + 1] *
                                prior_box_height +
                            prior_box_center_y;
    } else {
      target_box_width = exp(target_box_data[idx * len + 2]) * prior_box_width;
      target_box_height =
          exp(target_box_data[idx * len + 3]) * prior_box_height;
      target_box_center_x =
          target_box_data[idx * len] * prior_box_width + prior_box_center_x;
      target_box_center_y = target_box_data[idx * len + 1] * prior_box_height +
                            prior_box_center_y;
    }

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

    const T* prior_box_data = prior_box->data<T>();
    const T* target_box_data = target_box->data<T>();
    const T* prior_box_var_data = nullptr;
    if (prior_box_var) prior_box_var_data = prior_box_var->data<T>();

    if (target_box->lod().size()) {
      PADDLE_ENFORCE_EQ(target_box->lod().size(), 1,
                        "Only support 1 level of LoD.");
    }
    auto row = target_box->dims()[0];
    auto col = prior_box->dims()[0];
    auto len = prior_box->dims()[1];
    int block = 512;
    int grid = (row * col + block - 1) / block;
    auto& device_ctx = context.cuda_device_context();

    output_box->mutable_data<T>({row, col, len}, context.GetPlace());
    T* output = output_box->data<T>();

    auto code_type = GetBoxCodeType(context.Attr<std::string>("code_type"));
    bool normalized = context.Attr<bool>("box_normalized");
    if (code_type == BoxCodeType::kEncodeCenterSize) {
      EncodeCenterSizeKernel<T><<<grid, block, 0, device_ctx.stream()>>>(
          prior_box_data, prior_box_var_data, target_box_data, row, col, len,
          normalized, output);
    } else if (code_type == BoxCodeType::kDecodeCenterSize) {
      DecodeCenterSizeKernel<T><<<grid, block, 0, device_ctx.stream()>>>(
          prior_box_data, prior_box_var_data, target_box_data, row, col, len,
          normalized, output);
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
