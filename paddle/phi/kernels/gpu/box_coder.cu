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

#include "paddle/phi/kernels/box_coder_kernel.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/box_coder.h"

namespace phi {

template <typename T>
__global__ void EncodeCenterSizeKernel(const T *prior_box_data,
                                       const T *prior_box_var_data,
                                       const T *target_box_data,
                                       const int row,
                                       const int col,
                                       const int len,
                                       const bool normalized,
                                       const T prior_box_var_size,
                                       const float *variance,
                                       const int var_size,
                                       T *output) {
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
__global__ void DecodeCenterSizeKernel(const T *prior_box_data,
                                       const T *prior_box_var_data,
                                       const T *target_box_data,
                                       const int row,
                                       const int col,
                                       const int len,
                                       const bool normalized,
                                       const T prior_box_var_size,
                                       const float *variance,
                                       const int var_size,
                                       const int axis,
                                       T *output) {
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

template <typename T, typename Context>
void BoxCoderKernel(const Context &dev_ctx,
                    const DenseTensor &prior_box,
                    const paddle::optional<DenseTensor> &prior_box_var,
                    const DenseTensor &target_box,
                    const std::string &code_type_str,
                    bool normalized,
                    int axis,
                    const std::vector<float> &variance,
                    DenseTensor *output_box) {
  const T *prior_box_data = prior_box.template data<T>();
  const T *target_box_data = target_box.template data<T>();
  const T *prior_box_var_data = nullptr;
  auto prior_box_var_size = 0;
  if (prior_box_var) {
    PADDLE_ENFORCE_EQ(variance.empty(),
                      true,
                      phi::errors::InvalidArgument(
                          "Input 'PriorBoxVar' and attribute 'variance'"
                          " of BoxCoder operator should not be used at the "
                          "same time."));
    prior_box_var_data = prior_box_var->data<T>();
    prior_box_var_size = prior_box_var->dims().size();
  }
  if (!(variance.empty())) {
    PADDLE_ENFORCE_EQ(static_cast<int>(variance.size()),
                      4,
                      phi::errors::InvalidArgument(
                          "Size of attribute 'variance' in BoxCoder operator"
                          " should be 4. But received size is %d",
                          variance.size()));
  }

  if (target_box.lod().size()) {
    PADDLE_ENFORCE_EQ(target_box.lod().size(),
                      1,
                      phi::errors::InvalidArgument(
                          "Input 'TargetBox' of BoxCoder operator only"
                          " supports LoD with one level."));
  }
  const int var_size = static_cast<int>(variance.size());
  auto code_type = phi::funcs::GetBoxCodeType(code_type_str);
  auto row = target_box.dims()[0];
  auto col = prior_box.dims()[0];
  if (code_type == phi::funcs::BoxCodeType::kDecodeCenterSize) {
    col = target_box.dims()[1];
  }
  auto len = prior_box.dims()[1];
  int block = 512;
  int grid = (row * col + block - 1) / block;

  int bytes = var_size * sizeof(float);
  auto dev_var = paddle::memory::Alloc(
      dev_ctx.GetPlace(),
      bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  float *dev_var_data = reinterpret_cast<float *>(dev_var->ptr());
  auto cplace = phi::CPUPlace();
  const auto gplace = dev_ctx.GetPlace();
  paddle::memory::Copy(
      gplace, dev_var_data, cplace, &variance[0], bytes, dev_ctx.stream());

  output_box->Resize({row, col, len});
  dev_ctx.template Alloc<T>(output_box);
  T *output = output_box->data<T>();

  if (code_type == phi::funcs::BoxCodeType::kEncodeCenterSize) {
    EncodeCenterSizeKernel<T>
        <<<grid, block, 0, dev_ctx.stream()>>>(prior_box_data,
                                               prior_box_var_data,
                                               target_box_data,
                                               row,
                                               col,
                                               len,
                                               normalized,
                                               prior_box_var_size,
                                               dev_var_data,
                                               var_size,
                                               output);
  } else if (code_type == phi::funcs::BoxCodeType::kDecodeCenterSize) {
    DecodeCenterSizeKernel<T>
        <<<grid, block, 0, dev_ctx.stream()>>>(prior_box_data,
                                               prior_box_var_data,
                                               target_box_data,
                                               row,
                                               col,
                                               len,
                                               normalized,
                                               prior_box_var_size,
                                               dev_var_data,
                                               var_size,
                                               axis,
                                               output);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    box_coder, GPU, ALL_LAYOUT, phi::BoxCoderKernel, float, double) {}
