// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/kernels/arm/concat_compute.h"
#include <string>
#include <vector>
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

std::vector<size_t> stride_numel(const DDim& ddim) {
  std::vector<size_t> strides(ddim.size());
  strides[ddim.size() - 1] = ddim[ddim.size() - 1];
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i];
  }
  return strides;
}

void ConcatCompute::Run() {
  auto& param = Param<operators::ConcatParam>();
  std::vector<lite::Tensor*> inputs = param.x;
  auto* out = param.output;
  int axis = param.axis;
  out->mutable_data<float>();

  /// Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && inputs.size() < 10) {
    size_t output_offset = 0;
    for (auto* in : inputs) {
      auto in_stride = stride_numel(in->dims());
      auto out_stride = stride_numel(out->dims());
      void* dst = out->mutable_data<float>() + output_offset;
      const void* src = in->data<float>();
#if 0
      LOG(INFO) << "out_stride.size():" << out_stride.size();
      LOG(INFO) << "out_stride[0]" << out_stride[0];
      for (int i=0; i < out_stride.size(); ++i) {
        LOG(INFO) << "out_stride[" << i << "]:" << out_stride[i];
      }
      LOG(INFO) << "in_stride.size():" << in_stride.size();
      for (int i=0; i < in_stride.size(); ++i) {
        LOG(INFO) << "in_stride[" << i << "]:" << in_stride[i];
      }
#endif
      // src and dst tensor should have the same dims size.
      CHECK(in_stride.size() == out_stride.size());
      std::memcpy(dst, src, sizeof(float) * in_stride[0]);
      output_offset += in_stride[0];
    }
  } else {
    std::vector<lite::Tensor*> inputs_concat(inputs.size());
    for (int j = 0; j < inputs.size(); ++j) {
      inputs_concat[j] = inputs[j];
    }
    lite::arm::math::concat_func(inputs_concat, axis, out);
  }
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(concat, kARM, kFloat, kNCHW,
                     paddle::lite::kernels::arm::ConcatCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
