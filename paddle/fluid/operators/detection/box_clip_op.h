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

#pragma once
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detection/bbox_util.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class BoxClipKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input_box = context.Input<LoDTensor>("Input");
    auto* im_info = context.Input<LoDTensor>("ImInfo");
    auto* output_box = context.Output<LoDTensor>("Output");
    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    output_box->mutable_data<T>(context.GetPlace());
    if (input_box->lod().size()) {
      PADDLE_ENFORCE_EQ(input_box->lod().size(), 1UL,
                        platform::errors::InvalidArgument(
                            "Input(Input) of BoxClip only supports 1 level "
                            "of LoD. But received the "
                            "level = %d",
                            input_box->lod().size()));
    }
    auto box_lod = input_box->lod().back();
    int64_t n = static_cast<int64_t>(box_lod.size() - 1);
    for (int i = 0; i < n; ++i) {
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      Tensor box_slice = input_box->Slice(box_lod[i], box_lod[i + 1]);
      Tensor output_slice = output_box->Slice(box_lod[i], box_lod[i + 1]);
      ClipTiledBoxes<T>(dev_ctx, im_info_slice, box_slice, &output_slice);
    }
  }
};

}  // namespace operators
}  // namespace paddle
